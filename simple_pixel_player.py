#!/usr/bin/env python3
"""
Simplified Pixel Art Video Player
- Auto resolution based on terminal size
- Fixed optimized color palette
- No CRT effects
- Better video quality
"""

import cv2
import numpy as np
import time
import pygame
import shutil
import sys
from pathlib import Path
from typing import Optional, Tuple
import threading
import queue

class SimplePixelPlayer:
    """Simplified pixel art video player with sensible defaults"""
    
    # Optimized 32-color palette for good video representation
    PALETTE = np.array([
        # Grayscale (8 levels)
        [0, 0, 0],       [36, 36, 36],    [72, 72, 72],    [109, 109, 109],
        [145, 145, 145], [182, 182, 182], [218, 218, 218], [255, 255, 255],
        # Primary colors
        [255, 0, 0],     [0, 255, 0],     [0, 0, 255],     
        # Secondary colors  
        [255, 255, 0],   [255, 0, 255],   [0, 255, 255],
        # Warm tones
        [255, 128, 0],   [255, 192, 128], [139, 69, 19],   [205, 133, 63],
        # Cool tones
        [70, 130, 180],  [100, 149, 237], [0, 128, 128],   [46, 139, 87],
        # Skin tones
        [255, 220, 177], [255, 206, 180], [222, 171, 127], [188, 143, 107],
        # Additional colors
        [128, 0, 128],   [128, 128, 0],   [0, 128, 0],     [128, 0, 0],
        [0, 0, 128],     [64, 64, 64]
    ], dtype=np.uint8)
    
    def __init__(self):
        # Get terminal size and calculate optimal resolution
        self.term_cols, self.term_rows = shutil.get_terminal_size((80, 24))
        
        # Use reasonable resolution that won't overwhelm the terminal
        # Leave margin for UI and prevent overflow
        self.width = min(120, self.term_cols - 4)  # Reasonable max width
        self.height = min(40, (self.term_rows - 4) * 2)  # Account for half-blocks
        
        # Set minimum resolution for quality
        if self.width < 60:
            self.width = 60
        if self.height < 30:
            self.height = 30
            
        print(f"🎮 Resolution: {self.width}x{self.height // 2} characters")
        print(f"   Effective pixels: {self.width}x{self.height} (with half-blocks)")
        
        # Build color lookup for fast processing
        self._build_color_cache()
        
        # Frame timing - use lower FPS for terminal stability
        self.target_fps = 10  # Reduced for better terminal handling
        self.frame_time = 1.0 / self.target_fps
        
    def _build_color_cache(self):
        """Build KD-tree for fast color matching"""
        from scipy.spatial import KDTree
        self.color_tree = KDTree(self.PALETTE)
        
    def find_closest_color(self, rgb: np.ndarray) -> np.ndarray:
        """Find closest palette color"""
        _, idx = self.color_tree.query(rgb)
        return self.PALETTE[idx]
    
    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """Process a frame with simplified pipeline"""
        # 1. Resize to target resolution
        processed = cv2.resize(frame, (self.width, self.height), 
                               interpolation=cv2.INTER_AREA)
        
        # 2. Apply simple dithering (ordered dithering for speed)
        processed = self.apply_dithering(processed)
        
        # 3. Quantize to palette
        h, w = processed.shape[:2]
        processed_flat = processed.reshape(-1, 3)
        _, indices = self.color_tree.query(processed_flat)
        processed = self.PALETTE[indices].reshape(h, w, 3)
        
        return processed
    
    def apply_dithering(self, image: np.ndarray) -> np.ndarray:
        """Apply simple ordered dithering using vectorized operations"""
        # 4x4 Bayer matrix for ordered dithering
        bayer = np.array([
            [0, 8, 2, 10],
            [12, 4, 14, 6],
            [3, 11, 1, 9],
            [15, 7, 13, 5]
        ], dtype=np.float32) / 16.0 * 32  # Scale for our color space
        
        h, w = image.shape[:2]
        
        # Tile the Bayer matrix to match image dimensions
        bayer_tiled = np.tile(bayer, (h // 4 + 1, w // 4 + 1))[:h, :w]
        
        # Expand dimensions to match RGB channels
        bayer_tiled = np.expand_dims(bayer_tiled, axis=2)
        
        # Apply dithering in one vectorized operation
        dithered = image.astype(np.float32) + bayer_tiled - 16
        
        return np.clip(dithered, 0, 255).astype(np.uint8)
    
    def render_frame(self, frame: np.ndarray) -> str:
        """Render frame as colored half-blocks"""
        h, w = frame.shape[:2]
        output = []
        
        # Process two rows at a time for half-blocks
        for y in range(0, h, 2):
            row = []
            for x in range(w):
                # Get top and bottom pixels
                top = frame[y, x] if y < h else np.zeros(3)
                bottom = frame[y + 1, x] if y + 1 < h else np.zeros(3)
                
                # Create half-block character with colors
                char = self._colored_half_block(top, bottom)
                row.append(char)
            output.append(''.join(row))
        
        return '\n'.join(output)
    
    def _colored_half_block(self, top: np.ndarray, bottom: np.ndarray) -> str:
        """Create colored half-block character"""
        # Ensure RGB values are in valid range
        top = np.clip(top, 0, 255).astype(int)
        bottom = np.clip(bottom, 0, 255).astype(int)
        
        # Check if pixels are similar
        if np.allclose(top, bottom, rtol=0.1):
            r, g, b = top
            return f'\x1b[38;2;{r};{g};{b}m█\x1b[0m'
        
        tr, tg, tb = top
        br, bg, bb = bottom
        
        # Optimize for black pixels
        if np.mean(top) < 10 and np.mean(bottom) < 10:
            return ' '
        elif np.mean(top) < 10:
            return f'\x1b[38;2;{br};{bg};{bb}m▄\x1b[0m'
        elif np.mean(bottom) < 10:
            return f'\x1b[38;2;{tr};{tg};{tb}m▀\x1b[0m'
        else:
            # Use upper half block with foreground and background colors
            return f'\x1b[38;2;{tr};{tg};{tb}m\x1b[48;2;{br};{bg};{bb}m▀\x1b[0m'
    
    def play_video(self, video_path: str):
        """Play a video file"""
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            print(f"❌ Error: Cannot open video {video_path}")
            return
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps if fps > 0 else 0
        
        print(f"📹 Video: {frame_count} frames, {fps:.1f} fps, {duration:.1f}s")
        print(f"🎮 Playing at {self.target_fps} fps")
        print("   Press Ctrl+C to stop\n")
        
        # Extract and play audio in background
        audio_thread = self.start_audio(video_path)
        
        # Calculate frame skip for target FPS
        frame_skip = max(1, int(fps / self.target_fps))
        
        try:
            frame_num = 0
            start_time = time.time()
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Skip frames to match target FPS
                if frame_num % frame_skip != 0:
                    frame_num += 1
                    continue
                
                # Convert BGR to RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Process frame
                processed = self.process_frame(frame)
                
                # Render and display
                rendered = self.render_frame(processed)
                
                # Clear screen and display with better buffering
                sys.stdout.write('\x1b[2J\x1b[H' + rendered)
                sys.stdout.flush()
                
                # Frame timing
                elapsed = time.time() - start_time
                target_time = (frame_num / frame_skip) * self.frame_time
                if target_time > elapsed:
                    time.sleep(target_time - elapsed)
                
                frame_num += 1
                
        except KeyboardInterrupt:
            print("\n\n⏹️  Stopped")
        finally:
            cap.release()
            self.stop_audio()
    
    def start_audio(self, video_path: str) -> Optional[threading.Thread]:
        """Start audio playback in background"""
        try:
            # Extract audio to temp file
            import tempfile
            import subprocess
            
            audio_file = tempfile.mktemp(suffix='.mp3')
            cmd = [
                'ffmpeg', '-i', video_path,
                '-vn', '-acodec', 'mp3',
                '-ab', '192k', '-y',
                audio_file
            ]
            subprocess.run(cmd, capture_output=True, check=True)
            
            # Play audio
            def play_audio():
                pygame.mixer.init()
                pygame.mixer.music.load(audio_file)
                pygame.mixer.music.play()
                while pygame.mixer.music.get_busy():
                    time.sleep(0.1)
                Path(audio_file).unlink(missing_ok=True)
            
            thread = threading.Thread(target=play_audio)
            thread.start()
            return thread
            
        except Exception as e:
            print(f"⚠️  Audio disabled: {e}")
            return None
    
    def stop_audio(self):
        """Stop audio playback"""
        try:
            if pygame.mixer.get_init():
                pygame.mixer.music.stop()
                pygame.mixer.quit()
        except:
            pass

class VideoDownloader:
    """Simple YouTube downloader"""
    
    @staticmethod
    def download(url: str, quality: str = "720p") -> str:
        """Download video from YouTube"""
        import yt_dlp
        import tempfile
        
        output_dir = tempfile.gettempdir()
        
        ydl_opts = {
            'format': f'best[height<={quality[:-1]}][ext=mp4]/best[height<={quality[:-1]}]/best',
            'outtmpl': f'{output_dir}/%(title)s.%(ext)s',
            'quiet': False,
            'no_warnings': False,
        }
        
        print(f"⬇️  Downloading video...")
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            filename = ydl.prepare_filename(info)
            
            # Check for mp4 extension
            if not filename.endswith('.mp4'):
                base = filename.rsplit('.', 1)[0]
                if Path(f"{base}.mp4").exists():
                    filename = f"{base}.mp4"
        
        print(f"✅ Downloaded: {filename}")
        return filename