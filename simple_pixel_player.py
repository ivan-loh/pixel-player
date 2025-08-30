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
import signal
import atexit
from pathlib import Path
from typing import Optional, Tuple
import threading



class SimplePixelPlayer:
    """Simplified pixel art video player with sensible defaults"""
    
    def __init__(self, use_true_color=False):
        self.use_true_color = use_true_color
        
        # Default is adaptive 8-bit (256 colors)
        if not self.use_true_color:
            self.palette_size = 256
            self.adaptive_palette = True
            # Generate initial 256-color palette
            self.current_palette = self._generate_256_palette()
        else:
            # True color mode doesn't use palette
            self.palette_size = 0
            self.adaptive_palette = False
            self.current_palette = None
        
        self.frame_buffer = []
        
        self.term_cols, self.term_rows = shutil.get_terminal_size((80, 24))
        
        self.width = min(120, self.term_cols - 4)
        self.height = min(40, (self.term_rows - 4) * 2)
        
        if self.width < 60:
            self.width = 60
        if self.height < 30:
            self.height = 30
            
        print(f"🎮 Resolution: {self.width}x{self.height // 2} characters")
        print(f"   Effective pixels: {self.width}x{self.height} (with half-blocks)")
        
        if self.use_true_color:
            print(f"🎨 Color mode: True Color (16.7M colors)")
        else:
            print(f"🎨 Color mode: Adaptive 8-bit (256 colors)")
        
        if not self.use_true_color:
            self._build_color_cache()
        
        self.target_fps = 10
        self.frame_time = 1.0 / self.target_fps
        
    
    
    def _generate_256_palette(self):
        """Generate a well-distributed 256-color palette"""
        # Create a 6-6-6 RGB cube (216 colors) + 40 grayscale
        palette = []
        
        # RGB cube: 6 levels per channel = 216 colors
        for r in range(6):
            for g in range(6):
                for b in range(6):
                    palette.append([r * 51, g * 51, b * 51])
        
        # Add 40 grayscale levels for smooth gradients
        for i in range(40):
            gray = int(i * 255 / 39)
            palette.append([gray, gray, gray])
        
        return np.array(palette[:256], dtype=np.uint8)
    
    
    
    def _build_color_cache(self):
        from scipy.spatial import KDTree
        self.color_tree = KDTree(self.current_palette)
        
    
    
    def find_closest_color(self, rgb: np.ndarray) -> np.ndarray:
        _, idx = self.color_tree.query(rgb)
        return self.current_palette[idx]
    
    
    
    def generate_adaptive_palette(self, frame: np.ndarray):
        try:
            from sklearn.cluster import MiniBatchKMeans
        except ImportError:
            return
        
        h, w = frame.shape[:2]
        sample_size = min(10000, h * w)
        pixels = frame.reshape(-1, 3)
        
        indices = np.random.choice(pixels.shape[0], sample_size, replace=False)
        sample_pixels = pixels[indices]
        
        n_colors = self.palette_size
        # Use 5 iterations for 256 colors
        kmeans = MiniBatchKMeans(n_clusters=n_colors, random_state=42, n_init=5)
        kmeans.fit(sample_pixels)
        
        self.current_palette = kmeans.cluster_centers_.astype(np.uint8)
        self._build_color_cache()
    
    
    
    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        processed = cv2.resize(frame, (self.width, self.height), 
                               interpolation=cv2.INTER_AREA)
        
        if self.use_true_color:
            return processed
        
        if self.adaptive_palette and hasattr(self, 'frame_count'):
            # Update palette every 15 frames
            if self.frame_count % 15 == 0:
                self.generate_adaptive_palette(processed)
        
        processed = self.apply_dithering(processed)
        
        h, w = processed.shape[:2]
        processed_flat = processed.reshape(-1, 3)
        _, indices = self.color_tree.query(processed_flat)
        processed = self.current_palette[indices].reshape(h, w, 3)
        
        return processed
    
    
    
    def apply_dithering(self, image: np.ndarray) -> np.ndarray:
        # 4x4 Bayer matrix for ordered dithering
        bayer = np.array([
            [0, 8, 2, 10],
            [12, 4, 14, 6],
            [3, 11, 1, 9],
            [15, 7, 13, 5]
        ], dtype=np.float32) / 16.0 * 32
        
        h, w = image.shape[:2]
        bayer_tiled = np.tile(bayer, (h // 4 + 1, w // 4 + 1))[:h, :w]
        bayer_tiled = np.expand_dims(bayer_tiled, axis=2)
        
        dithered = image.astype(np.float32) + bayer_tiled - 16
        
        return np.clip(dithered, 0, 255).astype(np.uint8)
    
    
    
    def render_frame(self, frame: np.ndarray) -> str:
        h, w = frame.shape[:2]
        output = []
        
        for y in range(0, h, 2):
            row = []
            for x in range(w):
                top = frame[y, x] if y < h else np.zeros(3)
                bottom = frame[y + 1, x] if y + 1 < h else np.zeros(3)
                
                char = self._colored_half_block(top, bottom)
                row.append(char)
            output.append(''.join(row))
        
        return '\n'.join(output)
    
    
    
    def _colored_half_block(self, top: np.ndarray, bottom: np.ndarray) -> str:
        top = np.clip(top, 0, 255).astype(int)
        bottom = np.clip(bottom, 0, 255).astype(int)
        
        if np.allclose(top, bottom, rtol=0.1):
            r, g, b = top
            return f'\x1b[38;2;{r};{g};{b}m█\x1b[0m'
        
        tr, tg, tb = top
        br, bg, bb = bottom
        
        if np.mean(top) < 10 and np.mean(bottom) < 10:
            return ' '
        elif np.mean(top) < 10:
            return f'\x1b[38;2;{br};{bg};{bb}m▄\x1b[0m'
        elif np.mean(bottom) < 10:
            return f'\x1b[38;2;{tr};{tg};{tb}m▀\x1b[0m'
        else:
            return f'\x1b[38;2;{tr};{tg};{tb}m\x1b[48;2;{br};{bg};{bb}m▀\x1b[0m'
    
    
    
    def play_video(self, video_path: str):
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
        
        audio_thread, audio_started = self.start_audio(video_path)
        
        frame_duration_ms = 1000.0 / fps
        
        # Setup cleanup handler
        def cleanup_terminal():
            sys.stdout.write('\x1b[?25h')    # Show cursor
            sys.stdout.write('\x1b[?1049l')  # Return to main screen buffer
            sys.stdout.flush()
        
        # Register cleanup for unexpected exits
        atexit.register(cleanup_terminal)
        signal.signal(signal.SIGTERM, lambda s, f: sys.exit(0))
        
        # Setup terminal for flicker-free playback
        sys.stdout.write('\x1b[?1049h')  # Switch to alternate screen buffer
        sys.stdout.write('\x1b[?25l')    # Hide cursor
        sys.stdout.write('\x1b[2J')      # Clear screen once
        sys.stdout.flush()
        
        try:
            frame_num = 0
            self.frame_count = 0
            
            if audio_started:
                time.sleep(0.1)
            
            start_time = time.time()
            
            while True:
                if audio_started and pygame.mixer.get_init() and pygame.mixer.music.get_busy():
                    audio_pos_ms = pygame.mixer.music.get_pos()
                    if audio_pos_ms > 0:
                        target_frame = int(audio_pos_ms / frame_duration_ms)
                        
                        while frame_num < target_frame and cap.isOpened():
                            ret, frame = cap.read()
                            if not ret:
                                break
                            frame_num += 1
                        
                        if frame_num > target_frame + 2:
                            wait_time = (frame_num - target_frame) * frame_duration_ms / 1000.0
                            time.sleep(min(wait_time, 0.1))
                            continue
                
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                self.frame_count = frame_num
                processed = self.process_frame(frame)
                
                rendered = self.render_frame(processed)
                
                # Move cursor home without clearing screen
                sys.stdout.write('\x1b[H' + rendered)
                sys.stdout.flush()
                
                if not (audio_started and pygame.mixer.get_init() and pygame.mixer.music.get_busy()):
                    elapsed = time.time() - start_time
                    target_time = frame_num * frame_duration_ms / 1000.0
                    if target_time > elapsed:
                        time.sleep(target_time - elapsed)
                
                frame_num += 1
                
        except KeyboardInterrupt:
            pass  # Clean exit
        finally:
            cap.release()
            self.stop_audio()
            
            # Restore terminal
            cleanup_terminal()
            atexit.unregister(cleanup_terminal)
            print("\n⏹️  Stopped")
    
    
    
    def start_audio(self, video_path: str) -> Tuple[Optional[threading.Thread], bool]:
        try:
            import tempfile
            import subprocess
            
            audio_file = tempfile.mktemp(suffix='.mp3')
            cmd = [
                'ffmpeg', '-i', video_path,
                '-vn', '-acodec', 'mp3',
                '-ab', '192k', '-y',
                audio_file
            ]
            result = subprocess.run(cmd, capture_output=True)
            
            if result.returncode != 0:
                print(f"⚠️  Audio extraction failed, playing video without sound")
                return None, False
            
            pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=512)
            
            def play_audio():
                try:
                    pygame.mixer.music.load(audio_file)
                    pygame.mixer.music.play()
                    while pygame.mixer.music.get_busy():
                        time.sleep(0.1)
                except Exception as e:
                    print(f"⚠️  Audio playback error: {e}")
                finally:
                    try:
                        Path(audio_file).unlink(missing_ok=True)
                    except:
                        pass
            
            thread = threading.Thread(target=play_audio, daemon=True)
            thread.start()
            return thread, True
            
        except Exception as e:
            print(f"⚠️  Audio disabled: {e}")
            return None, False
    
    
    
    def stop_audio(self):
        try:
            if pygame.mixer.get_init():
                pygame.mixer.music.stop()
                pygame.mixer.quit()
        except Exception:
            pass



class VideoDownloader:
    """Simple YouTube downloader"""
    
    @staticmethod
    def download(url: str, quality: str = "720p") -> str:
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
            
            if not filename.endswith('.mp4'):
                base = filename.rsplit('.', 1)[0]
                if Path(f"{base}.mp4").exists():
                    filename = f"{base}.mp4"
        
        print(f"✅ Downloaded: {filename}")
        return filename