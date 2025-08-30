#!/usr/bin/env python3

import sys
import os
import signal
import atexit
import time
import shutil
import threading
import tempfile
import subprocess
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
import pygame

try:
    from scipy.spatial import KDTree
except ImportError:
    KDTree = None

try:
    from sklearn.cluster import MiniBatchKMeans
except ImportError:
    MiniBatchKMeans = None

try:
    import yt_dlp
except ImportError:
    yt_dlp = None

# Constants
DEFAULT_FPS = 10
MIN_WIDTH = 60
MIN_HEIGHT = 30
MAX_WIDTH = 120
MAX_HEIGHT = 40
PALETTE_SIZE = 256
PALETTE_UPDATE_INTERVAL = 15
BAYER_MATRIX = np.array([
    [0, 8, 2, 10],
    [12, 4, 14, 6],
    [3, 11, 1, 9],
    [15, 7, 13, 5]
], dtype=np.float32) / 16.0 * 32



def download_video(url: str, quality: str = "720p") -> str:
    if yt_dlp is None:
        raise ImportError("yt-dlp is required for YouTube downloads. Install with: pip install yt-dlp")
    
    output_dir = tempfile.gettempdir()
    
    ydl_opts = {
        'format': f'best[height<={quality[:-1]}][ext=mp4]/best[height<={quality[:-1]}]/best',
        'outtmpl': f'{output_dir}/%(title)s.%(ext)s',
        'quiet': False,
        'no_warnings': False,
    }
    
    print("Downloading video...")
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        filename = ydl.prepare_filename(info)
        
        if not filename.endswith('.mp4'):
            base = filename.rsplit('.', 1)[0]
            if Path(f"{base}.mp4").exists():
                filename = f"{base}.mp4"
    
    print(f"Downloaded: {filename}")
    return filename



class PixelPlayer:
    def __init__(self, use_true_color=False):
        self.use_true_color = use_true_color
        self.target_fps = DEFAULT_FPS
        self.frame_time = 1.0 / self.target_fps
        self.frame_buffer = []
        
        if not self.use_true_color:
            self.palette_size = PALETTE_SIZE
            self.adaptive_palette = True
            self.current_palette = self._generate_palette()
        else:
            self.palette_size = 0
            self.adaptive_palette = False
            self.current_palette = None
        
        self._setup_resolution()
        self._print_settings()
        
        if not self.use_true_color and KDTree is not None:
            self._build_color_cache()
    
    def _setup_resolution(self):
        cols, rows = shutil.get_terminal_size((80, 24))
        self.width = min(MAX_WIDTH, cols - 4)
        self.height = min(MAX_HEIGHT, (rows - 4) * 2)
        
        if self.width < MIN_WIDTH:
            self.width = MIN_WIDTH
        if self.height < MIN_HEIGHT:
            self.height = MIN_HEIGHT
    
    def _print_settings(self):
        print(f"Resolution: {self.width}x{self.height // 2} characters")
        print(f"Effective pixels: {self.width}x{self.height} (with half-blocks)")
        
        if self.use_true_color:
            print("Color mode: True Color (16.7M colors)")
        else:
            print("Color mode: Adaptive 8-bit (256 colors)")
    
    def _generate_palette(self):
        palette = []
        
        # RGB cube: 6 levels per channel = 216 colors
        for r in range(6):
            for g in range(6):
                for b in range(6):
                    palette.append([r * 51, g * 51, b * 51])
        
        # Add 40 grayscale levels
        for i in range(40):
            gray = int(i * 255 / 39)
            palette.append([gray, gray, gray])
        
        return np.array(palette[:PALETTE_SIZE], dtype=np.uint8)
    
    def _build_color_cache(self):
        if KDTree is not None:
            self.color_tree = KDTree(self.current_palette)
        else:
            self.color_tree = None
    
    def _generate_adaptive_palette(self, frame: np.ndarray):
        if MiniBatchKMeans is None:
            return
        
        h, w = frame.shape[:2]
        sample_size = min(10000, h * w)
        pixels = frame.reshape(-1, 3)
        
        indices = np.random.choice(pixels.shape[0], sample_size, replace=False)
        sample_pixels = pixels[indices]
        
        kmeans = MiniBatchKMeans(n_clusters=self.palette_size, random_state=42, n_init=5)
        kmeans.fit(sample_pixels)
        
        self.current_palette = kmeans.cluster_centers_.astype(np.uint8)
        self._build_color_cache()
    
    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        processed = cv2.resize(frame, (self.width, self.height), 
                               interpolation=cv2.INTER_AREA)
        
        if self.use_true_color:
            return processed
        
        if self.adaptive_palette and hasattr(self, 'frame_count'):
            if self.frame_count % PALETTE_UPDATE_INTERVAL == 0:
                self._generate_adaptive_palette(processed)
        
        processed = self._apply_dithering(processed)
        
        if self.color_tree is not None:
            h, w = processed.shape[:2]
            processed_flat = processed.reshape(-1, 3)
            _, indices = self.color_tree.query(processed_flat)
            processed = self.current_palette[indices].reshape(h, w, 3)
        
        return processed
    
    def _apply_dithering(self, image: np.ndarray) -> np.ndarray:
        h, w = image.shape[:2]
        bayer_tiled = np.tile(BAYER_MATRIX, (h // 4 + 1, w // 4 + 1))[:h, :w]
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
            print(f"Error: Cannot open video {video_path}")
            return
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps if fps > 0 else 0
        
        print(f"Video: {frame_count} frames, {fps:.1f} fps, {duration:.1f}s")
        print(f"Playing at {self.target_fps} fps")
        print("Press Ctrl+C to stop\n")
        
        audio_thread, audio_started = self._start_audio(video_path)
        frame_duration_ms = 1000.0 / fps
        
        self._setup_terminal()
        
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
                
                sys.stdout.write('\x1b[H' + rendered)
                sys.stdout.flush()
                
                if not (audio_started and pygame.mixer.get_init() and pygame.mixer.music.get_busy()):
                    elapsed = time.time() - start_time
                    target_time = frame_num * frame_duration_ms / 1000.0
                    if target_time > elapsed:
                        time.sleep(target_time - elapsed)
                
                frame_num += 1
                
        except KeyboardInterrupt:
            pass
        finally:
            cap.release()
            self._stop_audio()
            self._cleanup_terminal()
            print("\nStopped")
    
    def _setup_terminal(self):
        def cleanup():
            sys.stdout.write('\x1b[?25h')    # Show cursor
            sys.stdout.write('\x1b[?1049l')  # Return to main screen
            sys.stdout.flush()
        
        atexit.register(cleanup)
        signal.signal(signal.SIGTERM, lambda s, f: sys.exit(0))
        
        sys.stdout.write('\x1b[?1049h')  # Alternate screen
        sys.stdout.write('\x1b[?25l')    # Hide cursor
        sys.stdout.write('\x1b[2J')      # Clear screen
        sys.stdout.flush()
    
    def _cleanup_terminal(self):
        sys.stdout.write('\x1b[?25h')    # Show cursor
        sys.stdout.write('\x1b[?1049l')  # Return to main screen
        sys.stdout.flush()
        
        # Unregister cleanup handlers
        for handler in atexit._exithandlers[:]:
            if 'cleanup' in str(handler[0]):
                atexit._exithandlers.remove(handler)
    
    def _start_audio(self, video_path: str) -> Tuple[Optional[threading.Thread], bool]:
        try:
            audio_file = tempfile.mktemp(suffix='.mp3')
            cmd = [
                'ffmpeg', '-i', video_path,
                '-vn', '-acodec', 'mp3',
                '-ab', '192k', '-y',
                audio_file
            ]
            result = subprocess.run(cmd, capture_output=True)
            
            if result.returncode != 0:
                print("Audio extraction failed, playing video without sound")
                return None, False
            
            pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=512)
            
            def play_audio():
                try:
                    pygame.mixer.music.load(audio_file)
                    pygame.mixer.music.play()
                    while pygame.mixer.music.get_busy():
                        time.sleep(0.1)
                except Exception as e:
                    print(f"Audio playback error: {e}")
                finally:
                    try:
                        Path(audio_file).unlink(missing_ok=True)
                    except:
                        pass
            
            thread = threading.Thread(target=play_audio, daemon=True)
            thread.start()
            return thread, True
            
        except Exception as e:
            print(f"Audio disabled: {e}")
            return None, False
    
    def _stop_audio(self):
        try:
            if pygame.mixer.get_init():
                pygame.mixer.music.stop()
                pygame.mixer.quit()
        except Exception:
            pass



def show_usage():
    print("""
Pixel Art Video Player
Transform videos into terminal pixel art

Usage:
    python pixelplay.py <video_file or youtube_url> [options]
    
Options:
    --true-color    Use true 24-bit RGB colors (16.7M colors)
                    Default: Adaptive 256-color palette
    
Examples:
    python pixelplay.py video.mp4
    python pixelplay.py video.mp4 --true-color
    python pixelplay.py https://youtube.com/watch?v=VIDEO_ID
""")



def main():
    if len(sys.argv) < 2 or '--help' in sys.argv or '-h' in sys.argv:
        show_usage()
        sys.exit(0)
    
    input_source = sys.argv[1]
    use_true_color = '--true-color' in sys.argv
    
    player = PixelPlayer(use_true_color=use_true_color)
    
    if input_source.startswith(('http://', 'https://', 'www.')):
        print("YouTube URL detected")
        try:
            video_path = download_video(input_source)
            player.play_video(video_path)
            Path(video_path).unlink(missing_ok=True)
        except Exception as e:
            print(f"Error downloading video: {e}")
            sys.exit(1)
    else:
        if not Path(input_source).exists():
            print(f"Error: File not found: {input_source}")
            sys.exit(1)
        
        print(f"Local file: {input_source}")
        player.play_video(input_source)
    
    print("\nDone!")



if __name__ == "__main__":
    main()