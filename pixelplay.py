#!/usr/bin/env python3
"""
Simple Pixel Art Video Player
Usage:
    python pixelplay.py <youtube_url or video_file>
"""

import sys
import os
from pathlib import Path
from simple_pixel_player import SimplePixelPlayer, VideoDownloader

def print_banner():
    """Print a simple banner"""
    print("""
╔═══════════════════════════════════════╗
║     🎮 PIXEL VIDEO PLAYER 🎮          ║
║     Simple & High Quality             ║
╚═══════════════════════════════════════╝
""")

def main():
    if len(sys.argv) < 2:
        print("Usage: python pixelplay.py <youtube_url or video_file>")
        print("\nExamples:")
        print("  python pixelplay.py https://youtube.com/watch?v=...")
        print("  python pixelplay.py video.mp4")
        sys.exit(1)
    
    input_source = sys.argv[1]
    print_banner()
    
    # Create player
    player = SimplePixelPlayer()
    
    # Check if it's a URL or file
    if input_source.startswith(('http://', 'https://', 'www.')):
        # YouTube URL - download it
        print(f"🌐 YouTube URL detected")
        try:
            video_path = VideoDownloader.download(input_source)
            player.play_video(video_path)
            
            # Clean up downloaded file
            Path(video_path).unlink(missing_ok=True)
            
        except Exception as e:
            print(f"❌ Error downloading video: {e}")
            sys.exit(1)
    else:
        # Local file
        if not Path(input_source).exists():
            print(f"❌ Error: File not found: {input_source}")
            sys.exit(1)
        
        print(f"📁 Local file: {input_source}")
        player.play_video(input_source)
    
    print("\n✅ Done!")

if __name__ == "__main__":
    main()