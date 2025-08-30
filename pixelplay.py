#!/usr/bin/env python3
"""
Simple Pixel Art Video Player
Usage:
    python pixelplay.py <youtube_url or video_file> [options]
    
Options:
    --true-color    Use true 24-bit RGB colors (16.7M colors)
    --help          Show this help message
    
Default mode: Adaptive 8-bit palette (256 colors)
"""

import sys
import os
from pathlib import Path
from simple_pixel_player import SimplePixelPlayer, VideoDownloader



def print_banner():
    print("""
╔═══════════════════════════════════════╗
║     🎮 PIXEL VIDEO PLAYER 🎮          ║
║     Simple & High Quality             ║
╚═══════════════════════════════════════╝
""")



def print_help():
    print(__doc__)
    print("\nColor Modes:")
    print("  Default        : Adaptive 8-bit palette (256 colors)")
    print("  --true-color   : True 24-bit RGB (16.7M colors)")
    print("\nExamples:")
    print("  python pixelplay.py video.mp4")
    print("  python pixelplay.py video.mp4 --true-color")
    print("  python pixelplay.py https://youtube.com/watch?v=...")
    sys.exit(0)



def main():
    if len(sys.argv) < 2 or '--help' in sys.argv or '-h' in sys.argv:
        if len(sys.argv) < 2:
            print("Usage: python pixelplay.py <youtube_url or video_file> [options]")
            print("Use --help for more information")
        else:
            print_help()
        sys.exit(1)
    
    
    input_source = sys.argv[1]
    use_true_color = '--true-color' in sys.argv
    
    
    print_banner()
    player = SimplePixelPlayer(use_true_color=use_true_color)
    
    
    if input_source.startswith(('http://', 'https://', 'www.')):
        print(f"🌐 YouTube URL detected")
        try:
            video_path = VideoDownloader.download(input_source)
            player.play_video(video_path)
            Path(video_path).unlink(missing_ok=True)
        except Exception as e:
            print(f"❌ Error downloading video: {e}")
            sys.exit(1)
    else:
        if not Path(input_source).exists():
            print(f"❌ Error: File not found: {input_source}")
            sys.exit(1)
        
        print(f"📁 Local file: {input_source}")
        player.play_video(input_source)
    
    
    print("\n✅ Done!")



if __name__ == "__main__":
    main()