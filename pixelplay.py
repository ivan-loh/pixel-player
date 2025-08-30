#!/usr/bin/env python3
"""
Simple Pixel Art Video Player
Usage:
    python pixelplay.py <youtube_url or video_file> [options]
    
Options:
    --true-color    Use true 24-bit RGB colors (no palette)
    --adaptive      Use adaptive palette that changes with video content
    --help          Show this help message
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

def print_help():
    """Print help message"""
    print(__doc__)
    print("\nColor Modes:")
    print("  Default        : 64-color fixed palette (fast, good quality)")
    print("  --true-color   : True 24-bit RGB (best quality, may be slower)")  
    print("  --adaptive     : Adaptive palette (adjusts to video content)")
    print("\nExamples:")
    print("  python pixelplay.py video.mp4")
    print("  python pixelplay.py video.mp4 --true-color")
    print("  python pixelplay.py https://youtube.com/watch?v=... --adaptive")
    sys.exit(0)

def main():
    if len(sys.argv) < 2 or '--help' in sys.argv or '-h' in sys.argv:
        if len(sys.argv) < 2:
            print("Usage: python pixelplay.py <youtube_url or video_file> [options]")
            print("Use --help for more information")
        else:
            print_help()
        sys.exit(1)
    
    # Parse arguments
    input_source = sys.argv[1]
    use_true_color = '--true-color' in sys.argv
    use_adaptive = '--adaptive' in sys.argv
    
    # Check for conflicting options
    if use_true_color and use_adaptive:
        print("❌ Error: Cannot use both --true-color and --adaptive")
        print("Choose one color mode or use default")
        sys.exit(1)
    
    print_banner()
    
    # Create player with specified options
    player = SimplePixelPlayer(use_true_color=use_true_color, adaptive_palette=use_adaptive)
    
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