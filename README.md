# Pixel Art Video Player

A terminal-based video player that transforms videos into high-quality pixel art using Unicode half-block characters and ANSI colors. It can download YouTube videos or play local video files directly in your terminal with synchronized audio playback.

## Features

### Core Capabilities
- **YouTube Integration**: Automatically downloads and plays YouTube videos using yt-dlp
- **Local File Support**: Plays any local video file supported by OpenCV
- **Ultra-High Resolution**: Renders at 1120×560 characters (effectively 1120×1120 pixels using half-blocks)
- **Optimized Color Palette**: Uses a carefully crafted 32-color palette for accurate video representation
- **Advanced Dithering**: Implements Bayer matrix ordered dithering for smooth gradients
- **Synchronized Audio**: Plays audio through pygame with proper frame synchronization
- **Adaptive Performance**: Maintains consistent 15 FPS playback with intelligent frame skipping

### Visual Quality
- **Unicode Half-Blocks**: Uses `▀`, `▄`, and `█` characters for double vertical resolution
- **24-bit True Color**: Full RGB color support for modern terminals
- **Smart Color Quantization**: KD-tree based nearest neighbor search for fast palette matching
- **Optimized Rendering**: Special handling for black pixels and similar colors to reduce artifacts

## Installation

### Prerequisites
- Python 3.8 or higher
- Terminal with 24-bit color support (most modern terminals)
- FFmpeg installed on your system (for audio extraction)
- Sufficient terminal size (recommended: at least 120×60 characters)

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd pixel
```

2. Create and activate a virtual environment:
```bash
# On macOS/Linux
python3 -m venv venv
source venv/bin/activate

# On Windows
python -m venv venv
venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Verify FFmpeg installation:
```bash
ffmpeg -version
```

## Usage

### Basic Commands

**Play a YouTube video:**
```bash
python pixelplay.py "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
```

**Play a local video file:**
```bash
python pixelplay.py path/to/video.mp4
```

**Supported formats:** MP4, AVI, MOV, MKV, and any format supported by OpenCV

### Controls
- **Ctrl+C**: Stop playback and exit
- The player automatically cleans up temporary files after playback

### Examples

```bash
# Play a YouTube music video
python pixelplay.py "https://youtube.com/watch?v=VIDEO_ID"

# Play a local movie file
python pixelplay.py ~/Movies/sample.mp4

# Play a downloaded video
python pixelplay.py downloaded_video.mkv
```

## Technical Details

### Architecture

The project consists of two main components:

1. **pixelplay.py** - Entry point and command-line interface
   - Handles argument parsing
   - Detects URL vs local file input
   - Manages video download and cleanup
   - Provides user-friendly error messages

2. **simple_pixel_player.py** - Core video processing engine
   - `SimplePixelPlayer` class: Main video player implementation
   - `VideoDownloader` class: YouTube download functionality
   - Frame processing pipeline
   - Audio synchronization

### Video Processing Pipeline

1. **Frame Capture**: Uses OpenCV to read video frames
2. **Resolution Scaling**: Resizes frames to target resolution using INTER_AREA interpolation
3. **Dithering**: Applies 4×4 Bayer matrix ordered dithering for better gradients
4. **Color Quantization**: Maps pixels to nearest palette color using scipy KD-tree
5. **Half-Block Rendering**: Combines two vertical pixels into single Unicode character
6. **ANSI Color Output**: Generates escape sequences for terminal display

### Color Palette

The 32-color optimized palette includes:
- **8 Grayscale levels**: From pure black to white for accurate brightness
- **Primary colors**: Pure red, green, and blue
- **Secondary colors**: Yellow, magenta, and cyan
- **Warm tones**: Orange, browns, and skin tones
- **Cool tones**: Blues and teals
- **Additional colors**: For better coverage of common video content

### Performance Optimizations

- **Frame Skipping**: Intelligently skips frames to maintain target FPS
- **Color Caching**: Pre-built KD-tree for O(log n) color lookups
- **Black Pixel Optimization**: Special handling for dark areas
- **Batch Processing**: Vectorized NumPy operations for speed
- **Efficient Rendering**: Minimal ANSI escape sequences

## Dependencies

### Core Libraries
- **opencv-python** (≥4.8.0): Video processing and frame capture
- **numpy** (≥1.24.0): Numerical operations and array manipulation
- **pygame** (≥2.5.0): Audio playback
- **yt-dlp** (≥2024.1.0): YouTube video downloading

### Image Processing
- **Pillow** (≥10.0.0): Additional image operations
- **scipy** (≥1.11.0): KD-tree for color matching
- **scikit-image** (≥0.22.0): Advanced image processing

### Terminal UI
- **colorama** (≥0.4.6): Cross-platform terminal colors
- **blessed** (≥1.20.0): Terminal capabilities
- **click** (≥8.1.0): Command-line interface
- **tqdm** (≥4.66.0): Progress bars

### Performance
- **numba** (≥0.58.0): JIT compilation for critical loops

## Project Structure

```
pixel/
├── README.md                 # This documentation
├── pixelplay.py             # Main entry point and CLI
├── simple_pixel_player.py   # Core video player implementation
├── requirements.txt         # Python dependencies
└── venv/                    # Virtual environment (git-ignored)
```

## Troubleshooting

### Common Issues

**"Terminal doesn't support colors"**
- Ensure your terminal supports 24-bit true color
- Try modern terminals like iTerm2, Windows Terminal, or Kitty

**"Video playback is choppy"**
- Reduce terminal window size
- Close other terminal tabs/windows
- Ensure your system isn't under heavy load

**"Audio out of sync"**
- This can happen with variable framerate videos
- Try converting the video to constant framerate first

**"FFmpeg not found"**
- Install FFmpeg: 
  - macOS: `brew install ffmpeg`
  - Ubuntu/Debian: `sudo apt install ffmpeg`
  - Windows: Download from ffmpeg.org

**"Download fails"**
- Check your internet connection
- Verify the YouTube URL is valid
- Update yt-dlp: `pip install --upgrade yt-dlp`

### Performance Tips

1. **Terminal Size**: Larger terminals require more processing power
2. **Video Quality**: Lower resolution videos process faster
3. **Color Depth**: Some terminals perform better with reduced colors
4. **Background Processes**: Close unnecessary applications for smoother playback

## How It Works

### Half-Block Rendering Technique

The player uses Unicode half-block characters (`▀` and `▄`) to double the effective vertical resolution. Each character position represents two pixels:
- Upper pixel uses foreground color
- Lower pixel uses background color
- Full blocks (`█`) used when both pixels are similar
- Spaces used for very dark areas

### Bayer Dithering Algorithm

The 4×4 Bayer matrix creates the illusion of more colors by adding structured noise:
```
 0  8  2 10
12  4 14  6
 3 11  1  9
15  7 13  5
```
This pattern is tiled across the image and adds threshold values to pixel colors before quantization, creating smoother gradients with the limited palette.

### Frame Synchronization

The player maintains 15 FPS by:
1. Calculating frame skip ratio based on source video FPS
2. Using high-precision timing with `time.time()`
3. Sleeping between frames to maintain consistent timing
4. Running audio in a separate thread for uninterrupted playback

## License

This project is provided as-is for educational and entertainment purposes. Please respect copyright laws when downloading and playing video content.