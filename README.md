# Pixel Art Video Player

A terminal-based video player that transforms videos into high-quality pixel art using Unicode half-block characters and ANSI colors. It can download YouTube videos or play local video files directly in your terminal with synchronized audio playback.

## Features

### Core Capabilities
- **YouTube Integration**: Automatically downloads and plays YouTube videos using yt-dlp
- **Local File Support**: Plays any local video file supported by OpenCV
- **Ultra-High Resolution**: Renders at 1120×560 characters (effectively 1120×1120 pixels using half-blocks)
- **Multiple Color Modes**:
  - **Extended Palette**: Default 64-color optimized palette with improved color coverage
  - **True Color Mode**: Full 24-bit RGB without palette quantization for best quality
  - **Adaptive Palette**: Dynamic palette that adjusts to video content every 30 frames
- **Advanced Dithering**: Implements Bayer matrix ordered dithering for smooth gradients
- **Enhanced Audio/Video Synchronization**: Advanced sync mechanism using real-time audio position tracking
- **Adaptive Performance**: Maintains consistent 15 FPS playback with intelligent frame skipping
- **Robust Error Recovery**: Graceful fallback behaviors when audio or features are unavailable

### Visual Quality
- **Unicode Half-Blocks**: Uses `▀`, `▄`, and `█` characters for double vertical resolution
- **24-bit True Color**: Full RGB color support for modern terminals
- **Smart Color Quantization**: KD-tree based nearest neighbor search for fast palette matching
- **Optimized Rendering**: Special handling for black pixels and similar colors to reduce artifacts

## Recent Improvements

### Enhanced Audio/Video Synchronization
- Implemented real-time audio position tracking using `pygame.mixer.music.get_pos()`
- Dynamic frame adjustment that skips frames when video lags or waits when ahead
- Maintains perfect sync with 2-frame tolerance throughout playback
- Optimized pygame.mixer initialization (44100Hz, 16-bit, stereo, 512 buffer)

### Improved Error Handling
- Graceful fallback when audio extraction fails - video plays silently without interruption
- Adaptive palette mode automatically falls back to fixed palette if scikit-learn is unavailable
- Better pygame.mixer initialization with optimized audio settings
- Daemon threads ensure proper cleanup on exit

### Bug Fixes
- Fixed initialization order bug where `current_palette` was accessed before initialization
- Resolved pygame.mixer errors when audio extraction fails
- Improved audio thread management with comprehensive error handling
- Enhanced stability across different video formats and system configurations

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

**Note:** After updates, always run `pip install -r requirements.txt` again to ensure all dependencies are installed, including scikit-learn for adaptive palette mode.

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

### Command Line Options

- `--true-color`: Use true 24-bit RGB colors without palette quantization (best quality)
- `--adaptive`: Use adaptive palette that dynamically adjusts to video content
- `--help` or `-h`: Show help message with all available options

**Note:** You cannot use `--true-color` and `--adaptive` simultaneously.

### Controls
- **Ctrl+C**: Stop playback and exit
- The player automatically cleans up temporary files after playback

### Examples

```bash
# Play with default 64-color palette
python pixelplay.py "https://youtube.com/watch?v=VIDEO_ID"

# Play with true 24-bit RGB colors (best quality)
python pixelplay.py video.mp4 --true-color

# Play with adaptive palette that adjusts to content
python pixelplay.py movie.mkv --adaptive

# Play a local file with default settings
python pixelplay.py ~/Movies/sample.mp4

# Show help message
python pixelplay.py --help
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
4. **Color Processing**: Three modes available:
   - **Palette Mode** (default): Maps pixels to nearest color in 64-color palette using scipy KD-tree
   - **True Color Mode**: Preserves original 24-bit RGB values without quantization
   - **Adaptive Palette Mode**: Dynamically generates optimal palette using k-means clustering
5. **Half-Block Rendering**: Combines two vertical pixels into single Unicode character
6. **ANSI Color Output**: Generates escape sequences for terminal display

### Color Palette

The expanded 64-color optimized palette includes:
- **12 Grayscale levels**: Enhanced gradient representation from pure black to white
- **Primary colors**: Multiple saturation levels for red, green, and blue
- **Secondary colors**: Yellow, magenta, and cyan with variations
- **Orange spectrum**: 4 shades for warm tones and skin colors
- **Pink/Purple spectrum**: 4 shades for better color variety
- **Blue/Cyan spectrum**: 8 shades for sky and water scenes
- **Green spectrum**: 4 shades for nature content
- **Brown/Beige tones**: 8 shades for earth tones and skin colors

### Adaptive Palette Mode

When using `--adaptive`, the player:
1. Buffers 5 frames of video data
2. Every 30 frames, analyzes the buffered frames
3. Uses k-means clustering (via scikit-learn) to generate 64 optimal colors
4. Smoothly transitions to the new palette for better color representation
5. Particularly effective for videos with changing color schemes

**Note:** If scikit-learn is not installed, the player gracefully falls back to the fixed 64-color palette mode.

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
- **scikit-learn** (≥1.3.0): K-means clustering for adaptive palette generation

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
- The player now features enhanced audio/video synchronization that handles most sync issues automatically
- For persistent issues with variable framerate videos, try converting to constant framerate first

**"FFmpeg not found"**
- Install FFmpeg: 
  - macOS: `brew install ffmpeg`
  - Ubuntu/Debian: `sudo apt install ffmpeg`
  - Windows: Download from ffmpeg.org
- Note: If audio extraction fails, the video will play without sound

**"ImportError for scikit-learn"**
- Run `pip install -r requirements.txt` to install all dependencies
- The player will automatically fall back to fixed palette mode if scikit-learn is unavailable

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

### Color Processing Modes

**Default Palette Mode (64 colors)**:
- Uses a carefully crafted palette with improved color coverage
- Fast KD-tree based nearest neighbor search for color matching
- Provides good balance between quality and performance

**True Color Mode (`--true-color`)**:
- Bypasses palette quantization entirely
- Each pixel retains its original 24-bit RGB value
- Best visual quality but may be slower on some terminals
- Ideal for videos with subtle color gradients

**Adaptive Palette Mode (`--adaptive`)**:
- Analyzes video content in real-time
- Generates optimal 64-color palette every 30 frames
- Uses k-means clustering to find the most representative colors
- Excellent for videos with distinct color themes or scene changes

### Bayer Dithering Algorithm

The 4×4 Bayer matrix creates the illusion of more colors by adding structured noise:
```
 0  8  2 10
12  4 14  6
 3 11  1  9
15  7 13  5
```
This pattern is tiled across the image and adds threshold values to pixel colors before quantization, creating smoother gradients with the limited palette.

### Audio/Video Synchronization

The player features an advanced synchronization mechanism that ensures audio and video stay perfectly aligned:

1. **Real-time Audio Position Tracking**: Monitors actual audio playback position using `pygame.mixer.music.get_pos()`
2. **Dynamic Frame Adjustment**: 
   - Automatically skips frames when video lags behind audio
   - Pauses frame display when video is ahead of audio
   - Maintains sync throughout playback with a 2-frame tolerance
3. **Optimized Audio Settings**: Pygame mixer initialized with optimal parameters (44100Hz, 16-bit, stereo, 512 buffer)
4. **Frame Skip Calculation**: Intelligent frame skipping based on source video FPS
5. **Separate Audio Thread**: Runs audio in dedicated daemon thread for uninterrupted playback
6. **Graceful Degradation**: If audio extraction fails, video plays silently without interrupting the viewing experience

## License

This project is provided as-is for educational and entertainment purposes. Please respect copyright laws when downloading and playing video content.