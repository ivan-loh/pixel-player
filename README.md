# Pixel Art Video Player

A terminal-based video player that transforms videos into high-quality pixel art using Unicode half-block characters and ANSI colors. It can download YouTube videos or play local video files directly in your terminal with synchronized audio playback.

## Features

- **YouTube Integration**: Downloads and plays YouTube videos using yt-dlp
- **Local File Support**: Plays video files supported by OpenCV
- **Auto-Sizing Resolution**: Adapts to your terminal size for optimal display
- **Two Color Modes**: 256-color adaptive palette (default) or true 24-bit RGB
- **Half-Block Rendering**: Uses Unicode characters (`▀`, `▄`, `█`) for double vertical resolution
- **Audio/Video Sync**: Real-time synchronization with automatic frame adjustment
- **Bayer Dithering**: Ordered dithering for smoother gradients
- **Graceful Fallbacks**: Continues playback when audio extraction fails


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

- `--true-color`: Use true 24-bit RGB colors instead of 256-color adaptive palette
- `--help` or `-h`: Show help message with all available options

### Controls
- **Ctrl+C**: Stop playback and exit
- The player automatically cleans up temporary files after playback

### Examples

```bash
# Play with default 64-color palette
python pixelplay.py "https://youtube.com/watch?v=VIDEO_ID"

# Play with true 24-bit RGB colors (best quality)
python pixelplay.py video.mp4 --true-color

# Play with default 256-color adaptive palette
python pixelplay.py movie.mkv

# Play a local file with default settings
python pixelplay.py ~/Movies/sample.mp4

# Show help message
python pixelplay.py --help
```

## Technical Details

```mermaid
flowchart TD
    %% Input Sources
    youtube["YouTube URL"]
    localfile["Local Video File"]
    
    %% System Architecture
    subgraph system["System Architecture"]
        cli["pixelplay.py<br/>CLI Entry Point"]
        core["simple_pixel_player.py"]
        subgraph corecomp["Core Components"]
            player["SimplePixelPlayer<br/>Video Processing"]
            downloader["VideoDownloader<br/>URL Handling"]
        end
    end
    
    %% Video Processing Pipeline
    subgraph videopipe["Video Processing Pipeline"]
        capture["Frame Capture<br/>(OpenCV)"]
        autosize["Auto-Sizing<br/>(Terminal Dimensions)"]
        colormode{"Color Processing<br/>Mode?"}
        adaptive["256-Color Adaptive<br/>(K-means clustering<br/>every 15 frames)"]
        truecolor["True Color<br/>(24-bit RGB)"]
        dither["Bayer Dithering<br/>(4×4 matrix)"]
        render["Half-Block Rendering<br/>(Unicode characters)"]
    end
    
    %% Audio Processing
    subgraph audiopipe["Audio Processing"]
        extract["Audio Extraction<br/>(FFmpeg)"]
        playback["Audio Playback<br/>(pygame)"]
    end
    
    %% Output and Sync
    subgraph output["Output & Synchronization"]
        sync["Real-time Position<br/>Tracking"]
        decision{"Frame Timing"}
        skip["Skip Frame"]
        wait["Wait"]
        display["Terminal Display<br/>+ Audio Output"]
    end
    
    %% Main Flow Connections
    youtube --> cli
    localfile --> cli
    cli --> downloader
    cli --> player
    downloader --> capture
    player --> capture
    
    %% Video Processing Flow
    capture --> autosize
    autosize --> colormode
    colormode -->|"Adaptive"| adaptive
    colormode -->|"True Color"| truecolor
    adaptive --> dither
    truecolor --> dither
    dither --> render
    
    %% Audio Processing Flow
    capture --> extract
    extract --> playback
    
    %% Synchronization Flow
    render --> sync
    playback --> sync
    sync --> decision
    decision -->|"Behind"| skip
    decision -->|"Ahead"| wait
    decision -->|"In Sync"| display
    skip --> display
    wait --> display
    
    %% Styling
    classDef inputStyle fill:#e1f5fe,stroke:#0277bd,color:#000
    classDef systemStyle fill:#f3e5f5,stroke:#7b1fa2,color:#000
    classDef processStyle fill:#e8f5e8,stroke:#2e7d32,color:#000
    classDef decisionStyle fill:#fff3e0,stroke:#ef6c00,color:#000
    classDef outputStyle fill:#fce4ec,stroke:#c2185b,color:#000
    
    class youtube,localfile inputStyle
    class cli,core,player,downloader systemStyle
    class capture,autosize,adaptive,truecolor,dither,render,extract,playback,sync processStyle
    class colormode,decision decisionStyle
    class display,skip,wait outputStyle
```

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

## License

This project is provided as-is for educational and entertainment purposes. Please respect copyright laws when downloading and playing video content.