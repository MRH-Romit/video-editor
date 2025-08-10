# VideoFX Studio

A real-time video effects application built with Python, PySide6, and OpenCV. Apply stunning effects to your videos with live preview and export functionality.

![VideoFX Studio](https://img.shields.io/badge/Python-3.10+-blue.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![Platform](https://img.shields.io/badge/Platform-Windows-lightgrey.svg)

## Features

### 🎬 Core Functionality

- **Drag & Drop Video Loading** - Simply drag video files into the window
- **Live Side-by-Side Preview** - See original and processed video in real-time
- **Real-time Effects Processing** - Apply effects with instant feedback
- **Export Processed Videos** - Save your edited videos with background processing
- **Play/Pause Control** - Control playback with FPS display

### ✨ Available Effects

1. **Grayscale** - Convert to black and white
2. **Canny Edge Detection** - Highlight edges in the video
3. **Gaussian Blur** - Apply smooth blur effect
4. **Sharpen** - Enhance image sharpness using unsharp mask
5. **Sepia Tone** - Apply vintage sepia color effect
6. **Cartoon Effect** - Transform video into cartoon-like appearance
7. **Pixelate** - Create retro pixelated effect
8. **HSV Adjust** - Fine-tune Hue, Saturation, and Value
9. **Glitch Effect** - Digital corruption with channel shifts
10. **Motion Blur** - Apply directional motion blur

### 🎛️ Real-time Controls

- Effect selection dropdown
- Dynamic parameter sliders (intensity, strength, pixel size, etc.)
- Auto-hiding controls based on selected effect
- Live parameter adjustment with instant preview

## Project Structure

```
videofx-studio/
├── main.py              # Main application and UI
├── effects.py           # Video effects and image utilities
├── video_io.py          # Video reading and export functionality
├── requirements.txt     # Python dependencies
├── README.md           # Project documentation
├── LICENSE             # MIT License
├── .gitignore          # Git ignore patterns
└── dist/              # Built executables
    └── VideoFXStudio.exe
```

## Installation

### Prerequisites

- Python 3.10 or higher
- Windows OS (primary platform)

### Setup Instructions

1. **Clone the repository**
   
   ```bash
   git clone https://github.com/MRH-Romit/video-editor.git
   cd video-editor
   ```

2. **Create a virtual environment**
   
   ```bash
   python -m venv .venv
   .venv\Scripts\activate  # Windows
   ```

3. **Install dependencies**
   
   ```bash
   pip install -r requirements.txt
   ```
   
   Or manually:
   ```bash
   pip install --upgrade pip
   pip install PySide6 opencv-python pyinstaller
   ```

## Usage

### Running the Application

**Development Mode:**

```bash
python main.py
```

**Using the Executable:**

1. Build the executable (see Building section below)
2. Run `dist/VideoFXStudio.exe`

### How to Use

1. **Load Video**: Drag and drop a video file or use the "Open..." button
2. **Select Effect**: Choose from the dropdown menu
3. **Adjust Parameters**: Use the sliders to fine-tune the effect
4. **Preview**: Watch the real-time side-by-side comparison
5. **Export**: Click "Save Processed..." to export your edited video

### Supported Formats

- **Input**: MP4, MOV, AVI, MKV, WEBM
- **Output**: MP4 (MP4V codec)

## Building Executable

Create a standalone Windows executable:

```bash
pyinstaller --noconfirm --onefile --windowed --name "VideoFXStudio" main.py
```

The executable will be available in the `dist/` folder.

## Technical Details

### Architecture

- **Modular Design**: Separated into main UI, effects processing, and video I/O
- **UI Framework**: PySide6 (Qt for Python)
- **Video Processing**: OpenCV with NumPy for efficient array operations
- **Threading**: QThread for video reading and export processing
- **Real-time Processing**: Frame-by-frame effect application with parameter updates

### Code Organization

- **`main.py`**: Main application window, UI components, and event handling
- **`effects.py`**: Video effects implementation, image utilities, and effect parameters
- **`video_io.py`**: Video reading thread and background export worker

### Performance Notes

- Performance depends on CPU/GPU and video resolution
- For better performance on high-resolution videos, consider resizing the preview window
- Effects are applied in real-time, so complex effects may reduce playback smoothness

### Dependencies

```txt
PySide6>=6.0.0
opencv-python>=4.5.0
numpy>=1.20.0
pyinstaller>=5.0.0
```

## Contributing

Contributions are welcome! Here are some ways you can contribute:

1. **Add New Effects**: Implement additional video effects in `effects.py`
2. **Performance Improvements**: Optimize processing algorithms
3. **UI Enhancements**: Improve the user interface in `main.py`
4. **Cross-Platform Support**: Add Linux/macOS compatibility
5. **Bug Fixes**: Report and fix issues

### Development Setup

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## Roadmap

### Planned Features

- [ ] GPU acceleration (CUDA/OpenCL support)
- [ ] Additional effects (background blur, stabilization)
- [ ] Color LUT support
- [ ] Batch processing
- [ ] Timeline editing
- [ ] Audio processing
- [ ] Linux/macOS support
- [ ] Plugin system for custom effects

### Known Issues

- Some effects may be CPU-intensive on high-resolution videos
- Limited to MP4V codec for output (H.264 support requires additional OpenCV build)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Built with [PySide6](https://wiki.qt.io/Qt_for_Python) for the UI framework
- [OpenCV](https://opencv.org/) for video processing capabilities
- [NumPy](https://numpy.org/) for numerical operations

## Support

If you encounter any issues or have questions:

1. Check the [Issues](https://github.com/MRH-Romit/video-editor/issues) page
2. Create a new issue if your problem isn't already reported
3. Provide detailed information about your system and the issue

## Authors

- **MRH-Romit** - *Project Creator* - [MRH-Romit](https://github.com/MRH-Romit)

---

⭐ If you found this project helpful, please give it a star!
