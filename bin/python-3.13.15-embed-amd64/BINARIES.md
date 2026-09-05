# Embedded Python environment (not in git — assemble separately)

The complete `python-3.13.15-embed-amd64` runtime is intentionally not
included in this repository (~430 MB with packages). It ships only inside
the portable ZIP published under GitHub Releases.

## Base interpreter

- Download: https://www.python.org/ftp/python/3.13.15/python-3.13.15-embed-amd64.zip
- Unpack it so this folder contains `python.exe` (64-bit):

```text
bin/python-3.13.15-embed-amd64/python.exe   <- entry point used by start.bat
```

- Edit `python313._pth` inside that folder and uncomment the
  `import site` line, otherwise installed packages are not found.

## Required packages (minimum versions)

Install with pip into the embedded environment (the `start.bat` entry
point above must resolve every import in `src/`):

```text
gradio>=6.26.0
av>=18.1.0
opencv-python-headless>=5.0.0.93
numpy>=2.5.2
pillow>=12.3.0
pillow-heif>=1.6.0
rawpy>=0.27.1
resvg-py>=0.5.0
```

Validated on 2026-09-03 against: gradio 6.26.0, av 18.1.0,
opencv-python-headless 5.0.0.93, numpy 2.5.2, pillow 12.3.0,
pillow-heif 1.6.0, rawpy 0.27.1, resvg-py 0.5.0.

## Notes

- Python itself is under the PSF License; each package retains its own
  license — preserve the notices shipped with each distribution.
- Native media/image dependencies (FFmpeg libs via PyAV, OpenCV, NumPy)
  come from the packages above; no separate installs are needed besides
  the `bin/ffmpeg/` and `bin/runtime/` contents documented next to this file.
