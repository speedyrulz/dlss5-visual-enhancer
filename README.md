# DLSS 5 Visual Enhancer

[![Downloads](https://img.shields.io/github/downloads/Merserk/dlss5-visual-enhancer/total.svg?style=flat-square&label=Downloads)](https://github.com/Merserk/dlss5-visual-enhancer/releases) [![Patreon](https://img.shields.io/badge/Patreon-MM744-F96854?style=flat-square&logo=patreon&logoColor=white)](https://www.patreon.com/MM744) ![Platform](https://img.shields.io/badge/Platform-Windows-0078D4?style=flat-square&logo=windows11&logoColor=white) ![NVIDIA](https://img.shields.io/badge/NVIDIA-RTX-76B900?style=flat-square&logo=nvidia&logoColor=white) ![DLSS](https://img.shields.io/badge/DLSS-5-76B900?style=flat-square) ![DLSS Frame Generation](https://img.shields.io/badge/DLSS-Frame%20Generation-76B900?style=flat-square&logo=nvidia&logoColor=white) ![Type](https://img.shields.io/badge/Type-Portable-2EA44F?style=flat-square) [![License](https://img.shields.io/badge/License-MIT-007EC6?style=flat-square)](LICENSE)

Windows application for applying a DLSS 5 Neural Rendering feature-18 pipeline to images and video, and NVIDIA DLSS Frame Generation to video frame interpolation, through a local Gradio interface. It is an independent community project and is not affiliated with, sponsored by, or endorsed by NVIDIA, ReShade, RenoDX, FFmpeg, or their contributors.

<img width="1403" height="630" alt="image" src="https://github.com/user-attachments/assets/4530410a-af0f-42b5-9ba7-72106e1f5517" />

### Original

https://github.com/user-attachments/assets/f27f61a3-cad3-4278-af66-eb11c54600fb

### DLSS 5

https://github.com/user-attachments/assets/d91591a9-2df1-4b4b-b18f-bd4dff73d5bc

## Changes in this fork

This fork ([speedyrulz/dlss5-visual-enhancer](https://github.com/speedyrulz/dlss5-visual-enhancer)) tracks upstream (including its GPU selection and DLSS Frame Generation) and adds fixes and performance work on top:

- **Variable-frame-rate fix:** VFR videos (typical phone footage) crash upstream at a content-dependent frame with `Invalid argument ... returned 22`. The intermediate stream quantizes timestamps to average-frame-duration ticks, so two close frames can collide into one tick and the muxer rejects the duplicate. The encoder context now keeps the source's fine time base, preserving exact VFR timing; a safety guard bumps genuinely duplicated timestamps and reports the count as `pts_collisions_adjusted`.
- **Worker send/receive overlap:** a dedicated sender thread keeps one frame in flight inside the native worker's pipe while the main thread drains results, so the ~48 MB per-frame upload overlaps DLSS evaluation instead of serializing with it (measured ~20% per-frame improvement at 1836x3264 on top of the threaded pipeline).
- **Adapter verification:** after the worker handshake, the session reads back which Direct3D adapter the worker actually bound. An explicit AI Processing GPU selection the worker could not honor fails fast with a clear message instead of silently rendering on the wrong card; automatic selection corrects the reported GPU.
- **Smarter automatic Video Processing GPU:** with two or more cards, the automatic video-GPU choice prefers a card other than the AI Processing GPU when it has at least 2 GB of free VRAM, so NVENC does not compete with the DLSS render; single-GPU systems and explicit selections behave as upstream.
- **Better failure diagnostics:** if the video encoder process dies (video render or frame interpolation), the error names the encoder, exit code, and frame, and includes FFmpeg's stderr; failure reports gain an `encoder_log` section.

## Installation

1. Download the [latest release](https://github.com/Merserk/dlss5-visual-enhancer/releases/latest).
2. Unpack the downloaded ZIP archive.
3. Run `start.bat`.

## Main features

- **Images:** single-image and batch processing with per-file success/failure results, responsive input/output previews, individual downloads, a ZIP of successful outputs, batch manifests, and diagnostic JSON reports.
- **Image formats:** common Pillow formats plus HEIF/HEIC, SVG, and many camera RAW formats. Outputs are PNG, JPEG, WebP, AVIF, or TIFF.
- **Image handling:** EXIF orientation is applied; ICC input is converted to sRGB; supported EXIF, DPI, and XMP metadata is retained; alpha is preserved except when JPEG composites it over white. EXIF/TIFF rational values and other unusual metadata are converted safely for diagnostic JSON without modifying the metadata used to encode the output. Animated and multipage files use only the first frame/page.
- **Video:** single-video and reorderable batch-video Neural Rendering, plus one-frame and three-second previews for a single selected video. H.264, HEVC, AV1, and ProRes Proxy are available in MP4, MKV, or MOV where compatible.
- **Frame Interpolation:** single-video and reorderable batch-video processing with NVIDIA DLSS Frame Generation, selectable output rates from 23.976 to 120 FPS, and a three-second preview for a single selected video.
- **Sequential video batches:** Neural Rendering and Frame Interpolation batches run once, one by one, in the displayed uploader order using one settings snapshot. A failed item is recorded without preventing later videos from rendering. Successful outputs remain individually downloadable and every batch receives an ordered JSON manifest.
- **Single-only video previews:** input and output players are shown when exactly one video is selected. They are hidden for multi-video batches; batch results are provided through downloadable files and the per-video results table.
- **Media preservation:** frame timestamps and display rotation are handled; original metadata and chapters are copied. MKV copies compatible audio and subtitle streams, while MP4/MOV convert audio to 192 kbps AAC. Frame Interpolation also preserves supported text subtitles in MP4/MOV.
- **GPU selection:** AI Processing and Video Processing GPUs can be selected separately. The AI GPU is used for DLSS Neural Rendering and Frame Generation; the Video GPU is used for H.264, HEVC, and AV1 NVENC encoding. Automatic selection is available for both.
- **GPU compatibility:** supported RTX GPUs are detected by architecture and compute capability, covering Ampere, Ada, and Blackwell. RTX 30 uses the tested experimental Ampere Neural Rendering path and may be significantly slower than RTX 40/50.
- **Renaming:** Image, Video, and Frame Interpolation outputs support Auto timestamped naming, Copy naming with the original base filename, or a Custom suffix. Existing outputs are never overwritten.
- **Safety and diagnostics:** only one GPU render runs at a time. Stop cancels the active worker/encoder and removes only incomplete output/job data. In a batch, completed files are retained and queued items are marked skipped. Outputs are accepted only after the relevant render path and output properties are verified.
- **Persistent controls:** Image and Video tabs share neural settings and the DLSS Model Preset, including the experimental Automatic Mask toggle. Frame Interpolation settings and GPU selections are also saved in `config.ini`. Settings presets can be exported to or imported from JSON, and Reset restores the relevant controls to their defaults.

The application creates `outputs/`, `logs/`, and `jobs/` when needed. Successful media is written to `outputs/`, reports and manifests to `logs/`, and temporary active-render data to `jobs/`.

## Requirements

- 64-bit Windows 11 with Direct3D 12.
- NVIDIA GeForce RTX GPU based on a supported Ampere, Ada, or Blackwell architecture. RTX 40/50 are the primary Neural Rendering targets; RTX 30 is enabled as a slower experimental path using the tested compatible runtime pair. RTX 20 and non-RTX GPUs are rejected.
- Frame Interpolation requires a GPU and NVIDIA driver that report DLSS Frame Generation support, Hardware-accelerated GPU scheduling (HAGS) enabled in Windows, and the validated DLSSG runtime included with the release.
- Frame Interpolation accepts SDR video. PQ/HLG HDR input is rejected instead of being silently converted.

## Settings

| Neural control | Values | Default |
| --- | --- | --- |
| NR Preset | Default, Preset #1, Preset #2, Preset #3 | Default |
| NR Style | Default, Natural, Cinematic | Default |
| NR Intensity | 0.00–2.00 | 1.00 |
| Local Tone Strength | 0.00–2.00 | 1.00 |
| Local Structure Strength | 0.00–2.00 | 1.00 |
| Skin Structure Strength | -1.00–2.00 | -1.00 (native default) |
| Automatic Mask | Off, On | Off |

The preset numbers are experimental native model hints; their visual effect is content-dependent.

| DLSS Model Preset | SDK value | Behavior |
| --- | ---: | --- |
| Default | 0 | Applies no override and lets NVIDIA choose its normal mode-specific preset |
| J | 10 | Forces preset J for every supported DLSS scaling mode |
| K | 11 | Forces preset K for every supported DLSS scaling mode |
| L | 12 | Forces preset L for every supported DLSS scaling mode |
| M | 13 | Forces preset M for every supported DLSS scaling mode |

The DLSS Model Preset is independent of the experimental Neural Rendering **NR Preset** control.

| Upscaling mode | Factor | Behavior |
| --- | ---: | --- |
| DLAA / native | 1× | Keeps the source dimensions |
| Quality | 1.5× | Produces 1.5× output dimensions |
| Balanced | 1.724× | Produces approximately 1.724× output dimensions |
| Performance | 2× | Produces 2× output dimensions |
| Ultra Performance | 3× | Produces 3× output dimensions |

Output dimensions are rounded to even pixels and limited to a 7680×4320 boundary.

| Frame Interpolation setting | Choices and behavior | Default |
| --- | --- | --- |
| Output FPS | 23.976, 25, 29.97, 30, 50, 59.94, 60, 90, or 120 FPS | 60 |
| DLSS engine | Auto, Native DLSSG, or Cascade | Auto |
| Video codec | H.264, HEVC, AV1, or ProRes Proxy | H.264 |
| Container | MP4, MKV, or MOV; ProRes Proxy requires MKV or MOV | MP4 |
| Encoding quality | Auto (Default), Good, Best, or Max | Auto (Default) |
| Rename | Auto adds a DLSSFG timestamp; Copy keeps the original base name; Custom appends the entered suffix | Auto |

`Auto` uses a supported exact native DLSSG grid when possible and the cascade path when required. If the selected output FPS is equal to or below the source rate, source frames are resampled without generating additional frames.

| Output setting | Choices and behavior |
| --- | --- |
| Image format | PNG/TIFF are lossless; JPEG/WebP/AVIF use the 1–100 quality control (default 95) |
| Video codec | H.264, HEVC, AV1, or ProRes Proxy |
| Container | MP4, MKV, or MOV; ProRes Proxy requires MKV or MOV |
| Encoding quality | Auto (Default) uses resolution/FPS/codec; Good = Auto×2; Best = Auto×4; Max uses CQ/CRF 0; ProRes uses its fixed Proxy profile |
| Rename | Auto adds the DLSS5 timestamp; Copy keeps the original base name; Custom appends the entered suffix before the output extension |

H.264, HEVC, and AV1 require working NVENC support on the selected or automatically chosen Video Processing GPU at the requested output size. ProRes Proxy uses CPU-based 10-bit 4:2:2 encoding, although the verified Neural Rendering path remains RGBA8.

| Application setting | Choices and behavior |
| --- | --- |
| AI Processing GPU | Automatic or a compatible Ampere, Ada, or Blackwell RTX GPU; used for Neural Rendering and Frame Generation |
| Video Processing GPU | Automatic or an available NVIDIA GPU; used for H.264, HEVC, and AV1 NVENC encoding |
| Settings preset | Export all adjustable Image, Video, Frame Interpolation, and GPU settings to JSON, or import a preset to apply and save them |

Saved GPU selections use stable GPU identity. If a previously saved GPU is no longer available, that selection returns to Automatic rather than silently switching to another saved device.

## License and third-party notices

Original application code is licensed under the MIT License, copyright © 2026 Merserk. That license covers only original project code; it does not relicense or grant rights to any third-party software, model, binary, trademark, media, or other asset.

- **NVIDIA DLSS/NGX:** NVIDIA and its suppliers retain their rights in genuine NVIDIA SDK and runtime files, including DLSS Neural Rendering and DLSS Frame Generation components. Use and distribution are governed by the [NVIDIA RTX SDK License](https://github.com/NVIDIA/DLSS/blob/main/LICENSE.txt). Their presence in a portable package does not imply a standalone redistribution right, and this project must not be represented as NVIDIA-sponsored or endorsed.
- **FFmpeg:** the referenced Gyan.dev `9.0.1-full_build` was configured with GPL and version-3 components and is distributed under GPLv3. Its build information, license, and exact [corresponding FFmpeg source commit](https://github.com/FFmpeg/FFmpeg/commit/bf1b838f2a) are retained under `bin/ffmpeg/`. Anyone redistributing that binary must satisfy the applicable GPLv3 and corresponding-source obligations. See [FFmpeg licensing](https://github.com/FFmpeg/FFmpeg/blob/master/LICENSE.md).
- **ReShade:** copyright belongs to Patrick Mours and contributors; ReShade is available under the [BSD 3-Clause License](https://github.com/crosire/reshade).
- **RenoDX:** RenoDX core is copyright its authors and available under [MIT](https://github.com/clshortfuse/renodx/blob/main/LICENSE). This does not establish the license of the separate `renodx-dlss5.addon64` file.
- **Python and packages:** Python is provided under the [PSF License](https://docs.python.org/3.13/license.html). Gradio, Pillow, pillow-heif, rawpy, resvg-py, PyAV, OpenCV, NumPy, their transitive dependencies, and bundled codecs retain their own copyright and license terms; preserve the notices shipped with each distribution.

NVIDIA, GeForce RTX, NGX, and DLSS are trademarks and/or registered trademarks of NVIDIA Corporation. FFmpeg, ReShade, RenoDX, Python, and other names belong to their respective owners. Codec patent or other permissions may also be required depending on jurisdiction and use. Review the controlling licenses before building or distributing a complete package.
