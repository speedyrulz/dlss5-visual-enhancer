# FFmpeg binaries (not in git — download separately)

The executable binaries are intentionally not included in this repository.
Each file exceeds GitHub's 100 MB per-file push limit, so they ship only
inside the portable ZIP published under GitHub Releases.

## Tested build

- Source: https://www.gyan.dev/ffmpeg/builds/ (Windows 64-bit static)
- Version: `9.0.1-full_build-www.gyan.dev`
- License: GPLv3 — the `LICENSE` and `README.txt` files in this folder are
  part of that distribution; keep them next to the binaries. Anyone
  redistributing the binaries must satisfy the GPLv3 and
  corresponding-source obligations (the exact upstream source commit is
  recorded in `README.txt`).

## Required layout

Place the files exactly like this (paths are hardcoded in `src/core/paths.py`):

```text
bin/ffmpeg/LICENSE            <- already in this folder, keep it
bin/ffmpeg/README.txt         <- already in this folder, keep it
bin/ffmpeg/bin/ffmpeg.exe     <- from the gyan.dev full build
bin/ffmpeg/bin/ffprobe.exe    <- from the gyan.dev full build
```

Note: the executables live in the inner `bin/` subfolder, not directly
under `bin/ffmpeg/`.

## Integrity checklist (SHA256 of the tested files)

```text
ffmpeg.exe   57C56E369D5B4873B4D93FC1A1D833CB7CD8BC9325C14B05C34CE60B22842D8A
ffprobe.exe  AFE05347CAAABE479B3C4EAE71992B6EC1E11C57266A1D665DEB0F9FE9847208
```

A mismatch means a different build — the application may still work, but
encoding behavior (NVENC, libx264/libx265, libsvtav1, ProRes) was verified
only against the build above.
