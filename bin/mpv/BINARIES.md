# MPV player (not in git — download separately)

The executable player is intentionally not included in this repository.
It ships only inside the portable ZIP published under GitHub Releases.

## Tested build

- Source: https://mpv.io/ (Windows 64-bit)
- Version: `v0.41.0-dev-g41f6a6450` built on Dec 21 2025 (libplacebo v7.358.0)
- License: GPLv2+ by default, LGPLv2.1+ build mode — the `LICENSE.GPL`,
  `LICENSE.LGPL`, and `Copyright` files in this folder are part of that
  distribution; keep them next to the binary. They and their dependencies
  retain their own copyright and license terms.

## Required layout

Place the files exactly like this (paths are hardcoded in `src/core/paths.py`
and used by `src/live/player.py::launch_mpv()` / `check_live_binaries()`):

```text
bin/mpv/LICENSE.GPL    <- already in this folder, keep it
bin/mpv/LICENSE.LGPL   <- already in this folder, keep it
bin/mpv/Copyright      <- already in this folder, keep it
bin/mpv/mpv.exe        <- from the mpv Windows build
bin/mpv/vulkan-1.dll   <- from the mpv Windows build
bin/mpv/d3dcompiler_43.dll <- from the mpv Windows build
```

Only `mpv.exe` is required by the app (`MPV.is_file()` gate). The DLLs ship
with the same Windows build and are needed for playback on most systems.

## Integrity checklist (SHA256 of the tested files)

```text
mpv.exe   6145E63F026451A764077D53FD60860EC9F5C2BC76DCD6E62A88967AC375453D
```

## Notes

- Optional: only Live sessions with **Open in MPV** require it. Image, Video,
  and Frame Interpolation work without it.
- A mismatch means a different build — playback may still work, but Live
  behavior was verified only against the build above.
