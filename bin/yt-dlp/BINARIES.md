# yt-dlp resolver (not in git — download separately)

The executable resolver is intentionally not included in this repository.
It ships only inside the portable ZIP published under GitHub Releases.

## Tested build

- Source: https://github.com/yt-dlp/yt-dlp
- Version: `2026.08.19` (Windows standalone `yt-dlp.exe`)
- License: The Unlicense (public domain) — the `LICENSE` and
  `THIRD_PARTY_LICENSES.txt` files in this folder are part of that
  distribution; keep them next to the binary.

## Required layout

Place the file exactly like this (path is hardcoded in `src/core/paths.py`
and used by `src/live/source_resolver.py::resolve_source()` and
`src/live/player.py::check_live_binaries()`):

```text
bin/yt-dlp/LICENSE                  <- already in this folder, keep it
bin/yt-dlp/THIRD_PARTY_LICENSES.txt <- already in this folder, keep it
bin/yt-dlp/yt-dlp.exe               <- standalone Windows executable
```

## Integrity checklist (SHA256 of the tested file)

```text
yt-dlp.exe   66674953FE251B89F4D08C5F0E35E0728679BD67AB3D7D05C0562AF101DD3E7A
```

## Notes

- Optional: only Live YouTube/Twitch page URLs require it. Local files and
  direct stream URLs work without it.
- A mismatch means a different build — resolution may still work, but Live
  page resolution was verified only against the build above.
