# DLSS / ReShade runtime files (not in git — restore manually)

The proprietary, closed-source, or prebuilt Windows runtime files are
intentionally not included in this repository. Several exceed GitHub's
100 MB per-file push limit, so the full set ships only inside the portable
ZIP published under GitHub Releases. The repository's MIT license does not
cover these files and grants no right to obtain or redistribute them.

## Required layout

`src/core/runtime.py::validate_runtime_files()` enforces this exact
`host/` / `dlss/` / `dlssg/` split — flat layouts are rejected at startup:

```text
bin/runtime/host/dxgi.dll                <- ReShade 6.8.0 x64 WITH full add-on support
bin/runtime/host/LICENSE-ReShade.txt     <- BSD-3-Clause, committed in this folder
bin/runtime/host/renodx-dlss5.addon64    <- RenoDX nightly-era build v0.2026.0828.0517
bin/runtime/host/LICENSE-RenoDX.txt      <- MIT, committed in this folder
bin/runtime/host/nvngx_dlssnr.dll        <- copy of the selected dlssnr/ arch build below (RenoDX requires it next to itself)
bin/runtime/host/LICENSE-NVIDIA-DLSS.txt <- NVIDIA RTX SDKs License, committed in this folder
bin/runtime/host/nvngx.dll               <- project-built D3D12 worker (MIT, see below)
bin/runtime/dlss/nvngx_dlss.dll          <- NVIDIA DLSS 310.9.0.0 (Super Resolution)
bin/runtime/dlss/LICENSE-NVIDIA-DLSS.txt <- same NVIDIA text, keep a copy next to the DLL
bin/runtime/dlssg/nvngx_dlssg.dll        <- NVIDIA DLSS-G 310.9.0.0 (Frame Generation, NGX path)
bin/runtime/dlssg/dlssg-worker.exe       <- project-built DLSSG worker (MIT)
bin/runtime/dlssg/LICENSE-NVIDIA-DLSS.txt<- same NVIDIA text, keep a copy next to the DLLs
bin/runtime/dlssnr/Turing+/nvngx_dlssnr.dll         <- NVIDIA DLSSNR 310.8 (Turing and higher arch build)
bin/runtime/dlssnr/Turing+/LICENSE-NVIDIA-DLSS.txt  <- same NVIDIA text, keep a copy next to the DLL
bin/runtime/dlssnr/Ada Lovelace+/nvngx_dlssnr.dll         <- NVIDIA DLSSNR 310.8 (Ada Lovelace and higher arch build)
bin/runtime/dlssnr/Ada Lovelace+/LICENSE-NVIDIA-DLSS.txt  <- same NVIDIA text, keep a copy next to the DLL
bin/runtime/dlssnr/Blackwell+/nvngx_dlssnr.dll         <- NVIDIA DLSSNR 310.8 (Blackwell and higher arch build)
bin/runtime/dlssnr/Blackwell+/LICENSE-NVIDIA-DLSS.txt  <- same NVIDIA text, keep a copy next to the DLL
```

`ReShade.ini` is NOT listed: it is auto-generated runtime state. The
worker rewrites every `[RenoDX.DLSS5]` key from the UI settings on each
run — never commit it.

## Where to get each file

- **ReShade (`dxgi.dll`):** https://reshade.me/ — download **ReShade 6.8.0
  with full add-on support** (the standard build cannot load unsigned
  `.addon64` files). Rename the 64-bit DLL to `dxgi.dll`.
- **RenoDX add-on:** https://github.com/clshortfuse/renodx/releases —
  nightly-era build `v0.2026.0828.0517`, file `renodx-dlss5.addon64`.
  The RenoDX framework is MIT-licensed; this names the framework license,
  not a separate grant for the binary — use only builds you are
  authorized to redistribute.
- **NVIDIA DLLs (`nvngx_dlss.dll`, `nvngx_dlssg.dll`, `nvngx_dlssnr.dll`):**
  genuine NVIDIA SDK/driver/game distributions only
  (https://github.com/NVIDIA/DLSS, NVIDIA App, GeForce drivers). They are
  governed by the [NVIDIA RTX SDK License](https://github.com/NVIDIA/DLSS/blob/main/LICENSE.txt):
  no standalone redistribution, NVIDIA-GPU-only use, and no implying
  NVIDIA sponsorship. Tested FileVersions: DLSS `310.9.0.0`, DLSS-G
  `310.9.0.0`, DLSSNR Turing+ `310.8.SF.0` (`NVIDIA DLSSNR`), DLSSNR Ada /
  Blackwell `310.8.0.0` — verify authenticity before use.
- **DLSS Architecture (`dlssnr/` + `host/nvngx_dlssnr.dll`):** see
  `src/core/dlss_architecture.py`. `Auto` maps RTX 20/30 to `Turing+`,
  RTX 40 to `Ada Lovelace+`, RTX 50 to `Blackwell+` (unknown GPUs fall back
  to `Turing+`); explicit choices copy that subfolder over
  `host/nvngx_dlssnr.dll` at startup and before each render (identical
  content is skipped, failures only warn to `logs/dlssnr_architecture.log`).
  Missing arch DLLs are restored from `bin/Turing+.zip`,
  `bin/Ada Lovelace+.zip`, `bin/Blackwell+.zip` (each must contain only
  `nvngx_dlssnr.dll); the release ZIP already ships the extracted
  `dlssnr/` folders. `host/nvngx_dlssnr.dll` is therefore not an independent
  binary — its hash always equals one of the three arch hashes below.
- **Project-built (`host/nvngx.dll`, `dlssg/dlssg-worker.exe`):** built
  from the project's own MIT-licensed native sources
  (`DLSS5-Feeder`, `Frame-Interpolation`; not published in this repo).
  No download needed if you have the release ZIP; otherwise build them
  from the published sources.

## Integrity checklist (SHA256 of the tested files)

```text
dxgi.dll                0CEE63F9C9F13F3AC909C5B4903F4DBB4B719A7AB3B4F13B0DEAF83C814B94F7
renodx-dlss5.addon64    D5ADF82EB44B065F4C590AC91FE824BAB07AFEA0EB9F994BDE936710C8593952
nvngx_dlss.dll          07C7FEA19A24C75102BF34D1A4A775640CE323A0B2265D4887B331E62486C44F
nvngx_dlssg.dll         C64928FDB7C48A57722EA8EEF2662171EDC323473ADEA66C29A206A23F1A2BED
dlssnr Turing+          6EB209E764F39872625DEBD6ABAF45E2BB6322F6F270F781F70C059AE30B3927
dlssnr Ada Lovelace+    4B8D19BC3EFF58A084F5ECA7489C921501C203450169FB82FF4F649A4482BA05
dlssnr Blackwell+       E16BCF15E16E13F527491CDF7845B2FE6521A738D8F7C9C721866A8496E1FC8E
```

This project is not affiliated with or endorsed by NVIDIA, ReShade, or RenoDX.
