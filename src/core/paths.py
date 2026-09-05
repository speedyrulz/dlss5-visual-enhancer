from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RUNTIME = ROOT / "bin" / "runtime"
HOST_DIR = RUNTIME / "host"
DLSS_DIR = RUNTIME / "dlss"
DLSSG_DIR = RUNTIME / "dlssg"
# Per-architecture Neural Rendering builds (see src/core/dlss_architecture.py).
# The live copy stays at NEURAL_RUNTIME in host/ (RenoDX discovery rule).
DLSSNR_DIR = RUNTIME / "dlssnr"
FFMPEG = ROOT / "bin" / "ffmpeg" / "bin" / "ffmpeg.exe"
FFPROBE = ROOT / "bin" / "ffmpeg" / "bin" / "ffprobe.exe"
WORKER = HOST_DIR / "nvngx.dll"  # Signed-snippet caller checks require this image name.
ADDON = HOST_DIR / "renodx-dlss5.addon64"
HOST_DXGI = HOST_DIR / "dxgi.dll"
RESHADE_LOG = HOST_DIR / "ReShade.log"
DLSS_SUPERRES = DLSS_DIR / "nvngx_dlss.dll"
# The RenoDX add-on only discovers the signed NR runtime next to itself in
# host/, so NEURAL_RUNTIME lives there even though it is a DLSS component.
NEURAL_RUNTIME = HOST_DIR / "nvngx_dlssnr.dll"
# Live-tab externals (vendored, optional: only Live sessions require them).
MPV = ROOT / "bin" / "mpv" / "mpv.exe"
YTDLP = ROOT / "bin" / "yt-dlp" / "yt-dlp.exe"
# Ephemeral per-session HLS working dirs for Live (swept on stop/startup).
LIVE_DIR = ROOT / "live"
OUTPUTS = ROOT / "outputs"
LOGS = ROOT / "logs"
JOBS = ROOT / "jobs"
CONFIG_DIR = ROOT / "config"
CONFIG_PATH = CONFIG_DIR / "config.ini"
