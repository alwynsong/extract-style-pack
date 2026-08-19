#!/usr/bin/env python3
"""使用 FFmpeg 执行场景变化抽帧、均匀补帧，并调用质量筛选与去重。"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import hashlib
import io
import math

import cv2
import numpy as np
from PIL import Image, ImageCms, ImageOps

IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.webp', '.tif', '.tiff', '.bmp'}

@dataclass
class ImageMetrics:
    filename: str
    source_path: str
    sha256: str | None = None
    width: int | None = None
    height: int | None = None
    sharpness: float | None = None
    mean_luminance: float | None = None
    dark_pixel_share: float | None = None
    bright_pixel_share: float | None = None
    phash: str | None = None
    dhash: str | None = None
    status: str = 'pending'
    kept_path: str | None = None
    duplicate_of: str | None = None
    phash_distance: int | None = None
    dhash_distance: int | None = None
    histogram_correlation: float | None = None
    source_icc_profile: bool | None = None
    color_conversion: str | None = None
    review_required: bool = False
    filter_reason: str | None = None
    error: str | None = None

@dataclass
class AnalyzedImage:
    path: Path
    metrics: ImageMetrics
    phash_value: int
    dhash_value: int
    histogram: np.ndarray
    quality_score: float

def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        while (chunk := stream.read(4 * 1024 * 1024)):
            digest.update(chunk)
    return digest.hexdigest()

def hamming_distance(first: int, second: int) -> int:
    return (first ^ second).bit_count()

def hash_hex(value: int) -> str:
    return f'{value:016x}'

def perceptual_hash(gray: np.ndarray) -> int:
    resized = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32)
    dct = cv2.dct(resized)
    low = dct[:8, :8].flatten()
    median = float(np.median(low[1:]))
    bits = low > median
    value = 0
    for bit in bits:
        value = value << 1 | int(bit)
    return value

def difference_hash(gray: np.ndarray) -> int:
    resized = cv2.resize(gray, (9, 8), interpolation=cv2.INTER_AREA)
    bits = resized[:, 1:] > resized[:, :-1]
    value = 0
    for bit in bits.flatten():
        value = value << 1 | int(bit)
    return value

def color_histogram(rgb: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    histogram = cv2.calcHist([hsv], [0, 1], None, [36, 16], [0, 180, 0, 256])
    return cv2.normalize(histogram, histogram).flatten().astype(np.float32)

def normalized_rgb(path: Path) -> tuple[Image.Image, bool, str]:
    with Image.open(path) as source:
        oriented = ImageOps.exif_transpose(source)
        icc_profile = source.info.get('icc_profile')
        if isinstance(icc_profile, bytes) and icc_profile:
            try:
                source_profile = ImageCms.ImageCmsProfile(io.BytesIO(icc_profile))
                target_profile = ImageCms.createProfile('sRGB')
                converted = ImageCms.profileToProfile(oriented, source_profile, target_profile, outputMode='RGB')
                if converted is not None:
                    return (converted.copy(), True, 'icc_to_srgb')
                return (oriented.convert('RGB'), True, 'icc_fallback_rgb')
            except Exception:
                return (oriented.convert('RGB'), True, 'icc_fallback_rgb')
        return (oriented.convert('RGB'), False, 'assumed_srgb')

def image_array(path: Path) -> tuple[np.ndarray, bool, str]:
    image, has_icc, conversion = normalized_rgb(path)
    try:
        return (np.asarray(image, dtype=np.uint8), has_icc, conversion)
    finally:
        image.close()

def quality_score(width: int, height: int, sharpness: float) -> float:
    resolution = math.log2(max(width * height, 1))
    sharpness_bonus = min(math.log1p(max(sharpness, 0.0)), 6.0)
    return resolution * 10.0 + sharpness_bonus

def analyze_image(path: Path, blur_threshold: float, min_short_edge: int) -> AnalyzedImage:
    metrics = ImageMetrics(filename=path.name, source_path=str(path.resolve()))
    rgb, has_icc, conversion = image_array(path)
    metrics.source_icc_profile = has_icc
    metrics.color_conversion = conversion
    height, width = rgb.shape[:2]
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    mean_luminance = float(gray.mean())
    dark_share = float((gray < 12).mean())
    bright_share = float((gray > 243).mean())
    phash_value = perceptual_hash(gray)
    dhash_value = difference_hash(gray)
    metrics.sha256 = sha256_file(path)
    metrics.width = width
    metrics.height = height
    metrics.sharpness = round(sharpness, 4)
    metrics.mean_luminance = round(mean_luminance, 4)
    metrics.dark_pixel_share = round(dark_share, 6)
    metrics.bright_pixel_share = round(bright_share, 6)
    metrics.phash = hash_hex(phash_value)
    metrics.dhash = hash_hex(dhash_value)
    if mean_luminance < 6 and dark_share > 0.985:
        metrics.status = 'filtered'
        metrics.filter_reason = 'black_frame'
    elif mean_luminance > 249 and bright_share > 0.985:
        metrics.status = 'filtered'
        metrics.filter_reason = 'white_frame'
    elif min_short_edge and min(width, height) < min_short_edge:
        metrics.status = 'filtered'
        metrics.filter_reason = f'short_edge_below_{min_short_edge}'
    elif blur_threshold > 0 and sharpness < blur_threshold:
        metrics.review_required = True
    return AnalyzedImage(path=path, metrics=metrics, phash_value=phash_value, dhash_value=dhash_value, histogram=color_histogram(rgb), quality_score=quality_score(width, height, sharpness))

def image_files(input_dir: Path, recursive: bool) -> list[Path]:
    iterator = input_dir.rglob('*') if recursive else input_dir.glob('*')
    return sorted((path for path in iterator if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS), key=lambda item: str(item).casefold())

def unique_destination(directory: Path, name: str) -> Path:
    destination = directory / name
    if not destination.exists():
        return destination
    stem, suffix = (Path(name).stem, Path(name).suffix)
    index = 2
    while (directory / f'{stem}__{index}{suffix}').exists():
        index += 1
    return directory / f'{stem}__{index}{suffix}'

def save_normalized(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    image, _, _ = normalized_rgb(source)
    try:
        suffix = destination.suffix.lower()
        if suffix in {'.jpg', '.jpeg'}:
            image.save(destination, format='JPEG', quality=95, optimize=True)
        elif suffix == '.png':
            image.save(destination, format='PNG', optimize=True)
        elif suffix == '.webp':
            image.save(destination, format='WEBP', quality=95, method=6)
        elif suffix in {'.tif', '.tiff'}:
            image.save(destination, format='TIFF', compression='tiff_lzw')
        elif suffix == '.bmp':
            image.save(destination, format='BMP')
        else:
            image.save(destination, format='PNG')
    finally:
        image.close()

def copy_rejected(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)

def portable_path(path: str | None, root: Path | None) -> str | None:
    if not path:
        return None
    candidate = Path(path).resolve()
    if root is not None:
        try:
            return candidate.relative_to(root.resolve()).as_posix()
        except ValueError:
            pass
    return Path(path).name

def mark_exact_duplicates(items: list[AnalyzedImage]) -> None:
    by_hash: dict[str, list[AnalyzedImage]] = {}
    for item in items:
        if item.metrics.status == 'pending' and item.metrics.sha256:
            by_hash.setdefault(item.metrics.sha256, []).append(item)
    for group in by_hash.values():
        if len(group) < 2:
            continue
        keeper = max(group, key=lambda row: row.quality_score)
        for item in group:
            if item is keeper:
                continue
            item.metrics.status = 'duplicate'
            item.metrics.duplicate_of = keeper.metrics.filename
            item.metrics.filter_reason = 'exact_duplicate'

def mark_near_duplicates(items: list[AnalyzedImage], phash_limit: int, dhash_limit: int, histogram_limit: float) -> None:
    candidates = sorted((item for item in items if item.metrics.status == 'pending'), key=lambda row: row.quality_score, reverse=True)
    keepers: list[AnalyzedImage] = []
    for item in candidates:
        duplicate: tuple[AnalyzedImage, int, int, float] | None = None
        for keeper in keepers:
            p_distance = hamming_distance(item.phash_value, keeper.phash_value)
            if p_distance > phash_limit:
                continue
            d_distance = hamming_distance(item.dhash_value, keeper.dhash_value)
            if d_distance > dhash_limit:
                continue
            correlation = float(cv2.compareHist(item.histogram, keeper.histogram, cv2.HISTCMP_CORREL))
            if correlation >= histogram_limit:
                duplicate = (keeper, p_distance, d_distance, correlation)
                break
        if duplicate is None:
            keepers.append(item)
            continue
        keeper, p_distance, d_distance, correlation = duplicate
        item.metrics.status = 'duplicate'
        item.metrics.duplicate_of = keeper.metrics.filename
        item.metrics.phash_distance = p_distance
        item.metrics.dhash_distance = d_distance
        item.metrics.histogram_correlation = round(correlation, 6)
        item.metrics.filter_reason = 'near_duplicate'

def deduplicate_directory(input_dir: Path, output_dir: Path, rejected_dir: Path | None, *, recursive: bool=False, phash_distance: int=6, dhash_distance: int=8, histogram_correlation: float=0.85, blur_threshold: float=35.0, min_short_edge: int=360) -> list[ImageMetrics]:
    paths = image_files(input_dir, recursive)
    analyzed: list[AnalyzedImage] = []
    failures: list[ImageMetrics] = []
    for path in paths:
        try:
            analyzed.append(analyze_image(path, blur_threshold, min_short_edge))
        except Exception as exc:
            failures.append(ImageMetrics(filename=path.name, source_path=str(path.resolve()), status='failed', filter_reason='decode_error', error=str(exc)))
    mark_exact_duplicates(analyzed)
    mark_near_duplicates(analyzed, phash_distance, dhash_distance, histogram_correlation)
    for item in analyzed:
        if item.metrics.status == 'pending':
            destination = unique_destination(output_dir, item.path.name)
            save_normalized(item.path, destination)
            item.metrics.status = 'kept'
            item.metrics.kept_path = str(destination.resolve())
        elif rejected_dir is not None:
            destination = unique_destination(rejected_dir, item.path.name)
            copy_rejected(item.path, destination)
    return [item.metrics for item in analyzed] + failures

IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.webp', '.tif', '.tiff', '.bmp'}

SHOWINFO_TIME = re.compile(r"pts_time:([0-9]+(?:\.[0-9]+)?)")
IMAGE_NAME = re.compile(
    r"^(?P<video>VID-\d{4})__(?P<timestamp>\d{12})__(?P<method>scene|uniform|boundary)__(?P<sequence>\d{4})\.jpg$"
)
MACOS_EXECUTABLE_DIRS = (Path("/opt/homebrew/bin"), Path("/usr/local/bin"))

def windows_executable_candidates(command: str) -> list[Path]:
    executable = command if command.lower().endswith(".exe") else f"{command}.exe"
    candidates: list[Path] = []
    local_app_data = os.environ.get("LOCALAPPDATA")
    program_data = os.environ.get("PROGRAMDATA")
    program_files = os.environ.get("ProgramFiles")
    user_profile = os.environ.get("USERPROFILE")
    if program_data:
        candidates.append(Path(program_data) / "chocolatey" / "bin" / executable)
    if program_files:
        candidates.append(Path(program_files) / "ffmpeg" / "bin" / executable)
    if user_profile:
        candidates.append(Path(user_profile) / "scoop" / "shims" / executable)
    if local_app_data and command.lower().removesuffix(".exe") in {"ffmpeg", "ffprobe"}:
        package_root = Path(local_app_data) / "Microsoft" / "WinGet" / "Packages"
        if package_root.is_dir():
            for package_dir in sorted(package_root.glob("Gyan.FFmpeg*"), reverse=True):
                candidates.extend(sorted(package_dir.glob(f"ffmpeg-*/bin/{executable}"), reverse=True))
    return candidates


def find_executable(command: str) -> str | None:
    resolved = shutil.which(command)
    if resolved:
        return resolved
    candidate = Path(command).expanduser()
    if candidate.is_file():
        return str(candidate.resolve())
    if candidate.name == command:
        for directory in MACOS_EXECUTABLE_DIRS:
            fallback = directory / command
            if fallback.is_file():
                return str(fallback)
        if os.name == "nt":
            for fallback in windows_executable_candidates(command):
                if fallback.is_file():
                    return str(fallback.resolve())
    return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="从视频中提取可用于风格分析的代表帧")
    parser.add_argument("video", type=Path, nargs="?", help="待抽帧的本地视频；--check 时省略")
    parser.add_argument("--asset-id", help="视频素材 ID，例如 VID-0001")
    parser.add_argument("--output-dir", type=Path, help="该视频的抽帧目录")
    parser.add_argument("--check", action="store_true", help="检查 yt-dlp、FFmpeg 与 FFprobe 后退出")
    parser.add_argument("--yt-dlp", default="yt-dlp", help="yt-dlp 命令或绝对路径")
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--ffprobe", default="ffprobe")
    parser.add_argument("--scene-threshold", type=float, default=0.32)
    parser.add_argument("--max-kept", type=int, help="覆盖按时长计算的最终保留上限")
    parser.add_argument(
        "--blur-threshold",
        type=float,
        default=0.0,
        help="低于该清晰度时标记人工复核；0 表示关闭，不会因模糊自动过滤帧",
    )
    parser.add_argument("--phash-distance", type=int, default=6)
    parser.add_argument("--dhash-distance", type=int, default=8)
    parser.add_argument("--histogram-correlation", type=float, default=0.85)
    parser.add_argument("--overwrite", action="store_true", help="清理已有抽帧目录后重新生成")
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def run(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, capture_output=True, text=True, encoding="utf-8", errors="replace")



def first_output_line(completed: subprocess.CompletedProcess[str]) -> str:
    output = completed.stdout.strip() or completed.stderr.strip()
    return output.splitlines()[0] if output else "unknown"


def check_dependencies(args: argparse.Namespace) -> int:
    ffmpeg = find_executable(args.ffmpeg)
    ffprobe = find_executable(args.ffprobe)
    yt_dlp = find_executable(args.yt_dlp)
    yt_dlp_command: list[str] | None = [yt_dlp] if yt_dlp else None

    if yt_dlp_command is not None:
        executable_check = run([*yt_dlp_command, "--version"])
        if executable_check.returncode != 0:
            yt_dlp_command = None
    if yt_dlp_command is None:
        module_check = run([sys.executable, "-m", "yt_dlp", "--version"])
        if module_check.returncode == 0:
            yt_dlp_command = [sys.executable, "-m", "yt_dlp"]

    checks: list[tuple[str, list[str] | None]] = [
        ("yt-dlp", yt_dlp_command),
        ("ffmpeg", [ffmpeg] if ffmpeg else None),
        ("ffprobe", [ffprobe] if ffprobe else None),
    ]
    missing: list[str] = []
    for name, command in checks:
        label = name.upper().replace("-", "_")
        if command is None:
            missing.append(name)
            print(f"{label}\tmissing")
            continue
        version_args = ["--version"] if name == "yt-dlp" else ["-version"]
        completed = run([*command, *version_args])
        if completed.returncode != 0:
            missing.append(name)
            print(f"{label}\tfailed\t{' '.join(command)}")
            continue
        print(f"{label}\tready\t{' '.join(command)}\t{first_output_line(completed)}")

    if missing:
        print(f"MISSING\t{','.join(missing)}", file=sys.stderr)
        print(
            "RECOVERY\t安装缺失依赖后重新运行 --check；不要安装同名 ffmpeg Python 包",
            file=sys.stderr,
        )
        print("STATUS\tblocked")
        return 2
    print("STATUS\tready")
    return 0


def probe_video(video: Path, ffprobe: str) -> dict[str, Any]:
    completed = run(
        [
            ffprobe,
            "-v",
            "error",
            "-print_format",
            "json",
            "-show_format",
            "-show_streams",
            str(video),
        ]
    )
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr.strip() or "ffprobe 失败")
    payload = json.loads(completed.stdout)
    streams = payload.get("streams", [])
    videos = [row for row in streams if isinstance(row, dict) and row.get("codec_type") == "video"]
    if not videos:
        raise RuntimeError("no_video_stream")
    stream = videos[0]
    format_row = payload.get("format") if isinstance(payload.get("format"), dict) else {}
    raw_duration = stream.get("duration") or format_row.get("duration")
    try:
        duration = float(raw_duration)
    except (TypeError, ValueError) as exc:
        raise RuntimeError("无法确定视频时长") from exc
    if duration <= 0:
        raise RuntimeError("视频时长无效")
    return {
        "duration_seconds": duration,
        "width": int(stream.get("width") or 0),
        "height": int(stream.get("height") or 0),
        "codec": stream.get("codec_name"),
        "pixel_format": stream.get("pix_fmt"),
        "average_frame_rate": stream.get("avg_frame_rate"),
        "color_space": stream.get("color_space"),
        "color_transfer": stream.get("color_transfer"),
        "color_primaries": stream.get("color_primaries"),
    }


def target_counts(duration: float) -> tuple[int, int]:
    if duration <= 30:
        return 18, 18
    if duration <= 120:
        return 30, 28
    if duration <= 300:
        return 45, 40
    if duration <= 1200:
        return 75, 70
    if duration <= 3600:
        return 120, 100
    return 160, 120


def scene_thresholds(initial: float) -> list[float]:
    values = [initial, 0.26, 0.20, 0.38, 0.45, 0.55]
    unique: list[float] = []
    for value in values:
        normalized = min(0.95, max(0.05, value))
        if normalized not in unique:
            unique.append(normalized)
    return unique


def hdr_to_srgb_filter(probe: dict[str, Any]) -> tuple[str | None, str]:
    transfer = str(probe.get("color_transfer") or "").lower()
    primaries = str(probe.get("color_primaries") or "").lower()
    if transfer in {"smpte2084", "arib-std-b67"} or primaries == "bt2020":
        source_transfer = "smpte2084" if transfer == "smpte2084" else "arib-std-b67"
        return (
            f"zscale=transferin={source_transfer}:primariesin=bt2020:matrixin=bt2020nc:rangein=tv,"
            "zscale=transfer=linear:npl=100,tonemap=tonemap=hable:desat=0,"
            "zscale=transfer=bt709:primaries=bt709:matrix=bt709:range=tv,format=yuv420p",
            "hdr_to_bt709_srgb_hable",
        )
    return None, "sdr_passthrough"


def extract_scene_candidates(
    video: Path, directory: Path, ffmpeg: str, threshold: float, color_filter: str | None
) -> tuple[list[Path], list[float], str]:
    directory.mkdir(parents=True, exist_ok=True)
    pattern = directory / "scene_%05d.jpg"
    select_filter = f"select='gt(scene,{threshold:.4f})',showinfo"
    filter_value = f"{color_filter},{select_filter}" if color_filter else select_filter
    completed = run(
        [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "info",
            "-i",
            str(video),
            "-an",
            "-vf",
            filter_value,
            "-fps_mode",
            "vfr",
            "-q:v",
            "2",
            "-y",
            str(pattern),
        ]
    )
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr[-2000:] or "FFmpeg 场景抽帧失败")
    paths = sorted(directory.glob("scene_*.jpg"))
    timestamps = [float(match.group(1)) for match in SHOWINFO_TIME.finditer(completed.stderr)]
    if len(timestamps) != len(paths):
        timestamps = timestamps[: len(paths)]
        if len(timestamps) < len(paths):
            timestamps.extend([0.0] * (len(paths) - len(timestamps)))
    return paths, timestamps, completed.stderr


def choose_scene_run(
    video: Path,
    ffmpeg: str,
    temp_root: Path,
    initial: float,
    target: int,
    color_filter: str | None,
) -> tuple[float, list[Path], list[float]]:
    desired = max(4, round(target * 0.62))
    trials: list[tuple[float, list[Path], list[float]]] = []
    for threshold in scene_thresholds(initial):
        trial_dir = temp_root / f"scene_{threshold:.2f}"
        paths, timestamps, _ = extract_scene_candidates(
            video, trial_dir, ffmpeg, threshold, color_filter
        )
        trials.append((threshold, paths, timestamps))
        if desired * 0.65 <= len(paths) <= desired * 1.35:
            break
    return min(trials, key=lambda row: abs(len(row[1]) - desired))


def evenly_spaced_times(duration: float, count: int) -> list[float]:
    if count <= 0:
        return []
    margin = min(0.5, duration * 0.08)
    start, end = margin, max(margin, duration - margin)
    if count == 1:
        return [(start + end) / 2.0]
    step = (end - start) / (count + 1)
    return [start + step * (index + 1) for index in range(count)]


def far_from_existing(timestamp: float, existing: list[float], min_gap: float = 0.5) -> bool:
    return all(abs(timestamp - value) >= min_gap for value in existing)


def extract_timestamp(
    video: Path, output: Path, timestamp: float, ffmpeg: str, color_filter: str | None
) -> None:
    command = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-ss",
        f"{timestamp:.6f}",
        "-i",
        str(video),
        "-an",
    ]
    if color_filter:
        command.extend(("-vf", color_filter))
    command.extend(("-frames:v", "1", "-q:v", "2", "-y", str(output)))
    completed = run(command)
    if completed.returncode != 0 or not output.is_file():
        raise RuntimeError(completed.stderr.strip() or f"无法提取 {timestamp:.3f}s")


def frame_name(asset_id: str, timestamp: float, method: str, sequence: int) -> str:
    milliseconds = max(0, round(timestamp * 1000.0))
    return f"{asset_id}__{milliseconds:012d}__{method}__{sequence:04d}.jpg"


def copy_scene_frames(
    paths: list[Path], timestamps: list[float], output_dir: Path, asset_id: str, max_count: int
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    last_timestamp = -1000.0
    sequence = 1
    for source, timestamp in zip(paths, timestamps):
        if timestamp <= 0 or timestamp - last_timestamp < 0.35:
            continue
        destination = output_dir / frame_name(asset_id, timestamp, "scene", sequence)
        shutil.copy2(source, destination)
        records.append({"path": destination, "timestamp": timestamp, "method": "scene"})
        sequence += 1
        last_timestamp = timestamp
        if len(records) >= max_count:
            break
    return records


def limit_kept(records: list[ImageMetrics], max_kept: int) -> None:
    kept = [row for row in records if row.status == "kept"]
    if len(kept) <= max_kept:
        return
    kept.sort(key=lambda row: timestamp_from_name(row.filename))
    selected_indexes = {
        round(index * (len(kept) - 1) / (max_kept - 1)) if max_kept > 1 else len(kept) // 2
        for index in range(max_kept)
    }
    selected = {kept[index].filename for index in sorted(selected_indexes)}
    for row in kept:
        if row.filename not in selected:
            row.status = "filtered"
            row.filter_reason = "max_kept_limit"
            if row.kept_path:
                path = Path(row.kept_path)
                if path.exists():
                    path.unlink()
            row.kept_path = None


def timestamp_from_name(name: str) -> int:
    match = IMAGE_NAME.match(name)
    return int(match.group("timestamp")) if match else 0


def metadata_from_name(name: str) -> dict[str, Any]:
    match = IMAGE_NAME.match(name)
    if not match:
        return {"timestamp_ms": None, "method": None}
    return {
        "timestamp_ms": int(match.group("timestamp")),
        "method": match.group("method"),
    }


def portable_path(path: str | None, base: Path) -> str | None:
    if not path:
        return None
    candidate = Path(path)
    try:
        return candidate.resolve().relative_to(base.resolve()).as_posix()
    except ValueError:
        return candidate.name


def main() -> int:
    args = parse_args()
    if args.check:
        return check_dependencies(args)
    if args.video is None:
        print("缺少本地视频路径；仅检查环境请使用 --check", file=sys.stderr)
        return 2
    if args.asset_id is None:
        print("缺少 --asset-id，例如 VID-0001", file=sys.stderr)
        return 2
    if args.output_dir is None:
        print("缺少 --output-dir", file=sys.stderr)
        return 2
    video = args.video.expanduser().resolve()
    if not video.is_file():
        print(f"视频不存在：{video}", file=sys.stderr)
        return 2
    if not re.fullmatch(r"VID-\d{4}", args.asset_id):
        print("--asset-id 必须类似 VID-0001", file=sys.stderr)
        return 2
    ffmpeg = find_executable(args.ffmpeg)
    ffprobe = find_executable(args.ffprobe)
    if ffmpeg is None or ffprobe is None:
        missing = [name for name, value in (("ffmpeg", ffmpeg), ("ffprobe", ffprobe)) if value is None]
        print(f"缺少依赖：{', '.join(missing)}", file=sys.stderr)
        print("先运行 scripts/extract_video_frames.py --check 定位缺失项", file=sys.stderr)
        return 2

    output_dir = args.output_dir.expanduser().resolve()
    expected_parent = output_dir.parent
    if output_dir.name != args.asset_id or expected_parent.name != "视频抽帧":
        print(
            "--output-dir 必须是 <风格包>/参考素材/视频抽帧/<asset-id>，且目录名与 --asset-id 一致",
            file=sys.stderr,
        )
        return 2
    if output_dir.exists() and any(output_dir.iterdir()):
        if not args.overwrite:
            print(f"输出目录不是空目录；使用 --overwrite 才能重新生成：{output_dir}", file=sys.stderr)
            return 2
        if output_dir.is_symlink():
            print(f"拒绝覆盖符号链接目录：{output_dir}", file=sys.stderr)
            return 2
        shutil.rmtree(output_dir)
    raw_dir = output_dir / "原始抽帧"
    kept_dir = output_dir / "保留帧"
    rejected_dir = output_dir / "被过滤帧"
    for directory in (raw_dir, kept_dir, rejected_dir):
        directory.mkdir(parents=True, exist_ok=True)

    try:
        probe = probe_video(video, ffprobe)
        color_filter, color_conversion = hdr_to_srgb_filter(probe)
        probe["frame_color_conversion"] = color_conversion
        duration = float(probe["duration_seconds"])
        raw_target, default_max_kept = target_counts(duration)
        max_kept = max(3, args.max_kept or default_max_kept)
        frame_records: list[dict[str, Any]] = []

        with tempfile.TemporaryDirectory(prefix="extract-style-pack-") as temp_value:
            temp_root = Path(temp_value)
            threshold, scene_paths, scene_times = choose_scene_run(
                video, ffmpeg, temp_root, args.scene_threshold, raw_target, color_filter
            )
            frame_records.extend(
                copy_scene_frames(scene_paths, scene_times, raw_dir, args.asset_id, raw_target)
            )
            existing_times = [float(row["timestamp"]) for row in frame_records]
            uniform_needed = max(1, raw_target - len(frame_records))
            uniform_candidates = evenly_spaced_times(duration, max(uniform_needed * 2, uniform_needed))
            uniform_sequence = 1
            extraction_errors: list[dict[str, Any]] = []
            for timestamp in uniform_candidates:
                if len([row for row in frame_records if row["method"] == "uniform"]) >= uniform_needed:
                    break
                if not far_from_existing(timestamp, existing_times):
                    continue
                destination = raw_dir / frame_name(args.asset_id, timestamp, "uniform", uniform_sequence)
                try:
                    extract_timestamp(video, destination, timestamp, ffmpeg, color_filter)
                except RuntimeError as exc:
                    extraction_errors.append({"timestamp": round(timestamp, 6), "error": str(exc)})
                    continue
                frame_records.append({"path": destination, "timestamp": timestamp, "method": "uniform"})
                existing_times.append(timestamp)
                uniform_sequence += 1

        if not frame_records:
            raise RuntimeError("没有成功提取任何候选帧")

        metrics: list[ImageMetrics] = deduplicate_directory(
            raw_dir,
            kept_dir,
            rejected_dir,
            phash_distance=max(0, args.phash_distance),
            dhash_distance=max(0, args.dhash_distance),
            histogram_correlation=min(1.0, max(-1.0, args.histogram_correlation)),
            blur_threshold=max(0.0, args.blur_threshold),
            min_short_edge=0,
        )
        limit_kept(metrics, max_kept)

        kept_count = sum(row.status == "kept" for row in metrics)
        if kept_count < 3:
            raise RuntimeError(f"质量筛选后仅保留 {kept_count} 帧，少于最低 3 帧")

        base = output_dir.parent.parent.parent if len(output_dir.parents) >= 3 else output_dir.parent
        frames = []
        for index, metric in enumerate(sorted(metrics, key=lambda row: timestamp_from_name(row.filename)), 1):
            identity = metadata_from_name(metric.filename)
            row = asdict(metric)
            row.update(identity)
            row["frame_id"] = f"FRM-{args.asset_id}-{index:04d}"
            row["source_path"] = portable_path(row["source_path"], base)
            row["kept_path"] = portable_path(row["kept_path"], base)
            frames.append(row)

        scene_count = sum(row["method"] == "scene" for row in frame_records)
        uniform_count = sum(row["method"] == "uniform" for row in frame_records)
        payload = {
            "schema_version": "1.0",
            "created_at": utc_now(),
            "video_asset_id": args.asset_id,
            "source_video": portable_path(str(video), base),
            **probe,
            "frame_time_unit": "milliseconds",
            "extraction": {
                "scene_threshold": threshold,
                "target_raw_count": raw_target,
                "scene_candidates": scene_count,
                "uniform_candidates": uniform_count,
                "scene_not_applicable_reason": "未检测到明显场景切换" if scene_count == 0 else None,
                "uniform_not_applicable_reason": None,
                "raw_count": len(frame_records),
                "kept_count": kept_count,
                "max_kept": max_kept,
                "deduplication_performed": True,
                "extraction_errors": extraction_errors,
            },
            "frames": frames,
        }
        index_path = output_dir / "frames.json"
        index_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"OUTPUT\t{index_path}")
        print(f"RAW\t{len(frame_records)}")
        print(f"KEPT\t{kept_count}")
        print(f"SCENE_THRESHOLD\t{threshold:.2f}")
        return 0
    except Exception as exc:
        failure = {
            "schema_version": "1.0",
            "created_at": utc_now(),
            "video_asset_id": args.asset_id,
            "source_video": video.name,
            "status": "failed",
            "error": str(exc),
            "recovery": "修复原因后使用同一命令并添加 --overwrite 重新生成",
        }
        (output_dir / "frames.json").write_text(
            json.dumps(failure, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(f"抽帧失败：{exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
