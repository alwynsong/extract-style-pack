#!/usr/bin/env python3
"""检测风格包原始媒体，并标准化、筛选和去重图片。"""

from __future__ import annotations

import argparse
import hashlib
import json
import mimetypes
import re
import shutil
import subprocess
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path
from typing import Any

from PIL import Image, ImageOps

from extract_video_frames import ImageMetrics, deduplicate_directory, portable_path
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.webp', '.tif', '.tiff', '.bmp'}

def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec='seconds')

VIDEO_EXTENSIONS = {'.mp4', '.mov', '.mkv', '.avi', '.webm', '.m4v', '.mts', '.m2ts', '.wmv'}

SKIP_NAMES = {'__pycache__', '.git', '.svn'}

MACOS_EXECUTABLE_DIRS = (Path('/opt/homebrew/bin'), Path('/usr/local/bin'))

def media_find_executable(command: str) -> str | None:
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
    return None

def media_utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec='seconds')

def media_sha256_file(path: Path, chunk_size: int) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        while (chunk := stream.read(chunk_size)):
            digest.update(chunk)
    return digest.hexdigest()

def media_parse_fraction(value: object) -> float | None:
    if not isinstance(value, str) or not value or value in {'0/0', 'N/A'}:
        return None
    try:
        return float(Fraction(value))
    except (ValueError, ZeroDivisionError):
        return None

def media_parse_float(value: object) -> float | None:
    try:
        parsed = float(value)
        return parsed if parsed >= 0 else None
    except (TypeError, ValueError):
        return None

def media_relative_or_absolute(path: Path, output: Path) -> str:
    try:
        return path.resolve().relative_to(output.parent.parent.resolve()).as_posix()
    except ValueError:
        return path.name

def media_common_record(path: Path, output: Path, chunk_size: int) -> dict[str, Any]:
    stat = path.stat()
    return {'path': media_relative_or_absolute(path, output), 'original_name': path.name, 'extension': path.suffix.lower(), 'size_bytes': stat.st_size, 'modified_at': datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(timespec='seconds'), 'mime_type': mimetypes.guess_type(path.name)[0], 'sha256': media_sha256_file(path, chunk_size)}

def media_inspect_image(path: Path, output: Path, chunk_size: int) -> dict[str, Any]:
    record = media_common_record(path, output, chunk_size)
    record['media_type'] = 'image'
    try:
        with Image.open(path) as source:
            source.verify()
        with Image.open(path) as source:
            oriented = ImageOps.exif_transpose(source)
            record.update({'format': source.format, 'width': oriented.width, 'height': oriented.height, 'mode': oriented.mode, 'icc_profile_present': bool(source.info.get('icc_profile')), 'exif_orientation_applied': source.size != oriented.size, 'status': 'inspected', 'error': None})
    except Exception as exc:
        record.update({'status': 'failed', 'error': f'图片无法读取：{exc}'})
    return record

def media_run_ffprobe(path: Path, ffprobe: str) -> dict[str, Any]:
    command = [ffprobe, '-v', 'error', '-print_format', 'json', '-show_format', '-show_streams', str(path)]
    completed = subprocess.run(command, capture_output=True, text=True, encoding='utf-8', errors='replace')
    if completed.returncode != 0:
        message = completed.stderr.strip() or 'ffprobe 返回非零状态'
        raise RuntimeError(message)
    return json.loads(completed.stdout)

def media_rotation_from_stream(stream: dict[str, Any]) -> int:
    tags = stream.get('tags') if isinstance(stream.get('tags'), dict) else {}
    try:
        if 'rotate' in tags:
            return int(float(tags['rotate'])) % 360
    except (TypeError, ValueError):
        pass
    side_data = stream.get('side_data_list')
    if isinstance(side_data, list):
        for item in side_data:
            if isinstance(item, dict) and 'rotation' in item:
                try:
                    return int(float(item['rotation'])) % 360
                except (TypeError, ValueError):
                    continue
    return 0

def media_inspect_video(path: Path, output: Path, chunk_size: int, ffprobe: str) -> dict[str, Any]:
    record = media_common_record(path, output, chunk_size)
    record['media_type'] = 'video'
    try:
        payload = media_run_ffprobe(path, ffprobe)
        streams = payload.get('streams', [])
        video_streams = [item for item in streams if isinstance(item, dict) and item.get('codec_type') == 'video']
        audio_streams = [item for item in streams if isinstance(item, dict) and item.get('codec_type') == 'audio']
        if not video_streams:
            raise RuntimeError('no_video_stream')
        stream = video_streams[0]
        format_row = payload.get('format') if isinstance(payload.get('format'), dict) else {}
        duration = media_parse_float(stream.get('duration')) or media_parse_float(format_row.get('duration'))
        rotation = media_rotation_from_stream(stream)
        width = int(stream.get('width') or 0)
        height = int(stream.get('height') or 0)
        display_width, display_height = (height, width) if rotation in {90, 270} else (width, height)
        record.update({'container': format_row.get('format_name'), 'duration_seconds': duration, 'video_stream_count': len(video_streams), 'audio_stream_count': len(audio_streams), 'stream_index': stream.get('index'), 'codec': stream.get('codec_name'), 'codec_long_name': stream.get('codec_long_name'), 'pixel_format': stream.get('pix_fmt'), 'width': width, 'height': height, 'display_width': display_width, 'display_height': display_height, 'sample_aspect_ratio': stream.get('sample_aspect_ratio'), 'display_aspect_ratio': stream.get('display_aspect_ratio'), 'average_frame_rate_raw': stream.get('avg_frame_rate'), 'average_frame_rate': media_parse_fraction(stream.get('avg_frame_rate')), 'real_frame_rate_raw': stream.get('r_frame_rate'), 'real_frame_rate': media_parse_fraction(stream.get('r_frame_rate')), 'time_base': stream.get('time_base'), 'rotation_degrees': rotation, 'color_space': stream.get('color_space'), 'color_transfer': stream.get('color_transfer'), 'color_primaries': stream.get('color_primaries'), 'field_order': stream.get('field_order'), 'status': 'inspected', 'error': None})
    except Exception as exc:
        record.update({'status': 'failed', 'error': f'视频无法探测：{exc}'})
    return record

def media_iter_input_files(inputs: list[Path], recursive: bool) -> tuple[list[Path], list[dict[str, str]]]:
    files: set[Path] = set()
    errors: list[dict[str, str]] = []
    for raw in inputs:
        path = raw.expanduser()
        if not path.exists():
            errors.append({'path': str(path), 'error': '路径不存在'})
            continue
        if path.is_file():
            files.add(path.resolve())
            continue
        iterator = path.rglob('*') if recursive else path.glob('*')
        for child in iterator:
            if any((part in SKIP_NAMES for part in child.parts)):
                continue
            if child.is_file():
                files.add(child.resolve())
    return (sorted(files, key=lambda item: str(item).casefold()), errors)

def media_classify(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in IMAGE_EXTENSIONS:
        return 'image'
    if suffix in VIDEO_EXTENSIONS:
        return 'video'
    guessed = mimetypes.guess_type(path.name)[0] or ''
    if guessed.startswith('image/'):
        return 'image'
    if guessed.startswith('video/'):
        return 'video'
    return 'unsupported'

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='检测风格包原始媒体，并标准化、筛选和去重图片')
    parser.add_argument('style_pack', type=Path, help='风格包根目录')
    parser.add_argument('--ffprobe', default='ffprobe')
    parser.add_argument('--hash-chunk-mb', type=int, default=4)
    parser.add_argument('--phash-distance', type=int, default=6)
    parser.add_argument('--dhash-distance', type=int, default=8)
    parser.add_argument('--histogram-correlation', type=float, default=0.85)
    parser.add_argument('--blur-threshold', type=float, default=35.0)
    parser.add_argument('--min-short-edge', type=int, default=360)
    return parser.parse_args()

def stable_asset_id(path: Path, media_type: str, index: int) -> str:
    prefix = 'IMG' if media_type == 'image' else 'VID' if media_type == 'video' else 'UNSUPPORTED'
    match = re.search(f'(?i)\\b{prefix}-\\d{{4}}\\b', path.stem)
    return match.group(0).upper() if match else f'{prefix}-{index:04d}'

def reset_derived_directory(path: Path, style_pack: Path) -> None:
    resolved = path.resolve()
    resolved.relative_to(style_pack.resolve())
    if resolved.exists():
        shutil.rmtree(resolved)
    resolved.mkdir(parents=True, exist_ok=True)

def inspect_raw_media(style_pack: Path, image_dir: Path, video_dir: Path, report_path: Path, ffprobe_command: str, chunk_size: int) -> tuple[dict[str, Any], int]:
    ffprobe_path = media_find_executable(ffprobe_command)
    files, input_errors = media_iter_input_files([image_dir, video_dir], True)
    records: list[dict[str, Any]] = []
    counters = {'image': 0, 'video': 0, 'unsupported': 0}
    for path in files:
        media_type = media_classify(path)
        counters[media_type] += 1
        if media_type == 'image':
            record = media_inspect_image(path, report_path, chunk_size)
        elif media_type == 'video':
            if ffprobe_path is None:
                record = media_common_record(path, report_path, chunk_size)
                record.update({'media_type': 'video', 'status': 'failed', 'error': f'找不到 ffprobe：{ffprobe_command}'})
            else:
                record = media_inspect_video(path, report_path, chunk_size, ffprobe_path)
        else:
            record = media_common_record(path, report_path, chunk_size)
            record.update({'media_type': 'unsupported', 'status': 'unsupported', 'error': '不支持或无法根据扩展名/MIME 分类'})
        record['asset_id'] = stable_asset_id(path, media_type, counters[media_type])
        records.append(record)
    payload = {'schema_version': '1.0', 'created_at': utc_now(), 'ffprobe': Path(ffprobe_path).name if ffprobe_path else None, 'inputs': ['参考素材/原始文件/图片', '参考素材/原始文件/视频'], 'input_errors': [{'path': Path(row['path']).name, 'error': row['error']} for row in input_errors], 'summary': {'files': len(records), 'images': counters['image'], 'videos': counters['video'], 'unsupported': counters['unsupported'], 'failed': sum((row.get('status') == 'failed' for row in records))}, 'media': records}
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    failures = len(input_errors) + payload['summary']['failed']
    return (payload, failures)

def prepare_images(style_pack: Path, input_dir: Path, args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    output_dir = style_pack / '参考素材' / '标准化图片'
    rejected_dir = style_pack / '参考素材' / '被过滤图片'
    reset_derived_directory(output_dir, style_pack)
    reset_derived_directory(rejected_dir, style_pack)
    records = deduplicate_directory(input_dir, output_dir, rejected_dir, recursive=True, phash_distance=max(0, args.phash_distance), dhash_distance=max(0, args.dhash_distance), histogram_correlation=min(1.0, max(-1.0, args.histogram_correlation)), blur_threshold=max(0.0, args.blur_threshold), min_short_edge=max(0, args.min_short_edge))
    summary = {'total': len(records), 'kept': sum((row.status == 'kept' for row in records)), 'filtered': sum((row.status == 'filtered' for row in records)), 'duplicates': sum((row.status == 'duplicate' for row in records)), 'failed': sum((row.status == 'failed' for row in records))}
    for row in records:
        row.source_path = portable_path(row.source_path, style_pack) or row.filename
        row.kept_path = portable_path(row.kept_path, style_pack)
    payload: dict[str, Any] = {'schema_version': '1.0', 'created_at': utc_now(), 'input_dir': portable_path(str(input_dir), style_pack), 'output_dir': portable_path(str(output_dir), style_pack), 'rejected_dir': portable_path(str(rejected_dir), style_pack), 'settings': {'phash_distance': args.phash_distance, 'dhash_distance': args.dhash_distance, 'histogram_correlation': args.histogram_correlation, 'blur_threshold': args.blur_threshold, 'min_short_edge': args.min_short_edge}, 'summary': summary, 'images': [asdict(row) for row in records]}
    return (payload, summary['failed'])

def main() -> int:
    args = parse_args()
    style_pack = args.style_pack.expanduser().resolve()
    if not style_pack.is_dir():
        print(f'风格包目录不存在：{style_pack}', file=sys.stderr)
        return 2
    reference_root = style_pack / '参考素材'
    image_dir = reference_root / '原始文件' / '图片'
    video_dir = reference_root / '原始文件' / '视频'
    image_dir.mkdir(parents=True, exist_ok=True)
    video_dir.mkdir(parents=True, exist_ok=True)
    analysis_dir = style_pack / '分析'
    analysis_dir.mkdir(parents=True, exist_ok=True)
    media_report = analysis_dir / '媒体检测结果.json'
    media_payload, media_failures = inspect_raw_media(style_pack, image_dir, video_dir, media_report, args.ffprobe, max(1, args.hash_chunk_mb) * 1024 * 1024)
    image_payload, image_failures = prepare_images(style_pack, image_dir, args)
    image_report = analysis_dir / '去重与筛选记录.json'
    image_report.write_text(json.dumps(image_payload, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(f'MEDIA_REPORT\t{media_report}')
    print(f'IMAGE_REPORT\t{image_report}')
    print(f"MEDIA\t{media_payload['summary']['files']}")
    print(f"KEPT\t{image_payload['summary']['kept']}")
    print(f"FILTERED\t{image_payload['summary']['filtered']}")
    print(f"DUPLICATES\t{image_payload['summary']['duplicates']}")
    print(f'FAILED\t{media_failures + image_failures}')
    return 1 if media_failures or image_failures else 0

if __name__ == '__main__':
    sys.exit(main())
