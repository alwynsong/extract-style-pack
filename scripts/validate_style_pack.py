#!/usr/bin/env python3
"""完整验收提取风格包的来源、视频帧、分析、分组、联系表与双版色卡。"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any

from PIL import Image

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff", ".bmp"}
CARD_SIZE = (1920, 1080)
PALETTE_THUMB_SIZE = (960, 540)
PALETTE_OVERVIEW_COLUMNS = 2
PALETTE_OVERVIEW_GAP = 18
COLOR_MODEL = "3 主色 + 5 辅助色 + 2 点缀色"
EXPECTED_SLOTS = ("P1", "P2", "P3", "S1", "S2", "S3", "S4", "S5", "A1", "A2")
EXPECTED_ROLES = ("主色", "主色", "主色", "辅助色", "辅助色", "辅助色", "辅助色", "辅助色", "点缀色", "点缀色")
HEX_PATTERN = re.compile(r"#[0-9A-Fa-f]{6}")
UPPER_HEX_PATTERN = re.compile(r"^#[0-9A-F]{6}$")
GROUP_PATTERN = re.compile(r"^\d{2}-.+")
IMAGE_ID_PATTERN = re.compile(r"IMG-\d{4}", re.IGNORECASE)
FRAME_ID_PATTERN = re.compile(r"(?:FRM-)?VID-\d{4}", re.IGNORECASE)
WINDOWS_ABSOLUTE_PATTERN = re.compile(r"(?i)[A-Z]:\\(?:Users|Documents and Settings)\\")
UNIX_HOME_PATTERN = re.compile(r"/(?:Users|home)/[^/\s]+/")
SECRET_PATTERN = re.compile(
    r"(?i)(?:api[_-]?key|access[_-]?token|refresh[_-]?token|secret|password|authorization|cookie|bearer|private[_-]?key)\s*[:=]?\s*['\"]?[^\s'\"]{8,}"
)
EMAIL_PATTERN = re.compile(r"(?i)\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b")
SIGNED_URL_PATTERN = re.compile(
    r"(?i)(?:X-Amz-Signature|X-Goog-Signature|Signature|sig|token|Expires)=[^&\s\"']+"
)
PROMPT_HEADINGS = (
    "## 这个分组解决什么画面",
    "## 视觉锁",
    "## 可替换变量",
    "## 提示词",
    "### 风格环境",
    "### 主体与调度",
    "### 构图景别",
    "### 摄影参数",
    "### 光线色调",
    "### 材质细节",
    "### 氛围情绪",
    "### 画面限制",
    "## 视频提示词",
    "## 适合生成的镜头",
    "## 容易跑偏的地方",
    "## 自检清单",
)


@dataclass
class Validation:
    style_pack: Path
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    checks: dict[str, Any] = field(default_factory=dict)

    def error(self, message: str) -> None:
        self.errors.append(message)

    def warning(self, message: str) -> None:
        self.warnings.append(message)

    def check(self, name: str, value: Any) -> None:
        self.checks[name] = value


@dataclass(frozen=True)
class GroupInfo:
    name: str
    directory: Path
    references: tuple[Path, ...]
    card: Path
    pure_card: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="完整验收提取风格包")
    parser.add_argument("style_pack", type=Path)
    parser.add_argument("--report", type=Path, help="验收报告 JSON；默认写入分析/验收报告.json")
    parser.add_argument(
        "--visual-review-confirmed",
        action="store_true",
        help="记录人工已检查全部素材、分组和两套色卡总览",
    )
    parser.add_argument("--warnings-as-errors", action="store_true")
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def load_json(path: Path, validation: Validation, label: str) -> Any | None:
    if not path.is_file():
        validation.error(f"缺少{label}：{path}")
        return None
    if path.stat().st_size == 0:
        validation.error(f"{label}为空：{path}")
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        validation.error(f"{label}无法解析：{path} ({exc})")
        return None


def require_nonempty(path: Path, validation: Validation, label: str) -> str | None:
    if not path.is_file():
        validation.error(f"缺少{label}：{path}")
        return None
    try:
        text = path.read_text(encoding="utf-8")
    except Exception as exc:
        validation.error(f"{label}无法读取：{path} ({exc})")
        return None
    if not text.strip():
        validation.error(f"{label}为空：{path}")
        return None
    return text


def portable_relative(value: object) -> bool:
    if not isinstance(value, str) or not value.strip():
        return False
    if "\\" in value or re.match(r"^[A-Za-z]:", value) or value.startswith("/"):
        return False
    path = PurePosixPath(value)
    return ".." not in path.parts and "." not in path.parts


def resolve_portable(style_pack: Path, value: object) -> Path | None:
    if not portable_relative(value):
        return None
    return style_pack.joinpath(*PurePosixPath(str(value)).parts)


def inspect_image(path: Path, validation: Validation, label: str) -> tuple[tuple[int, int], str, str] | None:
    if not path.is_file():
        validation.error(f"缺少{label}：{path}")
        return None
    try:
        with Image.open(path) as image:
            image.load()
            return image.size, image.mode, str(image.format)
    except Exception as exc:
        validation.error(f"{label}无法读取：{path} ({exc})")
        return None


def require_jpeg(path: Path, validation: Validation, label: str) -> None:
    info = inspect_image(path, validation, label)
    if info is not None and info[2] != "JPEG":
        validation.error(f"{label}格式错误：{path} 为 {info[2]}，应为 JPEG")


def image_files(directory: Path, *, exclude_cards: bool = False) -> list[Path]:
    if not directory.is_dir():
        return []
    return sorted(
        path
        for path in directory.iterdir()
        if path.is_file()
        and path.suffix.lower() in IMAGE_EXTENSIONS
        and (not exclude_cards or "色卡" not in path.name)
    )


def overview_pages(base: Path) -> list[Path]:
    if base.is_file():
        return [base]
    return sorted(base.parent.glob(f"{base.stem}-[0-9][0-9]{base.suffix}"))


def display_path(path: Path, style_pack: Path) -> str:
    try:
        return path.resolve().relative_to(style_pack.resolve()).as_posix()
    except ValueError:
        return path.name


def sanitize_message(message: str, style_pack: Path) -> str:
    sanitized = message.replace(str(style_pack.resolve()), ".").replace(str(style_pack), ".")
    sanitized = re.sub(
        r"(?i)[A-Z]:\\(?:Users|Documents and Settings)\\[^\\\s\"']+(?:\\[^\s\"']*)?",
        "<private-path>",
        sanitized,
    )
    return re.sub(r"/(?:Users|home)/[^/\s]+(?:/[^\s\"']*)?", "<private-path>", sanitized)


def validate_deliverable_privacy(style_pack: Path, validation: Validation) -> None:
    report_path = style_pack / "分析" / "验收报告.json"
    for pattern in ("*.md", "*.json"):
        for path in style_pack.rglob(pattern):
            if path.resolve() == report_path.resolve():
                continue
            try:
                text = path.read_text(encoding="utf-8")
            except Exception as exc:
                validation.error(f"交付文本无法读取：{display_path(path, style_pack)} ({exc})")
                continue
            shown = display_path(path, style_pack)
            privacy_text = text.replace("\\\\", "\\")
            if WINDOWS_ABSOLUTE_PATTERN.search(privacy_text) or UNIX_HOME_PATTERN.search(privacy_text):
                validation.error(f"交付文件包含私人用户目录绝对路径：{shown}")
            if SECRET_PATTERN.search(text):
                validation.error(f"交付文件包含疑似凭据或密钥：{shown}")
            if EMAIL_PATTERN.search(text):
                validation.error(f"交付文件包含私人邮箱：{shown}")
            if SIGNED_URL_PATTERN.search(text):
                validation.error(f"交付文件包含临时签名 URL 参数：{shown}")


def validate_root_documents(style_pack: Path, validation: Validation) -> dict[str, str]:
    documents: dict[str, str] = {}
    for relative, label in (("00-说明.md", "项目说明"), ("design.md", "设计规范")):
        path = style_pack / relative
        text = require_nonempty(path, validation, label)
        if text is not None:
            documents[relative] = text
    design = documents.get("design.md")
    if design is not None:
        for concept in ("视觉", "构图", "色彩", "光", "镜头", "材质", "禁止"):
            if concept not in design:
                validation.warning(f"design.md 未明确覆盖“{concept}”相关规范")
    return documents


def validate_manifest(style_pack: Path, reference_root: Path, validation: Validation) -> tuple[dict[str, dict[str, Any]], list[str]]:
    payload = load_json(reference_root / "manifest.json", validation, "素材清单")
    if not isinstance(payload, dict):
        if payload is not None:
            validation.error("manifest.json 顶层必须是对象")
        return {}, []
    if payload.get("schema_version") != "1.0":
        validation.error(f"manifest schema_version 必须为 1.0：{payload.get('schema_version')}")
    if not isinstance(payload.get("project"), str) or not payload["project"].strip():
        validation.error("manifest 缺少非空 project")
    if not isinstance(payload.get("created_at"), str) or not payload["created_at"].strip():
        validation.error("manifest 缺少 created_at")
    sources = payload.get("sources")
    if not isinstance(sources, list) or not sources:
        validation.error("manifest.json sources 必须是非空数组")
        return {}, []

    by_id: dict[str, dict[str, Any]] = {}
    video_ids: list[str] = []
    allowed_source_types = {"web_search", "local_file", "uploaded_file", "media_url", "web_page"}
    allowed_statuses = {"discovered", "downloaded", "copied", "inspected", "processed", "filtered", "duplicate", "failed"}
    for index, raw in enumerate(sources):
        if not isinstance(raw, dict):
            validation.error(f"manifest sources[{index}] 不是对象")
            continue
        asset_id = raw.get("asset_id")
        if not isinstance(asset_id, str) or not asset_id:
            validation.error(f"manifest sources[{index}] 缺少 asset_id")
            continue
        if asset_id in by_id:
            validation.error(f"manifest 出现重复 asset_id：{asset_id}")
            continue
        by_id[asset_id] = raw
        if raw.get("source_type") not in allowed_source_types:
            validation.error(f"{asset_id} source_type 无效：{raw.get('source_type')}")
        if raw.get("status") not in allowed_statuses:
            validation.error(f"{asset_id} status 无效：{raw.get('status')}")
        media_type = raw.get("media_type")
        if media_type not in {"image", "video"}:
            validation.error(f"{asset_id} media_type 必须是 image 或 video：{media_type}")
        expected_id = r"IMG-\d{4}" if media_type == "image" else r"VID-\d{4}"
        if media_type in {"image", "video"} and not re.fullmatch(expected_id, asset_id):
            validation.error(f"{asset_id} 不符合 {media_type} 素材 ID 格式")
        if raw.get("status") == "processed" and not isinstance(raw.get("included_in_analysis"), bool):
            validation.error(f"{asset_id} processed 状态必须明确 included_in_analysis")
        if raw.get("included_in_analysis") is True and raw.get("status") != "processed":
            validation.error(f"{asset_id} included_in_analysis=true 时状态必须为 processed")
        if media_type == "video" and raw.get("status") not in {"failed", "filtered", "duplicate"}:
            if raw.get("status") != "processed" or raw.get("included_in_analysis") is not True:
                validation.error(
                    f"未排除的视频 {asset_id} 必须完成抽帧并标记 processed、included_in_analysis=true"
                )
            else:
                video_ids.append(asset_id)
        local_path = raw.get("local_path")
        if raw.get("status") not in {"failed", "discovered"}:
            resolved = resolve_portable(style_pack, local_path)
            if resolved is None:
                validation.error(f"{asset_id} local_path 不是相对 POSIX 路径：{local_path}")
            elif not resolved.is_file():
                validation.error(f"{asset_id} local_path 指向的文件不存在：{local_path}")
        sha256 = raw.get("sha256")
        if raw.get("status") not in {"failed", "discovered"} and (
            not isinstance(sha256, str) or not re.fullmatch(r"[0-9A-Fa-f]{64}", sha256)
        ):
            validation.error(f"{asset_id} 缺少有效 SHA-256")
        if raw.get("source_type") in {"web_search", "media_url", "web_page"}:
            if not any(isinstance(raw.get(key), str) and raw[key].startswith(("http://", "https://")) for key in ("page_url", "initial_url", "final_url")):
                validation.error(f"网络来源 {asset_id} 缺少可追溯 URL")
        if raw.get("status") == "failed" and not raw.get("failure_reason"):
            validation.error(f"失败来源 {asset_id} 缺少 failure_reason")

    validation.check("manifest_sources", len(by_id))
    validation.check("manifest_videos", len(video_ids))
    return by_id, video_ids


def validate_frame_index(
    style_pack: Path,
    reference_root: Path,
    asset_id: str,
    source: dict[str, Any],
    validation: Validation,
) -> set[str]:
    default_index = reference_root / "视频抽帧" / asset_id / "frames.json"
    index_value = source.get("frame_index")
    if index_value is not None:
        index_path = resolve_portable(style_pack, index_value)
        if index_path is None:
            validation.error(f"{asset_id} frame_index 不是相对 POSIX 路径：{index_value}")
            index_path = default_index
    else:
        index_path = default_index
        validation.error(f"视频 {asset_id} 缺少 frame_index")

    payload = load_json(index_path, validation, f"{asset_id} 抽帧索引")
    if not isinstance(payload, dict):
        return set()
    if payload.get("status") == "failed" or payload.get("error"):
        validation.error(f"视频 {asset_id} 抽帧失败：{payload.get('error')}")
        return set()
    if payload.get("video_asset_id") != asset_id:
        validation.error(f"{asset_id} frames.json 的 video_asset_id 不一致")
    if payload.get("frame_time_unit") != "milliseconds":
        validation.error(f"{asset_id} frame_time_unit 必须是 milliseconds")
    extraction = payload.get("extraction")
    frames = payload.get("frames")
    if not isinstance(extraction, dict) or not isinstance(frames, list):
        validation.error(f"{asset_id} frames.json 缺少 extraction 或 frames")
        return set()

    valid_statuses = {"kept", "filtered", "duplicate", "failed"}
    all_frame_ids: set[str] = set()
    for index, row in enumerate(frames):
        if not isinstance(row, dict):
            validation.error(f"{asset_id} frames[{index}] 必须是对象")
            continue
        frame_id = row.get("frame_id")
        if not isinstance(frame_id, str) or asset_id not in frame_id:
            validation.error(f"{asset_id} frames[{index}] frame_id 无效：{frame_id}")
        elif frame_id in all_frame_ids:
            validation.error(f"{asset_id} 出现重复 frame_id：{frame_id}")
        else:
            all_frame_ids.add(frame_id)
        if row.get("status") not in valid_statuses:
            validation.error(f"{asset_id} frames[{index}] status 无效：{row.get('status')}")
    kept_rows = [row for row in frames if isinstance(row, dict) and row.get("status") == "kept"]
    declared = extraction.get("kept_count")
    if declared != len(kept_rows):
        validation.error(f"{asset_id} kept_count 与 frames 记录不一致：{declared} != {len(kept_rows)}")
    manifest_kept = source.get("kept_frame_count")
    if manifest_kept != len(kept_rows):
        validation.error(f"{asset_id} manifest kept_frame_count 不一致：{manifest_kept} != {len(kept_rows)}")
    if len(kept_rows) < 3:
        validation.error(f"{asset_id} 最终保留帧少于 3：{len(kept_rows)}")
    scene_candidates = extraction.get("scene_candidates")
    uniform_candidates = extraction.get("uniform_candidates")
    if not isinstance(scene_candidates, int) or isinstance(scene_candidates, bool) or scene_candidates < 0:
        validation.error(f"{asset_id} scene_candidates 必须是非负整数：{scene_candidates}")
    elif scene_candidates == 0 and not extraction.get("scene_not_applicable_reason"):
        validation.error(f"{asset_id} 没有场景变化候选，且未记录不适用原因")
    if not isinstance(uniform_candidates, int) or isinstance(uniform_candidates, bool) or uniform_candidates <= 0:
        validation.error(f"{asset_id} uniform_candidates 必须是正整数：{uniform_candidates}")
    if extraction.get("deduplication_performed") is not True:
        validation.error(f"{asset_id} 未记录强制质量筛选与感知去重")

    filenames: set[str] = set()
    kept_paths: set[str] = set()
    for index, row in enumerate(kept_rows):
        frame_id = row.get("frame_id")
        filename = row.get("filename")
        source_path = row.get("source_path")
        timestamp = row.get("timestamp_ms")
        method = row.get("method")
        kept_path = row.get("kept_path")
        if not isinstance(filename, str) or not filename:
            validation.error(f"{asset_id} 保留帧[{index}] filename 无效：{filename}")
        if not portable_relative(source_path):
            validation.error(f"{asset_id} 保留帧[{index}] source_path 不是相对 POSIX 路径：{source_path}")
        if not isinstance(timestamp, int) or timestamp < 0:
            validation.error(f"{asset_id} 保留帧[{index}] timestamp_ms 无效：{timestamp}")
        if method not in {"scene", "uniform", "boundary"}:
            validation.error(f"{asset_id} 保留帧[{index}] method 无效：{method}")
        if isinstance(kept_path, str) and kept_path in kept_paths:
            validation.error(f"{asset_id} 出现重复 kept_path：{kept_path}")
        elif isinstance(kept_path, str):
            kept_paths.add(kept_path)
        resolved = resolve_portable(style_pack, kept_path)
        if resolved is None:
            validation.error(f"{asset_id} 保留帧路径不是相对 POSIX 路径：{kept_path}")
        elif not resolved.is_file():
            validation.error(f"{asset_id} 保留帧文件不存在：{kept_path}")
        else:
            if isinstance(filename, str) and filename != resolved.name:
                validation.error(f"{asset_id} filename 与 kept_path 文件名不一致：{filename} != {resolved.name}")
            filenames.add(resolved.name)

    kept_dir = reference_root / "视频抽帧" / asset_id / "保留帧"
    disk_names = {path.name for path in image_files(kept_dir)}
    if disk_names != filenames:
        validation.error(
            f"{asset_id} 保留帧磁盘文件与 frames.json 不一致：磁盘 {len(disk_names)}，JSON {len(filenames)}"
        )
    overview_files = overview_pages(reference_root / "视频抽帧总览" / f"{asset_id}.jpg")
    if not overview_files:
        validation.error(f"缺少 {asset_id} 抽帧总览或分页文件")
    for path in overview_files:
        require_jpeg(path, validation, f"{asset_id} 抽帧总览")
    validation.check(f"{asset_id}_overview_pages", len(overview_files))
    validation.check(f"{asset_id}_kept_frames", len(filenames))
    return filenames


def validate_videos(
    style_pack: Path,
    reference_root: Path,
    sources: dict[str, dict[str, Any]],
    video_ids: list[str],
    validation: Validation,
) -> set[str]:
    frame_names: set[str] = set()
    for asset_id in video_ids:
        source = sources[asset_id]
        if source.get("status") != "processed" or source.get("included_in_analysis") is not True:
            validation.error(f"正式视频 {asset_id} 必须标记 processed 且 included_in_analysis=true")
        frame_names.update(validate_frame_index(style_pack, reference_root, asset_id, source, validation))
    frame_root = reference_root / "视频抽帧"
    if frame_root.is_dir():
        disk_ids = {path.name for path in frame_root.iterdir() if path.is_dir()}
        extra = disk_ids - set(video_ids)
        if extra:
            validation.warning(f"存在 manifest 未登记的视频抽帧目录：{sorted(extra)}")
    validation.check("kept_video_frames", len(frame_names))
    return frame_names


def validate_analysis_assets(
    reference_root: Path, frame_names: set[str], min_images: int, validation: Validation
) -> tuple[set[str], int]:
    standardized = image_files(reference_root / "标准化图片")
    for path in standardized:
        if not IMAGE_ID_PATTERN.search(path.name):
            validation.error(f"标准化图片文件名无法追溯到 IMG 素材 ID：{path.name}")
        info = inspect_image(path, validation, f"标准化图片 {path.name}")
        if info is not None and info[1] != "RGB":
            validation.error(f"标准化图片必须为 RGB：{path.name} 实际 {info[1]}")
    all_names = {path.name for path in standardized} | frame_names
    total = len(all_names)
    if total < min_images:
        validation.error(f"正式分析画面不足 {min_images} 张：当前 {total}")
    master_pages = overview_pages(reference_root / "总览-contact-sheet.jpg")
    if not master_pages:
        validation.error("缺少全部正式分析素材总览：总览-contact-sheet.jpg 或分页文件")
    for path in master_pages:
        require_jpeg(path, validation, "全部素材总览")
    validation.check("standardized_images", len(standardized))
    validation.check("formal_analysis_images", total)
    validation.check("master_overview_pages", len(master_pages))
    return all_names, total


def expected_card_path(group_dir: Path) -> Path:
    display_name = re.sub(r"^\d{2}-", "", group_dir.name)
    return group_dir / f"00-{display_name}-色卡.png"


def expected_pure_card_path(group_dir: Path) -> Path:
    display_name = re.sub(r"^\d{2}-", "", group_dir.name)
    return group_dir / f"00-{display_name}-纯色色卡.png"


def validate_groups(reference_root: Path, formal_names: set[str], validation: Validation) -> dict[str, GroupInfo]:
    group_root = reference_root / "分组"
    if not group_root.is_dir():
        validation.error(f"缺少正式分组目录：{group_root}")
        return {}
    group_dirs = sorted(path for path in group_root.iterdir() if path.is_dir())
    if not group_dirs:
        validation.error("没有正式视觉分组")
        return {}
    result: dict[str, GroupInfo] = {}
    grouped_names: set[str] = set()
    for group_dir in group_dirs:
        if not GROUP_PATTERN.fullmatch(group_dir.name):
            validation.error(f"分组目录名称不符合 NN-[分组名]：{group_dir.name}")
        references = tuple(image_files(group_dir, exclude_cards=True))
        if len(references) < 2:
            validation.error(f"正式分组至少需要 2 个真实画面：{group_dir.name} 当前 {len(references)}")
        for path in references:
            if not (IMAGE_ID_PATTERN.search(path.name) or FRAME_ID_PATTERN.search(path.name)):
                validation.error(f"分组画面文件名无法追溯到素材或视频 ID：{group_dir.name}/{path.name}")
            if path.name not in formal_names:
                validation.error(f"分组含未登记为正式分析素材的文件：{group_dir.name}/{path.name}")
            grouped_names.add(path.name)
        pages = overview_pages(reference_root / "分组总览" / f"{group_dir.name}.jpg")
        if not pages:
            validation.error(f"缺少分组总览：{group_dir.name}")
        for page in pages:
            require_jpeg(page, validation, f"{group_dir.name} 分组总览")
        result[group_dir.name] = GroupInfo(
            name=group_dir.name,
            directory=group_dir,
            references=references,
            card=expected_card_path(group_dir),
            pure_card=expected_pure_card_path(group_dir),
        )
    ungrouped = formal_names - grouped_names
    if ungrouped:
        validation.warning(f"有 {len(ungrouped)} 个正式分析画面未进入任何视觉分组")
    validation.check("groups", len(result))
    validation.check("grouped_unique_images", len(grouped_names))
    return result


def normalized_label_text(value: object) -> str:
    if not isinstance(value, str):
        return ""
    return re.sub(r"\s+", " ", value.strip()).casefold()


def validate_image_labels(
    payload: object,
    style_pack: Path,
    formal_names: set[str],
    groups: dict[str, GroupInfo],
    validation: Validation,
) -> None:
    if not isinstance(payload, dict) or not isinstance(payload.get("items"), list):
        validation.error("单图标签顶层必须是对象，且包含 items 数组")
        return

    items = payload["items"]
    required_text_fields = (
        "asset_id",
        "file",
        "source_type",
        "space",
        "shot_and_subject_ratio",
        "composition",
        "lighting",
        "color",
        "camera_and_texture",
        "material",
        "mood",
        "avoid_identifiable_elements",
    )
    observation_fields = (
        "space",
        "shot_and_subject_ratio",
        "composition",
        "lighting",
        "color",
        "camera_and_texture",
        "material",
        "mood",
    )
    seen_ids: set[str] = set()
    seen_files: set[str] = set()
    observed_names: set[str] = set()
    observation_fingerprints: dict[tuple[str, ...], str] = {}
    evidence_fingerprints: dict[tuple[str, ...], str] = {}

    for index, item in enumerate(items):
        if not isinstance(item, dict):
            validation.error(f"单图标签第 {index + 1} 条必须是对象")
            continue
        item_id = item.get("asset_id")
        shown_id = item_id if isinstance(item_id, str) and item_id.strip() else f"第 {index + 1} 条"
        for field_name in required_text_fields:
            if not isinstance(item.get(field_name), str) or not item[field_name].strip():
                validation.error(f"{shown_id} 的逐图标签缺少非空字段：{field_name}")

        if isinstance(item_id, str) and item_id.strip():
            if item_id in seen_ids:
                validation.error(f"单图标签存在重复 asset_id：{item_id}")
            seen_ids.add(item_id)

        file_value = item.get("file")
        if isinstance(file_value, str) and file_value.strip():
            if file_value in seen_files:
                validation.error(f"单图标签存在重复 file：{file_value}")
            seen_files.add(file_value)
            if not portable_relative(file_value):
                validation.error(f"{shown_id} 的 file 必须为相对 POSIX 路径：{file_value}")
            else:
                image_path = resolve_portable(style_pack, file_value)
                file_name = PurePosixPath(file_value).name
                observed_names.add(file_name)
                if image_path is None or not image_path.is_file():
                    validation.error(f"{shown_id} 的逐图标签引用文件不存在：{file_value}")
                if file_name not in formal_names:
                    validation.error(f"{shown_id} 引用了非正式分析画面：{file_value}")

        visual_group = item.get("visual_group")
        if not isinstance(visual_group, str) or visual_group not in groups:
            validation.error(f"{shown_id} 的 visual_group 未回填或不是正式分组：{visual_group}")

        evidence = item.get("image_specific_evidence")
        if not isinstance(evidence, list) or len(evidence) < 2 or any(
            not isinstance(entry, str) or not entry.strip() for entry in evidence
        ):
            validation.error(f"{shown_id} 至少需要两条非空 image_specific_evidence")
        else:
            evidence_key = tuple(normalized_label_text(entry) for entry in evidence)
            previous = evidence_fingerprints.get(evidence_key)
            if previous is not None:
                validation.error(f"{shown_id} 与 {previous} 的逐图证据完全相同，疑似复制模板")
            else:
                evidence_fingerprints[evidence_key] = str(shown_id)

        features = item.get("transferable_features")
        if not isinstance(features, list) or not features or any(
            not isinstance(entry, str) or not entry.strip() for entry in features
        ):
            validation.error(f"{shown_id} 的 transferable_features 必须是非空字符串数组")

        composition = normalized_label_text(item.get("composition"))
        subject_ratio = normalized_label_text(item.get("shot_and_subject_ratio"))
        if composition and composition == subject_ratio:
            validation.error(f"{shown_id} 的 composition 与 shot_and_subject_ratio 完全相同")

        observation_key = tuple(normalized_label_text(item.get(field)) for field in observation_fields)
        if all(observation_key):
            previous = observation_fingerprints.get(observation_key)
            if previous is not None:
                validation.error(f"{shown_id} 与 {previous} 的完整逐图观察完全相同，疑似复制分组模板")
            else:
                observation_fingerprints[observation_key] = str(shown_id)

    missing = formal_names - observed_names
    extra = observed_names - formal_names
    if missing:
        validation.error(f"单图标签漏掉 {len(missing)} 个正式分析画面：{sorted(missing)}")
    if extra:
        validation.error(f"单图标签包含 {len(extra)} 个非正式分析画面：{sorted(extra)}")
    validation.check("image_label_items", len(items))
    validation.check("image_label_unique_files", len(seen_files))


def validate_analysis_documents(
    style_pack: Path,
    formal_names: set[str],
    groups: dict[str, GroupInfo],
    validation: Validation,
) -> dict[str, str]:
    analysis = style_pack / "分析"
    documents: dict[str, str] = {}
    for path, label in (
        (analysis / "媒体检测结果.json", "媒体检测结果"),
        (analysis / "去重与筛选记录.json", "去重与筛选记录"),
    ):
        payload = load_json(path, validation, label)
        if payload in ({}, [], None):
            validation.error(f"{label}必须包含有效记录：{path}")

    labels = load_json(analysis / "单图标签-第一版.json", validation, "单图标签")
    if labels in ({}, [], None):
        validation.error(f"单图标签必须包含有效记录：{analysis / '单图标签-第一版.json'}")
    else:
        validate_image_labels(labels, style_pack, formal_names, groups, validation)
    for relative, label in (
        ("风格规则-第一版.md", "风格规则"),
        ("分组总结.md", "分组总结"),
    ):
        path = analysis / relative
        text = require_nonempty(path, validation, label)
        if text is not None:
            documents[relative] = text
    summary = documents.get("分组总结.md", "")
    for group_name in groups:
        if group_name not in summary:
            validation.error(f"分组总结未覆盖正式分组：{group_name}")

    prompt_root = analysis / "分组提示词"
    for group_name in groups:
        path = prompt_root / f"{group_name}.md"
        text = require_nonempty(path, validation, f"{group_name} 分组提示词")
        if text is None:
            continue
        documents[f"分组提示词/{group_name}.md"] = text
        for heading in PROMPT_HEADINGS:
            if heading not in text:
                validation.error(f"{group_name} 分组提示词缺少章节：{heading}")
        movement_headings = len(re.findall(r"^###\s+.*运动方案", text, flags=re.MULTILINE))
        if movement_headings < 2:
            validation.error(f"{group_name} 分组提示词至少需要两套视频运动方案")
    return documents


def validate_card_png(path: Path, validation: Validation, label: str) -> Image.Image | None:
    info = inspect_image(path, validation, label)
    if info is None:
        return None
    if info[0] != CARD_SIZE:
        validation.error(f"{label}尺寸错误：{info[0]}，应为 {CARD_SIZE}")
    if info[1] != "RGB":
        validation.error(f"{label}色彩模式错误：{info[1]}，应为 RGB")
    if info[2] != "PNG":
        validation.error(f"{label}格式错误：{info[2]}，应为 PNG")
    try:
        with Image.open(path) as image:
            return image.convert("RGB").copy()
    except Exception:
        return None


def validate_color_row(group_name: str, row: object, slot: str, role: str, validation: Validation) -> str | None:
    if not isinstance(row, dict):
        validation.error(f"{group_name} {slot} 颜色记录不是对象")
        return None
    if row.get("slot") != slot:
        validation.error(f"{group_name} 色位顺序错误：应为 {slot}，实际 {row.get('slot')}")
    if row.get("role") != role:
        validation.error(f"{group_name} {slot} 角色错误：应为 {role}，实际 {row.get('role')}")
    if not isinstance(row.get("name"), str) or not row["name"].strip():
        validation.error(f"{group_name} {slot} 缺少中文色名")
    hex_value = row.get("hex")
    if not isinstance(hex_value, str) or not UPPER_HEX_PATTERN.fullmatch(hex_value):
        validation.error(f"{group_name} {slot} HEX 不是大写 #RRGGBB：{hex_value}")
        return None
    rgb = row.get("rgb")
    if not isinstance(rgb, list) or len(rgb) != 3 or any(
        not isinstance(value, int) or not 0 <= value <= 255 for value in rgb
    ):
        validation.error(f"{group_name} {slot} RGB 无效：{rgb}")
    else:
        expected_hex = "#{:02X}{:02X}{:02X}".format(*rgb)
        if expected_hex != hex_value:
            validation.error(f"{group_name} {slot} HEX/RGB 不一致：{hex_value} != {expected_hex}")
    for field_name in ("sample_share", "image_recurrence"):
        value = row.get(field_name)
        if not isinstance(value, (int, float)) or not 0 <= value <= 1:
            validation.error(f"{group_name} {slot} {field_name} 无效：{value}")
    return hex_value


def validate_pure_pixels(
    group_name: str, image: Image.Image | None, colors: list[object], validation: Validation
) -> None:
    if image is None or image.size != CARD_SIZE or image.mode != "RGB":
        return
    pixel_counts = image.getcolors(maxcolors=11)
    expected_count = (CARD_SIZE[0] // 5) * (CARD_SIZE[1] // 2)
    if pixel_counts is None or len(pixel_counts) != 10:
        actual = "超过 11" if pixel_counts is None else len(pixel_counts)
        validation.error(f"{group_name} 纯色色卡必须恰好 10 个 RGB 颜色，实际 {actual}")
    elif sorted(count for count, _ in pixel_counts) != [expected_count] * 10:
        validation.error(f"{group_name} 纯色色卡每种颜色必须精确占 {expected_count} 像素")
    if len(colors) != 10 or any(not isinstance(row, dict) for row in colors):
        return
    cell_width = CARD_SIZE[0] // 5
    cell_height = CARD_SIZE[1] // 2
    for index, row in enumerate(colors):
        rgb = row.get("rgb") if isinstance(row, dict) else None
        if not isinstance(rgb, list) or len(rgb) != 3:
            continue
        row_index, column = divmod(index, 5)
        center = (column * cell_width + cell_width // 2, row_index * cell_height + cell_height // 2)
        actual = image.getpixel(center)
        if actual != tuple(rgb):
            validation.error(f"{group_name} 纯色色卡 {EXPECTED_SLOTS[index]} 色位错误：{actual} != {tuple(rgb)}")


def validate_palette_overview(path: Path, group_count: int, label: str, validation: Validation) -> None:
    info = inspect_image(path, validation, label)
    if info is None:
        return
    rows = math.ceil(group_count / PALETTE_OVERVIEW_COLUMNS)
    expected_size = (
        PALETTE_OVERVIEW_COLUMNS * PALETTE_THUMB_SIZE[0]
        + (PALETTE_OVERVIEW_COLUMNS + 1) * PALETTE_OVERVIEW_GAP,
        rows * PALETTE_THUMB_SIZE[1] + (rows + 1) * PALETTE_OVERVIEW_GAP,
    )
    if info[0] != expected_size:
        validation.error(f"{label}尺寸错误：{info[0]}，应为 {expected_size}")
    if info[2] != "JPEG":
        validation.error(f"{label}格式错误：{info[2]}，应为 JPEG")


def validate_palette(
    style_pack: Path,
    reference_root: Path,
    groups: dict[str, GroupInfo],
    validation: Validation,
) -> dict[str, set[str]]:
    payload = load_json(style_pack / "分析" / "色卡数据.json", validation, "色卡数据")
    if not isinstance(payload, dict):
        return {}
    if not isinstance(payload.get("project"), str) or not payload["project"].strip():
        validation.error("色卡数据缺少 project")
    if payload.get("card_size") != list(CARD_SIZE):
        validation.error(f"色卡数据 card_size 错误：{payload.get('card_size')}")
    if payload.get("color_model") != COLOR_MODEL:
        validation.error(f"色卡数据 color_model 错误：{payload.get('color_model')}")
    rows = payload.get("groups")
    if not isinstance(rows, list):
        validation.error("色卡数据 groups 必须是数组")
        return {}
    by_name = {
        row.get("group"): row
        for row in rows
        if isinstance(row, dict) and isinstance(row.get("group"), str)
    }
    if set(by_name) != set(groups):
        validation.error(f"色卡 JSON 分组与目录不一致：JSON={sorted(by_name)} 目录={sorted(groups)}")

    colors_by_group: dict[str, set[str]] = {}
    for group_name, group in groups.items():
        card_like = sorted(path.name for path in group.directory.iterdir() if path.is_file() and "色卡" in path.name)
        expected_names = sorted((group.card.name, group.pure_card.name))
        if card_like != expected_names:
            validation.error(f"{group_name} 色卡文件必须且只能是 {expected_names}，实际 {card_like}")
        validate_card_png(group.card, validation, f"{group_name} 标注色卡")
        pure_image = validate_card_png(group.pure_card, validation, f"{group_name} 纯色色卡")
        row = by_name.get(group_name)
        if not isinstance(row, dict):
            continue
        if row.get("reference_count") != len(group.references):
            validation.error(
                f"{group_name} reference_count 错误：{row.get('reference_count')} != {len(group.references)}"
            )
        expected_card = group.card.resolve().relative_to(style_pack.resolve()).as_posix()
        expected_pure = group.pure_card.resolve().relative_to(style_pack.resolve()).as_posix()
        if row.get("card") != expected_card:
            validation.error(f"{group_name} card 路径错误：{row.get('card')} != {expected_card}")
        if row.get("pure_card") != expected_pure:
            validation.error(f"{group_name} pure_card 路径错误：{row.get('pure_card')} != {expected_pure}")
        if not isinstance(row.get("color_guidance"), str) or not row["color_guidance"].strip():
            validation.error(f"{group_name} 缺少 color_guidance")
        colors = row.get("colors")
        if not isinstance(colors, list) or len(colors) != 10:
            validation.error(f"{group_name} 必须正好 10 色")
            continue
        hex_values: list[str] = []
        for color, slot, role in zip(colors, EXPECTED_SLOTS, EXPECTED_ROLES):
            value = validate_color_row(group_name, color, slot, role, validation)
            if value is not None:
                hex_values.append(value)
        if len(set(hex_values)) != 10:
            validation.error(f"{group_name} 存在重复或无效 HEX：{hex_values}")
        colors_by_group[group_name] = set(hex_values)
        validate_pure_pixels(group_name, pure_image, colors, validation)
        if pure_image is not None:
            pure_image.close()

    validate_palette_overview(reference_root / "分组色卡总览.jpg", len(groups), "标注色卡总览", validation)
    validate_palette_overview(reference_root / "纯色色卡总览.jpg", len(groups), "纯色色卡总览", validation)
    validation.check("palette_groups", len(colors_by_group))
    return colors_by_group


def validate_document_colors(
    root_documents: dict[str, str],
    analysis_documents: dict[str, str],
    colors_by_group: dict[str, set[str]],
    validation: Validation,
) -> None:
    all_colors = set().union(*colors_by_group.values()) if colors_by_group else set()
    for name, text in root_documents.items():
        values = {value.upper() for value in HEX_PATTERN.findall(text)}
        unknown = values - all_colors
        if unknown:
            validation.error(f"{name} 包含不属于色卡数据的 HEX：{sorted(unknown)}")
        if name == "design.md" and not values:
            validation.warning("design.md 未写入任何实际色卡 HEX，请确认已同步真实色值")
    rules = analysis_documents.get("风格规则-第一版.md")
    if rules is not None:
        values = {value.upper() for value in HEX_PATTERN.findall(rules)}
        unknown = values - all_colors
        if unknown:
            validation.error(f"风格规则包含不属于色卡数据的 HEX：{sorted(unknown)}")
        if not values:
            validation.warning("风格规则未写入任何实际色卡 HEX，请确认已同步真实色值")
    for group_name, allowed in colors_by_group.items():
        key = f"分组提示词/{group_name}.md"
        text = analysis_documents.get(key)
        if text is None:
            continue
        values = {value.upper() for value in HEX_PATTERN.findall(text)}
        unknown = values - allowed
        if unknown:
            validation.error(f"{group_name} 分组提示词包含不属于本组色卡的 HEX：{sorted(unknown)}")
        if not values:
            validation.warning(f"{group_name} 分组提示词未写入任何本组实际色卡 HEX")


def validate_contact_report(style_pack: Path, groups: dict[str, GroupInfo], video_ids: list[str], validation: Validation) -> None:
    payload = load_json(style_pack / "分析" / "联系表生成记录.json", validation, "联系表生成记录")
    if not isinstance(payload, dict):
        return
    if payload.get("errors"):
        validation.error(f"联系表生成记录包含错误：{payload.get('errors')}")
    sheets = payload.get("sheets")
    if not isinstance(sheets, list):
        validation.error("联系表生成记录 sheets 必须是数组")
        return
    keys = {
        (row.get("kind"), row.get("name"))
        for row in sheets
        if isinstance(row, dict)
    }
    expected_counts: dict[tuple[str, str], int] = {
        ("master", "全部正式分析素材"): int(validation.checks.get("formal_analysis_images", 0)),
    }
    expected_counts.update(
        {
            ("video", video_id): int(validation.checks.get(f"{video_id}_kept_frames", 0))
            for video_id in video_ids
        }
    )
    expected_counts.update(
        {("group", group_name): len(group.references) for group_name, group in groups.items()}
    )
    settings = payload.get("settings") if isinstance(payload.get("settings"), dict) else {}
    columns = settings.get("columns")
    rows_per_page = settings.get("rows")
    page_capacity = columns * rows_per_page if isinstance(columns, int) and isinstance(rows_per_page, int) else 0
    for row in sheets:
        if not isinstance(row, dict):
            validation.error("联系表生成记录包含非对象条目")
            continue
        paths = row.get("paths")
        page_count = row.get("page_count")
        item_count = row.get("item_count")
        if not isinstance(paths, list) or not paths:
            validation.error(f"联系表记录缺少 paths：{row}")
            continue
        if page_count != len(paths):
            validation.error(f"联系表 page_count 与 paths 数量不一致：{row.get('name')}")
        if not isinstance(item_count, int) or item_count <= 0:
            validation.error(f"联系表 item_count 无效：{row.get('name')}")
        kind = row.get("kind")
        name = row.get("name")
        expected_count = expected_counts.get((kind, name)) if isinstance(kind, str) and isinstance(name, str) else None
        if expected_count is not None and item_count != expected_count:
            validation.error(f"联系表 {kind}/{name} item_count 与正式素材不一致：{item_count} != {expected_count}")
        if page_capacity <= 0:
            validation.error("联系表生成记录缺少有效 columns/rows 设置")
        elif isinstance(item_count, int) and page_count != math.ceil(item_count / page_capacity):
            validation.error(f"联系表 {kind}/{name} 分页数量与容量不一致")
        resolved_paths: list[Path] = []
        for value in paths:
            resolved = resolve_portable(style_pack, value)
            if resolved is None:
                validation.error(f"联系表路径不是相对 POSIX 路径：{value}")
            else:
                resolved_paths.append(resolved)
                require_jpeg(resolved, validation, f"联系表 {row.get('name')}")
        reference_root = style_pack / "参考素材"
        if kind == "master":
            expected_paths = overview_pages(reference_root / "总览-contact-sheet.jpg")
        elif kind == "video" and isinstance(name, str):
            expected_paths = overview_pages(reference_root / "视频抽帧总览" / f"{name}.jpg")
        elif kind == "group" and isinstance(name, str):
            expected_paths = overview_pages(reference_root / "分组总览" / f"{name}.jpg")
        else:
            expected_paths = []
        if {path.resolve() for path in resolved_paths} != {path.resolve() for path in expected_paths}:
            validation.error(f"联系表 {kind}/{name} 的 paths 与规范输出位置不一致")
    if not any(kind == "master" for kind, _ in keys):
        validation.error("联系表生成记录缺少 master 总览")
    for video_id in video_ids:
        if ("video", video_id) not in keys:
            validation.error(f"联系表生成记录缺少视频总览：{video_id}")
    for group_name in groups:
        if ("group", group_name) not in keys:
            validation.error(f"联系表生成记录缺少分组总览：{group_name}")


def main() -> int:
    args = parse_args()
    style_pack = args.style_pack.expanduser().resolve()
    if not style_pack.is_dir():
        print(f"风格包目录不存在：{style_pack}", file=sys.stderr)
        return 2
    reference_root = style_pack / "参考素材"
    analysis_root = style_pack / "分析"
    validation = Validation(style_pack)
    if not reference_root.is_dir():
        validation.error(f"缺少参考素材目录：{reference_root}")
    if not analysis_root.is_dir():
        validation.error(f"缺少分析目录：{analysis_root}")

    root_documents = validate_root_documents(style_pack, validation)
    sources, video_ids = validate_manifest(style_pack, reference_root, validation)
    frame_names = validate_videos(style_pack, reference_root, sources, video_ids, validation)
    formal_names, _ = validate_analysis_assets(reference_root, frame_names, 8, validation)
    groups = validate_groups(reference_root, formal_names, validation)
    analysis_documents = validate_analysis_documents(style_pack, formal_names, groups, validation)
    validate_contact_report(style_pack, groups, video_ids, validation)
    colors_by_group = validate_palette(style_pack, reference_root, groups, validation)
    validate_document_colors(root_documents, analysis_documents, colors_by_group, validation)
    validate_deliverable_privacy(style_pack, validation)

    if not args.visual_review_confirmed:
        validation.error("尚未记录人工目视验收；请检查全部素材、逐视频、分组和两套色卡总览")
    validation.check("visual_review_confirmed", bool(args.visual_review_confirmed))

    effective_errors = list(validation.errors)
    if args.warnings_as_errors:
        effective_errors.extend(f"[warning] {message}" for message in validation.warnings)
    status = "PASS" if not effective_errors else "FAIL"
    report_path = (args.report or analysis_root / "验收报告.json").expanduser().resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_errors = [sanitize_message(message, style_pack) for message in effective_errors]
    report_warnings = [sanitize_message(message, style_pack) for message in validation.warnings]
    payload = {
        "schema_version": "1.0",
        "validated_at": utc_now(),
        "style_pack": style_pack.name,
        "status": status,
        "automated_validation_passed": not effective_errors,
        "visual_review_confirmed": bool(args.visual_review_confirmed),
        "checks": validation.checks,
        "errors": report_errors,
        "warnings": report_warnings,
    }
    report_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    stream = sys.stdout if status == "PASS" else sys.stderr
    print(f"STYLE PACK VALIDATION {status}", file=stream)
    print(f"PACK\t{style_pack}", file=stream)
    for name, value in validation.checks.items():
        print(f"CHECK\t{name}\t{value}", file=stream)
    for warning in validation.warnings:
        print(f"WARNING\t{warning}", file=sys.stderr)
    for error in effective_errors:
        print(f"ERROR\t{error}", file=sys.stderr)
    print(f"REPORT\t{report_path}", file=stream)
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
