#!/usr/bin/env python3
"""生成风格包审核拼图、最终联系表与双版色卡。"""

from __future__ import annotations


import argparse

import json

import math

import re

import sys

from dataclasses import dataclass

from datetime import datetime, timezone

from pathlib import Path

from typing import Any, Iterable

from PIL import Image, ImageDraw, ImageFont, ImageOps

import colorsys

import io

import cv2

import numpy as np

from PIL import Image, ImageCms, ImageDraw, ImageFont, ImageOps

IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.webp', '.tif', '.tiff', '.bmp'}

CANVAS_SIZE = (1920, 1080)

MACOS_PINGFANG = Path('/System/Library/Fonts/PingFang.ttc')

FONT_CANDIDATES = (Path('C:/Windows/Fonts/msyh.ttc'), Path('C:/Windows/Fonts/msyhbd.ttc'), Path('C:/Windows/Fonts/simhei.ttf'), Path('C:/Windows/Fonts/simsun.ttc'), Path('/System/Library/Fonts/PingFang.ttc'), Path('/System/Library/Fonts/Hiragino Sans GB.ttc'), Path('/System/Library/Fonts/STHeiti Medium.ttc'), Path('/System/Library/Fonts/STHeiti Light.ttc'), Path('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'), Path('/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc'))

LATIN_FONT_CANDIDATES = (Path('C:/Windows/Fonts/arial.ttf'), Path('C:/Windows/Fonts/segoeui.ttf'), Path('/System/Library/Fonts/Supplemental/Arial.ttf'), Path('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'))

@dataclass(eq=False)
class Candidate:
    lab: np.ndarray
    rgb: tuple[int, int, int]
    weight: float
    recurrence: float
    chroma: float
    lightness: float

    @property
    def hex(self) -> str:
        return '#{:02X}{:02X}{:02X}'.format(*self.rgb)

def load_pingfang(size: int, index: int=0) -> ImageFont.ImageFont:
    if not MACOS_PINGFANG.is_file():
        raise RuntimeError(f'macOS 缺少系统 PingFang 字体：{MACOS_PINGFANG}')
    try:
        return ImageFont.truetype(str(MACOS_PINGFANG), size=size, index=index)
    except OSError as exc:
        raise RuntimeError(f'macOS 无法加载 PingFang 字体：{MACOS_PINGFANG}') from exc

def load_font(size: int, index: int=0) -> ImageFont.ImageFont:
    if sys.platform == 'darwin':
        return load_pingfang(size, index)
    for path in FONT_CANDIDATES:
        if path.is_file():
            try:
                return ImageFont.truetype(str(path), size=size, index=index)
            except OSError:
                continue
    raise RuntimeError('找不到可用中文字体；请安装微软雅黑、黑体、苹方或 Noto Sans CJK')

def load_latin_font(size: int) -> ImageFont.ImageFont:
    if sys.platform == 'darwin':
        return load_pingfang(size)
    for path in LATIN_FONT_CANDIDATES:
        if path.is_file():
            try:
                return ImageFont.truetype(str(path), size=size)
            except OSError:
                continue
    return load_font(size)

def open_srgb(path: Path) -> Image.Image:
    with Image.open(path) as source:
        image = ImageOps.exif_transpose(source)
        icc_profile = image.info.get('icc_profile')
        if icc_profile:
            try:
                src_profile = ImageCms.ImageCmsProfile(io.BytesIO(icc_profile))
                dst_profile = ImageCms.createProfile('sRGB')
                image = ImageCms.profileToProfile(image, src_profile, dst_profile, outputMode='RGB')
            except Exception:
                image = image.convert('RGB')
        else:
            image = image.convert('RGB')
        return image.copy()

def crop_letterbox(image: Image.Image) -> Image.Image:
    """仅裁掉几乎纯黑的连续边框，不会把普通暗场景当成黑边。"""
    array = np.asarray(image)
    luminance = 0.2126 * array[:, :, 0] + 0.7152 * array[:, :, 1] + 0.0722 * array[:, :, 2]
    dark_rows = (luminance < 6).mean(axis=1) > 0.985
    dark_cols = (luminance < 6).mean(axis=0) > 0.985

    def inward_limit(flags: np.ndarray, reverse: bool=False) -> int:
        values = flags[::-1] if reverse else flags
        count = 0
        for flag in values:
            if not flag:
                break
            count += 1
        return count
    top = inward_limit(dark_rows)
    bottom = inward_limit(dark_rows, reverse=True)
    left = inward_limit(dark_cols)
    right = inward_limit(dark_cols, reverse=True)
    if top + bottom >= image.height * 0.25 or left + right >= image.width * 0.25:
        return image
    return image.crop((left, top, image.width - right, image.height - bottom))

def image_files(group_dir: Path) -> list[Path]:
    return sorted((path for path in group_dir.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS and ('色卡' not in path.name)))

def sample_group(paths: list[Path], sample_per_image: int=3600) -> tuple[np.ndarray, np.ndarray, list[int]]:
    rng = np.random.default_rng(20260721)
    samples: list[np.ndarray] = []
    image_ids: list[np.ndarray] = []
    counts: list[int] = []
    for image_index, path in enumerate(paths):
        image = crop_letterbox(open_srgb(path))
        image.thumbnail((420, 420), Image.Resampling.LANCZOS)
        pixels = np.asarray(image, dtype=np.uint8).reshape(-1, 3)
        count = min(sample_per_image, len(pixels))
        indexes = rng.choice(len(pixels), size=count, replace=False)
        samples.append(pixels[indexes])
        image_ids.append(np.full(count, image_index, dtype=np.int32))
        counts.append(count)
    rgb = np.concatenate(samples, axis=0)
    ids = np.concatenate(image_ids, axis=0)
    return (rgb, ids, counts)

def cv_lab_to_standard(lab_cv: np.ndarray) -> np.ndarray:
    return np.array([lab_cv[0] * 100.0 / 255.0, lab_cv[1] - 128.0, lab_cv[2] - 128.0], dtype=np.float32)

def delta_e(first: Candidate, second: Candidate) -> float:
    return float(np.linalg.norm(first.lab - second.lab))

def cluster_candidates(rgb: np.ndarray, image_ids: np.ndarray, image_sample_counts: list[int]) -> list[Candidate]:
    lab = cv2.cvtColor(rgb.reshape(-1, 1, 3), cv2.COLOR_RGB2LAB).reshape(-1, 3)
    lab_float = lab.astype(np.float32)
    cluster_count = min(28, max(16, len(rgb) // 1800))
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 80, 0.2)
    cv2.setRNGSeed(20260721)
    _, labels, centers = cv2.kmeans(lab_float, cluster_count, None, criteria, 8, cv2.KMEANS_PP_CENTERS)
    labels = labels.reshape(-1)
    center_pixels = np.clip(np.rint(centers), 0, 255).astype(np.uint8).reshape(1, -1, 3)
    rgb_centers = cv2.cvtColor(center_pixels, cv2.COLOR_LAB2RGB).reshape(-1, 3)
    total = len(labels)
    candidates: list[Candidate] = []
    for cluster_id, center in enumerate(centers):
        mask = labels == cluster_id
        count = int(mask.sum())
        present = 0
        for image_index, image_count in enumerate(image_sample_counts):
            share = int(np.logical_and(mask, image_ids == image_index).sum()) / image_count
            if share >= 0.008:
                present += 1
        standard_lab = cv_lab_to_standard(center)
        rgb_tuple = tuple((int(value) for value in rgb_centers[cluster_id]))
        candidates.append(Candidate(lab=standard_lab, rgb=rgb_tuple, weight=count / total, recurrence=present / len(image_sample_counts), chroma=float(math.hypot(standard_lab[1], standard_lab[2])), lightness=float(standard_lab[0])))
    return sorted(candidates, key=lambda item: item.weight, reverse=True)

def min_distance(candidate: Candidate, selected: list[Candidate]) -> float:
    if not selected:
        return 100.0
    return min((delta_e(candidate, existing) for existing in selected))

def choose_with_relaxation(pool: list[Candidate], selected: list[Candidate], count: int, scorer, thresholds: tuple[float, ...]) -> list[Candidate]:
    chosen: list[Candidate] = []
    for threshold in thresholds:
        while len(chosen) < count:
            available = [item for item in pool if item not in selected and item not in chosen and (min_distance(item, selected + chosen) >= threshold)]
            if not available:
                break
            best = max(available, key=lambda item: scorer(item, selected + chosen))
            chosen.append(best)
        if len(chosen) == count:
            break
    return chosen

def rgb_hue_and_saturation(rgb: tuple[int, int, int]) -> tuple[float, float, float]:
    red, green, blue = (value / 255.0 for value in rgb)
    hue, lightness, saturation = colorsys.rgb_to_hls(red, green, blue)
    return (hue * 360.0, saturation, lightness)

HUE_FAMILIES = (('蓝', ('靛蓝', '青蓝', '孔雀蓝', '冷蓝', '浅蓝', '蓝'), 210.0), ('红', ('暗红', '砖红', '酒红', '红'), 4.0), ('橙', ('旧橙', '暖橙', '橙'), 28.0), ('黄', ('暖黄', '沙黄', '尘黄', '金黄', '黄'), 48.0), ('绿', ('墨绿', '橄榄绿', '绿'), 125.0), ('青', ('青绿', '青'), 180.0), ('紫', ('紫罗兰', '紫'), 282.0), ('粉', ('玫瑰粉', '粉'), 335.0))

def hue_targets(text: str) -> list[tuple[str, float]]:
    targets = [(family, target) for family, words, target in HUE_FAMILIES if any((word in text for word in words))]
    if any((word in text for word in ('青蓝', '孔雀蓝'))):
        targets = [item for item in targets if item[0] != '青']
    return targets

def circular_hue_distance(first: float, second: float) -> float:
    distance = abs(first - second)
    return min(distance, 360.0 - distance)

def palette_covers_hue(selected: list[Candidate], target: float) -> bool:
    for item in selected:
        hue, saturation, lightness = rgb_hue_and_saturation(item.rgb)
        if saturation >= 0.13 and lightness >= 0.045 and (circular_hue_distance(hue, target) <= 30.0):
            return True
    return False

def accent_hue_targets(text: str) -> list[tuple[str, float]]:
    markers = ('点缀', '小面积', '少量', '一块', '局部')
    ranked: list[tuple[int, str, float]] = []
    for clause in re.split('[，。；;]', text):
        marker_positions = [clause.find(marker) for marker in markers if marker in clause]
        if not marker_positions:
            continue
        for family, words, target in HUE_FAMILIES:
            positions = [clause.find(word) for word in words if word in clause]
            if not positions:
                continue
            distance = min((abs(color_pos - marker_pos) for color_pos in positions for marker_pos in marker_positions))
            ranked.append((distance, family, target))
    result: list[tuple[str, float]] = []
    for _, family, target in sorted(ranked):
        if family == '青' and any((existing == '蓝' for existing, _ in result)):
            continue
        if not any((existing == family for existing, _ in result)):
            result.append((family, target))
    return result

def candidate_for_hue(candidates: list[Candidate], target: float, excluded: list[Candidate] | None=None) -> Candidate | None:
    excluded = excluded or []
    viable: list[Candidate] = []
    for item in candidates:
        _, saturation, lightness = rgb_hue_and_saturation(item.rgb)
        if saturation >= 0.14 and 0.045 <= lightness <= 0.9 and (item.weight >= 0.003) and (min_distance(item, excluded) >= 7.0):
            viable.append(item)
    if not viable:
        return None

    def score(item: Candidate) -> float:
        hue, saturation, _ = rgb_hue_and_saturation(item.rgb)
        hue_distance = circular_hue_distance(hue, target)
        hue_match = max(0.0, 1.0 - hue_distance / 65.0)
        evidence = min(item.weight / 0.035, 1.0)
        return 0.72 * hue_match + 0.1 * min(saturation / 0.75, 1.0) + 0.08 * item.recurrence + 0.1 * evidence
    return max(viable, key=score)

def hinted_primary(candidates: list[Candidate], group_name: str) -> Candidate | None:
    """组名里明确出现的色相只作为候选色的语义纠偏，不会凭空造色。"""
    targets = hue_targets(group_name)
    target = targets[0][1] if targets else None
    if target is None:
        return None
    return candidate_for_hue(candidates, target)

def select_palette(candidates: list[Candidate], group_name: str, color_guidance: str) -> dict[str, list[Candidate]]:
    primaries: list[Candidate] = [candidates[0]]
    hinted = hinted_primary(candidates, group_name)
    if hinted is not None and min_distance(hinted, primaries) >= 7.0:
        primaries.append(hinted)

    def primary_score(item: Candidate, selected: list[Candidate]) -> float:
        diversity = min(min_distance(item, selected), 65.0) / 65.0
        return item.weight ** 0.52 * (0.68 + 0.32 * item.recurrence) * (0.55 + 0.45 * diversity)
    primaries.extend(choose_with_relaxation(candidates, primaries, 3 - len(primaries), primary_score, thresholds=(13.0, 10.0, 7.0, 4.0)))

    def accent_score(item: Candidate, selected: list[Candidate]) -> float:
        chroma = min(item.chroma / 72.0, 1.25)
        contrast = min(min_distance(item, primaries), 70.0) / 70.0
        evidence = min(item.weight / 0.012, 1.0)
        return 0.48 * chroma + 0.24 * contrast + 0.18 * item.recurrence + 0.1 * evidence
    accent_anchors: list[Candidate] = []
    for _, target in accent_hue_targets(color_guidance):
        if palette_covers_hue(accent_anchors, target):
            continue
        anchor = candidate_for_hue(candidates, target, primaries + accent_anchors)
        if anchor is not None and min_distance(anchor, primaries) >= 9.0:
            accent_anchors.append(anchor)
    semantic_anchors: list[Candidate] = []
    for family, target in hue_targets(color_guidance):
        if any((family == accent_family for accent_family, _ in accent_hue_targets(color_guidance))):
            continue
        if palette_covers_hue(primaries + accent_anchors + semantic_anchors, target):
            continue
        anchor = candidate_for_hue(candidates, target, primaries + semantic_anchors)
        if anchor is not None and min_distance(anchor, primaries) >= 9.0:
            semantic_anchors.append(anchor)
    accent_pool = [item for item in candidates if item not in primaries and item.weight >= 0.0035 and (item.recurrence >= 0.12)]
    accents = accent_anchors[:2]
    if len(accents) < 2:
        fallback_semantic = sorted(semantic_anchors, key=lambda item: (item.weight, -item.chroma))
        for item in fallback_semantic:
            if min_distance(item, primaries + accents) >= 7.0:
                accents.append(item)
            if len(accents) == 2:
                break
    accents.extend(choose_with_relaxation(accent_pool, primaries + accents, 2 - len(accents), accent_score, thresholds=(15.0, 12.0, 9.0, 6.0)))

    def secondary_score(item: Candidate, selected: list[Candidate]) -> float:
        diversity = min(min_distance(item, selected), 55.0) / 55.0
        weight = min(item.weight / 0.045, 1.0)
        chroma = min(item.chroma / 55.0, 1.0)
        return 0.28 * weight + 0.24 * item.recurrence + 0.38 * diversity + 0.1 * chroma
    secondaries = [item for item in semantic_anchors if item not in accents and min_distance(item, primaries + accents) >= 7.0][:5]
    secondaries.extend(choose_with_relaxation(candidates, primaries + accents + secondaries, 5 - len(secondaries), secondary_score, thresholds=(12.0, 9.0, 7.0, 4.0, 0.0)))
    selected = primaries + secondaries + accents
    if len(selected) != 10:
        raise RuntimeError(f'未能选满 10 个色位，当前为 {len(selected)}')
    if len({item.hex for item in selected}) != 10:
        raise RuntimeError('颜色取整后出现重复 HEX')
    return {'primary': sorted(primaries, key=lambda item: item.weight, reverse=True), 'secondary': sorted(secondaries, key=lambda item: item.lightness), 'accent': sorted(accents, key=lambda item: item.chroma, reverse=True)}

def chinese_color_name(rgb: tuple[int, int, int]) -> str:
    red, green, blue = (value / 255.0 for value in rgb)
    hue, lightness, saturation = colorsys.rgb_to_hls(red, green, blue)
    hue *= 360.0
    if max(rgb) <= 22:
        return '墨黑'
    if saturation < 0.1:
        if lightness < 0.12:
            return '墨黑'
        if lightness < 0.3:
            return '炭灰'
        if lightness < 0.58:
            return '中灰'
        if lightness < 0.82:
            return '浅灰'
        return '灰白'
    if hue < 15 or hue >= 345:
        base = '酒红' if lightness < 0.38 else '珊瑚红' if lightness > 0.62 else '砖红'
    elif hue < 45:
        if saturation < 0.28:
            base = '灰褐'
        elif hue < 25:
            base = '赤褐' if lightness < 0.48 else '陶红'
        elif hue < 36:
            base = '赭石棕' if lightness < 0.5 else '焦糖橙'
        else:
            base = '黄褐' if lightness < 0.5 else '驼棕'
    elif hue < 72:
        base = '卡其黄' if saturation < 0.42 else '金黄'
    elif hue < 155:
        base = '墨绿' if lightness < 0.34 else '橄榄绿' if saturation < 0.45 else '草绿'
    elif hue < 190:
        base = '深青' if lightness < 0.36 else '青绿'
    elif hue < 250:
        base = '靛蓝' if lightness < 0.32 else '钴蓝' if saturation > 0.48 else '雾蓝'
    elif hue < 290:
        base = '深紫' if lightness < 0.38 else '紫罗兰'
    elif hue < 345:
        base = '暗玫红' if lightness < 0.42 else '玫瑰粉'
    else:
        base = '中色'
    if lightness > 0.78:
        return '浅' + base
    if lightness < 0.18 and (not base.startswith('深')) and (not base.startswith('暗')):
        return '深' + base
    return base

def relative_luminance(rgb: tuple[int, int, int]) -> float:
    channels = []
    for value in rgb:
        normalized = value / 255.0
        channels.append(normalized / 12.92 if normalized <= 0.04045 else ((normalized + 0.055) / 1.055) ** 2.4)
    return 0.2126 * channels[0] + 0.7152 * channels[1] + 0.0722 * channels[2]

def render_card(project_title: str, group_name: str, palette: dict[str, list[Candidate]], output_path: Path) -> None:
    width, height = CANVAS_SIZE
    canvas = Image.new('RGB', CANVAS_SIZE, '#F1EEE7')
    draw = ImageDraw.Draw(canvas)
    title_font = load_font(52)
    group_font = load_font(31)
    meta_font = load_font(25)
    role_font = load_font(31)
    name_font = load_font(39)
    hex_font = load_latin_font(34)
    header_height = 188
    draw.rectangle((0, 0, width, header_height), fill='#F1EEE7')
    draw.text((52, 30), f'《{project_title}》色卡', fill='#171715', font=title_font)
    draw.text((54, 108), group_name, fill='#393833', font=group_font)
    right_meta = '10 色  ·  3 主色  ·  5 辅助色  ·  2 点缀色'
    meta_box = draw.textbbox((0, 0), right_meta, font=meta_font)
    draw.text((width - 54 - (meta_box[2] - meta_box[0]), 116), right_meta, fill='#6A6861', font=meta_font)
    palette_top = 214
    margin_x = 46
    gap_x = 14
    gap_y = 14
    bottom_margin = 40
    cell_width = (width - margin_x * 2 - gap_x * 4) // 5
    cell_height = (height - palette_top - bottom_margin - gap_y) // 2
    ordered: list[tuple[str, str, Candidate]] = []
    ordered.extend(((f'P{i}', '主色', item) for i, item in enumerate(palette['primary'], 1)))
    ordered.extend(((f'S{i}', '辅助色', item) for i, item in enumerate(palette['secondary'], 1)))
    ordered.extend(((f'A{i}', '点缀色', item) for i, item in enumerate(palette['accent'], 1)))
    for index, (slot, role, candidate) in enumerate(ordered):
        row, column = divmod(index, 5)
        x0 = margin_x + column * (cell_width + gap_x)
        y0 = palette_top + row * (cell_height + gap_y)
        x1 = x0 + cell_width
        y1 = y0 + cell_height
        draw.rounded_rectangle((x0, y0, x1, y1), radius=14, fill=candidate.rgb)
        text_color = '#FFFFFF' if relative_luminance(candidate.rgb) < 0.32 else '#151513'
        role_text = f'{slot}  {role}'
        name = chinese_color_name(candidate.rgb)
        draw.text((x0 + 28, y0 + 27), role_text, fill=text_color, font=role_font)
        draw.text((x0 + 28, y0 + 104), name, fill=text_color, font=name_font)
        draw.text((x0 + 28, y1 - 68), candidate.hex, fill=text_color, font=hex_font)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path, format='PNG', optimize=True, icc_profile=None)

def ordered_candidates(palette: dict[str, list[Candidate]]) -> list[Candidate]:
    return palette['primary'] + palette['secondary'] + palette['accent']

def render_pure_card(palette: dict[str, list[Candidate]], output_path: Path) -> None:
    """绘制只有颜色的满版 5×2 色卡，不允许出现任何非色块像素。"""
    width, height = CANVAS_SIZE
    cell_width = width // 5
    cell_height = height // 2
    canvas = Image.new('RGB', CANVAS_SIZE)
    for index, candidate in enumerate(ordered_candidates(palette)):
        row, column = divmod(index, 5)
        block = Image.new('RGB', (cell_width, cell_height), candidate.rgb)
        canvas.paste(block, (column * cell_width, row * cell_height))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path, format='PNG', optimize=True, icc_profile=None)

def palette_payload(palette: dict[str, list[Candidate]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    definitions = (('primary', 'P', '主色'), ('secondary', 'S', '辅助色'), ('accent', 'A', '点缀色'))
    for key, prefix, role in definitions:
        for index, candidate in enumerate(palette[key], 1):
            rows.append({'slot': f'{prefix}{index}', 'role': role, 'name': chinese_color_name(candidate.rgb), 'hex': candidate.hex, 'rgb': list(candidate.rgb), 'sample_share': round(candidate.weight, 4), 'image_recurrence': round(candidate.recurrence, 4)})
    return rows

def render_overview(card_paths: list[Path], output_path: Path) -> None:
    thumb_size = (960, 540)
    gap = 18
    columns = 2
    rows = math.ceil(len(card_paths) / columns)
    canvas = Image.new('RGB', (columns * thumb_size[0] + (columns + 1) * gap, rows * thumb_size[1] + (rows + 1) * gap), '#D8D4CC')
    for index, path in enumerate(card_paths):
        with Image.open(path) as image:
            thumb = image.convert('RGB').resize(thumb_size, Image.Resampling.LANCZOS)
        x = gap + index % columns * (thumb_size[0] + gap)
        y = gap + index // columns * (thumb_size[1] + gap)
        canvas.paste(thumb, (x, y))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path, format='JPEG', quality=92, optimize=True)

def portable_path(path: Path, style_pack: Path) -> str:
    return path.resolve().relative_to(style_pack.resolve()).as_posix()

def color_guidance_for_group(style_pack: Path, group_dir: Path) -> str:
    prompt_path = style_pack / '分析' / '分组提示词' / f'{group_dir.name}.md'
    if not prompt_path.is_file():
        return group_dir.name
    lines = prompt_path.read_text(encoding='utf-8').splitlines()
    for index, line in enumerate(lines):
        if re.match('^\\s*-\\s*\\*{0,2}色彩', line):
            return line.strip()
        if line.strip() == '## 视觉锁':
            for candidate in lines[index + 1:]:
                stripped = candidate.strip().strip('`')
                if not stripped:
                    continue
                if stripped.startswith('#'):
                    break
                return stripped
    return group_dir.name

CARD_MARKERS = ('-色卡', '-纯色色卡', '色卡总览')

VIDEO_ID_PATTERN = re.compile('^VID-\\d{4}$')

FRAME_NAME_PATTERN = re.compile('^(?P<video>VID-\\d{4})__(?P<timestamp>\\d{12})__(?P<method>scene|uniform|boundary)__(?P<sequence>\\d{4})')

@dataclass(frozen=True)
class SheetItem:
    path: Path
    label: str
    sort_key: tuple[object, ...]

@dataclass(frozen=True)
class RenderedSheet:
    paths: tuple[Path, ...]
    item_count: int
    page_count: int

def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec='seconds')

def is_image(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS

def is_generated_asset(path: Path) -> bool:
    name = path.name
    return any((marker in name for marker in CARD_MARKERS)) or name.startswith('总览-')

def image_paths(directory: Path, recursive: bool=False) -> list[Path]:
    if not directory.is_dir():
        return []
    iterator: Iterable[Path] = directory.rglob('*') if recursive else directory.glob('*')
    return sorted((path for path in iterator if is_image(path) and (not is_generated_asset(path))), key=lambda value: value.name.casefold())

def frame_identity(path: Path) -> tuple[int, str, str]:
    match = FRAME_NAME_PATTERN.match(path.name)
    if not match:
        return (0, 'frame', path.stem)
    milliseconds = int(match.group('timestamp'))
    method = match.group('method')
    return (milliseconds, method, match.group('video'))

def format_timestamp(milliseconds: int) -> str:
    total_seconds, millis = divmod(max(0, milliseconds), 1000)
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f'{hours:02d}:{minutes:02d}:{seconds:02d}.{millis:03d}'
    return f'{minutes:02d}:{seconds:02d}.{millis:03d}'

def discover_standardized_images(reference_root: Path) -> list[SheetItem]:
    directory = reference_root / '标准化图片'
    return [SheetItem(path=path, label=path.stem, sort_key=(0, path.name.casefold())) for path in image_paths(directory, recursive=True)]

def discover_kept_frames(reference_root: Path) -> list[SheetItem]:
    root = reference_root / '视频抽帧'
    if not root.is_dir():
        return []
    items: list[SheetItem] = []
    for video_dir in sorted((path for path in root.iterdir() if path.is_dir()), key=lambda p: p.name):
        kept_dir = video_dir / '保留帧'
        for path in image_paths(kept_dir):
            milliseconds, method, video_id = frame_identity(path)
            label = f'{video_id}  {format_timestamp(milliseconds)}  {method}'
            items.append(SheetItem(path=path, label=label, sort_key=(1, video_id, milliseconds, path.name.casefold())))
    return sorted(items, key=lambda item: item.sort_key)

def discover_video_items(video_dir: Path) -> list[SheetItem]:
    kept_dir = video_dir / '保留帧'
    files = image_paths(kept_dir)
    index_path = video_dir / 'frames.json'
    by_filename: dict[str, dict[str, Any]] = {}
    if index_path.is_file():
        try:
            payload = json.loads(index_path.read_text(encoding='utf-8'))
            frames = payload.get('frames', []) if isinstance(payload, dict) else []
            by_filename = {str(row.get('filename')): row for row in frames if isinstance(row, dict) and row.get('status') == 'kept' and row.get('filename')}
        except (OSError, json.JSONDecodeError):
            by_filename = {}
    items: list[SheetItem] = []
    for path in files:
        milliseconds, method, video_id = frame_identity(path)
        row = by_filename.get(path.name, {})
        timestamp_value = row.get('timestamp_ms')
        if isinstance(timestamp_value, int):
            milliseconds = timestamp_value
        method_value = row.get('method')
        if isinstance(method_value, str):
            method = method_value
        frame_id = row.get('frame_id') if isinstance(row.get('frame_id'), str) else path.stem
        label = f'{frame_id}  {format_timestamp(milliseconds)}  {method}'
        items.append(SheetItem(path=path, label=label, sort_key=(milliseconds, method, path.name.casefold())))
    return sorted(items, key=lambda item: item.sort_key)

def discover_group_items(group_dir: Path) -> list[SheetItem]:
    items: list[SheetItem] = []
    for path in image_paths(group_dir):
        milliseconds, method, video_id = frame_identity(path)
        if FRAME_NAME_PATTERN.match(path.name):
            label = f'{video_id}  {format_timestamp(milliseconds)}  {method}'
            sort_key: tuple[object, ...] = (1, video_id, milliseconds, path.name.casefold())
        else:
            label = path.stem
            sort_key = (0, path.name.casefold())
        items.append(SheetItem(path=path, label=label, sort_key=sort_key))
    return sorted(items, key=lambda item: item.sort_key)

def fit_image(path: Path, size: tuple[int, int]) -> Image.Image:
    with Image.open(path) as source:
        image = ImageOps.exif_transpose(source).convert('RGB')
        return ImageOps.contain(image, size, Image.Resampling.LANCZOS)

def ellipsize(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont, max_width: int) -> str:
    if draw.textbbox((0, 0), text, font=font)[2] <= max_width:
        return text
    suffix = '…'
    result = text
    while result and draw.textbbox((0, 0), result + suffix, font=font)[2] > max_width:
        result = result[:-1]
    return result + suffix if result else suffix

def page_path(base: Path, page_index: int, page_count: int) -> Path:
    if page_count <= 1:
        return base
    return base.with_name(f'{base.stem}-{page_index:02d}{base.suffix}')

def existing_page_paths(base: Path) -> list[Path]:
    pages = sorted(base.parent.glob(f'{base.stem}-[0-9][0-9]{base.suffix}'))
    if base.exists():
        pages.insert(0, base)
    return pages

def prepare_outputs(base: Path, overwrite: bool) -> None:
    existing = existing_page_paths(base)
    if existing and (not overwrite):
        raise FileExistsError(f'目标总览已存在，使用 --overwrite 才能覆盖：{existing[0]}')
    if overwrite:
        for path in existing:
            path.unlink()

def render_contact_sheet(items: list[SheetItem], output: Path, title: str, *, columns: int, rows: int, thumb_size: tuple[int, int], gap: int, margin: int, quality: int, background: str, labels: bool, overwrite: bool) -> RenderedSheet:
    if not items:
        raise ValueError(f'没有可用于总览的图片：{title}')
    output.parent.mkdir(parents=True, exist_ok=True)
    prepare_outputs(output, overwrite)
    label_height = 36 if labels else 0
    header_height = 78
    cell_width = thumb_size[0]
    cell_height = thumb_size[1] + label_height
    per_page = columns * rows
    page_count = math.ceil(len(items) / per_page)
    canvas_width = margin * 2 + columns * cell_width + (columns - 1) * gap
    canvas_height = margin * 2 + header_height + rows * cell_height + (rows - 1) * gap
    title_font = load_font(30)
    meta_font = load_font(18)
    label_font = load_font(16)
    saved: list[Path] = []
    for page_index in range(page_count):
        page_items = items[page_index * per_page:(page_index + 1) * per_page]
        canvas = Image.new('RGB', (canvas_width, canvas_height), background)
        draw = ImageDraw.Draw(canvas)
        draw.text((margin, margin), title, fill='#171715', font=title_font)
        meta = f'{len(items)} 项  ·  第 {page_index + 1}/{page_count} 页'
        meta_bbox = draw.textbbox((0, 0), meta, font=meta_font)
        draw.text((canvas_width - margin - (meta_bbox[2] - meta_bbox[0]), margin + 9), meta, fill='#605E58', font=meta_font)
        grid_top = margin + header_height
        for index, item in enumerate(page_items):
            row, column = divmod(index, columns)
            x = margin + column * (cell_width + gap)
            y = grid_top + row * (cell_height + gap)
            draw.rectangle((x, y, x + cell_width, y + thumb_size[1]), fill='#C8C4BC')
            try:
                image = fit_image(item.path, thumb_size)
                paste_x = x + (cell_width - image.width) // 2
                paste_y = y + (thumb_size[1] - image.height) // 2
                canvas.paste(image, (paste_x, paste_y))
            except Exception as exc:
                error_text = ellipsize(draw, f'读取失败：{exc}', label_font, cell_width - 16)
                draw.text((x + 8, y + 8), error_text, fill='#8B1E1E', font=label_font)
            if labels:
                label = ellipsize(draw, item.label, label_font, cell_width - 12)
                draw.text((x + 6, y + thumb_size[1] + 8), label, fill='#252421', font=label_font)
        target = page_path(output, page_index + 1, page_count)
        canvas.save(target, format='JPEG', quality=quality, optimize=True)
        saved.append(target)
    return RenderedSheet(paths=tuple(saved), item_count=len(items), page_count=page_count)

def relative_paths(paths: tuple[Path, ...], style_pack: Path) -> list[str]:
    result: list[str] = []
    for path in paths:
        try:
            result.append(path.resolve().relative_to(style_pack.resolve()).as_posix())
        except ValueError:
            result.append(str(path.resolve()))
    return result

def sheet_record(kind: str, name: str, rendered: RenderedSheet, style_pack: Path) -> dict[str, Any]:
    return {'kind': kind, 'name': name, 'item_count': rendered.item_count, 'page_count': rendered.page_count, 'paths': relative_paths(rendered.paths, style_pack)}

def make_master(style_pack: Path, reference_root: Path, settings: dict[str, Any]) -> dict[str, Any]:
    items = discover_standardized_images(reference_root) + discover_kept_frames(reference_root)
    items.sort(key=lambda item: item.sort_key)
    rendered = render_contact_sheet(items, reference_root / '总览-contact-sheet.jpg', f'《{style_pack.name}》全部正式分析素材', **settings)
    return sheet_record('master', '全部正式分析素材', rendered, style_pack)

def make_video_sheets(style_pack: Path, reference_root: Path, settings: dict[str, Any]) -> tuple[list[dict[str, Any]], list[str]]:
    frame_root = reference_root / '视频抽帧'
    if not frame_root.is_dir():
        return ([], [])
    records: list[dict[str, Any]] = []
    warnings: list[str] = []
    output_root = reference_root / '视频抽帧总览'
    for video_dir in sorted((path for path in frame_root.iterdir() if path.is_dir()), key=lambda p: p.name):
        if not VIDEO_ID_PATTERN.fullmatch(video_dir.name):
            warnings.append(f'跳过不符合视频 ID 规范的目录：{video_dir.name}')
            continue
        items = discover_video_items(video_dir)
        if not items:
            warnings.append(f'视频没有最终保留帧，未生成总览：{video_dir.name}')
            continue
        rendered = render_contact_sheet(items, output_root / f'{video_dir.name}.jpg', f'{video_dir.name} 抽帧总览', **settings)
        records.append(sheet_record('video', video_dir.name, rendered, style_pack))
    return (records, warnings)

def make_group_sheets(style_pack: Path, reference_root: Path, settings: dict[str, Any]) -> tuple[list[dict[str, Any]], list[str]]:
    group_root = reference_root / '分组'
    if not group_root.is_dir():
        return ([], [f'缺少分组目录：{group_root}'])
    records: list[dict[str, Any]] = []
    warnings: list[str] = []
    output_root = reference_root / '分组总览'
    group_dirs = sorted((path for path in group_root.iterdir() if path.is_dir()), key=lambda p: p.name)
    for group_dir in group_dirs:
        if not re.fullmatch('\\d{2}-.+', group_dir.name):
            warnings.append(f'分组目录名称不规范：{group_dir.name}')
        items = discover_group_items(group_dir)
        if not items:
            warnings.append(f'分组没有正式图片，未生成总览：{group_dir.name}')
            continue
        rendered = render_contact_sheet(items, output_root / f'{group_dir.name}.jpg', f'视觉分组 · {group_dir.name}', **settings)
        records.append(sheet_record('group', group_dir.name, rendered, style_pack))
    return (records, warnings)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="生成风格包审核拼图、最终联系表与双版色卡")
    parser.add_argument("style_pack", type=Path, help="风格包根目录")
    parser.add_argument("--mode", choices=("review", "final"), default="final")
    parser.add_argument("--title", help="色卡标题；省略时读取 manifest.project 或目录名")
    parser.add_argument("--recompute-palette", action="store_true", help="忽略已有色卡数据并重新从分组画面取色")
    parser.add_argument("--columns", type=int, default=5)
    parser.add_argument("--rows", type=int, default=6)
    parser.add_argument("--thumb-width", type=int, default=320)
    parser.add_argument("--thumb-height", type=int, default=220)
    parser.add_argument("--gap", type=int, default=12)
    parser.add_argument("--margin", type=int, default=24)
    parser.add_argument("--jpeg-quality", type=int, default=88)
    parser.add_argument("--background", default="#E7E3DB")
    parser.add_argument("--no-labels", action="store_true")
    parser.add_argument("--report", type=Path)
    return parser.parse_args()


def project_title(style_pack: Path, explicit: str | None) -> str:
    if explicit and explicit.strip():
        return explicit.strip()
    manifest = style_pack / "参考素材" / "manifest.json"
    if manifest.is_file():
        try:
            payload = json.loads(manifest.read_text(encoding="utf-8"))
            value = payload.get("project") if isinstance(payload, dict) else None
            if isinstance(value, str) and value.strip():
                return value.strip()
        except (OSError, json.JSONDecodeError):
            pass
    return re.sub(r"-风格包(?:-v\d+)?$", "", style_pack.name) or style_pack.name


def write_contact_report(
    style_pack: Path,
    report_path: Path,
    mode: str,
    settings: dict[str, Any],
    records: list[dict[str, Any]],
    warnings: list[str],
    errors: list[str],
) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": "1.0",
        "created_at": utc_now(),
        "style_pack": style_pack.name,
        "mode": mode,
        "settings": {
            "columns": settings["columns"],
            "rows": settings["rows"],
            "thumb_width": settings["thumb_size"][0],
            "thumb_height": settings["thumb_size"][1],
            "gap": settings["gap"],
            "margin": settings["margin"],
            "jpeg_quality": settings["quality"],
            "labels": settings["labels"],
        },
        "sheets": records,
        "warnings": warnings,
        "errors": errors,
    }
    report_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def candidate_from_color_row(row: dict[str, object]) -> Candidate:
    rgb_value = row.get("rgb")
    if not isinstance(rgb_value, list) or len(rgb_value) != 3:
        raise RuntimeError(f"色卡行缺少有效 RGB：{row}")
    rgb = tuple(int(value) for value in rgb_value)
    if any(value < 0 or value > 255 for value in rgb):
        raise RuntimeError(f"色卡 RGB 越界：{rgb}")
    lab_cv = cv2.cvtColor(np.array([[rgb]], dtype=np.uint8), cv2.COLOR_RGB2LAB)[0, 0]
    lab = cv_lab_to_standard(lab_cv.astype(np.float32))
    return Candidate(
        lab=lab,
        rgb=rgb,
        weight=float(row.get("sample_share") or 0.0),
        recurrence=float(row.get("image_recurrence") or 0.0),
        chroma=float(math.hypot(lab[1], lab[2])),
        lightness=float(lab[0]),
    )


def render_palettes_from_data(style_pack: Path, title: str) -> None:
    data_path = style_pack / "分析" / "色卡数据.json"
    try:
        payload = json.loads(data_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"无法读取已有色卡数据：{exc}") from exc
    groups = payload.get("groups") if isinstance(payload, dict) else None
    if not isinstance(groups, list):
        raise RuntimeError("已有色卡数据缺少 groups 数组")

    by_name = {
        row.get("group"): row
        for row in groups
        if isinstance(row, dict) and isinstance(row.get("group"), str)
    }
    group_root = style_pack / "参考素材" / "分组"
    group_dirs = sorted(path for path in group_root.iterdir() if path.is_dir() and image_files(path))
    disk_names = {path.name for path in group_dirs}
    if disk_names != set(by_name):
        raise RuntimeError(
            "已有色卡数据与当前分组不一致；确认分组变化后使用 --recompute-palette"
        )

    card_paths: list[Path] = []
    pure_card_paths: list[Path] = []
    for group_dir in group_dirs:
        row = by_name[group_dir.name]
        colors = row.get("colors")
        if not isinstance(colors, list):
            raise RuntimeError(f"{group_dir.name} 缺少 colors 数组")
        by_slot = {
            color.get("slot"): color
            for color in colors
            if isinstance(color, dict) and isinstance(color.get("slot"), str)
        }
        expected = [f"P{i}" for i in range(1, 4)] + [f"S{i}" for i in range(1, 6)] + [f"A{i}" for i in range(1, 3)]
        if set(by_slot) != set(expected):
            raise RuntimeError(f"{group_dir.name} 色位必须恰好为 {expected}")
        palette = {
            "primary": [candidate_from_color_row(by_slot[f"P{i}"]) for i in range(1, 4)],
            "secondary": [candidate_from_color_row(by_slot[f"S{i}"]) for i in range(1, 6)],
            "accent": [candidate_from_color_row(by_slot[f"A{i}"]) for i in range(1, 3)],
        }
        display_name = re.sub(r"^\d{2}-", "", group_dir.name)
        card_path = group_dir / f"00-{display_name}-色卡.png"
        pure_path = group_dir / f"00-{display_name}-纯色色卡.png"
        render_card(title, group_dir.name, palette, card_path)
        render_pure_card(palette, pure_path)
        card_paths.append(card_path)
        pure_card_paths.append(pure_path)
        print(f"OK\tpalette-from-json\t{group_dir.name}\t{card_path.name}\t{pure_path.name}")

    reference_root = style_pack / "参考素材"
    render_overview(card_paths, reference_root / "分组色卡总览.jpg")
    render_overview(pure_card_paths, reference_root / "纯色色卡总览.jpg")


def generate_palettes(style_pack: Path, title: str) -> None:
    group_root = style_pack / "参考素材" / "分组"
    if not group_root.is_dir():
        raise RuntimeError(f"缺少正式分组目录：{group_root}")
    group_dirs = sorted(path for path in group_root.iterdir() if path.is_dir() and image_files(path))
    if not group_dirs:
        raise RuntimeError(f"没有找到含正式图片或视频帧的分组：{group_root}")

    analysis_dir = style_pack / "分析"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    groups_payload: list[dict[str, object]] = []
    card_paths: list[Path] = []
    pure_card_paths: list[Path] = []

    for group_dir in group_dirs:
        paths = image_files(group_dir)
        rgb, image_ids, image_sample_counts = sample_group(paths)
        candidates = cluster_candidates(rgb, image_ids, image_sample_counts)
        guidance = color_guidance_for_group(style_pack, group_dir)
        palette = select_palette(candidates, group_dir.name, guidance)
        display_name = re.sub(r"^\d{2}-", "", group_dir.name)
        card_path = group_dir / f"00-{display_name}-色卡.png"
        pure_path = group_dir / f"00-{display_name}-纯色色卡.png"
        render_card(title, group_dir.name, palette, card_path)
        render_pure_card(palette, pure_path)
        card_paths.append(card_path)
        pure_card_paths.append(pure_path)
        groups_payload.append({
            "group": group_dir.name,
            "reference_count": len(paths),
            "color_guidance": guidance,
            "card": portable_path(card_path, style_pack),
            "pure_card": portable_path(pure_path, style_pack),
            "colors": palette_payload(palette),
        })
        print(f"OK\tpalette\t{group_dir.name}\t{card_path.name}\t{pure_path.name}")

    data_path = analysis_dir / "色卡数据.json"
    data_path.write_text(
        json.dumps({
            "project": title,
            "card_size": list(CANVAS_SIZE),
            "color_model": "3 主色 + 5 辅助色 + 2 点缀色",
            "groups": groups_payload,
        }, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    reference_root = style_pack / "参考素材"
    render_overview(card_paths, reference_root / "分组色卡总览.jpg")
    render_overview(pure_card_paths, reference_root / "纯色色卡总览.jpg")
    print(f"DATA\t{data_path}")


def main() -> int:
    args = parse_args()
    style_pack = args.style_pack.expanduser().resolve()
    reference_root = style_pack / "参考素材"
    if not style_pack.is_dir():
        print(f"风格包目录不存在：{style_pack}", file=sys.stderr)
        return 2
    if not reference_root.is_dir():
        print(f"缺少参考素材目录：{reference_root}", file=sys.stderr)
        return 2

    settings: dict[str, Any] = {
        "columns": min(8, max(1, args.columns)),
        "rows": min(20, max(1, args.rows)),
        "thumb_size": (max(120, args.thumb_width), max(90, args.thumb_height)),
        "gap": max(0, args.gap),
        "margin": max(0, args.margin),
        "quality": min(100, max(60, args.jpeg_quality)),
        "background": args.background,
        "labels": not args.no_labels,
        "overwrite": True,
    }
    records: list[dict[str, Any]] = []
    warnings: list[str] = []
    errors: list[str] = []

    try:
        if args.mode == "review":
            video_records, video_warnings = make_video_sheets(style_pack, reference_root, settings)
            records.extend(video_records)
            warnings.extend(video_warnings)
            if not records:
                errors.append("没有可生成审核拼图的视频保留帧")
        else:
            records.append(make_master(style_pack, reference_root, settings))
            video_records, video_warnings = make_video_sheets(style_pack, reference_root, settings)
            group_records, group_warnings = make_group_sheets(style_pack, reference_root, settings)
            records.extend(video_records)
            records.extend(group_records)
            warnings.extend(video_warnings)
            warnings.extend(group_warnings)
    except (FileExistsError, ValueError, OSError) as exc:
        errors.append(str(exc))

    report_path = args.report or style_pack / "分析" / "联系表生成记录.json"
    write_contact_report(style_pack, report_path, args.mode, settings, records, warnings, errors)

    for record in records:
        print(f"OK\t{record['kind']}\t{record['name']}\t{record['item_count']}\t{record['page_count']}")
    for warning in warnings:
        print(f"WARNING\t{warning}", file=sys.stderr)
    for error in errors:
        print(f"ERROR\t{error}", file=sys.stderr)
    print(f"REPORT\t{report_path}")

    if errors:
        return 1
    if args.mode == "final":
        try:
            title = project_title(style_pack, args.title)
            palette_data = style_pack / "分析" / "色卡数据.json"
            if palette_data.is_file() and not args.recompute_palette:
                render_palettes_from_data(style_pack, title)
            else:
                generate_palettes(style_pack, title)
        except Exception as exc:
            print(f"ERROR\t色卡生成失败：{exc}", file=sys.stderr)
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
