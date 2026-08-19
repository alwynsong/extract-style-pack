# extract-style-pack

一个用于从图片、本地视频或公开网络视频中提取可复用视觉风格包的通用 Agent Skill。

它把参考素材整理为可追溯的正式画面、视觉分组、联系表、双版色卡、风格规则和生成提示词。脚本处理确定性媒体任务，Agent 负责来源判断、逐图观察、分组、归纳和最终视觉验收。

> 适用于能够读取 `SKILL.md`、执行 Python 脚本并调用本地命令的 Agent；具体安装目录、权限与工具调用方式以所用 Agent 环境为准。

## 主要能力

- 接收用户图片、本地视频和公开网络视频
- 使用场景变化与均匀采样结合的视频抽帧
- 图片方向校正、媒体检测、质量筛选和感知去重
- 基于逐图证据的视觉分组，而不是按剧情幕或时间段分组
- 生成全部素材、逐视频和逐分组联系表
- 为每组生成 10 色标注色卡与无文字纯色色卡
- 生成可迁移的风格规则与图片/视频提示词
- 独立校验目录、来源、逐图标签、色卡、文档颜色和隐私信息

## 快速使用

安装 Skill 并完成下方环境配置后，直接向 Agent 描述目标即可，通常不需要手动运行脚本。

按作品名称提取：

```text
提取诺兰《奥德赛》的视觉风格，优先使用官方公开视频抽帧。
```

从上传视频提取：

```text
提取我上传视频的视觉风格，选择代表性画面，并生成视觉分组、色卡、风格规则和提示词。
```

还可以补充分析重点：

```text
重点分析摄影、光线、色彩和镜头质感，忽略剧情与角色身份。
```

具备本地命令与安装权限的 Agent 会检查视频工具、准备素材、抽取代表帧并完成后续分析；权限不足时应报告缺失依赖和恢复方式。

## 目录

```text
extract-style-pack/
├── SKILL.md
├── README.md
├── requirements.txt
├── assets/
├── references/
│   ├── 输入与输出契约.md
│   ├── 网络视频与抽帧.md
│   ├── 视觉分析与分组.md
│   └── 色卡与验收.md
├── scripts/
│   ├── prepare_media.py
│   ├── extract_video_frames.py
│   ├── render_style_pack.py
│   └── validate_style_pack.py
└── evals/
```

`SKILL.md` 是 Agent 的执行入口；`references/` 按阶段提供详细契约；`README.md` 只面向使用者，不是 Agent 的必读文件。

## 环境要求

- Python 3.10+
- yt-dlp（随 Python 依赖安装）
- FFmpeg 与 FFprobe（必须是可执行程序，不是同名 Python 包）
- Python 依赖：

```bash
python -m pip install --upgrade -r requirements.txt
```

根据操作系统安装 FFmpeg：

```text
Windows:       winget install --id Gyan.FFmpeg --exact
macOS:         brew install ffmpeg
Ubuntu/Debian: sudo apt update && sudo apt install ffmpeg
```

安装后进行强制预检：

```bash
python -X utf8 scripts/extract_video_frames.py --check
```

只有输出 `STATUS\tready` 才表示完整网络视频流程就绪。脚本会优先查找 PATH，也会识别 `python -m yt_dlp`、macOS 常用目录和 Windows 常见 Winget、Chocolatey、Scoop 安装目录。

处理 YouTube、哔哩哔哩等公开网络视频时，yt-dlp、FFmpeg 与 FFprobe 都视为核心依赖。下载能力仍由成熟的 yt-dlp 承担，Skill 不维护站点解析规则；抽帧脚本只接收本地视频，因此可被其他 Skill 复用。若 YouTube 提示缺少 JavaScript runtime 或 EJS 支持，请按 yt-dlp 官方依赖说明安装 Deno/Node 等兼容运行时。

具备安装权限的 Agent 应在缺少依赖时主动定位或发起安装；网络访问、系统级写入或管理员权限可能需要用户授权。不具备相应权限时，应给出明确的手动安装方式。脚本不会静默修改系统环境。

## 使用方式

把本目录安装或复制到所用 Agent 可读取的 Skill 目录后，用自然语言触发，例如：

```text
提取《某部电影》的视觉风格，优先使用公开视频抽帧。
```

```text
根据这些参考图生成一个可复用风格包，包括分组、色卡和提示词。
```

Agent 会按 `SKILL.md` 路由到所需 Reference 和脚本。通常不需要手动调用脚本；调试或单独复用时可使用下面的入口。

### 1. 准备媒体

将原始文件放入风格包的：

```text
参考素材/原始文件/图片/
参考素材/原始文件/视频/
```

然后运行：

```bash
python -X utf8 scripts/prepare_media.py "<风格包目录>"
```

该命令检测图片和视频，并重建脚本管理的 `标准化图片/`、`被过滤图片/` 与两份分析报告。

### 2. 视频抽帧

先检查完整媒体环境：

```bash
python -X utf8 scripts/extract_video_frames.py --check
```

网络视频通过合法方式下载为本地文件后再抽帧：

```bash
python -X utf8 scripts/extract_video_frames.py \
  "<本地视频>" \
  --asset-id "VID-0001" \
  --output-dir "<风格包目录>/参考素材/视频抽帧/VID-0001"
```

抽帧脚本依赖 FFmpeg/FFprobe，并输出候选帧、保留帧、过滤帧与 `frames.json`。

### 3. 生成审核拼图

```bash
python -X utf8 scripts/render_style_pack.py "<风格包目录>" --mode review
```

Agent 查看逐视频拼图并人工剔除坏帧。自动质量指标不替代视觉判断。

### 4. 生成最终视觉资产

逐图标注和视觉分组完成后运行：

```bash
python -X utf8 scripts/render_style_pack.py \
  "<风格包目录>" \
  --mode final \
  --title "<项目标题>"
```

该命令生成全部正式联系表、分组联系表、每组双版色卡、两套色卡总览以及对应 JSON 记录。已有 `色卡数据.json` 会作为人工调整后的事实源保留并重渲染；分组或素材改变时使用 `--recompute-palette` 重新取色。

### 5. 验收

```bash
python -X utf8 scripts/validate_style_pack.py \
  "<风格包目录>" \
  --visual-review-confirmed
```

只有在 Agent 确实看过全部素材、分组和色卡后，才使用 `--visual-review-confirmed`。退出码非 0 表示仍需修复。

## 分组原则

视觉分组不是电影的“幕”或剧情段落。它按空间、色彩、光线、构图、镜头、材质和氛围等可迁移特征归类。因此，不同时间点的场景可以进入同一视觉组；同一时间段也可能包含多个视觉组。

## 网络来源边界

本 Skill 只处理公开且允许访问的来源，不绕过 DRM、登录、付费墙、验证码、地区限制或其他访问控制，也不把 Cookie、令牌或临时签名 URL 写入交付物。来源不可合法取得时，应改用其他公开来源或请用户提供本地文件。

## Reference 导航

- `输入与输出契约.md`：来源、目录、命名、manifest、路径和隐私
- `网络视频与抽帧.md`：网络视频获取边界、FFmpeg 抽帧和人工复核
- `视觉分析与分组.md`：逐图标签、视觉分组、规则和提示词
- `色卡与验收.md`：联系表、双版色卡、颜色同步和验收

## 当前限制

- 网络视频站点规则变化频繁，下载由 yt-dlp 承担；出现解析器错误时应先升级 yt-dlp，最多重试一次。
- YouTube 的完整支持可能额外依赖 yt-dlp-ejs 与 JavaScript runtime。
- 自动筛选不能可靠识别所有字幕、转场和叙事重复，必须人工审核。
- 聚类色卡是审美判断的起点，最终颜色仍需结合画面证据复核。
- 风格提取应迁移视觉机制，不应复制受保护角色、演员肖像、Logo 或原作专属元素。

## 许可证

本项目采用 [MIT License](LICENSE)。
