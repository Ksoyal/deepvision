---
name: deepvision
description: 当当前模型无法直接读图、图片未进入上下文、用户询问剪贴板/截图内容，或任务需要 OCR、坐标、稳定元素 ID、空间关系、计数、定位、复核、公式、表格、文档、UI、图表或结构化视觉抽取时使用 DeepVision。纯剪贴板场景直接用 `deepvision -c`，不要自行保存临时文件；精确任务添加 `--intent count|ocr|locate|inspect`，能提取关注对象时加短 `--target`。同一图片已有有效 DeepVision 结果时优先复用。
---

# DeepVision

使用 DeepVision 将图片转换为锚点化视觉结构：先读语义单元，再按需引用带坐标的视觉基元。

## 调用模式

1. 选择图片来源：
   - 明确路径、URL 或 data URI：`deepvision <input>`。
   - 只提到剪贴板/截图且没有明确路径：`deepvision -c`。
   - 需要程序化输出：添加 `--json`。
2. 添加任务参数：

| 用户需求 | 参数 |
|---|---|
| 快速概览 | `--detail brief` |
| 数量、多少、比较数量 | `--intent count` |
| 读文字、OCR、公式、表格、代码 | `--intent ocr` |
| 位置、方向、第几个、坐标 | `--intent locate` |
| 复核、找异常、找差异 | `--intent inspect` |
| 有明确关注对象 | `--target <短目标词>` |
| 需要局部放大或低置信细化 | `--refine` |

`--target` 只放短目标词，不放完整问题。示例：`--intent count --target 手指`、`--intent ocr --target 公式`。

## 剪贴板规则

- 纯剪贴板/截图请求先运行 `deepvision -c`。
- 精确剪贴板任务在目标清楚时用 `deepvision -c --intent count|ocr|locate|inspect --target <短目标词>`。
- 不要用 PowerShell、Python 或其他工具把剪贴板图片保存成临时文件。
- 不要在 `deepvision -c` 前自行探测或转换剪贴板内容。
- 环境已提供明确图片路径时，使用该路径，不走剪贴板规则。
- `deepvision -c` 报告剪贴板无图时，请用户重新复制图片或提供有效路径。

## 结果处理

- 同一图片已有有效结果时优先复用；除非用户要求重析、图片变化或结果不足。
- 基于结果完成用户原始任务，不要只倾倒工具输出。
- 优先读 `# 语义单元` 理解整体；需要精确位置或细节时再读 `# 视觉基元` 和关系。
- `count` 以候选实例为主证据；`visual_markers` 只是可选彩色标记线索，只在对应目标时采用其数量。

## 成本与可靠性

- 默认使用缓存；仅在用户要求重析、图片变化但疑似旧缓存、当前结果不足或调试 prompt/模型时加 `--no-cache`。
- `--refine` 是局部细化，不是抗幻觉核查；不要默认开启。
- 自动点击、关键数据提取等高风险操作使用 `--verify`。
- 只有需要逐个核查所有基元时才用 `--verify full`；该模式慢且请求多。

## MCP 对应

- 明确路径：`describe_image_structured(path, detail, intent, target, refine)`。
- 剪贴板：`describe_clipboard_image(detail, intent, target, refine)`。
- MCP 默认值：`detail="standard"`、`intent="general"`、`target=""`、`refine=false`。
- MCP 不暴露 `verify` 或 `no-cache`；需要这些能力时使用 CLI。

## 错误

- 命令不存在：提示安装 DeepVision 并确认 `deepvision` 在 `PATH` 中。
- 配置缺失：提示配置 `DEEPVISION_API_KEY`、`DEEPVISION_BASE_URL`、`DEEPVISION_MODEL`，或运行 `deepvision init`。
- 路径无效：报告该输入无效，并请用户提供有效路径。
