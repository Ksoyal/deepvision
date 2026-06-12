"""MCP server:把 DeepVision 暴露为工具,供 Claude Desktop / Cherry Studio 等调用。

依赖 `mcp` 包(pip install "deepvision[mcp]")。提供一个工具:
- describe_image_structured: 返回带坐标锚点的结构化文本
- describe_clipboard_image: 读取剪贴板图片并返回结构化文本
"""

from __future__ import annotations

try:
    from mcp.server.fastmcp import FastMCP
except ImportError:  # pragma: no cover
    raise SystemExit("需要 mcp 包:pip install \"deepvision[mcp]\"")

from .config import Config
from .vision import describe_structured
from .refine import auto_refine
from .cli import _grab_clipboard

mcp = FastMCP("deepvision")


@mcp.tool()
def describe_image_structured(path: str,
                              detail: str = "standard",
                              intent: str = "general",
                              target: str = "",
                              refine: bool = False) -> str:
    """把本地图片解析成带坐标锚点的结构化视觉表示,供精确空间推理。

    Args:
        path: 图片文件路径。
        detail: 输出粒度:brief、standard 或 fine。
        intent: 任务意图:general、count、ocr、locate 或 inspect。
        target: 任务关注的短目标词,如 手指、公式、表格;不要传完整问题。
        refine: 是否对低置信区域做局部放大重识别。
    """
    cfg = Config.load()
    scene = describe_structured(path, cfg, detail=detail, intent=intent, target=target)
    if refine:
        scene = auto_refine(path, scene, cfg=cfg)
    return scene.to_anchored_text()


@mcp.tool()
def describe_clipboard_image(detail: str = "standard",
                             intent: str = "general",
                             target: str = "",
                             refine: bool = False) -> str:
    """读取剪贴板图片并解析成带坐标锚点的结构化视觉表示。

    Args:
        detail: 输出粒度:brief、standard 或 fine。
        intent: 任务意图:general、count、ocr、locate 或 inspect。
        target: 任务关注的短目标词,如 手指、公式、表格;不要传完整问题。
        refine: 是否对低置信区域做局部放大重识别。
    """
    try:
        items = _grab_clipboard()
    except Exception as e:  # noqa: BLE001
        return f"错误[剪贴板]: {e}"

    cfg = Config.load()
    outputs = []
    for src, image in items:
        scene = describe_structured(image, cfg, detail=detail, intent=intent, target=target)
        if refine:
            scene = auto_refine(image, scene, cfg=cfg)
        if len(items) > 1:
            outputs.append(f"# === {src} ===\n{scene.to_anchored_text()}")
        else:
            outputs.append(scene.to_anchored_text())
    return "\n\n".join(outputs)


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
