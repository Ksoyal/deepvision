"""MCP server:把 DeepVision 暴露为工具,供 Claude Desktop / Cherry Studio 等调用。

依赖 `mcp` 包(pip install "deepvision[mcp]")。提供一个工具:
- describe_image_structured: 返回带坐标锚点的结构化文本
"""

from __future__ import annotations

try:
    from mcp.server.fastmcp import FastMCP
except ImportError:  # pragma: no cover
    raise SystemExit("需要 mcp 包:pip install \"deepvision[mcp]\"")

from .config import Config
from .vision import describe_structured
from .refine import auto_refine

mcp = FastMCP("deepvision")


@mcp.tool()
def describe_image_structured(path: str, refine: bool = False) -> str:
    """把本地图片解析成带坐标锚点的结构化视觉表示,供精确空间推理。

    Args:
        path: 图片文件路径。
        refine: 是否对低置信区域做局部放大重识别。
    """
    cfg = Config.load()
    scene = describe_structured(path, cfg)
    if refine:
        scene = auto_refine(path, scene, cfg=cfg)
    return scene.to_anchored_text()


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
