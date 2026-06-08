"""MCP server:把 DeepVision 暴露为工具,供 Claude Desktop / Cherry Studio 等调用。

依赖 `mcp` 包(pip install "deepvision[mcp]")。提供两个工具:
- describe_image_structured: 返回带坐标锚点的结构化文本
- describe_image_flat:       返回扁平描述
"""

from __future__ import annotations

import base64

try:
    from mcp.server.fastmcp import FastMCP
except ImportError:  # pragma: no cover
    raise SystemExit("需要 mcp 包:pip install \"deepvision[mcp]\"")

from .config import Config
from .vision import describe, describe_structured
from .refine import auto_refine

mcp = FastMCP("deepvision")


@mcp.tool()
def describe_image_structured(path: str, question: str = "",
                              refine: bool = False) -> str:
    """把本地图片解析成带坐标锚点的结构化视觉表示,供精确空间推理。

    Args:
        path: 图片文件路径。
        question: 可选,聚焦的问题。
        refine: 是否对低置信区域做局部放大重识别。
    """
    cfg = Config.load()
    scene = describe_structured(path, question, cfg)
    if refine:
        scene = auto_refine(path, scene, cfg=cfg)
    return scene.to_anchored_text()


@mcp.tool()
def describe_image_flat(path: str, question: str = "") -> str:
    """把本地图片转成一段扁平文字描述(简单场景用)。"""
    return describe(path, question, Config.load())


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
