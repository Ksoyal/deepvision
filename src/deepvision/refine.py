"""局部细化:对低置信度区域做 crop + 放大 + 重新识别。

这是对论文 "point-while-reasoning" 思想的工程近似:当下游推理对
某个区域把握不足时,回头把那块 bbox 裁出来放大,再问一次多模态
API,相当于"用手指点过去看清楚"。不依赖未开源的模型权重,用通用
多模态 API 即可获得论文的大部分收益。
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Optional, Union

from PIL import Image

from .config import Config
from .schema import Scene, Primitive
from .vision import _to_data_uri, _call_api, _extract_json, _load_prompt, _load_image


def _crop_bbox(img: Image.Image, box, pad: float = 0.05) -> Image.Image:
    """按归一化 bbox(带 padding)裁剪原图。"""
    w, h = img.size
    x0, y0, x1, y1 = box
    x0 = max(0, (x0 - pad)) * w
    y0 = max(0, (y0 - pad)) * h
    x1 = min(1, (x1 + pad)) * w
    y1 = min(1, (y1 + pad)) * h
    return img.crop((int(x0), int(y0), int(x1), int(y1)))


def refine_region(
    image: Union[str, Path, bytes],
    primitive: Primitive,
    question: str = "",
    cfg: Optional[Config] = None,
) -> Scene:
    """放大单个基元区域重新解析,返回该局部的细化 Scene。

    局部 Scene 的坐标是相对裁剪区域的;调用方如需可映射回全局。
    """
    if primitive.type != "bbox" or not primitive.box:
        raise ValueError("只能细化带 bbox 的基元")
    cfg = cfg or Config.load()

    try:
        img = _load_image(image)
    except ValueError as e:
        raise ValueError("refine 需要文件路径、URL、bytes 或 data URI 输入") from e

    crop = _crop_bbox(img, primitive.box)
    buf = io.BytesIO()
    crop.save(buf, format="PNG")
    data_uri, w, h = _to_data_uri(buf.getvalue(), cfg.max_edge)

    prompt = _load_prompt("structured.md")
    q = question or f"放大查看「{primitive.label}」区域,精确解析其中的细节。"
    raw = _call_api(cfg, prompt, data_uri, q)
    scene = Scene.from_dict(_extract_json(raw))
    scene.width, scene.height = w, h
    scene.meta["refined_from"] = primitive.id
    return scene


def auto_refine(
    image: Union[str, Path, bytes],
    scene: Scene,
    threshold: float = 0.5,
    cfg: Optional[Config] = None,
) -> Scene:
    """对场景里所有低置信度 bbox 基元自动做一轮局部细化。

    把细化结果作为子场景挂到 meta['refinements'],保留原始结构。
    """
    cfg = cfg or Config.load()
    refinements = {}
    for p in scene.primitives:
        if p.type == "bbox" and p.box and (p.confidence or 1.0) < threshold:
            try:
                sub = refine_region(image, p, cfg=cfg)
                refinements[p.id] = sub.to_dict()
            except Exception as e:  # noqa: BLE001 — 单点失败不应中断整体
                refinements[p.id] = {"error": str(e)}
    scene.meta["refinements"] = refinements
    return scene
