"""核心视觉解析:图像 -> 结构化 Scene。

提供两个层次的接口:
- describe():           返回扁平文本描述(兼容 OpenVL 的简单用法)
- describe_structured(): 返回带坐标锚点的 Scene(本项目的核心价值)
"""

from __future__ import annotations

import base64
import io
import json
import re
import time
from pathlib import Path
from typing import Optional, Union

import requests
from PIL import Image

from .config import Config
from .schema import Scene


_PROMPT_DIR = Path(__file__).parent / "prompts"


def _normalize_coords(scene: Scene) -> Scene:
    """把模型可能返回的非 [0,1] 坐标统一压回 [0,1]。

    模型常无视归一化约定,返回三种标度之一:
      - [0,1]:        已正确,不动
      - 0~1000:       Qwen-VL / Gemini 系常见,除以 1000
      - 真实像素:      除以图片宽高
    用"出现过的最大坐标值"来判定标度,避免逐元素误判。
    """
    vals = []
    for p in scene.primitives:
        if p.box:
            vals.extend(p.box)
        if p.point:
            vals.extend(p.point)
    if not vals:
        return scene
    mx = max(vals)
    if mx <= 1.5:
        return scene  # 已是归一化坐标

    if mx <= 1000:
        sx = sy = 1000.0  # 0~1000 标度
    else:
        sx = float(scene.width or mx)   # 像素标度
        sy = float(scene.height or mx)

    def _clamp(v):
        return min(1.0, max(0.0, v))

    for p in scene.primitives:
        if p.box:
            x0, y0, x1, y1 = p.box
            p.box = (_clamp(x0 / sx), _clamp(y0 / sy),
                     _clamp(x1 / sx), _clamp(y1 / sy))
        if p.point:
            x, y = p.point
            p.point = (_clamp(x / sx), _clamp(y / sy))
    scene.meta["coord_rescaled"] = {"detected_max": mx, "scale_x": sx, "scale_y": sy}
    return scene



def _load_prompt(name: str) -> str:
    return (_PROMPT_DIR / name).read_text(encoding="utf-8")


def _to_data_uri(image: Union[str, Path, bytes], max_edge: int) -> tuple[str, int, int]:
    """把各种输入统一成缩放后的 PNG data URI,返回 (uri, width, height)。"""
    if isinstance(image, (str, Path)) and Path(image).is_file():
        img = Image.open(image)
    elif isinstance(image, bytes):
        img = Image.open(io.BytesIO(image))
    elif isinstance(image, str) and image.startswith("data:"):
        header, b64 = image.split(",", 1)
        img = Image.open(io.BytesIO(base64.b64decode(b64)))
    else:
        raise ValueError("不支持的图像输入:需要文件路径、bytes 或 data URI")

    img = img.convert("RGB")
    w, h = img.size
    scale = min(1.0, max_edge / max(w, h))
    if scale < 1.0:
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    uri = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
    return uri, img.width, img.height


def _call_api(cfg: Config, prompt: str, data_uri: str, question: str = "") -> str:
    """调用 OpenAI 兼容的多模态 chat 接口。

    内容排布:系统提示在前、用户问题居中、图片在后,有利于 API
    前缀缓存命中(同一张图配不同问题时复用前缀)。
    """
    cfg.require_key()
    user_content = []
    if question:
        user_content.append({"type": "text", "text": question})
    user_content.append({"type": "image_url", "image_url": {"url": data_uri}})

    payload = {
        "model": cfg.model,
        "temperature": cfg.temperature,
        "messages": [
            {"role": "system", "content": prompt},
            {"role": "user", "content": user_content},
        ],
    }
    url = f"{cfg.base_url.rstrip('/')}/chat/completions"
    headers = {
        "Authorization": f"Bearer {cfg.api_key}",
        "Content-Type": "application/json",
    }
    # 免费档常返回 429,做指数退避重试(尊重 Retry-After 头)
    last_exc = None
    for attempt in range(cfg.max_retries + 1):
        resp = requests.post(url, headers=headers, json=payload, timeout=cfg.timeout)
        if resp.status_code == 429 and attempt < cfg.max_retries:
            ra = resp.headers.get("Retry-After")
            wait = float(ra) if ra and ra.replace(".", "", 1).isdigit() else 2 ** attempt
            time.sleep(min(wait, 30))
            last_exc = requests.HTTPError(f"429 (第 {attempt + 1} 次,退避 {wait}s)")
            continue
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]
    raise last_exc or RuntimeError("API 调用失败")


def _extract_json(text: str) -> dict:
    """从模型输出里稳健地抠出 JSON(容忍 ```json 包裹或前后赘述)。"""
    text = text.strip()
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fence:
        text = fence.group(1)
    else:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1:
            text = text[start : end + 1]
    return json.loads(text)


def describe(image: Union[str, Path, bytes], question: str = "",
             cfg: Optional[Config] = None) -> str:
    """扁平文本描述(兼容 OpenVL 式简单用法)。"""
    cfg = cfg or Config.load()
    data_uri, _, _ = _to_data_uri(image, cfg.max_edge)
    prompt = _load_prompt("describe.md")
    return _call_api(cfg, prompt, data_uri, question)


def describe_structured(image: Union[str, Path, bytes], question: str = "",
                        cfg: Optional[Config] = None) -> Scene:
    """核心接口:返回带坐标锚点的结构化 Scene。"""
    cfg = cfg or Config.load()
    data_uri, w, h = _to_data_uri(image, cfg.max_edge)
    prompt = _load_prompt("structured.md")
    raw = _call_api(cfg, prompt, data_uri, question)
    data = _extract_json(raw)
    scene = Scene.from_dict(data)
    scene.width, scene.height = w, h
    return _normalize_coords(scene)
