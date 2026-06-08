"""视觉表示的数据结构。

这是 DeepVision 区别于扁平描述方案的核心:图像被表示为
一组带归一化坐标的视觉基元(Primitive)和它们之间的关系
(Relation),下游推理可以通过 id 精确引用任意元素。

坐标约定:全部使用归一化坐标 [0,1],原点在左上角。
- point: (x, y)
- bbox:  (x0, y0, x1, y1),x0<=x1, y0<=y1
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Tuple, Dict, Any


@dataclass
class Primitive:
    """一个视觉基元:图中可被精确引用的最小单位。"""

    id: str
    type: str  # "point" | "bbox"
    label: str
    box: Optional[Tuple[float, float, float, float]] = None
    point: Optional[Tuple[float, float]] = None
    text: Optional[str] = None  # OCR / 可读文本
    confidence: Optional[float] = None

    def center(self) -> Tuple[float, float]:
        """返回基元中心点,用于关系推理。"""
        if self.point is not None:
            return self.point
        if self.box is not None:
            x0, y0, x1, y1 = self.box
            return ((x0 + x1) / 2, (y0 + y1) / 2)
        raise ValueError(f"primitive {self.id} 既无 point 也无 box")


@dataclass
class Relation:
    """两个基元之间的空间或逻辑关系。

    rel 取值约定(可扩展):
      空间: above | below | left_of | right_of | contains | inside | overlaps
      逻辑: labels | points_to | part_of | aligned_with
    """

    subj: str  # 主体基元 id
    rel: str
    obj: str  # 客体基元 id
    note: Optional[str] = None


@dataclass
class Scene:
    """一张图的完整结构化表示。"""

    summary: str
    primitives: List[Primitive] = field(default_factory=list)
    relations: List[Relation] = field(default_factory=list)
    width: Optional[int] = None  # 原图像素宽,便于反归一化
    height: Optional[int] = None
    meta: Dict[str, Any] = field(default_factory=dict)

    def by_id(self, pid: str) -> Optional[Primitive]:
        for p in self.primitives:
            if p.id == pid:
                return p
        return None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Scene":
        prims = [Primitive(**p) for p in d.get("primitives", [])]
        rels = [Relation(**r) for r in d.get("relations", [])]
        return cls(
            summary=d.get("summary", ""),
            primitives=prims,
            relations=rels,
            width=d.get("width"),
            height=d.get("height"),
            meta=d.get("meta", {}),
        )

    def to_anchored_text(self) -> str:
        """渲染成"带坐标锚点"的文本,供下游文本模型在推理时引用。

        这是消除 Reference Gap 的关键产物:每个元素都带稳定 id 和
        坐标,模型可以 point-while-reasoning,而不是面对一段模糊散文。
        """
        lines = [f"# 场景总览\n{self.summary}\n", "# 视觉基元(归一化坐标,原点左上)"]
        for p in self.primitives:
            if p.type == "bbox" and p.box:
                x0, y0, x1, y1 = p.box
                loc = f"bbox=({x0:.3f},{y0:.3f},{x1:.3f},{y1:.3f})"
            elif p.point:
                loc = f"point=({p.point[0]:.3f},{p.point[1]:.3f})"
            else:
                loc = "无坐标"
            txt = f' text="{p.text}"' if p.text else ""
            lines.append(f"- [{p.id}] {p.label} {loc}{txt}")
        if self.relations:
            lines.append("\n# 关系")
            for r in self.relations:
                note = f" ({r.note})" if r.note else ""
                lines.append(f"- {r.subj} --{r.rel}--> {r.obj}{note}")
        return "\n".join(lines)
