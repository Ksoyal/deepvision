"""视觉表示的数据结构。

这是 DeepVision 区别于扁平描述方案的核心:图像被表示为
一组聚合语义单元(Composite)、带归一化坐标的视觉基元(Primitive)
和它们之间的关系(Relation),下游推理可以通过 id 精确引用任意元素。

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
    role: Optional[str] = None  # 通用语义角色,如 problem_number / button / cell
    parent: Optional[str] = None  # 所属 Composite id,用于限制跨块关系噪声

    def center(self) -> Tuple[float, float]:
        """返回基元中心点,用于关系推理。"""
        if self.point is not None:
            return self.point
        if self.box is not None:
            x0, y0, x1, y1 = self.box
            return ((x0 + x1) / 2, (y0 + y1) / 2)
        raise ValueError(f"primitive {self.id} 既无 point 也无 box")


@dataclass
class Composite:
    """由多个视觉基元组成的语义单元,如题目、表格、表单、流程节点组。"""

    id: str
    type: str = "composite"
    label: str = ""
    role: Optional[str] = None
    box: Optional[Tuple[float, float, float, float]] = None
    text: Optional[str] = None
    children: List[str] = field(default_factory=list)
    parent: Optional[str] = None
    confidence: Optional[float] = None

    def center(self) -> Tuple[float, float]:
        if self.box is not None:
            x0, y0, x1, y1 = self.box
            return ((x0 + x1) / 2, (y0 + y1) / 2)
        raise ValueError(f"composite {self.id} 无 box")


@dataclass
class Relation:
    """两个基元之间的关系。

    模型只产出**语义关系**(无法从坐标推导的那部分):
      labels(A 是 B 的标签) / points_to(有向连接) / part_of(语义从属)

    几何关系(above/below/left_of/right_of/contains/inside/aligned_with)
    不进模型输出,而是由 derive_geometric_relations() 按坐标确定性推导——
    坐标算出来的方向永远正确,也省掉模型在几何上编造方向的幻觉。
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
    composites: List[Composite] = field(default_factory=list)
    width: Optional[int] = None  # 原图像素宽,便于反归一化
    height: Optional[int] = None
    meta: Dict[str, Any] = field(default_factory=dict)

    def by_id(self, pid: str) -> Optional[Any]:
        for p in self.primitives:
            if p.id == pid:
                return p
        for c in self.composites:
            if c.id == pid:
                return c
        return None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Scene":
        prims = [Primitive(**_clean_dataclass_input(p, Primitive))
                 for p in d.get("primitives", [])]
        comps = [Composite(**_clean_dataclass_input(c, Composite))
                 for c in d.get("composites", [])]
        rels = [Relation(**_clean_dataclass_input(r, Relation))
                for r in d.get("relations", [])]
        return cls(
            summary=d.get("summary", ""),
            primitives=prims,
            relations=rels,
            composites=comps,
            width=d.get("width"),
            height=d.get("height"),
            meta=d.get("meta", {}),
        )

    def to_anchored_text(self) -> str:
        """渲染成"带坐标锚点"的文本,供下游文本模型在推理时引用。

        这是消除 Reference Gap 的关键产物:每个元素都带稳定 id 和
        坐标,模型可以 point-while-reasoning,而不是面对一段模糊散文。
        """
        lines = [f"# 场景总览\n{self.summary}\n"]
        if self.composites:
            lines.append("# 语义单元(聚合结构,优先供下游理解)")
            for c in self.composites:
                loc = _loc_of(c)
                role = f" role={c.role}" if c.role else ""
                txt = f' text="{c.text}"' if c.text else ""
                children = f" children=[{', '.join(c.children)}]" if c.children else ""
                parent = f" parent={c.parent}" if c.parent else ""
                lines.append(f"- [{c.id}] {c.label}{role} {loc}{txt}{children}{parent}")
            lines.append("")

        lines.append("# 视觉基元(归一化坐标,原点左上)")
        for p in self.primitives:
            loc = _loc_of(p)
            role = f" role={p.role}" if p.role else ""
            parent = f" parent={p.parent}" if p.parent else ""
            txt = f' text="{p.text}"' if p.text else ""
            lines.append(f"- [{p.id}] {p.label}{role} {loc}{parent}{txt}")
        if self.relations:
            lines.append("\n# 关系")
            for r in self.relations:
                note = f" ({r.note})" if r.note else ""
                lines.append(f"- {r.subj} --{r.rel}--> {r.obj}{note}")
        return "\n".join(lines)


def _clean_dataclass_input(data: Dict[str, Any], cls) -> Dict[str, Any]:
    """按 dataclass 字段过滤模型输出,兼容 bbox 别名和额外字段。"""
    item = dict(data)
    if "bbox" in item and "box" not in item:
        item["box"] = item["bbox"]
    if cls is Composite and item.get("children") is None:
        item["children"] = []
    known = set(cls.__dataclass_fields__)
    return {k: v for k, v in item.items() if k in known}


def _loc_of(item: Any) -> str:
    if getattr(item, "box", None):
        x0, y0, x1, y1 = item.box
        return f"bbox=({x0:.3f},{y0:.3f},{x1:.3f},{y1:.3f})"
    if getattr(item, "point", None):
        x, y = item.point
        return f"point=({x:.3f},{y:.3f})"
    return "无坐标"


# ---------------------------------------------------------------------------
# 几何关系推导:从坐标确定性地算出空间关系,不依赖模型。
# 模型只输出语义关系(labels/points_to/part_of),几何方向由这里补齐。
# ---------------------------------------------------------------------------

def _rect_of(p: Primitive) -> Optional[Tuple[float, float, float, float]]:
    """取基元的矩形;point 退化成零面积矩形,便于统一比较。"""
    if p.box is not None:
        return p.box
    if p.point is not None:
        x, y = p.point
        return (x, y, x, y)
    return None


def _overlap_1d(a0: float, a1: float, b0: float, b1: float) -> float:
    """两个区间的重叠长度(<=0 表示不相交)。"""
    return min(a1, b1) - max(a0, b0)


def _contains(outer: Tuple[float, float, float, float],
              inner: Tuple[float, float, float, float],
              eps: float = 1e-6) -> bool:
    """outer 是否(非退化地)包含 inner。"""
    ox0, oy0, ox1, oy1 = outer
    ix0, iy0, ix1, iy1 = inner
    if (ox1 - ox0) <= eps or (oy1 - oy0) <= eps:
        return False  # outer 无面积,不算容器
    return (ox0 <= ix0 + eps and oy0 <= iy0 + eps and
            ox1 + eps >= ix1 and oy1 + eps >= iy1 and
            (ox1 - ox0) * (oy1 - oy0) > (ix1 - ix0) * (iy1 - iy0) + eps)


def _same_parent(a: Primitive, b: Primitive) -> bool:
    return a.parent == b.parent


def _is_formula_part(p: Primitive) -> bool:
    return p.role == "formula_part"


def _is_horizontal_rule(p: Primitive) -> bool:
    r = _rect_of(p)
    if r is None:
        return False
    x0, y0, x1, y1 = r
    w, h = x1 - x0, y1 - y0
    return w > 0 and h > 0 and h <= 0.02 and (w / h) >= 8


def _keep_direction_relation(subj: Primitive, rel: str, obj: Primitive) -> bool:
    if not (_is_formula_part(subj) and _is_formula_part(obj)):
        return True
    if rel == "left_of":
        return False
    if rel == "above":
        return _is_horizontal_rule(subj) or _is_horizontal_rule(obj)
    return True


def derive_geometric_relations(primitives: List[Primitive],
                               align_tol: float = 0.02) -> List[Relation]:
    """从坐标推导一组**有界**的几何关系,并限制在同一父容器内。

    产出两类:
      - contains:每个基元只连到它**最小的直接容器**(包含森林,非传递闭包)。
      - left_of / above:每个基元在水平/垂直方向上,只连到投影重叠的**最近邻**。
        right_of / below 是它们的逆,不重复产出。

    这样可以避免不同题目、表格、面板之间产生低价值的跨块关系。
    """
    rects: List[Tuple[Primitive, Tuple[float, float, float, float]]] = []
    for p in primitives:
        r = _rect_of(p)
        if r is not None:
            rects.append((p, r))

    relations: List[Relation] = []

    # 包含:为每个 inner 找最小的直接容器
    for pi, ri in rects:
        best_parent: Optional[Primitive] = None
        best_area = float("inf")
        for pj, rj in rects:
            if pi is pj:
                continue
            if not _same_parent(pi, pj):
                continue
            if _contains(rj, ri):
                area = (rj[2] - rj[0]) * (rj[3] - rj[1])
                if area < best_area:
                    best_area = area
                    best_parent = pj
        if best_parent is not None:
            relations.append(Relation(subj=best_parent.id, rel="contains", obj=pi.id))

    contained = {r.obj for r in relations if r.rel == "contains"}

    # 方向:水平方向最近邻(left_of),只在垂直投影有重叠时成立
    for pi, ri in rects:
        if pi.id in contained:
            continue
        best_right: Optional[Primitive] = None
        best_gap = float("inf")
        for pj, rj in rects:
            if pi is pj or pj.id in contained:
                continue
            if not _same_parent(pi, pj):
                continue
            if _overlap_1d(ri[1], ri[3], rj[1], rj[3]) <= 0:
                continue  # 垂直不重叠,不算同一行
            gap = rj[0] - ri[2]  # pj 在 pi 右侧的水平间隙
            if gap >= -align_tol and gap < best_gap:
                best_gap = gap
                best_right = pj
        if best_right is not None:
            if _keep_direction_relation(pi, "left_of", best_right):
                relations.append(Relation(subj=pi.id, rel="left_of", obj=best_right.id))

    # 方向:垂直方向最近邻(above),只在水平投影有重叠时成立
    for pi, ri in rects:
        if pi.id in contained:
            continue
        best_below: Optional[Primitive] = None
        best_gap = float("inf")
        for pj, rj in rects:
            if pi is pj or pj.id in contained:
                continue
            if not _same_parent(pi, pj):
                continue
            if _overlap_1d(ri[0], ri[2], rj[0], rj[2]) <= 0:
                continue  # 水平不重叠,不算同一列
            gap = rj[1] - ri[3]  # pj 在 pi 下方的垂直间隙
            if gap >= -align_tol and gap < best_gap:
                best_gap = gap
                best_below = pj
        if best_below is not None:
            if _keep_direction_relation(pi, "above", best_below):
                relations.append(Relation(subj=pi.id, rel="above", obj=best_below.id))

    return relations


def sort_reading_order(primitives: List[Primitive],
                       row_tol: float = 0.03) -> List[Primitive]:
    """按阅读顺序(上→下、行内左→右)排序基元。

    相关元素在序列里聚簇,缓解下游模型的 Lost-in-the-Middle。
    用"行带"分组:中心 y 相近(差值 <= row_tol)的归为同一行,再按 x 排。
    无坐标的基元保持稳定,沉到末尾。
    """
    def center(p: Primitive) -> Optional[Tuple[float, float]]:
        try:
            return p.center()
        except ValueError:
            return None

    located = [p for p in primitives if center(p) is not None]
    unlocated = [p for p in primitives if center(p) is None]
    located.sort(key=lambda p: center(p)[1])  # 先按 y 粗排

    rows: List[List[Primitive]] = []
    for p in located:
        cy = center(p)[1]
        if rows and cy - center(rows[-1][0])[1] <= row_tol:
            rows[-1].append(p)
        else:
            rows.append([p])

    ordered: List[Primitive] = []
    for row in rows:
        row.sort(key=lambda p: center(p)[0])
        ordered.extend(row)
    ordered.extend(unlocated)
    return ordered
