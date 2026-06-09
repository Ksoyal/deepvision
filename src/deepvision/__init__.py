"""DeepVision: 让文本模型看懂图。

不同于把图片翻译成扁平散文描述,DeepVision 产出带坐标锚点的
结构化视觉表示(visual primitives + relations),让下游推理可以
精确引用图中元素的位置,从根本上缓解 Reference Gap。
"""

from .config import Config
from .schema import (
    Scene,
    Primitive,
    Relation,
    derive_geometric_relations,
    sort_reading_order,
)
from .vision import describe_structured

__all__ = [
    "Config",
    "Scene",
    "Primitive",
    "Relation",
    "derive_geometric_relations",
    "sort_reading_order",
    "describe_structured",
]

__version__ = "0.1.0"
