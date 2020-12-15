from typing import Any, Dict, Tuple, Union

JsonDict = Dict[str, Any]
Shape = Union[Tuple[int, ...], Tuple[None]]
Dtype = str


__all__ = [
    "Dtype",
    "JsonDict",
    "Shape",
]
