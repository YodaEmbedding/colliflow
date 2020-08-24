import json
from queue import Queue
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    Iterator,
    List,
    Sequence,
    Set,
    Tuple,
)

import rx

from colliflow.modules import InputLayer, Module
from colliflow.tensors import SymbolicTensor, Tensor


class Model:
    def __init__(
        self, *, inputs: List[SymbolicTensor], outputs: List[SymbolicTensor]
    ):
        self._inputs = inputs
        self._outputs = outputs
        self.modules = list(self._compute_order())

    def __call__(self, *inputs: Tensor) -> Sequence[Tensor]:
        return self._predict(*inputs)

    def __repr__(self) -> str:
        def _fmt(module, d):
            return (
                f"{d['inputs']} -> {d['id']} -> {d['outputs']}",
                f"{d['id']}: {module}",
            )

        rows = [_fmt(m, d) for m, d in self._serialize_pairs()]
        left_col = max(len(p) for p, _ in rows)
        return "\n".join(f"{p:{left_col}}  {m}" for p, m in rows)

    def to_rx(self, *inputs: rx.Observable) -> List[rx.Observable]:
        func = lambda module, xs: module.to_rx(*xs)
        return self._forward_graph(inputs, func)

    def serialize(self) -> str:
        """Serialize model to JSON."""
        return json.dumps(self.serialize_dict())

    def serialize_dict(self) -> List[Dict[str, Any]]:
        """Serialize model to JSON-serializable structure."""
        return [d for _, d in self._serialize_pairs()]

    @classmethod
    def deserialize(cls, model_config: str) -> "Model":
        """Deserialize model from JSON."""
        return cls.deserialize_dict(json.loads(model_config))

    @staticmethod
    def deserialize_dict(model_config: List[Dict[str, Any]]) -> "Model":
        """Deserialize model from JSON-serializable structure."""
        model_inputs = []
        model_outputs = []
        outputs: Dict[int, SymbolicTensor] = {}
        fringe = _Fringe()
        discovered = set()

        for node_cfg in model_config:
            if node_cfg["name"] == InputLayer.name:
                node_id = node_cfg["id"]
                fringe.put(node_id)
                discovered.add(node_id)

        node_configs = {node_cfg["id"]: node_cfg for node_cfg in model_config}

        while not fringe.empty():
            node_id = fringe.get()
            node_cfg = node_configs[node_id]
            is_input = node_cfg["name"] == InputLayer.name
            is_output = len(node_cfg["outputs"]) == 0
            ready = is_input or all(x in outputs for x in node_cfg["inputs"])

            if not ready:
                fringe.put_waiting(node_id)
                continue

            module = Module.from_config(node_cfg)
            inputs = (
                [SymbolicTensor(module.shape, module.dtype)]
                if is_input
                else [outputs[x] for x in node_cfg["inputs"]]
            )
            outputs[node_id] = module(*inputs)

            for nid in node_cfg["outputs"]:
                if nid in discovered:
                    continue
                fringe.put(nid)
                discovered.add(nid)

            if is_input:
                model_inputs.append(outputs[node_id])

            if is_output:
                model_outputs.append(outputs[node_id])

        return Model(inputs=model_inputs, outputs=model_outputs)

    def _compute_order(self) -> Iterator[Module]:
        visited = set()
        input_nodes = {x.parent for x in self._inputs}
        for output in self._outputs:
            output_node = output.parent
            for node in self._output_visiting_order(output_node, input_nodes):
                if node in visited:
                    continue
                visited.add(node)
                yield node

    def _flatten_graph(self) -> Iterator[Module]:
        node_set: Set[Module] = set()
        for x in self._inputs:
            input_module = x.parent
            yield from self._flatten(input_module, node_set)

    def _predict(self, *inputs: Tensor) -> List[Tensor]:
        apply_module = lambda module, xs: module(*xs)
        return self._forward_graph(inputs, apply_module)

    def _forward_graph(
        self,
        inputs: Sequence[Any],
        func: Callable[[Module, Sequence[Any]], Any],
    ) -> List[Any]:
        """Run forwards through the graph, applying given func."""
        if len(inputs) != len(self._inputs):
            raise ValueError("Wrong number of inputs provided")

        input_nodes = [x.parent for x in self._inputs]
        outputs = dict(zip(input_nodes, inputs))
        remaining = set(self.modules) - set(input_nodes)
        output_parents = [x.parent for x in self._outputs]

        for module in self.modules:
            if module in input_nodes:
                continue

            inputs = [outputs[x] for x in module.input_nodes]
            outputs[module] = func(module, inputs)
            remaining.remove(module)

            # Release reference to stored output if it is no longer needed
            for node in module.input_nodes:
                if any(x in remaining for x in node.output_nodes):
                    continue
                if node in outputs and node not in output_parents:
                    del outputs[node]

        return [outputs[x] for x in output_parents]

    def _serialize_pairs(
        self, skip_unneeded: bool = True
    ) -> Iterator[Tuple[Module, Dict[str, Any]]]:
        nodes = self._flatten_graph()
        if skip_unneeded:
            valid = set(self.modules)
            nodes = (x for x in nodes if x in valid)
        node_lut = {node: i for i, node in enumerate(nodes)}
        return ((node, node.config(node_lut)) for node in node_lut)

    @classmethod
    def _flatten(cls, node: Module, nodes: Set[Module]) -> Iterator[Module]:
        """Visits all nodes in graph."""
        if node is None or node in nodes:
            return
        yield node
        nodes.add(node)
        for x in node.output_nodes:
            yield from cls._flatten(x, nodes)

    @classmethod
    def _output_visiting_order(
        cls, output_node: Module, input_nodes: Iterable[Module]
    ) -> Iterator[Module]:
        """Yields node computation order for output node.

        This is done via post-order DFS traversal over the inverted tree.
        """
        if output_node in input_nodes:
            yield output_node
            return

        for node in output_node.input_nodes:
            yield from cls._output_visiting_order(node, input_nodes)

        yield output_node


class _Fringe:
    """Manage fringe for correct order of node expansion.

    The "fringe" consists of the list of nodes to be expanded next. If a
    particular node does not have all its inputs available during an
    attempt at node expansion, it is placed into the `_fringe_wait`
    queue. Once `_fringe` is emptied, we attempt to expand the nodes
    within `_fringe_wait`. If none of those nodes can be expanded
    either, then we raise an exception because the graph is not
    constructible.
    """

    def __init__(self):
        self._fringe = Queue()
        self._fringe_wait = Queue()
        self._wait_count = 0

    def empty(self):
        return self._fringe.empty() and self._fringe_wait.empty()

    def get(self):
        if not self._fringe.empty():
            self._wait_count = 0
            return self._fringe.get()

        if self._wait_count < len(self._fringe_wait):
            self._wait_count += 1
            return self._fringe_wait.get()

        raise Exception("No further items within fringe can be processed.")

    def put(self, item):
        self._fringe.put(item)

    def put_waiting(self, item):
        self._fringe_wait.put(item)


# def split_model():
#     pass
