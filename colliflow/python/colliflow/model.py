import asyncio
import json
from asyncio.events import AbstractEventLoop
from collections import abc
from queue import Queue
from typing import (
    Any,
    AsyncIterator,
    Callable,
    Dict,
    Iterable,
    Iterator,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
    Type,
    TypeVar,
    Union,
)

import rx
from rx import operators as ops
from rx.core.notification import Notification, OnError, OnNext
from rx.scheduler import ThreadPoolScheduler
from rx.scheduler.eventloop import AsyncIOScheduler

from colliflow.modules.input import InputModule
from colliflow.modules.module import InputAsyncModule, Module
from colliflow.tensors import SymbolicTensor, Tensor
from colliflow.typing import JsonDict

T = TypeVar("T")
MaybeSequence = Union[T, Sequence[T]]


def _coerce_sequence_type(x: MaybeSequence[T], t: Type[T]) -> Sequence[T]:
    if isinstance(x, t):
        return (x,)
    if isinstance(x, abc.Sequence) and all(isinstance(y, t) for y in x):
        return x
    raise TypeError(
        f"Cannot coerce type {type(x).__name__} to Sequence[{t.__name__}]."
    )


class Model:
    """Model representing a collaborative graph.

    Attributes:
        inputs: tensors whose parents are input modules
        outputs: tensors whose parents are the output modules
    """

    modules: List[Module]
    _inputs: List[SymbolicTensor]
    _outputs: List[SymbolicTensor]

    def __init__(
        self, *, inputs: List[SymbolicTensor], outputs: List[SymbolicTensor]
    ):
        self._inputs = inputs
        self._outputs = outputs
        self.modules = list(_compute_order(self._inputs, self._outputs))

    def __call__(self, *inputs: MaybeSequence[Tensor]) -> Sequence[Tensor]:
        """Synchronously run model prediction.

        For practical applications, asynchronous modes of operation
        should be preferred. See `to_rx` for more information.
        """
        xs = [_coerce_sequence_type(x, Tensor) for x in inputs]
        return self._predict(*xs)

    def __repr__(self) -> str:
        def _fmt(module, d):
            return (
                f"{d['inputs']} -> {d['id']} -> {d['outputs']}",
                f"{d['id']}: {module}",
            )

        rows = [_fmt(m, d) for m, d in self._serialize_pairs()]
        left_col = max(len(p) for p, _ in rows)
        return "\n".join(f"{p:{left_col}}  {m}" for p, m in rows)

    async def setup(self) -> AsyncIterator[Tuple[int, Any]]:
        """Sets up modules in parallel and yields their results."""
        async for result_pair in _rx_to_async_iter(self._setup_parallel()):
            yield result_pair

    def setup_blocking(self, parallel: bool = False) -> List[Tuple[int, Any]]:
        """Sets up modules and yields their results (non-async version)."""
        if not parallel:
            return [
                (i, module.setup()) for i, module in enumerate(self.modules)
            ]
        results = asyncio.run(self._setup_collect())
        return [results[i] for i in range(len(self.modules))]

    def to_rx(
        self, *inputs: MaybeSequence[rx.Observable]
    ) -> List[rx.Observable]:
        """Construct asynchronous pipeline using given input streams.

        Makes use of the Reactive Extensions (Rx) observable design
        pattern. Every module within the graph represents an observable
        and/or observer. Observable modules produce a stream of tensors;
        observers consume streams of tensors.
        """
        func = lambda module, xs: module.to_rx(*xs)
        xs = [_coerce_sequence_type(x, rx.Observable) for x in inputs]
        return self._forward_graph(xs, func)

    def serialize(self) -> str:
        """Serialize model to JSON-formatted string."""
        return json.dumps(self.serialize_dict())

    def serialize_dict(self) -> List[JsonDict]:
        """Serialize model to JSON-serializable structure."""
        return [d for _, d in self._serialize_pairs()]

    @classmethod
    def deserialize(cls, model_config: str) -> "Model":
        """Deserialize model from JSON-formatted string."""
        return cls.deserialize_dict(json.loads(model_config))

    @staticmethod
    def deserialize_dict(model_config: List[JsonDict]) -> "Model":
        """Deserialize model from JSON-serializable structure."""
        return _deserialize_dict(model_config)

    def _setup_parallel(self) -> rx.Observable:
        def setup_indexed(i: int, module: Module) -> Callable[[], Any]:
            return lambda: (i, module.setup())

        io_scheduler = ThreadPoolScheduler()
        observables = [
            rx.from_callable(setup_indexed(i, module), scheduler=io_scheduler)
            for i, module in enumerate(self.modules)
        ]
        return rx.from_iterable(observables).pipe(ops.merge_all())

    async def _setup_collect(self) -> Dict[int, Any]:
        results = {}
        async for module_id, result in self.setup():
            results[module_id] = result
        return results

    def _predict(self, *inputs: Sequence[Tensor]) -> List[Tensor]:
        apply_module = lambda module, xs: module(*xs)
        return self._forward_graph(inputs, apply_module)

    def _forward_graph(
        self,
        inputs: Sequence[Sequence[T]],
        func: Callable[[Module, Sequence[T]], T],
    ) -> List[T]:
        """Run forwards through the graph, applying given func."""
        if len(inputs) != len(self._inputs):
            raise ValueError(
                f"Wrong number of inputs provided: "
                f"expected {len(self._inputs)}, actual {len(inputs)}."
            )

        input_nodes: List[Module] = [x.parent for x in self._inputs]
        outputs = {n: func(n, xs) for n, xs in zip(input_nodes, inputs)}
        remaining = set(self.modules) - set(input_nodes)
        output_parents = [x.parent for x in self._outputs]

        for module in self.modules:
            if module in input_nodes:
                continue

            xs = [outputs[x] for x in module.input_nodes]
            outputs[module] = func(module, xs)
            remaining.remove(module)

            # Release reference to stored output if it is no longer needed
            for node in module.input_nodes:
                if any(x in remaining for x in node.output_nodes):
                    continue
                if node in outputs and node not in output_parents:
                    del outputs[node]

        return [outputs[x] for x in output_parents]

    def _serialize_pairs(self) -> Iterator[Tuple[Module, JsonDict]]:
        """Yields pairs of modules and serialized dicts."""
        valid_nodes = set(self.modules)
        nodes = (x for x in _flatten_graph(self._inputs) if x in valid_nodes)
        node_lut = {node: i for i, node in enumerate(nodes)}
        return ((node, node.config(node_lut)) for node in node_lut)


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

        if self._wait_count < self._fringe_wait.qsize():
            self._wait_count += 1
            return self._fringe_wait.get()

        raise Exception("No further items within fringe can be processed.")

    def put(self, item):
        self._fringe.put(item)

    def put_waiting(self, item):
        self._fringe_wait.put(item)


def _deserialize_dict(model_config: List[JsonDict]) -> Model:
    model_inputs = []
    model_outputs = []
    outputs: Dict[int, SymbolicTensor] = {}
    fringe = _Fringe()
    discovered = set()

    # Initialize fringe with input modules
    for node_cfg in model_config:
        if _is_input(node_cfg):
            node_id = node_cfg["id"]
            fringe.put(node_id)
            discovered.add(node_id)

    node_configs = {node_cfg["id"]: node_cfg for node_cfg in model_config}

    while not fringe.empty():
        node_id = fringe.get()
        node_cfg = node_configs[node_id]
        is_input = _is_input(node_cfg)
        is_output = len(node_cfg["outputs"]) == 0
        ready = is_input or all(x in outputs for x in node_cfg["inputs"])

        # If input tensors are not yet available, save node for later expansion
        if not ready:
            fringe.put_waiting(node_id)
            continue

        # Create module and connect it with input nodes
        outputs[node_id] = _create_and_init_module(node_cfg, outputs, is_input)

        # Append node outputs to fringe
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


def _is_input(node_cfg: JsonDict):
    module_type = Module.name_to_module[node_cfg["name"]]
    return issubclass(module_type, (InputModule, InputAsyncModule))


def _create_and_init_module(
    node_cfg: JsonDict, outputs: Dict[int, SymbolicTensor], is_input: bool
) -> SymbolicTensor:
    # Create module and determine its inputs
    module = Module.from_config(node_cfg)
    inputs = (
        [SymbolicTensor(s, d) for s, d in node_cfg["tensor_inputs"]]
        if is_input
        else [outputs[x] for x in node_cfg["inputs"]]
    )

    # Use inputs to instantiate module
    try:
        return module(*inputs)
    except Exception as e:
        raise Exception(
            f"{module} could not be called with the inputs {inputs}."
        ) from e


def _flatten_graph(inputs: List[SymbolicTensor]) -> Iterator[Module]:
    """Yields all nodes in graph.

    The parents of the input tensors are the "input modules".
    """
    node_set: Set[Module] = set()
    for x in inputs:
        input_module = x.parent
        yield from _flatten(input_module, node_set)


def _flatten(node: Module, nodes: Set[Module]) -> Iterator[Module]:
    if node is None or node in nodes:
        return
    yield node
    nodes.add(node)
    for x in node.output_nodes:
        yield from _flatten(x, nodes)


def _compute_order(
    inputs: List[SymbolicTensor], outputs: List[SymbolicTensor]
) -> Iterator[Module]:
    """Yields modules in computation order."""
    visited = set()
    input_nodes = {x.parent for x in inputs}
    for output in outputs:
        output_node = output.parent
        for node in _output_visiting_order(output_node, input_nodes):
            if node in visited:
                continue
            visited.add(node)
            yield node


def _output_visiting_order(
    output_node: Module, input_nodes: Iterable[Module]
) -> Iterator[Module]:
    """Yields node computation order for output node.

    This is done via post-order DFS traversal over the inverted tree.
    """
    if output_node in input_nodes:
        yield output_node
        return

    for node in output_node.input_nodes:
        yield from _output_visiting_order(node, input_nodes)

    yield output_node


async def _rx_to_async_iter(
    observable: rx.Observable,
    loop: Optional[AbstractEventLoop] = None,
) -> AsyncIterator[Any]:
    queue: asyncio.Queue[Notification] = asyncio.Queue()
    if loop is None:
        loop = asyncio.get_event_loop()

    def on_next(x):
        assert loop is not None
        asyncio.run_coroutine_threadsafe(queue.put(x), loop)

    disposable = observable.pipe(ops.materialize()).subscribe(
        on_next=on_next, scheduler=AsyncIOScheduler(loop=loop)
    )

    while True:
        x = await queue.get()
        if isinstance(x, OnNext):
            yield x.value
            queue.task_done()
        elif isinstance(x, OnError):
            disposable.dispose()
            raise RuntimeError(x.value)
        else:
            disposable.dispose()
            break


__all__ = [
    "Model",
]
