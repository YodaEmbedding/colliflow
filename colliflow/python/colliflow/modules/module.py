import inspect
from dataclasses import dataclass, field
from time import time
from typing import Any, Callable, Dict, List, Optional, Type

import rx
from rx import operators as ops

from colliflow.tensors import SymbolicTensor
from colliflow.typing import Dtype, JsonDict, Shape

epoch = time()


def get_time():
    return time() - epoch


def _forward_unimplemented(self, *inputs: Any) -> Any:
    """
    `forward` is not provided as a regular function in the
    `ForwardModule` class to ensure that type-checkers and linters
    (e.g. mypy and pylint)
    don't raise warnings when a deriving class provides a
    `forward` method that accepts a fixed number of arguments, instead
    of the varargs suggested by this signature. This technique bypasses
    the contravariance and Liskov substitutability checks, as described
    in the following issues:

     - https://github.com/python/mypy/issues/8795
     - https://github.com/pytorch/pytorch/issues/35566
    """
    # pass  # pylint: disable=unnecessary-pass
    raise NotImplementedError


@dataclass(eq=False)
class Node:
    input_nodes: List["Module"] = field(default_factory=list)
    output_nodes: List["Module"] = field(default_factory=list)
    input_shapes: List[Shape] = field(default_factory=list)
    input_dtypes: List[Dtype] = field(default_factory=list)
    output_shapes: List[Shape] = field(default_factory=list)
    output_dtypes: List[Dtype] = field(default_factory=list)


class Module(Node):
    """Defines a node in the collaborative intelligence graph.

    Inheriting classes must:

    1. Call `__init__`.
    2. Override `inner_config` for serialization.
    3. Either override `dtype`, and `shape` or set their backing fields,
       `_dtype`, and `_shape` for serialization.
    4. Set `name` to a unique identifier for the class.
    5. Either override `forward` to define the synchronous execution, or
       override `forward_rx` to define a custom asynchronous method.

    Optionally:

    6. Override `set_props_hook` in order to set `_dtype` or `_shape`
       during a symbolic tensor call.
    """

    name: Optional[str] = None
    name_to_module: Dict[str, Type["Module"]] = {}
    registered_modules: List[Type["Module"]] = []

    def __init__(self, shape: Shape, dtype: Dtype):
        super().__init__()
        self.output_shapes = [shape]
        self.output_dtypes = [dtype]
        self._is_used_in_static_graph: bool = False

    def __init_subclass__(cls, **kwargs):
        """Registers module classes (required for serializability).

        If the class is registered as an "__AbstractModule",
        then the name registration is deferred to concrete submodules.
        """
        if cls.name == "__AbstractModule":
            cls.name = None
            super().__init_subclass__(**kwargs)
            return
        if cls.name is None:
            cls.name = cls.__name__
        cls.registered_modules.append(cls)
        cls.name_to_module[cls.name] = cls
        super().__init_subclass__(**kwargs)

    def __call__(self, *inputs: SymbolicTensor) -> SymbolicTensor:
        """Connects module instances into a static graph."""
        return self._set_graph_inputs(*inputs)

    def __repr__(self) -> str:
        kwargs_str = ", ".join(
            f"{k}={v!r}" for k, v in self.inner_config().items()
        )
        return f"{self.name}({kwargs_str})"

    def inner_config(self) -> JsonDict:
        """Override this to serialize custom module parameters."""
        raise NotImplementedError

    def setup(self) -> Any:
        """Override this to run prior to observable graph construction.

        Though users may override this method with synchronous code,
        the setup will occur in parallel on various threads.
        This method should try to contain only IO-bound actions.

        May return a result indicating success/failure.
        """
        return None

    def to_rx(self, *inputs: rx.Observable) -> rx.Observable:
        """Produces output observable from input observables.

        See abstract subclass (e.g. `ForwardModule`) docstring
        for further details.
        """
        raise NotImplementedError

    def config(self, node_lut: Dict["Module", int]) -> JsonDict:
        """Returns serializable JSON dictionary.

        JSON dictionary describes graph connections and parameters
        so that a deserializer can reconstruct the module.
        """
        return {
            "id": node_lut[self],
            "name": self.name,
            "inputs": [node_lut.get(x, None) for x in self.input_nodes],
            "outputs": [node_lut.get(x, None) for x in self.output_nodes],
            "tensor_inputs": list(zip(self.input_shapes, self.input_dtypes)),
            "config": self.inner_config(),
        }

    @classmethod
    def from_config(cls, node_config: JsonDict) -> "Module":
        """Constructs module using JSON config."""
        name = node_config["name"]
        module_config = node_config["config"]
        create_module = cls.name_to_module[name]
        return create_module(**module_config)

    def _check_num_inputs(
        self,
        n_input: int,
        check_nodes=False,
        check_signature=False,  # pylint: disable=unused-argument
    ):
        n_nodes = len(self.input_nodes)
        if check_nodes and n_input != n_nodes:
            raise ValueError(
                f"Length mismatch: expected {n_nodes} inputs, "
                f"actual {n_input}."
            )

    def _set_graph_inputs(self, *inputs: SymbolicTensor) -> SymbolicTensor:
        if self._is_used_in_static_graph:
            raise RuntimeError(
                "Cannot reuse a module instance in a static graph."
            )

        self._check_num_inputs(len(inputs), check_signature=True)

        # Link static graph nodes together
        for tensor in inputs:
            parent = tensor.parent
            self.input_nodes.append(parent)
            self.input_dtypes.append(tensor.dtype)
            self.input_shapes.append(tensor.shape)
            if parent is None:
                continue
            parent.output_nodes.append(self)

        self._is_used_in_static_graph = True

        outputs = [
            SymbolicTensor(shape, dtype, self)
            for shape, dtype in zip(self.output_shapes, self.output_dtypes)
        ]
        return outputs[0]


class ForwardModule(Module):  # pylint: disable=abstract-method
    """Function from M tensors to N tensors. M, N > 0."""

    name = "__AbstractModule"

    def to_rx(self, *inputs: rx.Observable) -> rx.Observable:
        """Produces output observable from input observables.

        Zips together the input observables,
        puts the resulting observable on the correct scheduler,
        and then runs `forward`.
        The output observable is multicast so it can be reused
        without worrying about recomputation.
        """
        self._check_num_inputs(
            len(inputs), check_nodes=self._is_used_in_static_graph
        )
        observable = _zip_observables(*inputs)
        observable = observable.pipe(
            self._forward_to_rx_op(),
            ops.share(),
        )
        return observable

    forward: Callable[..., Any] = _forward_unimplemented
    """Override to define this module's core functionality."""

    def _forward_to_rx_op(self) -> Callable[[rx.Observable], rx.Observable]:
        return ops.map(lambda xs: self.forward(*xs))

    def _check_num_inputs(
        self, n_input: int, check_nodes=False, check_signature=False
    ):
        super()._check_num_inputs(n_input, check_nodes=check_nodes)
        n_forward = self._num_forward_params()
        if check_signature and n_input != n_forward:
            raise ValueError(
                f"Length mismatch: expected {n_forward} inputs, "
                f"actual {n_input}."
            )

    def _num_forward_params(self):
        return len(inspect.signature(self.forward).parameters)


def _forward_async_unimplemented(
    self, *inputs: rx.Observable
) -> rx.Observable:
    """See docstring of `_forward_unimplemented`."""
    raise NotImplementedError


class ForwardAsyncModule(Module):  # pylint: disable=abstract-method
    """Function from M observables to N observables. M, N > 0."""

    name = "__AbstractModule"

    forward: Callable[..., rx.Observable] = _forward_async_unimplemented
    """Override to define this module's core functionality.

    Should return observables
    after transforming the input observables in some manner.
    """

    def to_rx(self, *inputs: rx.Observable) -> rx.Observable:
        return self.forward(*inputs)


class InputAsyncModule(Module):
    """Produces Rx observables."""

    name = "__AbstractModule"

    def produce(self) -> rx.Observable:
        """Override to define this module's core functionality.

        Should create new observables and return them.
        """
        raise NotImplementedError

    def to_rx(self, *inputs: rx.Observable) -> rx.Observable:
        # NOTE If Module is not attached to graph, do not check len(inputs)
        # TODO Does it make sense for module to be detached from graph?
        if self._is_used_in_static_graph:
            self._check_num_inputs(len(inputs), check_nodes=True)
        return self.produce().pipe(ops.share())


class OutputAsyncModule(Module):
    """Consumes Rx observables."""

    name = "__AbstractModule"

    def consume(self, *inputs: rx.Observable):
        """Override to define this module's core functionality.

        Should subscribe to the input observables.
        """
        raise NotImplementedError

    def to_rx(self, *inputs: rx.Observable) -> rx.Observable:
        # NOTE If Module is not attached to graph, do not check len(inputs)
        # TODO Does it make sense for module to be detached from graph?
        if self._is_used_in_static_graph:
            self._check_num_inputs(len(inputs), check_nodes=True)
        self.consume(*inputs)
        # Create dummy observable to satisfy type signature
        return rx.from_iterable([])


def _zip_observables(*inputs: rx.Observable) -> rx.Observable:
    if len(inputs) == 0:
        raise ValueError("Requires at least one input.")
    if len(inputs) == 1:
        return inputs[0].pipe(ops.map(lambda x: (x,)))
    return rx.zip(*inputs)


__all__ = [
    "Module",
    "ForwardModule",
    "ForwardAsyncModule",
    "InputAsyncModule",
    "OutputAsyncModule",
]
