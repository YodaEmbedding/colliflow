import inspect
from time import time
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, cast

import rx
from rx import operators as ops

from colliflow.schedulers import schedulers
from colliflow.tensors import SymbolicTensor, Tensor

JsonDict = Dict[str, Any]

epoch = time()


def get_time():
    return time() - epoch


def _forward_unimplemented(self, *inputs: Any) -> Any:
    """
    `forward` is not provided as a regular function in the `Module`
    class to ensure that type-checkers and linters (e.g. mypy and
    pylint) don't raise warnings when a deriving class provides a
    `forward` method that accepts a fixed number of arguments, instead
    of the varargs suggested by this signature. This technique bypasses
    the contravariance and Liskov substitutability checks, as described
    in the following issues:

     - https://github.com/python/mypy/issues/8795
     - https://github.com/pytorch/pytorch/issues/35566
    """
    # pass  # pylint: disable=unnecessary-pass
    raise NotImplementedError


def _set_props_hook_unimplemented(
    self, *inputs: SymbolicTensor  # pylint: disable=unused-argument
):
    """
    `set_props_hook` is not provided as a regular function in the
    `Module` class for the same reasons as `forward`.
    """


class Module:
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

    registered_modules: List[Type["Module"]] = []
    name_to_module: Dict[str, Type["Module"]] = {}
    name: Optional[str] = None

    def __init__(
        self,
        shape: Tuple[int] = None,
        dtype: str = None,
        scheduler: str = None,
    ):
        self._shape = shape
        self._dtype = dtype
        self._scheduler = scheduler
        self.input_nodes: List["Module"] = []
        self.output_nodes: List["Module"] = []
        self.input_tensors: List[SymbolicTensor] = []
        self._is_used_in_static_graph = False

    def __init_subclass__(cls, **kwargs):
        if cls.name == "__AbstractModule":
            cls.name = None
            super().__init_subclass__(**kwargs)
            return
        if cls.name is None:
            cls.name = cls.__name__
        cls.registered_modules.append(cls)
        cls.name_to_module[cls.name] = cls
        super().__init_subclass__(**kwargs)

    def __call__(self, *inputs: Tensor, force_eval: bool = False):
        n_input = len(inputs)
        n_input_forward = self._num_forward_params()
        if n_input != n_input_forward:
            raise ValueError(
                f"Length mismatch: expected {n_input_forward} inputs, "
                f"actual {n_input}."
            )

        is_syms = [isinstance(tensor, SymbolicTensor) for tensor in inputs]

        if any(is_syms) and not all(is_syms):
            raise NotImplementedError(
                "Mixed symbolic and non-symbolic tensors are not supported."
            )

        if all(is_syms) and not force_eval:
            inputs = cast(Tuple[SymbolicTensor, ...], inputs)
            return self._set_graph_inputs(*inputs)

        if any(is_syms):
            raise ValueError("Cannot evaluate with symbolic tensors.")

        return self.forward(*inputs)

    def __repr__(self) -> str:
        kwargs_str = ", ".join(
            f"{k}={v!r}" for k, v in self.inner_config().items()
        )
        return f"{self.name}({kwargs_str})"

    @property
    def dtype(self) -> str:
        if self._dtype is None:
            raise ValueError("Please initialize dtype property.")
        return self._dtype

    @property
    def shape(self) -> Tuple[int]:
        if self._shape is None:
            raise ValueError("Please initialize shape property.")
        return self._shape

    def config(self, node_lut):
        return {
            "id": node_lut[self],
            "name": self.name,
            "inputs": [node_lut.get(x, None) for x in self.input_nodes],
            "outputs": [node_lut.get(x, None) for x in self.output_nodes],
            "tensor_inputs": [x.config for x in self.input_tensors],
            "config": self.inner_config(),
        }

    def inner_config(self) -> JsonDict:
        raise NotImplementedError

    @classmethod
    def from_config(cls, node_config: JsonDict) -> "Module":
        name = node_config["name"]
        module_config = node_config["config"]
        create_module = cls.name_to_module[name]
        return create_module(**module_config)

    forward: Callable[..., Any] = _forward_unimplemented
    """Override to define this module's core functionality."""

    set_props_hook: Callable[..., Any] = _set_props_hook_unimplemented
    """Override to set props like shape and dtype on symbolic call."""

    def forward_rx(
        self, *inputs: rx.Observable
    ) -> rx.core.ConnectableObservable:
        """
        Produces output observable from input observables.

        Override if desired. The default implementation zips together
        the input observables, puts the resulting observable on the
        correct scheduler, and then runs `forward`. The output
        observable is multicast so it can be reused without worrying
        about recomputation.
        """
        self._check_num_inputs(*inputs)
        observable = _zip_observables(*inputs)
        if self._scheduler is not None:
            observable = observable.pipe(
                ops.observe_on(schedulers[self._scheduler])
            )
        observable = observable.pipe(
            ops.do_action(lambda xs: print(f"{get_time():.1f}  {self}{xs}")),
            self._forward_to_rx_op(),
            ops.publish(),
        )
        observable = cast(rx.core.ConnectableObservable, observable)
        observable.connect()
        return observable

        # TODO QUESTION: Tensor type checks...
        # Tensor[type] -> Tensor[type]
        # Tensor[subtype] (covariance/contravariance)
        # Implement whatever works and worry about static problems later?

        # NOTE Define tensor as Any type since graph is deserialized anyways
        # Let the runtime do type-checking/casting if needed

        # Synchronous: forward: Sequence[Tensor] -> Tensor
        # Generator:   forward: Sequence[Tensor] with len=0 -> Tensor
        # Multi-output: forward: Sequence[Tensor] -> Sequence[Tensor]
        # Observable: forward:

        # def forward_async?? or forward_rx???
        # should an async take observableS as input? or a single zipped one?
        # TCP module probably doesn't want to zip the input observables...
        # but what about other applications?
        # do we need code around forward_rx? (e.g. scheduler, argument checks?)
        # does scheduler even make sense for multi-stream modules?

        # should modules be composable in heirarchy?

    def _check_forward_signature(self):
        n_input_nodes = len(self.input_nodes)
        n_input_forward = self._num_forward_params()
        if n_input_nodes != n_input_forward:
            raise ValueError(
                f"Length mismatch: forward requires {n_input_forward} inputs, "
                f"but module only has {n_input_nodes} input nodes."
            )

    def _check_num_inputs(self, *inputs):
        if len(inputs) != len(self.input_nodes):
            raise ValueError(
                f"Length mismatch: expected {len(self.input_nodes)} inputs, "
                f"actual {len(inputs)}"
            )

    def _forward_to_rx_op(self) -> Callable[[rx.Observable], rx.Observable]:
        flatten = inspect.isgeneratorfunction(self.forward)
        forward_op = ops.flat_map if flatten else ops.map
        return forward_op(lambda xs: self.forward(*xs))

    def _set_graph_inputs(self, *inputs: SymbolicTensor) -> SymbolicTensor:
        if self._is_used_in_static_graph:
            raise RuntimeError(
                "Cannot reuse a module instance more than once in a static "
                "graph."
            )

        # Link static graph nodes together
        for tensor in inputs:
            parent = tensor.parent
            self.input_nodes.append(parent)
            self.input_tensors.append(tensor)
            if parent is None:
                continue
            parent.output_nodes.append(self)

        self._check_forward_signature()
        self._is_used_in_static_graph = True
        self.set_props_hook(*inputs)

        return SymbolicTensor(self.shape, self.dtype, self)

    def _num_forward_params(self):
        return len(inspect.signature(self.forward).parameters)


class InputModule(Module):
    name = "__AbstractModule"


def _zip_observables(*inputs: rx.Observable) -> rx.Observable:
    if len(inputs) == 0:
        raise ValueError("Requires at least one input.")
    if len(inputs) == 1:
        return inputs[0].pipe(ops.map(lambda x: (x,)))
    return rx.zip(*inputs)
