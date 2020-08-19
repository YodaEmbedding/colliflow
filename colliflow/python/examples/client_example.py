import inspect
import json
import multiprocessing
import random
import socket
from pprint import pprint
from queue import Queue
from time import sleep, time
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    Iterator,
    List,
    Sequence,
    Tuple,
)

import rx
from rx import operators as ops
from rx.scheduler import ThreadPoolScheduler

cpu_scheduler = ThreadPoolScheduler(multiprocessing.cpu_count())
"""Thread scheduler with as many threads as CPU cores.

Note that full parallel multi-core usage may not occur if there is more
than one task that requires continuous usage of the GIL. However, if all
CPU-bound tasks release the GIL when doing heavy computations (e.g. by
off-loading work to external libraries such as NumPy), then CPU cores
will be utilized to better effect. If this is not the case, one should
use a scheduler that utilizes multiple *processes* instead of threads.
"""

io_scheduler = ThreadPoolScheduler()
"""Thread scheduler for IO-bound tasks.

The number of workers is set to the default value given by
`concurrent.futures.ThreadPoolExecutor`, which can be found in the
official Python documentation for your Python version.
"""

schedulers = {"cpu": cpu_scheduler, "io": io_scheduler}

epoch = time()


def get_time():
    return time() - epoch


class Tensor:  # pylint: disable=too-few-public-methods
    def __init__(self, shape, dtype):
        if shape is None or dtype is None:
            raise ValueError("Please ensure shape and dtype are correct.")

        self.shape = tuple(shape)
        self.dtype = dtype

        # TODO hold some actual data!
        # self.data = data

    def __repr__(self) -> str:
        return f"Tensor(shape={self.shape}, dtype={self.dtype})"


class SymbolicTensor(Tensor):  # pylint: disable=too-few-public-methods
    def __init__(self, shape, dtype, parent=None):
        super().__init__(shape, dtype)
        self.parent = parent

    def __repr__(self) -> str:
        s = f"shape={self.shape}, dtype={self.dtype}, parent={self.parent!r}"
        return f"SymbolicTensor({s})"


class Module:
    """Defines a node in the collaborative intelligence graph.

    Inheriting classes must:

    1. Override `forward` for execution.
    2. Override `inner_config` for serialization.
    3. Either override `dtype`, and `shape` or set their backing fields,
       `_dtype`, and `_shape` for serialization.
    4. Set `name` to a unique identifier for the class.
    """

    _registered_modules: List["Module"] = []
    _name_to_module: Dict[str, "Module"] = {}
    name: str = None

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
        self._is_used_in_static_graph = False

    def __init_subclass__(cls, **kwargs):
        if cls.name is None:
            cls.name = cls.__name__
        cls._registered_modules.append(cls)
        cls._name_to_module[cls.name] = cls
        super().__init_subclass__(**kwargs)

    def __call__(self, *inputs):
        is_syms = [isinstance(tensor, SymbolicTensor) for tensor in inputs]

        if all(is_syms):
            return self._forward_symbolic(*inputs)

        if any(is_syms):
            raise NotImplementedError(
                "Mixed symbolic and non-symbolic tensors are not supported."
            )

        return self.forward(*inputs)

    def __repr__(self) -> str:
        kwargs_str = ", ".join(
            f"{k}={v!r}" for k, v in self.inner_config().items()
        )
        return f"{self.name}({kwargs_str})"

    @property
    def dtype(self) -> str:
        return self._dtype

    @property
    def shape(self) -> Tuple[int]:
        return self._shape

    def config(self, node_lut):
        return {
            "id": node_lut[self],
            "name": self.name,
            "inputs": [node_lut.get(x, None) for x in self.input_nodes],
            "outputs": [node_lut.get(x, None) for x in self.output_nodes],
            "config": self.inner_config(),
        }

    def forward(self, *inputs) -> Any:
        raise NotImplementedError

    def inner_config(self) -> Dict[str, Any]:
        raise NotImplementedError

    @classmethod
    def from_config(cls, node_config: Dict[str, Any]) -> "Module":
        name = node_config["name"]
        module_config = node_config["config"]
        create_module = cls._name_to_module[name]
        return create_module(**module_config)

    def to_rx(self, *inputs):
        self._check_signature()
        if len(inputs) != len(self.input_nodes):
            raise ValueError(
                f"Length mismatch: Expected {len(self.input_nodes)} inputs, "
                f"actual {len(inputs)}"
            )
        # TODO Should 0 inputs be an error? If not, what is the observable?
        if len(inputs) == 0:
            raise ValueError("Requires at least one input")
        if len(inputs) == 1:
            observable = inputs[0].pipe(ops.map(lambda x: (x,)))
        else:
            observable = rx.zip(*inputs)
        flatten = inspect.isgeneratorfunction(self.forward)
        forward_op = ops.flat_map if flatten else ops.map
        ops_pre = []
        if self._scheduler is not None:
            ops_pre.append(ops.observe_on(schedulers[self._scheduler]))
        observable = observable.pipe(
            *ops_pre,
            ops.do_action(lambda xs: print(f"{get_time():.1f}  {self}{xs}")),
            forward_op(lambda xs: self.forward(*xs)),
            ops.publish(),
        )
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

    def set_props_hook(self, *inputs: Sequence[SymbolicTensor]):
        """Override to set props like shape and dtype on symbolic call."""

    def _check_signature(self):
        n_input_nodes = len(self.input_nodes)
        n_input_forward = len(inspect.signature(self.forward).parameters)
        if n_input_nodes != n_input_forward:
            raise ValueError(
                f"Length mismatch: forward requires {n_input_forward} inputs, "
                f"but module only has {n_input_nodes} input nodes"
            )

    def _forward_symbolic(
        self, *inputs: Sequence[SymbolicTensor]
    ) -> SymbolicTensor:
        if self._is_used_in_static_graph:
            raise RuntimeError(
                "Cannot reuse a module instance more than once in a static "
                "graph."
            )

        # Link static graph nodes together
        for tensor in inputs:
            parent = tensor.parent
            self.input_nodes.append(parent)
            if parent is None:
                continue
            parent.output_nodes.append(self)

        self._is_used_in_static_graph = True
        self.set_props_hook(*inputs)

        return SymbolicTensor(self.shape, self.dtype, self)


def Input(shape, dtype, scheduler=None):  # pylint: disable=invalid-name
    x = SymbolicTensor(shape, dtype)
    return InputLayer(shape, dtype, scheduler=scheduler)(x)


class InputLayer(Module):
    name = "Input"

    def __init__(self, shape, dtype, **kwargs):
        super().__init__(shape, dtype, **kwargs)

    def inner_config(self):
        return {"shape": self.shape, "dtype": self.dtype}

    def forward(self, tensor: Tensor):  # pylint: disable=arguments-differ
        return tensor


class Preprocessor(Module):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def inner_config(self):
        return {}

    def forward(self, tensor: Tensor):  # pylint: disable=arguments-differ
        return tensor

    def set_props_hook(self, tensor):  # pylint: disable=arguments-differ
        self._shape = tensor.shape
        self._dtype = tensor.dtype


class ClientInferenceModel(Module):
    def __init__(self, func=None, shape=None, dtype=None, **kwargs):
        super().__init__(shape, dtype, **kwargs)
        self.func = func

    def inner_config(self):
        return {"shape": self.shape, "dtype": self.dtype}

    def forward(self, tensor: Tensor):  # pylint: disable=arguments-differ
        sleep(0.7)
        return self.func(tensor)


class ServerInferenceModel(Module):
    def __init__(self, func=None, shape=None, dtype=None, **kwargs):
        super().__init__(shape, dtype, **kwargs)
        self.func = func

    def inner_config(self):
        return {"shape": self.shape, "dtype": self.dtype}

    def forward(self, tensor: Tensor):  # pylint: disable=arguments-differ
        sleep(0.5)
        return self.func(tensor)


class Postencoder(Module):
    def __init__(self, **kwargs):
        super().__init__((None,), "uint8", **kwargs)

    def inner_config(self):
        return {}

    def forward(self, tensor: Tensor):  # pylint: disable=arguments-differ
        return tensor


class Predecoder(Module):
    def __init__(self, shape, dtype, **kwargs):
        super().__init__(shape, dtype, **kwargs)

    def inner_config(self):
        return {"shape": self.shape, "dtype": self.dtype}

    def forward(self, tensor: Tensor):  # pylint: disable=arguments-differ
        return tensor


# Not really needed since Postencoder already converts to uint8
# class TcpSender(Module):
#     def __init__(self):
#         super().__init__((None,), "uint8")
#
#     def inner_config(self):
#         return {}
#
#     def forward(self, tensor: Tensor):  # pylint: disable=arguments-differ
#         return Tensor(self.shape, self.dtype)


class TcpClient(Module):
    def __init__(self, hostname=None, port=None, **kwargs):
        super().__init__((None,), "uint8", **kwargs)
        self.hostname = hostname
        self.port = port

    def inner_config(self):
        return {}

    def forward(self, tensor: Tensor):  # pylint: disable=arguments-differ
        return tensor


# class TcpServer(Module):
#     # so... this should what? keep a bind listener open? then wait for input?
#     # input should be in a "streaming" tensor format...! (byte header/etc)
#
#     def __init__(self, hostname=None, port=None, **kwargs):
#         super().__init__((None,), "uint8", **kwargs)
#         self.hostname = hostname
#         self.port = port
#         # wait... what about "async" server that we wrote? nevermind that?
#         # that's useful for multiclient architecture... but forget that for now
#
#         # perhaps pass a prepared socket in? is that what the executor's job is?
#         # Meh... actually, it's fine to do this I think but not IMMEDIATELY?
#         # only when model is "initialized" properly, and we send a start signal
#         # to all modules for any initialization code?
#
#
#         # does this module really need an "input"? or can it be a "producer"
#         # node (e.g. like InputLayer)
#
#         # also, this is all running in a separate thread...
#         # OH SO THATS WHAT EXECUTOR DOES! MANAGE THREADS! and pools!
#
#         # Design all these things on Surface...
#
#         self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
#         self.sock.bind((hostname, port))
#         self.sock.listen(1)
#         # self.conn, self.addr = self.sock.accept()
#         # self.conn.recv()
#
#         # TODO
#
#     def inner_config(self):
#         return {}
#
#     def forward(self, tensor: Tensor):  # pylint: disable=arguments-differ
#         return tensor


class Model:
    def __init__(
        self, *, inputs: List[SymbolicTensor], outputs: List[SymbolicTensor]
    ):
        self._inputs = inputs
        self._outputs = outputs
        self.modules = list(self._compute_order())

    def __call__(self, *inputs: Sequence[Tensor]) -> Sequence[Tensor]:
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

    def to_rx(self, *inputs: Sequence[rx.Observable]) -> List[rx.Observable]:
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
        outputs = {}
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

    def _compute_order(self) -> Iterable[Module]:
        visited = set()
        input_nodes = {x.parent for x in self._inputs}
        for output in self._outputs:
            output_node = output.parent
            for node in self._output_visiting_order(output_node, input_nodes):
                if node in visited:
                    continue
                visited.add(node)
                yield node

    def _flatten_graph(self) -> Iterable[Module]:
        node_set = set()
        for x in self._inputs:
            input_module = x.parent
            yield from self._flatten(input_module, node_set)

    def _predict(self, *inputs: Sequence[Tensor]) -> List[Tensor]:
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
    def _flatten(cls, node, nodes):
        """Visits all nodes in graph."""
        if node is None or node in nodes:
            return
        yield node
        nodes.add(node)
        for x in node.output_nodes:
            yield from cls._flatten(x, nodes)

    @classmethod
    def _output_visiting_order(cls, output_node, input_nodes):
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


class Executor:
    # What exactly does this class do?
    # Is it only useful for actual real execution, not simulations?
    #
    # port, host... localhost? only needed when network is...?
    # perhaps pass a [virtual] connection object instead!
    # what is the protocol of the object? TCP/UDP? What module node should
    # each one go to? idk... assume single for now.
    #
    # self.conn = conn # Don't handle actual connecting/etc inside class!
    # just provide an interface for STREAMS or PACKETS?
    #
    # or alternatively, just have a "UdpPacketizer" and "UdpSender/etc"
    # as separate modules! (the sender has an actual connection object)
    #
    # how to deal with multithreading/multiprocessing?
    # perhaps each module specifies which pool it wants to be in
    # (CPU/IO/SINGLE)
    #
    # OK, but what about messages? or model switching?
    # Who handles all that?
    #
    # So... what's the point of Executor?
    # Backpressure handling?
    # Auto-model switching?
    # "Outer" comm protocol? (which is used for... what?)
    #
    def __init__(self, model):
        self.model = model


def simple_model():
    client_func = lambda x: Tensor(shape=(14, 14, 512), dtype="uint8")
    server_func = lambda x: Tensor(shape=(1000,), dtype="float32")

    inputs = [Input(shape=(224, 224, 3), dtype="uint8")]

    x = inputs[0]
    x = Preprocessor()(x)
    x = ClientInferenceModel(
        func=client_func, shape=(14, 14, 512), dtype="uint8"
    )(x)
    x = Postencoder()(x)
    x = Predecoder(shape=(14, 14, 512), dtype="uint8")(x)
    x = ServerInferenceModel(  #
        func=server_func, shape=(1000,), dtype="float32"
    )(x)

    outputs = [x]

    return Model(inputs=inputs, outputs=outputs)


class RandomMerge(Module):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def inner_config(self):
        return {}

    def forward(self, left: Tensor, right: Tensor):
        return left if random.random() < 0.5 else right

    def set_props_hook(self, left: Tensor, right: Tensor):
        assert left.shape == right.shape
        assert left.dtype == right.dtype
        self._shape = left.shape
        self._dtype = left.dtype


def multi_branch_model():
    client_func = lambda x: Tensor(shape=(14, 14, 512), dtype="uint8")
    server_func = lambda x: Tensor(shape=(1000,), dtype="float32")

    inputs = [Input(shape=(224, 224, 3), dtype="uint8", scheduler="io")]
    a = inputs[0]

    x = Postencoder(scheduler="cpu")(a)
    x = Predecoder(shape=(224, 224, 3), dtype="uint8")(x)
    b = x
    c = RandomMerge()(a, b)
    c = RandomMerge()(c, c)

    x = Preprocessor(scheduler="cpu")(c)
    x = ClientInferenceModel(
        func=client_func, shape=(14, 14, 512), dtype="uint8", scheduler="cpu"
    )(x)
    x = Postencoder()(x)
    x = Predecoder(shape=(14, 14, 512), dtype="uint8")(x)
    x = ServerInferenceModel(  #
        func=server_func, shape=(1000,), dtype="float32", scheduler="cpu"
    )(x)

    # outputs = [x, a, b, c]
    outputs = [x, c]
    # outputs = [x]

    return Model(inputs=inputs, outputs=outputs)


def model_client_server():
    client_func = lambda x: Tensor(shape=(14, 14, 512), dtype="uint8")
    server_func = lambda x: Tensor(shape=(1000,), dtype="float32")

    inputs = [Input(shape=(224, 224, 3), dtype="uint8")]
    x = inputs[0]
    x = Preprocessor()(x)
    x = ClientInferenceModel(
        func=client_func, shape=(14, 14, 512), dtype="uint8"
    )(x)
    x = Postencoder()(x)
    # x = TcpSender()(x)
    outputs = [x]
    model_client = Model(inputs=inputs, outputs=outputs)

    inputs = [Input(shape=(None,), dtype="uint8")]
    x = inputs[0]
    x = Predecoder(shape=(14, 14, 512), dtype="uint8")(x)
    x = ServerInferenceModel(  #
        func=server_func, shape=(1000,), dtype="float32"
    )(x)
    # x = TcpSender()(x)
    outputs = [x]
    model_server = Model(inputs=inputs, outputs=outputs)

    return model_client, model_server


def model_from_config(model_config):
    client_func = lambda x: Tensor(shape=(14, 14, 512), dtype="uint8")
    server_func = lambda x: Tensor(shape=(1000,), dtype="float32")

    model = Model.deserialize_dict(model_config)
    x = next(x for x in model.modules if isinstance(x, ClientInferenceModel))
    x.func = client_func
    x = next(x for x in model.modules if isinstance(x, ServerInferenceModel))
    x.func = server_func

    return model


def main():
    print("\nSIMPLE MODEL CONSTRUCTION TEST")
    model = simple_model()
    print(model)

    print("\nSIMPLE SYNCHRONOUS PREDICTION TEST")
    preds = model(Tensor(shape=(224, 224, 3), dtype="uint8"))
    print(preds)

    print("\nSERIALIZE TEST")
    model_config = model.serialize_dict()
    pprint(model_config)
    # pprint(json.loads(json.dumps(model_config)))

    print("\nDESERIALIZE TEST")
    model = model_from_config(model_config)
    preds = model(Tensor(shape=(224, 224, 3), dtype="uint8"))
    print(preds)

    print("\nMODEL CLIENT/SERVER TEST")
    model_client, model_server = model_client_server()
    print(model_client)
    print(model_server)

    # print("\nCLIENT/SERVER EXECUTOR TEST")
    # client_executor = Executor(model=model_client)
    # server_executor = Executor(model=model_server)


def main_rx():
    frames = rx.interval(1).pipe(
        ops.do_action(lambda x: print(f"\n{get_time():.1f}  Frame {x}\n")),
        ops.map(lambda x: Tensor((224, 224, 3), "uint8")),
        ops.publish(),
    )
    model = simple_model()
    model = multi_branch_model()
    print(model)

    # preds = model(Tensor((224, 224, 3), "uint8"))
    # print(preds)

    observables = model.to_rx(frames)
    observable = observables[0]
    observable.subscribe(lambda x: print(f"{get_time():.1f}  Result"))

    frames.connect()
    sleep(10)

    # TODO
    # buffer_size, drop=True, etc
    # backpressure
    # time slice scheduling
    # scheduler messaging
    # tcp messaging
    # tcp client/server
    # ordering: "tensor indexes" (e.g. tcp "enforces" order; udp order lost tensors)
    # ...

    # NOTE
    # predictive early-dropping frames + scheduling matters for low-latency
    # ...not really for high-throughput only, though

    # TEST
    # multicasting
    # various architectures, fuzzy timings/delays
    # verify computation completes, no deadlocks, ...


if __name__ == "__main__":
    # main()
    main_rx()


# TODO flow of actual data...? ehhhh

# TODO
# Test entire model (with local TCP)

# TODO
# Communication protocol... does TcpSender handle it? Or does the actual
# executor/context determine what to do? What's the purpose of TcpSender, then?
#
# Doesn't it make the API cleaner to let TcpSender handle certain details?
#
# But it also makes sense to have the "Executor" switch between models
# (receiving/sending configs), sending stats/etc, and so on.
# Perhaps TcpSender is then just some sort of interface that outputs "packets"
# (or in this case, a data stream) that the graph executor itself then relays.

# TODO
# Should we have each "Module" node report estimated compute times?
# That way, we can schedule better?
