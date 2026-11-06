"""ONNX export and TensorRT engines for the two per-frame networks (RAFT stereo, SegFormer).

Each network is wrapped so the exported graph does everything up to a tensor the Python side can use
directly: SegFormer returns the full-resolution instrument probability, RAFT returns the final flow for a
batch of (left->right, right->left) pairs, so the left-right consistency check stays identical to
`sr3d.depth.raft.RAFTStereo`. TensorRT 10+ builds strongly typed networks, so FP16 engines come from FP16
ONNX graphs. Inputs have fixed shapes (one camera, one resolution).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from ..depth.raft import RAFTStereo
from ..segmentation.data import IMAGENET_MEAN, IMAGENET_STD


class SegFormerProb(torch.nn.Module):
    """(1, 3, H, W) ImageNet-normalised image -> (1, H, W) instrument probability."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        logits = F.interpolate(self.model(pixel_values=x).logits, size=x.shape[-2:], mode="bilinear",
                               align_corners=False)
        return logits.softmax(1)[:, 1]


RAFT_FP32_OPS = ("Add", "Sub", "Mul", "Div", "GridSample", "Floor", "Ceil", "Clip", "Where", "Expand", "Range",
                 "ConstantOfShape", "Concat", "Reshape", "Unsqueeze", "Slice", "Gather")


class RAFTFlow(torch.nn.Module):
    """(2, 3, H, W) first images, (2, 3, H, W) second images in [-1, 1] -> (2, 2, H, W) final flow."""

    def __init__(self, model, iters: int):
        super().__init__()
        self.model, self.iters = model, iters

    def forward(self, a, b):
        return self.model(a, b, num_flow_updates=self.iters)[-1]


def export_onnx(module: torch.nn.Module, inputs: tuple, path: str | Path, names: list[str], fp16: bool,
                fp32_ops: tuple[str, ...] = ()) -> Path:
    """Export in float32; for `fp16` the graph is converted afterwards (inputs/outputs stay float32), keeping
    `fp32_ops` in float32. RAFT needs that: its pixel coordinates reach ~80 at 1/8 resolution, where float16
    steps are 1/16, i.e. 0.5 px at full resolution, so coordinate arithmetic and sampling stay float32 while
    the convolutions run in float16."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    module = module.eval().cuda()
    inputs = tuple(t.cuda() for t in inputs)
    with torch.inference_mode():
        torch.onnx.export(module, inputs, str(path), input_names=names, output_names=["out"], opset_version=17,
                          dynamo=False)
    if fp16:
        import onnx
        from onnxconverter_common import float16
        block = list(float16.DEFAULT_OP_BLOCK_LIST) + list(fp32_ops)
        model = float16.convert_float_to_float16(onnx.load(str(path)), keep_io_types=True, op_block_list=block)
        onnx.save(model, str(path))
    return path


def raft_corr_volume(onnx_path: str | Path) -> str:
    """Name of RAFT's full-resolution correlation volume: the data input of the first GridSample."""
    import onnx
    graph = onnx.load(str(onnx_path), load_external_data=False).graph
    return next(n.input[0] for n in graph.node if n.op_type == "GridSample")


def build_engine(onnx_path: str | Path, engine_path: str | Path, workspace_gb: float = 4.0,
                 pin_tensors: tuple[str, ...] = ()) -> Path:
    """Build a strongly typed engine. `pin_tensors` are intermediate tensors marked as extra outputs so they
    get a dedicated buffer for the whole inference.

    RAFT FP32 needs its correlation volume pinned (`raft_corr_volume`). Without it, TensorRT 11.4 returns
    wrong flow from the third refinement iteration on (P99 29 px after 12 iterations, any builder
    optimisation level), while ONNX Runtime on the same file is exact. The volume is computed once and read
    by every iteration. Marking it, or the first lookup that reads it, as an output restores the correct
    result; marking other tensors moves the error around. This looks like the volume's memory being reused
    while it is still needed, but that was not verified inside TensorRT."""
    import tensorrt as trt
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    parser = trt.OnnxParser(network, logger)
    if not parser.parse(Path(onnx_path).read_bytes()):
        raise RuntimeError("\n".join(str(parser.get_error(i)) for i in range(parser.num_errors)))
    if pin_tensors:
        found = {}
        for i in range(network.num_layers):
            layer = network.get_layer(i)
            for j in range(layer.num_outputs):
                if layer.get_output(j).name in pin_tensors:
                    found[layer.get_output(j).name] = layer.get_output(j)
        missing = set(pin_tensors) - set(found)
        if missing:
            raise RuntimeError(f"tensors to pin not found in network: {sorted(missing)}")
        for t in found.values():
            network.mark_output(t)
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, int(workspace_gb * (1 << 30)))
    blob = builder.build_serialized_network(network, config)
    if blob is None:
        raise RuntimeError(f"TensorRT build failed for {onnx_path}")
    Path(engine_path).write_bytes(bytes(blob))
    return Path(engine_path)


class TRTModule:
    """Runs a serialized engine on torch CUDA tensors (inputs by name, one output named 'out')."""

    def __init__(self, engine_path: str | Path):
        import tensorrt as trt
        self.runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
        self.engine = self.runtime.deserialize_cuda_engine(Path(engine_path).read_bytes())
        self.context = self.engine.create_execution_context()
        self.stream = torch.cuda.Stream()     # TensorRT adds a sync per call on the legacy default stream
        names = [self.engine.get_tensor_name(i) for i in range(self.engine.num_io_tensors)]
        self.inputs = [n for n in names if self.engine.get_tensor_mode(n) == trt.TensorIOMode.INPUT]
        dt = {trt.DataType.FLOAT: torch.float32, trt.DataType.HALF: torch.float16}
        self.in_dtype = {n: dt[self.engine.get_tensor_dtype(n)] for n in self.inputs}
        # 'out' plus any pinned intermediates (see `build_engine`); pinned buffers are written, never read
        self.buffers = {n: torch.empty(tuple(self.engine.get_tensor_shape(n)),
                                       dtype=dt[self.engine.get_tensor_dtype(n)], device="cuda")
                        for n in names if n not in self.inputs}
        for n, buf in self.buffers.items():
            self.context.set_tensor_address(n, buf.data_ptr())
        self.out = self.buffers["out"]

    def __call__(self, **tensors: torch.Tensor) -> torch.Tensor:
        keep = []
        for n in self.inputs:
            t = tensors[n].to(self.in_dtype[n]).contiguous()
            keep.append(t)
            self.context.set_tensor_address(n, t.data_ptr())
        caller = torch.cuda.current_stream()
        self.stream.wait_stream(caller)          # inputs are ready
        self.context.execute_async_v3(self.stream.cuda_stream)
        caller.wait_stream(self.stream)          # later torch work on the caller's stream sees the output
        for t in keep:
            t.record_stream(self.stream)         # keep converted inputs alive until the engine has read them
        return self.out


class TRTSegmenter:
    """Drop-in for `InstrumentSegmenter.prob` backed by a TensorRT engine."""

    def __init__(self, engine_path: str | Path):
        self.net = TRTModule(engine_path)
        self.mean = torch.tensor(IMAGENET_MEAN, device="cuda").view(3, 1, 1)
        self.std = torch.tensor(IMAGENET_STD, device="cuda").view(3, 1, 1)

    @torch.inference_mode()
    def prob(self, rgb: np.ndarray) -> np.ndarray:
        x = torch.from_numpy(rgb).cuda().permute(2, 0, 1).float() / 255
        x = ((x - self.mean) / self.std)[None]
        return self.net(x=x)[0].float().cpu().numpy()


class TRTStereo(RAFTStereo):
    """`RAFTStereo` with the flow network replaced by a TensorRT engine; the left-right check is unchanged."""

    def __init__(self, engine_path: str | Path, lr_check_px: float = 1.0):   # no torch model is loaded
        self.device, self.lr_check_px, self.last_vertical_flow = "cuda", lr_check_px, None
        self.net = TRTModule(engine_path)

    def _flow(self, a, b):
        return self.net(a=a, b=b).float()
