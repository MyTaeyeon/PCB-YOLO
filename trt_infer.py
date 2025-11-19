import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit  
import numpy as np
import threading

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

class TRTInfer:
    def __init__(self, engine_path: str, use_cuda_stream: bool = True):
        self.engine_path = engine_path
        self.runtime = trt.Runtime(TRT_LOGGER)
        self._load_engine()
        self.context = self.engine.create_execution_context()
        # if engine uses dynamic shapes, may need to set binding shapes per-input before enqueue
        self.stream = cuda.Stream() if use_cuda_stream else None
        # precompute host/device buffers
        self._allocate_buffers()
        # lock to make inference thread-safe
        self.lock = threading.Lock()

    def _load_engine(self):
        with open(self.engine_path, "rb") as f:
            engine_data = f.read()
        self.engine = self.runtime.deserialize_cuda_engine(engine_data)
        if self.engine is None:
            raise RuntimeError("Failed to deserialize TensorRT engine")

    def _allocate_buffers(self):
        self.host_inputs = []
        self.device_inputs = []
        self.host_outputs = []
        self.device_outputs = []
        # build a list of binding info dicts
        bindings = []

        # determine number of bindings by probing indices
        idx = 0
        while True:
            try:
                name = self.engine.get_binding_name(idx)
            except Exception:
                break
            is_input = self.engine.binding_is_input(idx)
            # get dtype and shape if available (some TRT builds may throw)
            try:
                dtype = trt.nptype(self.engine.get_binding_dtype(idx))
            except Exception:
                dtype = np.float32  # fallback
            try:
                shape = tuple(self.engine.get_binding_shape(idx))
            except Exception:
                # fallback: unknown shape, mark dynamic
                shape = None

            binfo = {"idx": idx, "name": name, "is_input": is_input,
                    "dtype": dtype, "shape": shape, "dynamic": False}
            # treat as dynamic if shape is None or any dim == -1
            if shape is None or any([s == -1 for s in shape]):
                binfo["dynamic"] = True
                binfo["host_mem"] = None
                binfo["device_mem"] = None
            else:
                # allocate host pinned and device mem
                nelems = int(np.prod(shape))
                nbytes = nelems * np.dtype(dtype).itemsize
                host_mem = cuda.pagelocked_empty(nelems, dtype)
                device_mem = cuda.mem_alloc(nbytes)
                binfo.update({"host_mem": host_mem, "device_mem": device_mem})
                if is_input:
                    self.host_inputs.append(host_mem)
                    self.device_inputs.append(device_mem)
                else:
                    self.host_outputs.append(host_mem)
                    self.device_outputs.append(device_mem)

            bindings.append(binfo)
            idx += 1

        if len(bindings) == 0:
            raise RuntimeError("No bindings found in engine (check engine file).")

        self.bindings = bindings


    def _ensure_dynamic_bindings(self, inputs):
        # inputs: dict {binding_name: np.array} OR single input np.array if single input model
        for idx_info in self.bindings:
            if not idx_info.get("dynamic", False):
                continue
            idx = idx_info["idx"]
            name = idx_info["name"]
            is_input = idx_info["is_input"]
            if name not in inputs:
                raise RuntimeError(f"Dynamic binding {name} expects data but not found in inputs")
            arr = inputs[name]
            arr = np.ascontiguousarray(arr)
            shape = arr.shape
            # set binding shape on context
            self.context.set_binding_shape(idx, shape)
            dtype = idx_info["dtype"]
            nbytes = arr.size * np.dtype(dtype).itemsize
            host_mem = cuda.pagelocked_empty(arr.size, dtype)
            np.copyto(np.frombuffer(host_mem, dtype=dtype).reshape(shape), arr.ravel().view(dtype).reshape(shape))
            device_mem = cuda.mem_alloc(nbytes)
            # replace binding info
            idx_info.update({"host_mem": host_mem, "device_mem": device_mem, "shape": shape, "dynamic": False})
            # also set in bindings list for actual enqueue
            self.bindings[idx] = idx_info

    def infer(self, input_array: np.ndarray, input_binding_name: str = None):
        """
        input_array: preprocessed input ready for engine (NCHW, dtype matching engine)
        input_binding_name: the binding name of the model input (if multiple)
        Returns: list of numpy arrays for outputs (in host order)
        """
        # prepare input dict keyed by binding name to support dyn
