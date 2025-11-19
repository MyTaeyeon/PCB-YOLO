# inspect_engine.py
import tensorrt as trt
import sys, traceback

TRT_LOGGER = trt.Logger(trt.Logger.INFO)
engine_path = r"C:\Users\Default\Giang_space\meiko\PCB-YOLO\runs\detect\train\weights\model_fp16.trt"

def safe_print(title, v):
    print(f"\n=== {title} ===")
    try:
        print(v)
    except Exception as e:
        print("Error printing value:", e)

def try_call(fn, *args, **kw):
    try:
        return fn(*args, **kw)
    except Exception as e:
        print(f"EXCEPTION calling {getattr(fn,'__name__',str(fn))}: {e}")
        traceback.print_exc()
        return None

print("Loading engine:", engine_path)
with open(engine_path, "rb") as f:
    buf = f.read()

print("engine file size (bytes):", len(buf))

print("\n-- try deserialize --")
try:
    runtime = trt.Runtime(TRT_LOGGER)
    engine = runtime.deserialize_cuda_engine(buf)
    print("Engine object:", type(engine))
except Exception as e:
    print("DESERIALIZE FAILED:", e)
    traceback.print_exc()
    sys.exit(1)

# print attributes and dir
safe_print("dir(engine) (first 200 chars)", str(dir(engine))[:2000])
safe_print("repr(engine) (short)", repr(engine)[:1000])

# Try common ways to get binding count
for name in ("num_bindings", "num_bindings()", "get_nb_bindings", "nb_bindings", "get_nb_bindings()", "nbBindings"):
    print(f"\n-- trying attribute/call: {name} --")
    if hasattr(engine, "num_bindings"):
        try:
            print("engine.num_bindings:", engine.num_bindings)
        except Exception as e:
            print("engine.num_bindings raised:", e)
    if hasattr(engine, "get_nb_bindings"):
        try:
            print("engine.get_nb_bindings():", engine.get_nb_bindings())
        except Exception as e:
            print("engine.get_nb_bindings() raised:", e)

# Try brute-force get_binding_name for index range 0..31
print("\n-- brute force get_binding_name 0..31 --")
for i in range(0, 32):
    try:
        name = engine.get_binding_name(i)
        is_input = engine.binding_is_input(i)
        print(f"idx {i}: name={name}  input={is_input}")
        # try dtype/shape if available
        try:
            dtype = engine.get_binding_dtype(i)
            print("   dtype:", dtype)
        except Exception as e:
            print("   get_binding_dtype() raised:", e)
        try:
            shape = engine.get_binding_shape(i)
            print("   shape:", shape)
        except Exception as e:
            print("   get_binding_shape() raised:", e)
    except Exception as e:
        print(f"idx {i} -> exception: {e}")
        # don't break — continue to show all
        continue

# Try introspect other possibly helpful methods
for fn in ("get_binding_name", "binding_is_input", "get_binding_dtype", "get_binding_shape"):
    print(f"\n-- try call {fn} with idx=0 --")
    try:
        f = getattr(engine, fn)
        print("callable OK, result:", try_call(lambda: f(0)))
    except Exception as e:
        print(f"cannot call {fn}: {e}")

print("\n-- done inspect_engine.py --")
