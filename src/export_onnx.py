"""Export checkpoint -> ONNX (for browser inference via onnxruntime-web).
Exports: logits [1,T,V], pooled [1,D] (memory embeddings), intent_logits [1,5]."""
import os, sys, json
import torch
sys.path.insert(0, os.path.dirname(__file__))
from model import Transformer, configs
from tokenizer import Tokenizer

HERE = os.path.dirname(__file__)
OUT = os.path.join(HERE, "..", "out")
DATA = os.path.join(HERE, "..", "data")
DOCS_MODELS = os.path.join(HERE, "..", "docs", "models")

def export(stage):
    ck = torch.load(os.path.join(OUT, f"{stage}.pt"), map_location="cpu", weights_only=False)
    cfg = configs()[stage]
    m = Transformer(cfg)
    m.load_state_dict(ck["model"])
    m.eval()

    class Wrapper(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m
        def forward(self, idx):
            o = self.m(idx)
            return o["logits"], o["pooled"], o["intent_logits"]

    w = Wrapper(m)
    dummy = torch.randint(0, cfg.vocab_size, (1, 16))
    os.makedirs(DOCS_MODELS, exist_ok=True)
    path = os.path.join(DOCS_MODELS, f"{stage}.onnx")
    torch.onnx.export(
        w, (dummy,), path,
        input_names=["idx"],
        output_names=["logits", "pooled", "intent_logits"],
        dynamic_axes={"idx": {1: "T"}, "logits": {1: "T"}},
        opset_version=14, do_constant_folding=True,
    )
    size = os.path.getsize(path) / 1e6
    # inline external weights into a single .onnx file (PWA-friendly)
    import onnx
    from onnx.external_data_helper import convert_model_from_external_data
    m2 = onnx.load(path, load_external_data=True)
    for t in m2.graph.initializer:
        if t.data_location == onnx.TensorProto.EXTERNAL:
            t.data_location = onnx.TensorProto.DEFAULT
            t.ClearField("external_data")
    onnx.save(m2, path)
    size = os.path.getsize(path) / 1e6
    # validate
    try:
        import onnxruntime as ort
        sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
        tk = Tokenizer.load(os.path.join(DATA, "vocab.json"))
        ids = tk.encode("<bos><pers:sigma><user>hi<ai>")
        lg, pl, il = sess.run(None, {"idx": [ids]})
        nxt = int(lg[0, -1].argmax())
        print(f"OK {stage}.onnx  {size:.1f} MB  next-token={tk.itos[nxt]!r}  pooled={pl.shape} intent={il.shape}")
    except ImportError:
        print(f"OK {stage}.onnx  {size:.1f} MB  (onnxruntime not installed, skipped validation)")
    return path

if __name__ == "__main__":
    for s in sys.argv[1:]:
        export(s)
