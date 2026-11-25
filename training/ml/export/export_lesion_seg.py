"""Export the lesion segmentation net to ExecuTorch (XNNPACK, FP32) into $AGROARMOUR_WORK/models/lesion/.

  python ml/export/export_lesion_seg.py --weights runs/lesion/L1/best.pt     # trained
  python ml/export/export_lesion_seg.py --smoke                              # untrained: lowering check only

Writes model.pte, io_spec.json, labels.json, and checks the .pte against eager PyTorch on random inputs.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "train"))
from common import MODELS  # noqa: E402
from lesion_net import CLASSES, LesionNet  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--imgsz", type=int, default=512)
    ap.add_argument("--name", default="lesion")
    a = ap.parse_args()
    from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
    from executorch.exir import to_edge_transform_and_lower
    from executorch.runtime import Runtime

    model = LesionNet(pretrained_backbone=False)
    if a.weights:
        model.load_state_dict(torch.load(a.weights, map_location="cpu")["model"])
    elif not a.smoke:
        sys.exit("--weights or --smoke")
    model.eval()
    x = torch.rand(1, 3, a.imgsz, a.imgsz)
    edge = to_edge_transform_and_lower(torch.export.export(model, (x,)), partitioner=[XnnpackPartitioner()])
    nodes = [n.target.__name__ if hasattr(n.target, "__name__") else str(n.target)
             for n in edge.exported_program().graph.nodes if n.op == "call_function"]
    undelegated = sorted({n for n in nodes if "executorch_call_delegate" not in n and "getitem" not in n})
    out = MODELS / (a.name + ("_smoke" if a.smoke else ""))
    out.mkdir(parents=True, exist_ok=True)
    (out / "model.pte").write_bytes(edge.to_executorch().buffer)

    method = Runtime.get().load_program(str(out / "model.pte")).load_method("forward")
    diffs, ms = [], []
    with torch.no_grad():
        for _ in range(3):
            xi = torch.rand(1, 3, a.imgsz, a.imgsz)
            t0 = time.perf_counter(); y = method.execute([xi])[0]; ms.append((time.perf_counter() - t0) * 1000)
            ref = model(xi)
            diffs.append(float((y - ref).abs().max()))
            agree = float((y.argmax(1) == ref.argmax(1)).float().mean())
    spec = {"input": [1, 3, a.imgsz, a.imgsz], "output": list(y.shape), "classes": CLASSES, "normalised_in_model": True}
    (out / "io_spec.json").write_text(json.dumps(spec, indent=1))
    (out / "labels.json").write_text(json.dumps(CLASSES))
    print(json.dumps({"pte": str(out / "model.pte"), "bytes": (out / "model.pte").stat().st_size, "undelegated_ops": undelegated,
                      "max_abs_diff": max(diffs), "argmax_agree_last": agree, "host_ms": [round(m) for m in ms], **spec}))


if __name__ == "__main__":
    main()
