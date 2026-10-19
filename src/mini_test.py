"""Fast milestone check for mid-training 50m checkpoints: OOV name echo +
recall + dog-fact copy only. Full behavioral suite = chat_test.py at the end."""
import os, sys
sys.path.insert(0, os.path.dirname(__file__))
from chat_test import load, chat

stage = sys.argv[1] if len(sys.argv) > 1 else "50m"
model, tk, npar = load(stage)
print(f"== mini {stage}: {npar:,} params, ckpt step check")
for nm in ["joninibanana", "qwilmazok"]:
    h, mr = [], []
    a, _, mm = chat(model, tk, "fake", h, f"my name is {nm}", max_new=48, mem_rows=mr)
    ok = "OK" if nm in a.lower() else "FAIL"
    print(f"[U] my name is {nm}\n[A] {a}   <echo {ok}>\n[M] {mm}")
    h.append((f"my name is {nm}", a)); mr += mm
    a2, _, _ = chat(model, tk, "fake", h, "what's my name?", max_new=48, mem_rows=mr)
    ok2 = "OK" if nm in a2.lower() else "FAIL"
    print(f"[U] what's my name?\n[A] {a2}   <recall {ok2}>\n")
h, mr = [], []
a, _, mm = chat(model, tk, "sigma", h, "my dog is called vandross", max_new=48, mem_rows=mr)
ok = "OK" if "vandross" in a.lower() or any("v a n d r o s s" in w for w in mm) else "FAIL"
print(f"[U] my dog is called vandross\n[A] {a}   <fact {ok}>\n[M] {mm}")
h.append(("my dog is called vandross", a)); mr += mm
a2, _, _ = chat(model, tk, "sigma", h, "what's my dog's name?", max_new=48, mem_rows=mr)
ok2 = "OK" if "vandross" in a2.lower() else "FAIL"
print(f"[U] what's my dog's name?\n[A] {a2}   <recall {ok2}>")
