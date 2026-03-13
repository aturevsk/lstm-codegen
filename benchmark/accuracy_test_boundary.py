#!/usr/bin/env python3
"""
accuracy_test_boundary.py
Find 100 inputs that produce non-Class-0 predictions (classes 1-4),
then verify all four C implementations agree on those boundary cases.

These are the hardest cases for numerical correctness — close logit margins
near decision boundaries where small floating-point differences could flip a class.

Run from project root:
    python3 benchmark/accuracy_test_boundary.py
"""

import os, sys, subprocess, numpy as np

PROJECT    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MATLAB_INC = "/Applications/MATLAB_R2026a.app/extern/include"
CC         = "gcc"
CFLAGS     = ["-O3", "-std=c99", "-ffast-math"]

HARNESS_OPT1 = r"""
#include <stdio.h>
#include <stdint.h>
#include "lstm_model.h"
int main(void) {
    float input[225]; int32_t output[75];
    for (int i = 0; i < 225; i++) scanf("%f", &input[i]);
    lstm_seq_to_seq(input, 75, output);
    for (int t = 0; t < 75; t++) printf("%d\n", output[t]);
    return 0;
}
"""
HARNESS_OPT2 = r"""
#include <stdio.h>
#include "codegen_opt2/lstm_infer_opt2.h"
#include "codegen_opt2/lstm_infer_opt2_initialize.h"
#include "codegen_opt2/lstm_infer_opt2_terminate.h"
int main(void) {
    float rowmaj[225], in1[225]; int pred[75];
    for (int i = 0; i < 225; i++) scanf("%f", &rowmaj[i]);
    for (int t = 0; t < 75; t++)
        for (int f = 0; f < 3; f++)
            in1[t + f*75] = rowmaj[t*3 + f];
    lstm_infer_opt2_initialize();
    lstm_infer_opt2(in1, pred);
    lstm_infer_opt2_terminate();
    for (int t = 0; t < 75; t++) printf("%d\n", pred[t]);
    return 0;
}
"""
HARNESS_OPT3 = r"""
#include <stdio.h>
#include "codegen_opt3/lstm_infer_opt3.h"
#include "codegen_opt3/lstm_infer_opt3_initialize.h"
#include "codegen_opt3/lstm_infer_opt3_terminate.h"
int main(void) {
    float input[225]; int pred[75];
    for (int i = 0; i < 225; i++) scanf("%f", &input[i]);
    lstm_infer_opt3_initialize();
    lstm_infer_opt3(input, pred);
    lstm_infer_opt3_terminate();
    for (int t = 0; t < 75; t++) printf("%d\n", pred[t]);
    return 0;
}
"""
HARNESS_OPT4 = r"""
#include <stdio.h>
#include "codegen_opt4/lstm_infer_opt4.h"
#include "codegen_opt4/lstm_infer_opt4_initialize.h"
#include "codegen_opt4/lstm_infer_opt4_terminate.h"
int main(void) {
    float input[225]; int pred[75];
    for (int i = 0; i < 225; i++) scanf("%f", &input[i]);
    lstm_infer_opt4_initialize();
    lstm_infer_opt4(input, pred);
    lstm_infer_opt4_terminate();
    for (int t = 0; t < 75; t++) printf("%d\n", pred[t]);
    return 0;
}
"""

SPECS = [
    {"opt":1,"label":"Hand-written C",
     "harness":HARNESS_OPT1,"harness_file":"option1_handwritten_c/main_acc2.c",
     "bin":"option1_handwritten_c/acc2_opt1",
     "srcs_extra":["option1_handwritten_c/lstm_model.c"],
     "flags":["-Ioption1_handwritten_c"]},
    {"opt":2,"label":"MATLAB Coder (PyTorch Pkg)",
     "harness":HARNESS_OPT2,"harness_file":"option2_matlab_pytorch_coder/main_acc2.c",
     "bin":"option2_matlab_pytorch_coder/acc2_opt2",
     "srcs_extra":["option2_matlab_pytorch_coder/codegen_opt2/lstm_infer_opt2.c",
                   "option2_matlab_pytorch_coder/codegen_opt2/lstm_infer_opt2_initialize.c",
                   "option2_matlab_pytorch_coder/codegen_opt2/lstm_infer_opt2_terminate.c"],
     "flags":["-Ioption2_matlab_pytorch_coder","-Ioption2_matlab_pytorch_coder/codegen_opt2",f"-I{MATLAB_INC}"]},
    {"opt":3,"label":"MATLAB Coder (dlnet/PT)",
     "harness":HARNESS_OPT3,"harness_file":"option3_matlab_dlnetwork/main_acc2.c",
     "bin":"option3_matlab_dlnetwork/acc2_opt3",
     "srcs_extra":["option3_matlab_dlnetwork/codegen_opt3/lstm_infer_opt3.c",
                   "option3_matlab_dlnetwork/codegen_opt3/callPredict.c",
                   "option3_matlab_dlnetwork/codegen_opt3/lstm_infer_opt3_data.c",
                   "option3_matlab_dlnetwork/codegen_opt3/lstm_infer_opt3_emxutil.c",
                   "option3_matlab_dlnetwork/codegen_opt3/lstm_infer_opt3_initialize.c",
                   "option3_matlab_dlnetwork/codegen_opt3/lstm_infer_opt3_terminate.c",
                   "option3_matlab_dlnetwork/codegen_opt3/minOrMax.c",
                   "option3_matlab_dlnetwork/codegen_opt3/rtGetInf.c",
                   "option3_matlab_dlnetwork/codegen_opt3/rtGetNaN.c",
                   "option3_matlab_dlnetwork/codegen_opt3/rt_nonfinite.c"],
     "flags":["-Ioption3_matlab_dlnetwork","-Ioption3_matlab_dlnetwork/codegen_opt3",f"-I{MATLAB_INC}"]},
    {"opt":4,"label":"MATLAB Coder (dlnet/ONNX)",
     "harness":HARNESS_OPT4,"harness_file":"option4_onnx_matlab/main_acc2.c",
     "bin":"option4_onnx_matlab/acc2_opt4",
     "srcs_extra":["option4_onnx_matlab/codegen_opt4/lstm_infer_opt4.c",
                   "option4_onnx_matlab/codegen_opt4/callPredict.c",
                   "option4_onnx_matlab/codegen_opt4/lstm_infer_opt4_data.c",
                   "option4_onnx_matlab/codegen_opt4/lstm_infer_opt4_emxutil.c",
                   "option4_onnx_matlab/codegen_opt4/lstm_infer_opt4_initialize.c",
                   "option4_onnx_matlab/codegen_opt4/lstm_infer_opt4_terminate.c",
                   "option4_onnx_matlab/codegen_opt4/minOrMax.c",
                   "option4_onnx_matlab/codegen_opt4/rtGetInf.c",
                   "option4_onnx_matlab/codegen_opt4/rtGetNaN.c",
                   "option4_onnx_matlab/codegen_opt4/rt_nonfinite.c"],
     "flags":["-Ioption4_onnx_matlab","-Ioption4_onnx_matlab/codegen_opt4",f"-I{MATLAB_INC}"]},
]


def compile_all(specs):
    for spec in specs:
        with open(os.path.join(PROJECT, spec["harness_file"]), "w") as f:
            f.write(spec["harness"])
        srcs = [os.path.join(PROJECT, spec["harness_file"])] + \
               [os.path.join(PROJECT, s) for s in spec["srcs_extra"]]
        cmd = [CC] + CFLAGS + srcs + ["-o", os.path.join(PROJECT, spec["bin"]), "-lm"] + spec["flags"]
        r = subprocess.run(cmd, capture_output=True, text=True, cwd=PROJECT)
        if r.returncode != 0:
            print(f"COMPILE FAILED opt{spec['opt']}:\n{r.stderr[:600]}")
            sys.exit(1)
        print(f"  Opt {spec['opt']}: OK")


def run_bin(bin_path, flat):
    stdin_str = "\n".join(f"{v:.8f}" for v in flat)
    r = subprocess.run([bin_path], input=stdin_str, capture_output=True,
                       text=True, timeout=10)
    return list(map(int, r.stdout.strip().split()))


def main():
    N_TARGET = 200   # boundary inputs to find
    N_SWEEP  = 5000  # max random inputs to sweep through opt1

    print("=" * 70)
    print("  LSTM CodeGen — Boundary Accuracy Validation")
    print(f"  Target: {N_TARGET} inputs that produce non-Class-0 predictions")
    print("=" * 70)

    # ── Compile ──────────────────────────────────────────────────────────────
    print("\n[1/4] Compiling test harnesses ...")
    compile_all(SPECS)
    bins = {s["opt"]: os.path.join(PROJECT, s["bin"]) for s in SPECS}

    # ── Sweep opt1 to find boundary inputs ───────────────────────────────────
    print(f"\n[2/4] Sweeping up to {N_SWEEP} random inputs through Opt 1 ...")
    print(f"      Looking for {N_TARGET} inputs with at least one non-Class-0 prediction ...\n")

    rng = np.random.default_rng(seed=1234)
    boundary_inputs = []
    class_found = {1:0, 2:0, 3:0, 4:0}
    swept = 0

    for _ in range(N_SWEEP):
        if len(boundary_inputs) >= N_TARGET:
            break
        scale = rng.choice([0.5, 1.0, 2.0, 3.0, 5.0])
        x = rng.standard_normal((75, 3)).astype(np.float32) * scale
        preds = run_bin(bins[1], x.flatten().tolist())
        swept += 1
        non_zero = [p for p in preds if p != 0]
        if non_zero:
            boundary_inputs.append((x, preds))
            for c in set(non_zero):
                class_found[c] = class_found.get(c, 0) + 1
            if len(boundary_inputs) % 10 == 0:
                print(f"  Found {len(boundary_inputs):3d}/{N_TARGET}  "
                      f"(swept {swept} inputs,  classes seen: {dict(class_found)})")

    print(f"\n  Swept {swept} inputs, found {len(boundary_inputs)} with non-Class-0 predictions")
    if len(boundary_inputs) < N_TARGET:
        print(f"  WARNING: only found {len(boundary_inputs)} boundary inputs in {N_SWEEP} sweeps")

    # ── Run all 4 options on boundary inputs ──────────────────────────────────
    actual = len(boundary_inputs)
    print(f"\n[3/4] Running {actual} boundary inputs through all 4 options ...")

    mismatches = []
    class_counts = {c: 0 for c in range(5)}
    total_preds = 0

    for i, (x, ref_preds) in enumerate(boundary_inputs):
        flat = x.flatten().tolist()
        results = {1: ref_preds}
        for opt_n in [2, 3, 4]:
            results[opt_n] = run_bin(bins[opt_n], flat)

        for cls in ref_preds:
            class_counts[cls] += 1
        total_preds += 75

        for opt_n in [2, 3, 4]:
            if results[opt_n] != results[1]:
                wrong = sum(a != b for a, b in zip(results[opt_n], results[1]))
                mismatches.append({"test": i+1, "opt": opt_n, "steps_wrong": wrong,
                                   "ref": results[1], "got": results[opt_n]})

        if (i + 1) % 10 == 0:
            status = "✓" if not any(m["test"]==i+1 for m in mismatches) else "✗"
            print(f"  Test {i+1:3d}/{actual}  {status}")

    # ── Results ───────────────────────────────────────────────────────────────
    print()
    print("=" * 70)
    print("  RESULTS")
    print("=" * 70)

    if not mismatches:
        print(f"\n  ✅  ALL {actual} BOUNDARY TESTS PASSED")
        print(f"  {total_preds:,} predictions compared — 100.00% match across all 4 options")
    else:
        n_fail = len(set(m["test"] for m in mismatches))
        print(f"\n  ❌  {n_fail}/{actual} tests had mismatches:")
        for m in mismatches[:10]:
            print(f"     Test {m['test']:3d}  Opt {m['opt']}  "
                  f"{m['steps_wrong']}/75 timesteps wrong")
            # show first differing timestep
            for t, (a, b) in enumerate(zip(m["ref"], m["got"])):
                if a != b:
                    print(f"       first diff at t={t}: ref={a} got={b}")
                    break

    print(f"\n  Class distribution across {actual} boundary tests (Opt1 reference):")
    for cls, count in sorted(class_counts.items()):
        pct = 100.0 * count / total_preds
        bar = "█" * int(pct / 2)
        print(f"    Class {cls}: {count:5d} / {total_preds}  ({pct:5.1f}%)  {bar}")

    non0_total = sum(v for k,v in class_counts.items() if k != 0)
    print(f"\n  Non-Class-0 predictions: {non0_total} / {total_preds} "
          f"({100*non0_total/total_preds:.1f}%)  ← vs 1.9% in random test")

    # ── Cleanup ───────────────────────────────────────────────────────────────
    for spec in SPECS:
        for f in [spec["bin"], spec["harness_file"]]:
            try: os.remove(os.path.join(PROJECT, f))
            except FileNotFoundError: pass

    return len(mismatches) == 0


if __name__ == "__main__":
    ok = main()
    sys.exit(0 if ok else 1)
