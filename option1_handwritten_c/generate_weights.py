#!/usr/bin/env python3
"""
generate_weights.py
====================
Extracts weights from a PyTorch ExportedProgram (LSTMSeqToSeqModel.pt2) and
writes them as a C header file (lstm_weights.h) suitable for inclusion in the
hand-written STM32 LSTM inference engine.

Also saves a test input / expected-output pair as NumPy .npy files so that
the C implementation can be validated against PyTorch.

Usage
-----
    python3 generate_weights.py [--model MODEL.pt2] [--out-dir DIR]

Defaults:
    --model    LSTMSeqToSeqModel.pt2   (in the current directory)
    --out-dir  .                        (write files next to this script)

Requirements
------------
    pip install torch numpy

Model assumptions
-----------------
The model is expected to expose the following state_dict keys (PyTorch default
LSTM naming convention):

    lstm.weight_ih_l0   shape [4*H, I]   combined input-hidden weights
    lstm.weight_hh_l0   shape [4*H, H]   combined hidden-hidden weights
    lstm.bias_ih_l0     shape [4*H]       input-hidden bias
    lstm.bias_hh_l0     shape [4*H]       hidden-hidden bias
    fc.weight           shape [C, H]      fully-connected weights
    fc.bias             shape [C]         fully-connected bias

Where:
    I = input_size  = 3
    H = hidden_size = 50
    C = num_classes = 5

Gate ordering within the PyTorch LSTM weight matrix is: i, f, g, o
(rows 0:H = input gate, H:2H = forget gate, 2H:3H = cell gate, 3H:4H = output gate)

Output header layout
--------------------
The C arrays are named and sized as follows (all row-major):

    lstm_Wi_i [H*I  = 150 ]  input  weights, input  gate
    lstm_Wi_f [H*I  = 150 ]  input  weights, forget gate
    lstm_Wi_g [H*I  = 150 ]  input  weights, cell   gate
    lstm_Wi_o [H*I  = 150 ]  input  weights, output gate

    lstm_Wh_i [H*H  = 2500]  hidden weights, input  gate
    lstm_Wh_f [H*H  = 2500]  hidden weights, forget gate
    lstm_Wh_g [H*H  = 2500]  hidden weights, cell   gate
    lstm_Wh_o [H*H  = 2500]  hidden weights, output gate

    lstm_b_i  [H    = 50  ]  combined bias (bias_ih + bias_hh), input  gate
    lstm_b_f  [H    = 50  ]  combined bias, forget gate
    lstm_b_g  [H    = 50  ]  combined bias, cell   gate
    lstm_b_o  [H    = 50  ]  combined bias, output gate

    fc_W      [C*H  = 250 ]  FC weight matrix, row-major
    fc_b      [C    = 5   ]  FC bias

Combining the two bias vectors (bias_ih + bias_hh) into one removes an
addition per gate per timestep in the C inference loop with no loss of
accuracy.
"""

import argparse
import os
import textwrap
from datetime import datetime, timezone

import numpy as np

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Extract LSTM weights to C header")
    p.add_argument("--model",   default="LSTMSeqToSeqModel.pt2",
                   help="Path to the ExportedProgram .pt2 file")
    p.add_argument("--out-dir", default=".",
                   help="Directory where lstm_weights.h and .npy files are written")
    p.add_argument("--seq-len", type=int, default=75,
                   help="Sequence length for the test input (default: 75)")
    p.add_argument("--seed",    type=int, default=42,
                   help="RNG seed for the random test input (default: 42)")
    return p.parse_args()

# ---------------------------------------------------------------------------
# Helper: format a NumPy array as a C initialiser list
# ---------------------------------------------------------------------------

def fmt_array(arr: np.ndarray, cols: int = 8) -> str:
    """Return a C initialiser-list string, 'cols' values per line."""
    flat = arr.ravel().astype(np.float32)
    parts = []
    for i, v in enumerate(flat):
        # Use %g-style hex for exact round-trip; fall back to %.8f for
        # readability.  We use %.8f here so the output is human-legible.
        parts.append(f"{v:.8f}f")
    lines = []
    for i in range(0, len(parts), cols):
        lines.append("    " + ", ".join(parts[i:i+cols]))
    return ",\n".join(lines)

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    # -----------------------------------------------------------------------
    # 1. Load the ExportedProgram
    # -----------------------------------------------------------------------
    import torch

    print(f"[1/5] Loading ExportedProgram from '{args.model}' ...")
    try:
        ep = torch.export.load(args.model)
    except Exception as exc:
        # Fallback: try torch.load (works for older pt2 saves / scripted models)
        print(f"      torch.export.load failed ({exc}); trying torch.load ...")
        ep = torch.load(args.model, map_location="cpu", weights_only=False)

    # -----------------------------------------------------------------------
    # 2. Extract state_dict
    #
    # ExportedProgram stores parameters in ep.state_dict; a scripted/traced
    # model exposes them via .state_dict() directly.
    # -----------------------------------------------------------------------
    print("[2/5] Extracting state_dict ...")
    if hasattr(ep, "state_dict"):
        sd = ep.state_dict
        if callable(sd):
            sd = sd()           # scripted model — call the method
    elif hasattr(ep, "named_parameters"):
        sd = {k: v for k, v in ep.named_parameters()}
    else:
        raise RuntimeError("Cannot extract state_dict from the loaded object.")

    # Convert all tensors to float32 NumPy arrays
    np_sd = {k: v.detach().float().numpy() for k, v in sd.items()}

    # Print discovered keys for debugging
    print("      Keys found:")
    for k, v in np_sd.items():
        print(f"        {k:40s}  {v.shape}")

    # -----------------------------------------------------------------------
    # 3. Extract and verify shapes
    # -----------------------------------------------------------------------
    print("[3/5] Verifying shapes ...")

    # Locate LSTM keys (handle both 'lstm.' and bare naming)
    def _get(d, *candidates):
        for k in candidates:
            if k in d:
                return d[k]
        raise KeyError(f"None of {candidates} found in state_dict")

    weight_ih = _get(np_sd, "lstm.weight_ih_l0", "weight_ih_l0")
    weight_hh = _get(np_sd, "lstm.weight_hh_l0", "weight_hh_l0")
    bias_ih   = _get(np_sd, "lstm.bias_ih_l0",   "bias_ih_l0")
    bias_hh   = _get(np_sd, "lstm.bias_hh_l0",   "bias_hh_l0")
    fc_weight = _get(np_sd, "fc.weight", "classifier.weight", "linear.weight")
    fc_bias   = _get(np_sd, "fc.bias",   "classifier.bias",   "linear.bias")

    # Derive dimensions from the weight matrices
    four_H, I = weight_ih.shape   # [4*H, I]
    H = four_H // 4
    C = fc_weight.shape[0]        # [C, H]

    assert four_H == 4 * H,        f"weight_ih rows ({four_H}) not divisible by 4"
    assert weight_hh.shape == (4*H, H), f"weight_hh shape mismatch: {weight_hh.shape}"
    assert bias_ih.shape   == (4*H,),   f"bias_ih shape mismatch: {bias_ih.shape}"
    assert bias_hh.shape   == (4*H,),   f"bias_hh shape mismatch: {bias_hh.shape}"
    assert fc_weight.shape == (C, H),   f"fc_weight shape mismatch: {fc_weight.shape}"
    assert fc_bias.shape   == (C,),     f"fc_bias shape mismatch: {fc_bias.shape}"

    print(f"      input_size={I}, hidden_size={H}, num_classes={C}")

    # -----------------------------------------------------------------------
    # 4. Split combined weight matrices by gate
    #
    # PyTorch concatenates gates in order [i, f, g, o] along axis 0.
    # After slicing we get H rows for each gate.
    # -----------------------------------------------------------------------
    Wi_i, Wi_f, Wi_g, Wi_o = np.split(weight_ih, 4, axis=0)  # each [H, I]
    Wh_i, Wh_f, Wh_g, Wh_o = np.split(weight_hh, 4, axis=0) # each [H, H]
    bi_i, bi_f, bi_g, bi_o = np.split(bias_ih, 4, axis=0)    # each [H]
    bh_i, bh_f, bh_g, bh_o = np.split(bias_hh, 4, axis=0)   # each [H]

    # Combine ih + hh biases so the inference loop does one addition instead of two
    b_i = bi_i + bh_i
    b_f = bi_f + bh_f
    b_g = bi_g + bh_g
    b_o = bi_o + bh_o

    # -----------------------------------------------------------------------
    # 5. Write lstm_weights.h
    # -----------------------------------------------------------------------
    out_h = os.path.join(args.out_dir, "lstm_weights.h")
    print(f"[4/5] Writing '{out_h}' ...")

    now_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    with open(out_h, "w", encoding="utf-8") as fh:
        fh.write(textwrap.dedent(f"""\
            /* Auto-generated weights from {os.path.basename(args.model)} */
            /* Generated: {now_utc} */
            /* LSTM: input_size={I}, hidden_size={H} | FC: {H}->{C} | batch_first=True, seq_len={args.seq_len} */
            #ifndef LSTM_WEIGHTS_H
            #define LSTM_WEIGHTS_H

            #define LSTM_INPUT_SIZE   {I}
            #define LSTM_HIDDEN_SIZE  {H}
            #define LSTM_NUM_CLASSES  {C}
            #define LSTM_SEQ_LEN      {args.seq_len}

            /* Input gate weights: Wi_i[HIDDEN][INPUT], Wh_i[HIDDEN][HIDDEN] */
            static const float lstm_Wi_i[{H * I}] = {{
            {fmt_array(Wi_i)}
            }};

            static const float lstm_Wi_f[{H * I}] = {{
            {fmt_array(Wi_f)}
            }};

            static const float lstm_Wi_g[{H * I}] = {{
            {fmt_array(Wi_g)}
            }};

            static const float lstm_Wi_o[{H * I}] = {{
            {fmt_array(Wi_o)}
            }};

            static const float lstm_Wh_i[{H * H}] = {{
            {fmt_array(Wh_i)}
            }};

            static const float lstm_Wh_f[{H * H}] = {{
            {fmt_array(Wh_f)}
            }};

            static const float lstm_Wh_g[{H * H}] = {{
            {fmt_array(Wh_g)}
            }};

            static const float lstm_Wh_o[{H * H}] = {{
            {fmt_array(Wh_o)}
            }};

            /* Combined bias (bias_ih + bias_hh) per gate [HIDDEN] */
            static const float lstm_b_i[{H}] = {{
            {fmt_array(b_i)}
            }};

            static const float lstm_b_f[{H}] = {{
            {fmt_array(b_f)}
            }};

            static const float lstm_b_g[{H}] = {{
            {fmt_array(b_g)}
            }};

            static const float lstm_b_o[{H}] = {{
            {fmt_array(b_o)}
            }};

            /* FC layer: fc_W[NUM_CLASSES][HIDDEN], fc_b[NUM_CLASSES] */
            static const float fc_W[{C * H}] = {{
            {fmt_array(fc_weight)}
            }};

            static const float fc_b[{C}] = {{
            {fmt_array(fc_bias)}
            }};

            #endif /* LSTM_WEIGHTS_H */
        """))

    print(f"      Written {os.path.getsize(out_h):,} bytes.")

    # -----------------------------------------------------------------------
    # 6. Generate test input and expected output
    #
    # We draw a random float32 input from N(0,1) with the specified seed,
    # run it through the PyTorch model, and save:
    #   test_input.npy          float32, shape [seq_len, input_size]
    #   test_output_expected.npy int64,   shape [seq_len]
    #
    # The C benchmark can load test_input.npy to verify numerical correctness.
    # -----------------------------------------------------------------------
    print(f"[5/5] Generating seed-{args.seed} test input / expected output ...")

    rng = np.random.default_rng(args.seed)
    x_np = rng.standard_normal((args.seq_len, I)).astype(np.float32)

    # Run the PyTorch model for the reference output
    ep.eval()
    with torch.no_grad():
        # lstm_seq_to_seq expects batch_first=True input: [batch=1, seq_len, input_size]
        x_t = torch.from_numpy(x_np).unsqueeze(0)   # [1, 75, 3]
        try:
            logits = ep(x_t)                          # ExportedProgram is callable
        except Exception:
            # Some ExportedPrograms expose a .module() wrapper
            logits = ep.module()(x_t)

        # logits shape: [1, seq_len, num_classes]  or  [seq_len, num_classes]
        if logits.dim() == 3:
            logits = logits.squeeze(0)               # [seq_len, num_classes]
        preds = logits.argmax(dim=-1).numpy()        # [seq_len]

    np.save(os.path.join(args.out_dir, "test_input.npy"),           x_np)
    np.save(os.path.join(args.out_dir, "test_output_expected.npy"), preds)

    print(f"      test_input.npy          : shape {x_np.shape}, dtype {x_np.dtype}")
    print(f"      test_output_expected.npy: shape {preds.shape}, dtype {preds.dtype}")
    print(f"      First 10 predictions    : {preds[:10].tolist()}")
    print()
    print("Done.  To use in main_benchmark.c:")
    print("  python3 -c \"")
    print("  import numpy as np")
    print("  x = np.load('test_input.npy')")
    print("  print(','.join(f'{v:.8f}f' for v in x.ravel()))")
    print("  \"")
    print("and replace the test_input[] array in main_benchmark.c with the output.")


if __name__ == "__main__":
    main()
