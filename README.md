# LSTM CodeGen — C Code Generation for STM32

Four approaches to deploying a PyTorch LSTM Seq-to-Seq model as bare-metal C on STM32F746G-Discovery, benchmarked with Apple Clang 17 (-O3 -ffast-math).

## Interactive App

Open `webapp/index.html` in a browser.
Password: **lstmcode**

## Options

| Option | Method | Mean (µs) | Flash |
|--------|--------|-----------|-------|
| 1 | Hand-written C | **53.6** | 12 KB |
| 2 | MATLAB Coder via PyTorch Support Pkg (MLIR/TOSA) | 87.5 | 210 KB |
| 3 | MATLAB Coder via `importNetworkFromPyTorch` → dlnetwork | 93.5 | 215 KB |
| 4 | MATLAB Coder via ONNX → `importNetworkFromONNX` → dlnetwork | 93.2 | 215 KB |

All four options produce **100% identical predictions**.

## Model

- Architecture: LSTM(in=3, hidden=50) → FC(50→5) → ArgMax
- Sequence length: 75 timesteps
- Parameters: 11,255 (44 KB float32)
- Source: `LSTMSeqToSeqModel.pt2`
- Target: ARM Cortex-M7 @ 216 MHz, FPv5-D16

## Benchmark

```bash
python3 benchmark/benchmark_all.py
```

Requires: MATLAB R2026a (for Options 2–4 codegen), Apple Clang / GCC.

## Project Structure

```
option1_handwritten_c/      Hand-written LSTM inference + weight export
option2_matlab_pytorch_coder/  MATLAB Coder via PyTorch Support Package
option3_matlab_dlnetwork/   MATLAB Coder via importNetworkFromPyTorch
option4_onnx_matlab/        MATLAB Coder via ONNX import
benchmark/                  Benchmark harness + results JSON
report/                     PDF technical report + generator script
webapp/                     Interactive browser app (password: lstmcode)
```
