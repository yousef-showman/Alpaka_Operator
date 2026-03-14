# Alpaka_Operator

ONNX operator implementations for GPU/CPU inference using the [alpaka](https://github.com/alpaka-group/alpaka) heterogeneous library, as part of the SOFIE project in ROOT/TMVA.

## Structure

```
Alpaka_Operator/
├── kernels/          # Alpaka kernel headers (one per ONNX operator)
│   └── Sigmoid_kernel.hxx
└── tests/            # Unit tests (to be added)
```

## Implemented Operators

| Operator | File |
|----------|------|
| Sigmoid  | `kernels/Sigmoid_kernel.hxx` |

## GSoC 2026

This repository is part of the evaluation for the project:
**ML Inference on heterogeneous architectures using SOFIE**
under CERN-HSF (ML4EP).
