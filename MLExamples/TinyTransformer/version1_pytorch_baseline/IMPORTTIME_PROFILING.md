# Python Import Time Profiling

IMPORTTIME_PROFILING.md from `HPCTrainingExamples/MLExamples/TinyTransformer/version1_pytorch_baseline` in the Training Examples repository.

## Overview

The `python -X importtime` flag provides detailed timing information about module imports during Python script execution. This is useful for identifying slow imports that can impact startup time.

## Basic Usage

```
python -X importtime script.py
```

This outputs a hierarchical tree showing import time for each module in microseconds.

## Output Format

```
import time: self [us] | cumulative | imported package
import time:       150 |        150 |   _frozen_importlib_external
import time:        89 |         89 |     _codecs
import time:       658 |        747 |   codecs
import time:       597 |        597 |   encodings.aliases
import time:      1521 |       2865 | encodings
```

- **self [us]**: Time spent in the module itself (microseconds)
- **cumulative**: Total time including all sub-imports (microseconds)
- **imported package**: Module name with indentation showing import hierarchy

## Example: Profiling TinyLlama V1

Redirect import timing output to a file for analysis:

```
python -X importtime tiny_llama_v1.py 2> import_times.txt
```

Analyze PyTorch import time:

```
python -X importtime -c "import torch" 2>&1 | grep -E "torch|time:"
```

## Common Import Time Bottlenecks

| Package | Typical Import Time | Notes |
|---------|-------------------|-------|
| PyTorch (torch) | 500ms - 2000ms | Loads CUDA/ROCm libraries, operator registry |
| Transformers | 300ms - 1000ms | Loads tokenizers, model architectures |
| DeepSpeed | 200ms - 800ms | Distributed training components |
| NumPy/SciPy | 50ms - 200ms | Optimized BLAS/LAPACK libraries |

## Generate Import Time Report

```
python -X importtime tiny_llama_v1.py 2>&1 | \
    grep "import time:" | \
    sort -k3 -n -r | \
    head -20 > top_imports.txt
```

## ROCm/PyTorch Considerations

Reduce logging overhead during import analysis:

```
AMD_LOG_LEVEL=0 MIOPEN_LOG_LEVEL=0 python -X importtime script.py
```

Skip GPU initialization during import analysis:

```
HIP_VISIBLE_DEVICES=-1 python -X importtime script.py
```

## Additional Resources

- [Python -X Options Documentation](https://docs.python.org/3/using/cmdline.html#id5)
- [PyTorch Performance Tuning Guide](https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html)
