# pycuts

<div style="display: flex; gap: 7px;">
  <a href="https://pypi.org/project/pycuts/" target="_blank"><img alt="PyPI" src="https://img.shields.io/pypi/v/pycuts"></a>
  <a href="https://huggingface.co/spaces/dwancin/" target="_blank"><img alt="Hugging Face" src="https://img.shields.io/badge/%F0%9F%A4%97%20HuggingFace-%23097EFF?style=flat&logoColor=black"></a>
  <a href="https://github.com/dwancin/pycuts" target="_blank"><img alt="Static Badge" src="https://img.shields.io/badge/Repository-white?logo=github&logoColor=black"></a>
</div>

**pycuts** is a small Python library that provides a collection of shortcut functions for common operations across various libraries, particularly for PyTorch and Hugging Face Hub environments.

## Installation

You can install pycuts directly from PyPI:

```bash
pip install pycuts
```

## Shortcuts

| **Function**          | **Return Type**       | **Description**                                                        |
|-----------------------|-----------------------|------------------------------------------------------------------------|
| `device()`            | `torch.device`        | Determines the best device to use (`cuda`, `mps`, or `cpu`).            |
| `gpu()`               | `bool`                | Returns `True` if a GPU ("cuda" or "mps") is available, otherwise `False`. |
| `torch_dtype()`       | `torch.dtype`         | Determines the appropriate tensor precision based on the device.        |
| `synchronize()`       | `None`                | Waits for all kernels in all streams on the given device to complete.   |
| `empty_cache()`       | `None`                | Clears the GPU memory to prevent out-of-memory errors.                  |
| `device_count()`      | `int`                 | Returns the number of available devices (e.g., number of GPUs).         |
| `manual_seed(seed)`   | `None`                | Sets the random seed for reproducible behavior across CPU/GPU.          |
| `is_spaces()`         | `bool`                | Returns `True` if running in a Hugging Face Space, otherwise `False`.   |
| `is_zero_gpu_space()` | `bool`                | Returns `True` if running in a zero-GPU Hugging Face Space.             |

## Examples

```python
import pycuts
print(f"current device is: {pycuts.device()}")
```
