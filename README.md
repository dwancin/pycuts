# `pycuts`
<div style="display: flex; gap: 7px;">
  <a href="https://pypi.org/project/pycuts/" target="_blank"><img alt="PyPI" src="https://img.shields.io/pypi/v/pycuts"></a>
  <a href="https://huggingface.co/spaces/dwancin/" target="_blank"><img alt="Hugging Face" src="https://img.shields.io/badge/%F0%9F%A4%97%20HuggingFace-%23097EFF?style=flat&logoColor=black"></a>
  <a href="https://github.com/dwancin/pycuts" target="_blank"><img alt="Static Badge" src="https://img.shields.io/badge/Repository-white?logo=github&logoColor=black"></a>
</div>

**pycuts** is a small Python library that provides a collection of shortcut functions for common operations across various libraries.

## Installation

You can install pycuts directly from PyPI:

```bash
pip install pycuts
```

## Shortcuts

| **Function**          | **Type**       | **Description**                                                        |
|-----------------------|----------------|------------------------------------------------------------------------|
| `device()`            | `torch.device` | Determines which device to use. ("cuda", "mps", or "cpu").             |
| `gpu()`               | `bool`         | Whether a GPU ("cuda" or "mps") is available or not.                   |
| `synchronize()`       | `torch.device` | Waits for all kernels in all streams on the given device to complete.  |
| `empty_cache()`       | `torch.device` | Clears the GPU memory to prevent out-of-memory errors.                 |
| `device_count()`      | `int`          | Returns the number of available devices.                               |
| `manual_seed()`       | `int`          | Sets the seed for generating random numbers for reproducible behavior. |
| `is_spaces()`         | `bool`         | Checks if the current script is running on a Hugging Face Space.       |
| `is_zero_gpu_space()` | `bool`         | Checks if the Hugging Face Space runs on Zero GPU.                     |
