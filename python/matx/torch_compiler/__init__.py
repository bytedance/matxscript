minimum_torch_version = '2.0.0a0'

try:
    import torch

    assert torch.__version__ >= minimum_torch_version

except ModuleNotFoundError:
    print(f'torch is not installed. matx.inductor requires torch >= {minimum_torch_version}')
    raise
except AssertionError:
    print(f'matx.inductor requires torch >= {minimum_torch_version}')
    raise
