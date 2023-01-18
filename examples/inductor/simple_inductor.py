import matx
import torch
import numpy as np
import json


def kernel(a: matx.NDArray, b: matx.NDArray):
    c = a + b
    c = torch.nn.functional.relu(c)
    return c,


add_kernel = matx.inductor(kernel, example_inputs=[
    torch.randn(5),
    torch.randn(5)
])


@matx.script
def add_json(a: str, b: str) -> str:
    """
    Assume a and b is a json containing 10 digits. We would like to add them and return another json
    """
    a_list = json.loads(a)
    b_list = json.loads(b)

    a_tensor = matx.NDArray(arr=a_list, shape=[5], dtype='float32')
    b_tensor = matx.NDArray(arr=b_list, shape=[5], dtype='float32')
    c_tensor = matx.NDArray(arr=b_list, shape=[5], dtype='float32')

    add_kernel(a_tensor, b_tensor, c_tensor)

    result_lst = c_tensor.tolist()

    return json.dumps(result_lst)


if __name__ == '__main__':
    print(f'Pytorch version {torch.__version__}')

    a_np = np.random.randn(5).astype(np.float32)
    b_np = np.random.randn(5).astype(np.float32)
    c_np = np.random.randn(5).astype(np.float32)

    a = matx.NDArray([], a_np.shape, str(a_np.dtype))
    a.from_numpy(a_np)

    b = matx.NDArray([], b_np.shape, str(b_np.dtype))
    b.from_numpy(b_np)

    c = matx.NDArray([], c_np.shape, str(c_np.dtype))
    c.from_numpy(c_np)

    print(a)
    print(b)
    print(c)

    print(kernel(a.torch(), b.torch()))

    d = add_kernel(a, b, c)
    print(c)

    a = json.dumps([1, 2, 3, 4, 5])
    b = json.dumps([6, 7, 8, 9, 10])
    result = add_json(a, b)
    print(result)
