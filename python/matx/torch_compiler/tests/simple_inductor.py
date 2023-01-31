import json

import matx
import torch


@matx.inductor(example_inputs=[torch.randn(5), torch.randn(5)])
def add_relu(a: matx.NDArray, b: matx.NDArray):
    c = a + b
    c = torch.nn.functional.relu(c)
    return c,


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

    add_relu(a_tensor, b_tensor, c_tensor)

    result_lst = c_tensor.tolist()

    return json.dumps(result_lst)


if __name__ == '__main__':
    print(f'Pytorch version {torch.__version__}')
    a = json.dumps([1, 2, 3, 4, 5])
    b = json.dumps([6, 7, 8, 9, 10])
    result = add_json(a, b)
    print(result)
