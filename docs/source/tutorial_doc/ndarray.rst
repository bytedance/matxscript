################################
NDArray
################################

Like numpy.array and torch.Tensor, NDArray is a data structure Matx uses to represent a tensor. Currently, we only support simple constructors and data manipulation, and NDArray is primarily used to perform data transfer from Matx to Pytorch/TensorFlow/TVM.

********************************
Constructor
********************************
The constructor of NDArray has 4 arguments:

+--------+------+--------------------------------------------------------------------------------------------------------------------------------------------------------+
| Args   | Type | Description                                                                                                                                            |
+========+======+========================================================================================================================================================+
| arr    | List | Construct a NDArray from arr                                                                                                                           |
+--------+------+--------------------------------------------------------------------------------------------------------------------------------------------------------+
| shape  | List | The shape of the NDArray. It is equivalent to np.array(arr).reshape(shape). If shape is [], the shape will be the same as arr.                         |
+--------+------+--------------------------------------------------------------------------------------------------------------------------------------------------------+
| dtype  | str  | The type of the data stored in NDArray. Currently, we support int32, int64, float32, float64, uint8 and bool.                                          |
+--------+------+--------------------------------------------------------------------------------------------------------------------------------------------------------+
| device | str  | The device where the NDArray is stored. Supported type: "cpu“, “cuda:%d” and “gpu:%d”, where d is the device number. The default device is "cpu".      |
+--------+------+--------------------------------------------------------------------------------------------------------------------------------------------------------+


Example 1: Pass in a flat list of size 4 and reshape it into a 2x2 matrix.
============================================================================================================

.. code-block:: python3 

    >>> import matx
    >>> nd = matx.NDArray([1,2,3,4], [2, 2], "int32")
    >>> nd
    [
    [ 1 2 ]
    [ 3 4 ]
    ]

    >>> nd.shape()
    [2, 2]
    >>> nd.dtype()
    'int32'
    >>>

Example 2: Pass in shape as [].
====================================

.. code-block:: python3 

    >>> import matx
    >>> nd = matx.NDArray([[1,2],[3,4]], [], "int32")
    >>> nd
    [
    [ 1 2 ]
    [ 3 4 ]
    ]

    >>> nd.shape()
    [2, 2]
    >>> nd.dtype()
    'int32'
    >>>

Please refer to the API documentation for more details.