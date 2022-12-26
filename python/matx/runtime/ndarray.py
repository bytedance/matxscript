# Copyright 2022 ByteDance Ltd. and/or its affiliates.
#
# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
# pylint: disable=invalid-name, unused-import
"""Runtime NDArray API"""
import ctypes
import numpy as np
from .. import _ffi

from .._ffi.base import _LIB, check_call, c_array, string_types
from .._ffi.runtime_ctypes import DataType, MATXScriptDevice, MATXArray, MATXArrayHandle
from .._ffi.runtime_ctypes import DataTypeCode, matx_shape_index_t

# pylint: disable=wrong-import-position
from . import _ffi_api
from typing import Union
from . import container
from .object import Object


__all__ = [
    "NDArray",
    "add",
    "sub",
    "div",
    "mul",
    "rand",
    "concatenate",
    "stack",
    "from_numpy",
    "from_dlpack",
]


def _ndarray_callback(obj):
    obj.__init_self__()
    return obj


@_ffi.register_object("runtime.NDArray", _ndarray_callback)
class NDArray(Object):

    """Lightweight NDArray implementation for matx runtime

    The structure is currently just a container for a multi-dimensional array, without defining various types of transformations and arithmetic methods.
    The goal of the structure is to serve as a bridge tool and other machine learning frameworks (pytorch tensorflow) for the conversion of multidimensional arrays

    Args:
        arr (List): Constructing the contents of an NDArray
        shape (List): Shape of the constructed NDArray
        dtype (str): The type of the constructed NDArray element, supporting "int32" "int64" "float32" "float64"

    Construction method 1
        arr is a one-dimensional List, the shape is not empty, producing NDArray with the content of arr and the shape of the given shape


        Examples:
            >>> import matx
            >>> nd = matx.NDArray([1,2,3,4,5,6], [3,2], "int32")
            >>> nd
            [
             [ 1 2 ]
             [ 3 4 ]
             [ 5 6 ]
            ]

    Construction method 2
        arr is a List of arbitrary dimensions, shape is an empty List, producing NDArray with the same shape as arr and the same content as arr

        Examples:
            >>> import matx
            >>> nd = matx.NDArray([[1, 2, 3,], [4, 5, 6]], [], "int32")
            >>> nd
            [
             [ 1 2 3 ]
             [ 4 5 6 ]
            ]

            >>> nd = matx.NDArray([1, 2, 3, 4, 5, 6], [], "int32")
            >>> nd
            [1, 2, 3, 4, 5, 6]

    Construction method 3
         arr is empty, shape is not empty, return a NDArray corresponding to the random initialization content of the shape

        Examples:
            >>> imoprt matx
            >>> nd = matx.NDArray([], [2, 3], "float32")
            >>> nd
            [
             [ 4.46171e-08 3.08678e-41 2.25609e-43 ]
             [ 0 8.15743e-06 4.5723e-41 ]
            ]
    """

    __hash__ = None

    def __init__(self,
                 arr: Union[int, float, list, container.List],
                 shape: Union[list, container.List],
                 dtype: str,
                 device: str = "cpu"):
        new_shape = [x for x in shape]
        self.__init_handle_by_constructor__(
            _ffi_api.NDArray,
            arr,
            new_shape,
            dtype,
            device)
        self.__init_self__()

    def __init_self__(self):
        tensor_handle = MATXArrayHandle()
        handle = self.handle
        if not isinstance(handle, ctypes.c_void_p):
            handle = ctypes.c_void_p(handle)
        check_call(_LIB.MATXScriptGetDLTensor(handle, ctypes.byref(tensor_handle)))
        self.tensor_handle = tensor_handle

    def __setstate__(self, state):
        assert isinstance(state, (bytes, bytearray))
        arr = _ffi_api.msgpack_loads(state)
        assert isinstance(arr, NDArray), "internal error"
        handle, code = _ffi.matx_script_api.steal_object_handle(arr)
        self.handle = handle
        self.type_code = code
        self.__init_self__()

    def __getstate__(self):
        return _ffi_api.msgpack_dumps(self)

    # -------------------- [begin] methods can be used in script --------------------
    def to_list(self):
        """Convert a NDArray to a matx.List corresponding to the shape

        Returns:
            matx.List

        Examples:
            >>> import matx
            >>> nd = matx.NDArray([], [2, 3], "float32")
            >>> nd
            [
             [ 1.35134e-24 4.55548e-41 1.35134e-24 ]
             [ 4.55548e-41 7.41622e+14 3.06142e-41 ]
            ]
            >>> l = nd.to_list()
            >>> l
            [[1.35134e-24, 4.55548e-41, 1.35134e-24], [4.55548e-41, 7.41622e+14, 3.06142e-41]]
        """
        return _ffi_api.NDArrayToList(self)

    def tolist(self):
        """Convert a NDArray to a matx.List corresponding to the shape

        Returns:
            matx.List

        Examples:
            >>> import matx
            >>> nd = matx.NDArray([], [2, 3], "float32")
            >>> nd
            [
             [ 1.35134e-24 4.55548e-41 1.35134e-24 ]
             [ 4.55548e-41 7.41622e+14 3.06142e-41 ]
            ]
            >>> l = nd.tolist()
            >>> l
            [[1.35134e-24, 4.55548e-41, 1.35134e-24], [4.55548e-41, 7.41622e+14, 3.06142e-41]]
        """
        return _ffi_api.NDArrayToList(self)

    def shape(self):
        """Returns the current NDArray's shape, unlike numpy, this is a method and not a property

        Returns:
            matx.List

        Examples:
            >>> import matx
            >>> x = matx.NDArray([], [3,2], "int32")
            >>> x.shape()
            [3, 2]
        """
        return [self.tensor_handle.contents.shape[i]
                for i in range(self.tensor_handle.contents.ndim)]

    def dtype(self):
        """Returns the dtype of the current NDArray as a string

        Returns:
            str: "int32" "int64" "float32" "float64"

        Examples:
            >>> import matx
            >>> x = matx.NDArray([], [3,2], "int32")
            >>> x.dtype()
            'int32'
        """
        return str(self.tensor_handle.contents.dtype)

    def dim(self):
        """Returns the number of array dimensions. Unlike numpy, this is a method and not a property.

        Returns:
            int

        Examples:
            >>> import matx
            >>> x = matx.NDArray([], [3,2], "int32")
            >>> x.dim()
            2

        """
        return self.tensor_handle.contents.ndim

    def is_contiguous(self):
        """Returns a int indicating if the underlying data is contiguous in memory.
        The continuity of array changes when its stride changes.

        Returns:
            int

        Examples:
            >>> import matx
            >>> x = matx.NDArray([1,2,3,4], [2,2], "int32")
            >>> y = x.transpose()
            >>> y.is_contiguous()
            0

        """
        return _ffi_api.NDArrayIsContiguous(self)

    def contiguous(self):
        """Returns a copy of the ndarray with contiguous memory if the adarray is not contiguous.
        Otherwise, the original one is returned.

        Returns:
            matx.NDArray

        Examples:
            >>> import matx
            >>> x = matx.NDArray([1,2,3,4], [2,2], "int32")
            >>> y = x.transpose()
            >>> z = y.contiguous()
            [[1, 3],
             [2, 4]]
            >>> z.is_contiguous()
            1

        """
        return _ffi_api.NDArrayContiguous(self)

    def reshape(self, newshape: Union[tuple, list]):
        return _ffi_api.NDArrayReshape(self, newshape)

    def squeeze(self, axis: tuple = ()):
        return _ffi_api.NDArraySqueeze(self, axis)

    def unsqueeze(self, dim: int):
        return _ffi_api.NDArrayUnsqueeze(self, dim)

    def device(self):
        """Returns the current NDArray device as a string"""
        if self.tensor_handle.contents.device.device_type == 1:
            return "cpu"
        return "%s:%s" % (
            MATXScriptDevice.MASK2STR[self.tensor_handle.contents.device.device_type],
            self.tensor_handle.contents.device.device_id)

    def stride(self):
        """Returns List of bytes to step in each dimension when traversing an array.

        Returns
             matx.List

        Examples:
            >>> import matx
            >>> x = matx.NDArray([1,2,3,4], [2,2], "int32")
            >>> y = x.transpose()
            >>> y.stride()
            [1, 2]
        """
        return _ffi_api.NDArrayStride(self)

    def __len__(self):
        return 0 if self.tensor_handle.contents.ndim <= 0 else self.tensor_handle.contents.shape[0]

    # -------------------- [end] methods can be used in script --------------------

    # @property
    # def device(self):
    #    """context of this array"""
    #    return self.tensor_handle.contents.device

    # def __hash__(self):
    #    return ctypes.cast(self.tensor_handle, ctypes.c_void_p).value

    # def __eq__(self, other):
    #    return self.same_as(other)

    # def __ne__(self, other):
    #    return not self.__eq__(other)

    def __repr__(self):
        res = "<matx.NDArray shape={0}, {1}>\n".format(
            tuple(self.shape()), self.tensor_handle.contents.device)
        res += self.asnumpy().__repr__()
        return res

    # def __str__(self):
    #     return str(self.asnumpy())

    def from_numpy(self, source_array):
        """Copy data from a numpy.ndarray to the current NDArray, requiring both to have the same size.
           Note! This method cannot be compiled for use in matx.script

        Args:
            source_array (numpy.ndarray)

        Raises:
            ValueError

        Returns:
            matx.NDArray: self

        Examples:
            >>> import matx
            >>> import numpy
            >>> arr = numpy.random.rand(2, 3)
            >>> arr
            array([[0.0402096 , 0.99905783, 0.85840985],
                [0.89764146, 0.25342801, 0.566187  ]])
            >>> nd = matx.NDArray([], arr.shape, str(arr.dtype))
            >>> nd.from_numpy(arr)
            [
             [0.0402096 , 0.99905783, 0.85840985]
             [0.89764146, 0.25342801, 0.566187  ]
            ]
        """

        shape, dtype = tuple(self.shape()), self.dtype()

        t = DataType(dtype)
        if t.lanes > 1:
            shape = shape + (t.lanes,)
            t.lanes = 1
            dtype = str(t)

        if source_array.shape != shape:
            raise ValueError("array shape do not match the shape of NDArray {0} vs {1}".format(
                source_array.shape, shape))
        source_array = np.ascontiguousarray(source_array, dtype=dtype)
        assert source_array.flags['C_CONTIGUOUS']
        data = source_array.ctypes.data_as(ctypes.c_void_p)
        nbytes = ctypes.c_size_t(source_array.size * source_array.dtype.itemsize)
        check_call(_LIB.MATXScriptArrayCopyFromBytes(self.tensor_handle, data, nbytes))
        return self

    def asnumpy(self):
        """Construct a numpy.ndarray from the current NDArray.
        Note! This method cannot be compiled for use in matx.script

        Returns:
            numpy.ndarray

        Examples:
            >>> import matx
            >>> import numpy
            >>> nd = matx.NDArray([[1,2,3],[4,5,6]], [], "int32")
            >>> nd
            [
             [1, 2, 3]
             [4, 5, 6]
            ]
            >>> arr = nd.asnumpy()
            >>> arr
            array([[1, 2, 3],
                [4, 5, 6]], dtype=int32)
        """
        t = DataType(self.dtype())
        shape, dtype = tuple(self.shape()), self.dtype()
        if t.lanes > 1:
            shape = shape + (t.lanes,)
            t.lanes = 1
            dtype = str(t)
        np_arr = np.empty(shape, dtype=dtype)
        assert np_arr.flags['C_CONTIGUOUS']
        data = np_arr.ctypes.data_as(ctypes.c_void_p)
        nbytes = ctypes.c_size_t(np_arr.size * np_arr.dtype.itemsize)
        check_call(_LIB.MATXScriptArrayCopyToBytes(self.tensor_handle, data, nbytes))
        return np_arr

    def numpy(self):
        """Construct a numpy.ndarray from the current NDArray.
        Note! This method cannot be compiled for use in matx.script

        Returns:
            numpy.ndarray

        Examples:
            >>> import matx
            >>> import numpy
            >>> nd = matx.NDArray([[1,2,3],[4,5,6]], [], "int32")
            >>> nd
            [
             [1, 2, 3]
             [4, 5, 6]
            ]
            >>> arr = nd.numpy()
            >>> arr
            array([[1, 2, 3],
                [4, 5, 6]], dtype=int32)
        """
        return self.asnumpy()

    # def from_array(self, source_array):
    #    if not isinstance(source_array, np.ndarray):
    #        try:
    #            source_array = np.array(source_array, dtype=dtype)
    #        except:
    #            raise TypeError('array must be an array_like data,' +
    #                            'type %s is not supported' % str(type(source_array)))
    #    return self.from_numpy(source_array)

    # def _copyto(self, target_nd):
    #    """Internal function that implements copy to target ndarray."""
    #    check_call(
    #        _LIB.MATXScriptArrayCopyFromTo(self.tensor_handle, target_nd.tensor_handle, None)
    #    )
    #    return target_nd

#    def copyto(self, target):
#        """Copy array to target
#
#        Parameters
#        ----------
#        target : NDArray
#            The target array to be copied, must have same shape as this array.
#        """
#        if isinstance(target, NDArray):
#            return self._copyto(target)
#        if isinstance(target, MATXScriptDevice):
#            res = empty(tuple(self.shape()), self.dtype(), target)
#            return self._copyto(res)
#        raise ValueError("Unsupported target type %s" % str(type(target)))

#    def tensor_addr(self):
#        return ctypes.cast(self.tensor_handle, ctypes.c_void_p).value

#    def same_as(self, other):
#        """Check object identity equality
#
#        Parameters
#        ----------
#        other : object
#            The other object to compare to
#
#        Returns
#        -------
#        same : bool
#            Whether other is same as self.
#        """
#        if not isinstance(other, NDArray):
#            return False
#        return self.tensor_addr == other.tensor_addr

#    def __setitem__(self, in_slice, value):
#        """Set ndarray value"""
#        if (not isinstance(in_slice, slice) or
#                in_slice.start is not None
#                or in_slice.stop is not None):
#            raise ValueError('Array only support set from numpy array')
#        if isinstance(value, NDArray):
#            if value.handle is not self.handle:
#                value.copyto(self)
#        elif isinstance(value, (np.ndarray, np.generic)):
#            self.copyfrom(value)
#        else:
#            raise TypeError('type %s not supported' % str(type(value)))
    def __getitem__(self, idx):
        length = len(self)
        if isinstance(idx, slice):
            start = idx.start if idx.start is not None else 0
            stop = idx.stop if idx.stop is not None else length
            step = idx.step if idx.step is not None else 1
            return _ffi_api.NDArrayGetSlice(self, start, stop, step)
        else:
            if idx < 0:
                idx += length
            if idx < -length or idx >= length:
                raise IndexError("Index out of range. size: {}, got index {}"
                                 .format(length, idx))
            return _ffi_api.NDArrayGetItem(self, idx)

    def __setitem__(self, idx, item):
        if isinstance(idx, slice):
            start = idx.start if idx.start is not None else 0
            stop = idx.stop if idx.stop is not None else len(self)
            return _ffi_api.NDArraySetSlice(self, start, stop, item)
        else:
            return _ffi_api.NDArraySetItem(self, idx, item)

    def transpose(self, axes=None):
        """Reverse or permute the axes of an array

        Args:
            axes (list of ints)

        Returns :
            the given with its axes permuted. A view is returned whenever possible

        """
        return _ffi_api.NDArrayTranspose(self, axes)

    def as_type(self, dtype: str):
        return _ffi_api.NDArrayAsType(self, dtype)

    def release(self):
        _ffi.matx_script_api.release_object_handle(self)

    def to_dlpack(self):
        return _ffi.matx_script_api._to_dlpack(self)

    def _copy_torch(self):
        import torch
        with torch.no_grad():
            return self._torch().clone()

    def _torch(self):
        _ffi_api.CurrentThreadStreamSync(
            self.tensor_handle.contents.device.device_type,
            self.tensor_handle.contents.device.device_id
        )
        import torch
        import torch.utils.dlpack
        with torch.no_grad():
            return torch.utils.dlpack.from_dlpack(self.to_dlpack())

    def torch(self, copy=True):
        """convert NDArray to torch.Tensor, make sure NDArray is synchronized

        Returns:
            torch.Tensor
        """
        if copy:
            return self._copy_torch()
        else:
            return self._torch()


def add(lhs, rhs):
    """Supports addition between NDArray and NDArray
       Supports addition between NDArray and numbers

    Supports broadcasting(https://numpy.org/doc/stable/user/basics.broadcasting.html)

    Specific use of the interface: matx.array.add

    Args:
        lhs (matx.NDArray or number): Left operand
        rhs (matx.NDArray or number): Right operand

    Returns:
        matx.NDArray

    Examples:
        >>> import matx
        >>> nd1 = matx.array.rand([2, 3]) # Analogous to numpy.random.rand(2, 3)
        >>> nd1
        [
         [ 0.767908 0.0856769 0.93522 ]
         [ 0.91489 0.347408 0.896765 ]
        ]

        >>> nd2 = matx.array.rand([2, 1, 3])
        >>> nd2
        [
         [
          [ 0.811942 0.00867774 0.2914 ]
         ]
         [
          [ 0.186276 0.200477 0.650708 ]
         ]
        ]

        >>> nd3 = matx.array.add(nd1, nd2)
        >>> nd3
        [
         [
          [ 1.57985 0.0943547 1.22662 ]
          [ 1.72683 0.356086 1.18816 ]
         ]
         [
          [ 0.954184 0.286154 1.58593 ]
          [ 1.10117 0.547885 1.54747 ]
         ]
        ]

        >>> nd3.shape()
        [2, 2, 3]
        >>>
        >>> nd3 = matx.array.add(nd1, 2.0)
        >>> nd3
        [
         [ 2.69112 2.45002 2.5514 ]
         [ 2.22683 2.73329 2.75127 ]
        ]
    """
    return _ffi_api.NDArrayAdd(lhs, rhs)


def sub(lhs, rhs):
    """Support subtraction between NDArray and NDArray
       Support subtraction between NDArray and numbers

    Support broadcasting(https://numpy.org/doc/stable/user/basics.broadcasting.html)

    Specific use of the interface: matx.array.sub

    Args:
        lhs (matx.NDArray or number): Left operand
        rhs (matx.NDArray or number): Right operand

    Returns:
        matx.NDArray

    Examples:
        >>> import matx
        >>> nd1 = matx.array.rand([2, 3]) # Analogous to numpy.random.rand(2, 3)
        >>> nd1
        [
         [ 0.767908 0.0856769 0.93522 ]
         [ 0.91489 0.347408 0.896765 ]
        ]

        >>> nd2 = matx.array.rand([2, 1, 3])
        >>> nd2
        [
         [
          [ 0.811942 0.00867774 0.2914 ]
         ]
         [
          [ 0.186276 0.200477 0.650708 ]
         ]
        ]

        >>> nd3 = matx.array.sub(nd1, nd2)
        >>> nd3
        [
         [
          [ -0.120825 0.441345 0.260003 ]
          [ -0.58511 0.724615 0.459874 ]
         ]
         [
          [ 0.504841 0.249546 -0.0993055 ]
          [ 0.0405553 0.532816 0.100566 ]
         ]
        ]

        >>> nd3.shape()
        [2, 2, 3]
        >>>
        >>> nd3 = matx.array.sub(nd1, 2.0)
        >>> nd3
        [
         [ -1.30888 -1.54998 -1.4486 ]
         [ -1.77317 -1.26671 -1.24873 ]
        ]
    """
    return _ffi_api.NDArraySub(lhs, rhs)


def div(lhs, rhs):
    """Support division between NDArray and NDArray
       Support division between NDArray and numbers


    Supports broadcasting(https://numpy.org/doc/stable/user/basics.broadcasting.html)

    Specific use of the interface: matx.array.div

    Args:
        lhs (matx.NDArray or number): Left operand
        rhs (matx.NDArray or number): Right operand

    Returns:
        matx.NDArray

    Examples:
        >>> import matx
        >>> nd1 = matx.array.rand([2, 3]) # Analogous to numpy.random.rand(2, 3)
        >>> nd1
        [
         [ 0.767908 0.0856769 0.93522 ]
         [ 0.91489 0.347408 0.896765 ]
        ]

        >>> nd2 = matx.array.rand([2, 1, 3])
        >>> nd2
        [
         [
          [ 0.811942 0.00867774 0.2914 ]
         ]
         [
          [ 0.186276 0.200477 0.650708 ]
         ]
        ]

        >>> nd3 = matx.array.div(nd1, nd2)
        >>> nd3
        [
         [
          [ 0.85119 51.8594 1.89225 ]
          [ 0.279369 84.5028 2.57815 ]
         ]
         [
          [ 3.71017 2.24476 0.847389 ]
          [ 1.21772 3.65775 1.15455 ]
         ]
        ]

        >>> nd3.shape()
        [2, 2, 3]
        >>>
        >>> nd3 = matx.array.div(nd1, 2)
        >>> nd3
        [
         [ 0.345559 0.225011 0.275701 ]
         [ 0.113416 0.366647 0.375637 ]
        ]
    """
    return _ffi_api.NDArrayDiv(lhs, rhs)


def mul(lhs, rhs):
    """Support multiplication between NDArray and NDArray
       Support multiplication between NDArray and numbers

    support broadcasting(https://numpy.org/doc/stable/user/basics.broadcasting.html)

    Specific use of the interface: matx.array.mul

    Args:
        lhs (matx.NDArray or number): Left operand
        rhs (matx.NDArray or number): Right operand

    Returns:
        matx.NDArray

    Examples:
        >>> import matx
        >>> nd1 = matx.array.rand([2, 3]) # Analogous to numpy.random.rand(2, 3)
        >>> nd1
        [
         [ 0.767908 0.0856769 0.93522 ]
         [ 0.91489 0.347408 0.896765 ]
        ]

        >>> nd2 = matx.array.rand([2, 1, 3])
        >>> nd2
        [
         [
          [ 0.811942 0.00867774 0.2914 ]
         ]
         [
          [ 0.186276 0.200477 0.650708 ]
         ]
        ]

        >>> nd3 = matx.array.mul(nd1, nd2)
        >>> nd3
        [
         [
          [ 0.561147 0.00390518 0.160679 ]
          [ 0.184174 0.00636333 0.218921 ]
         ]
         [
          [ 0.128739 0.0902191 0.358802 ]
          [ 0.0422533 0.147008 0.48886 ]
         ]
        ]

        >>> nd3.shape()
        [2, 2, 3]
        >>>
        >>> nd3 = matx.array.mul(nd1, 2)
        >>> nd3
        [
         [ 1.38223 0.900045 1.1028 ]
         [ 0.453663 1.46659 1.50255 ]
        ]
    """
    return _ffi_api.NDArrayMul(lhs, rhs)


def rand(shape):
    """Returns a NDArray filled with random numbers from a uniform distribution on the interval[0, 1), similar to numpy.rando.rand

    Analogous to: matx.array.rand

    Args:
        shape (List)

    Returns:
        matx.NDArray

    Examples:
        >>> import matx
        >>> nd = marx.nd_rand([2, 3])
        >>> nd
       [
        [ 0.145728 0.357696 0.574777 ]
        [ 0.437011 0.0172242 0.704895 ]
       ]

       >>> nd.shape()
       [2, 3]
       >>> nd.dtype()
       'float32'
    """
    return _ffi_api.NDArrayRand(shape)


def concatenate(seq, axes=0):
    """Concatenates the given sequence of seq tensors in the given dimension, Similar to numpy.concatenate

    Specific use of the interface: matx.array.concatenate

    Args:
        seq (List[matx.NDArray] or Tuple[matx.NDArray])
        axes (int, optional): dim

    Returns:
        matx.NDArray

    Examples:
        >>> import matx
        >>> nd = matx.NDArray([0, 1, 2, 3, 4, 5, 6, 7], [2, 2, 2], "int32")
        >>> matx.array.concatenate([nd, nd], 0)
        [
         [
          [ 0 1 ]
          [ 2 3 ]
         ]
         [
          [ 4 5 ]
          [ 6 7 ]
         ]
         [
          [ 0 1 ]
          [ 2 3 ]
         ]
         [
          [ 4 5 ]
          [ 6 7 ]
         ]
        ]

        >>> matx.array.concatenate([nd, nd], 1)
        [
         [
          [ 0 1 ]
          [ 2 3 ]
          [ 0 1 ]
          [ 2 3 ]
         ]
         [
          [ 4 5 ]
          [ 6 7 ]
          [ 4 5 ]
          [ 6 7 ]
         ]
        ]

        >>> matx.array.concatenate([nd, nd], 2)
        [
         [
          [ 0 1 0 1 ]
          [ 2 3 2 3 ]
         ]
         [
          [ 4 5 4 5 ]
          [ 6 7 6 7 ]
         ]
        ]
    """
    return _ffi_api.NDArrayConcatenate(seq, axes)


def stack(seq, axes=0):
    """Concatenates a sequence of NDArray along a new dimension.

    Specific use of the interface: matx.array.stack

    Args:
        seq (List[matx.NDArray] or Tuple[matx.NDArray])
        axes (int, optional): dim

    Returns:
        matx.NDArray

    Examples:
        >>> import matx
        >>> nd = matx.NDArray([0, 1, 2, 3, 4, 5, 6, 7], [2, 2, 2], "int32")
        >>> matx.array.stack([nd, nd], 0)
        [
         [
          [
            [ 0 1 ]
            [ 2 3 ]
          ]
          [
            [ 4 5 ]
            [ 6 7 ]
          ]
         ]
         [
          [
            [ 0 1 ]
            [ 2 3 ]
          ]
          [
            [ 4 5 ]
            [ 6 7 ]
          ]
         ]
        ]

        >>> matx.array.stack([nd, nd], 1)
        [
         [
          [
            [ 0 1 ]
            [ 2 3 ]
          ]
          [
            [ 0 1 ]
            [ 2 3 ]
          ]
         ]
         [
          [
            [ 4 5 ]
            [ 6 7 ]
          ]
          [
            [ 4 5 ]
            [ 6 7 ]
          ]
         ]
        ]

        >>> matx.array.stack([nd, nd], 2)
        [
         [
          [
            [ 0 1 ]
            [ 0 1 ]
          ]
          [
            [ 2 3 ]
            [ 2 3 ]
          ]
         ]
         [
          [
            [ 4 5 ]
            [ 4 5 ]
          ]
          [
            [ 6 7 ]
            [ 6 7 ]
          ]
         ]
        ]
    """
    return _ffi_api.NDArrayStack(seq, axes)


# def device(dev_type, dev_id=0):
#    """Constructs the MATX context with the given dev_type dev_id.
#       Note! This method cannot be compiled for use in matx.script
#
#    Args
#        dev_type (int or str): The device type mask or name of the device.
#
#        dev_id (int, optional): The integer device id
#
#    Returns
#        MATXScriptDevice
#
#    Examples
#        >>> import matx
#        >>> assert matx.context("cpu", 1) == matx.cpu(1)
#        >>> assert matx.context("gpu", 0) == matx.gpu(0)
#        >>> assert matx.context("cuda", 0) == matx.gpu(0)
#    """
#    if isinstance(dev_type, string_types):
#        if '-device=micro_dev' in dev_type:
#            dev_type = MATXScriptDevice.STR2MASK['micro_dev']
#        else:
#            dev_type = dev_type.split()[0]
#            if dev_type not in MATXScriptDevice.STR2MASK:
#                raise ValueError("Unknown device type %s" % dev_type)
#            dev_type = MATXScriptDevice.STR2MASK[dev_type]
#    return MATXScriptDevice(dev_type, dev_id)
#
#
# def cpu(dev_id=0):
#    """Constructing the cpu device.
#       Note! This method cannot be compiled for use in matx.script
#
#    Args:
#        dev_id (int, optional): Construct cpu device, device_id is 0 by default
#
#    Returns
#        MATXScriptDevice
#    """
#    return MATXScriptDevice(1, dev_id)
#
#
# def gpu(dev_id=0):
#    """Constructing the gpu device.
#       Note! This method cannot be compiled for use in matx.script
#
#    Args:
#        dev_id (int, optional): Construct gpu device, device_id is 0 by default
#
#    Returns:
#        [type]: [description]
#    """
#    return MATXScriptDevice(2, dev_id)
#
#
# def ndarray(arr: List, shape: List, dtype: str):
#    """Construct the module method of matx.NDArray
#       Note! This method cannot be compiled for use in matx.script
#
#    Args:
#        arr (List)
#        shape (List)
#        dtype (str)
#
#    Returns:
#        matx.NDArray
#    """
#    return _ffi_api.NDArray(arr, shape, dtype)
#

def from_numpy(arr, device="cpu"):
    """Construct a module method for matx.NDArray from numpy.ndarray.
       Note! This method cannot be compiled for use in matx.script

    Args:
        arr (numpy.ndarray)
        device (MATXScriptDevice, optional
             The device context to create the array)

    Returns:
        matx.NDArray

    Examples:
        >>> import numpy
        >>> import matx
        >>> arr = numpy.random.rand(2, 3)
        >>> nd = matx.array.from_numpy(arr)
        >>> nd
        [
         [0.39801043, 0.5075788 , 0.58423371]
         [0.34059181, 0.90339341, 0.72762747]
        ]
    """
    return NDArray(
        container.List(), container.List(arr.shape), str(arr.dtype), device
    ).from_numpy(arr)

# def numpyasarray(np_data):
#    """Return a MATXArray representation of a numpy array.
#    """
#    data = np_data
#    assert data.flags['C_CONTIGUOUS']
#    arr = MATXArray()
#    shape = c_array(matx_shape_index_t, data.shape)
#    arr.data = data.ctypes.data_as(ctypes.c_void_p)
#    arr.shape = shape
#    arr.strides = None
#    arr.dtype = DataType(np.dtype(data.dtype).name)
#    arr.ndim = data.ndim
#    # CPU device
#    arr.device = context(1, 0)
#    return arr, shape


# def empty(shape, dtype="float32", device=context(1, 0)):
#    """Create an empty array given shape and device
#
#    Parameters
#    ----------
#    shape : tuple of int
#        The shape of the array
#
#    dtype : type or str
#        The data type of the array.
#
#    device : MATXScriptDevice
#        The context of the array
#
#    Returns
#    -------
#    arr : nd.NDArray
#        The array matx supported.
#    """
#    shape = c_array(matx_shape_index_t, shape)
#    ndim = ctypes.c_int(len(shape))
#    handle = ctypes.c_void_p()
#    dtype = DataType(dtype)
#    check_call(_LIB.NDArrayAlloc(
#        shape, ndim,
#        ctypes.c_int(dtype.type_code),
#        ctypes.c_int(dtype.bits),
#        ctypes.c_int(dtype.lanes),
#        device.device_type,
#        device.device_id,
#        ctypes.byref(handle)))
#    obj = NDArray.__new__(NDArray)
#    obj.handle = handle
#    obj.__init_self__()
#    return obj


def from_dlpack(dltensor):
    """Produce an array from a DLPack tensor without memory copy.
    Retreives the underlying DLPack tensor's pointer to create an array from the
    data. Removes the original DLPack tensor's destructor as now the array is
    responsible for destruction.

    Parameters
    ----------
    dltensor : DLPack tensor
        Input DLManagedTensor, can only be consumed once.

    Returns
    -------
    arr: nd.NDArray
        The array view of the tensor data.
    """
    return _ffi.matx_script_api._from_dlpack(dltensor)


# def array(arr, device=cpu(0)):
#    """Create an array from source arr.
#
#    Parameters
#    ----------
#    arr : numpy.ndarray
#        The array to be copied from
#
#    device : MATXScriptDevice, optional
#        The device context to create the array
#
#    Returns
#    -------
#    ret : NDArray
#        The created array
#    """
#    if not isinstance(arr, (np.ndarray, NDArray)):
#        arr = np.array(arr)
#    return empty(arr.shape, arr.dtype, device).copyfrom(arr)


# Register back to FFI
# _set_class_ndarray(NDArray)
