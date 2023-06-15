# Copyright 2022 ByteDance Ltd. and/or its affiliates.
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Acknowledgement: The structure of the graph is inspired by AITemplate.
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
# pylint: disable=invalid-name
from __future__ import annotations

from abc import ABC, abstractmethod
from numbers import Number
from pprint import pformat
from typing import Any, Dict, Iterable, List, Optional, Set, Union

from matx.kernel import symbolic
from matx.kernel.op_registry import OP_REGISTRY

import math
import copy
import sympy


# TODO: Introduce networkx

class Node(ABC):
    """Base class of Tensor, Operator, etc."""

    def __init__(self) -> None:
        """
        Initializes self._attrs field, which is a dict that stores
        all attributes for this Node.
        Basic attributes include:
            * name: str, name of the node.
            * depth: int, depth of the node in a graph. None if this is not applicable.
            * nop: bool, marks whether this node is a no-operation.
        Child classes add their own attributes to this dict.
        """
        super().__init__()
        self._attrs: Dict[str, Any] = {"name": None, "depth": 0, "nop": False}

    def __str__(self) -> str:
        """Returns a string version of this object."""
        return pformat(self._attrs, indent=2, depth=2)

    def __repr__(self) -> str:
        """Returns a string containing a printable representation of this object."""
        return self.__str__()

    @abstractmethod
    def pseudo_code(self, with_shape: bool = False) -> str:
        """Returns a string containing pseudo code of this object.

        Parameters
        ----------
        with_shape: bool
            Marks whether to include shape info in the returned pseudo code.

        Returns
        ----------
        str
            Pseudo code.
        """
        pass


class IntVar(Node):
    """
    An IntVar represents a dynamic dimension.
    IntVar and IntImm (see below) are used together to represent a Tensor's shape.

    IntVar supports basic arithmetic operations, and returns the most conservative
    IntVar w.r.t. range of _attrs["values"].
    """

    def __init__(
            self,
            values: List[int],
            name: str = None,
            symbolic_value: Optional[sympy.Basic] = None,
    ) -> None:
        """Initializes an IntVar.

        Parameters
        ----------
        values : List[int]
            A list of possible values of this dynamic dimension.
            len(values) must be >= 2.

            When len(values) == 2, the values are treated as a lower bound and an upper bound.
            Both upper bound and lower bound are inclusive.
            This is the default use case.

            When len(values) > 2, the first / last values are treated as lower / upper bounds,
            and the other values are used for internal profiling purpose.
            This is a legacy use case.

        name : str, optional
            Name of this dimension, by default None.
            This field must be set for dims which are used by input tensors.

        symbolic_value: sympy.Basic, optional
            The symbolic value for this IntVar. If None is provided, we will generate a symbol for this IntVar.
        """
        super().__init__()
        self._attrs["name"] = name

        if values is None or len(values) < 2:
            raise RuntimeError(
                "IntVar 'values' field must have at least 2 values! values: {}, name: {}".format(
                    values, name
                )
            )
        if min(values) < 0:
            raise RuntimeError(
                "IntVar has < 0 value! values: {}, name: {}".format(values, name)
            )
        self._attrs["values"] = sorted(set(values))
        if len(self._attrs["values"]) == 1:
            self._attrs["symbolic_value"] = self._attrs["values"][0]
            self._attrs["values"] = self._attrs["values"] * 2
        else:
            if symbolic_value is None:
                symbolic_value = symbolic.create_new_symbol(name, values)
                symbolic.store_intvar(symbolic_value.name, self)
            self._attrs["symbolic_value"] = symbolic_value

    def __str__(self) -> str:
        return pformat(self._attrs, indent=2)

    def __eq__(self, another: Any) -> bool:
        return (
                isinstance(another, IntVar)
                and self._attrs["symbolic_value"] == another._attrs["symbolic_value"]
        )

    def __hash__(self) -> int:
        return hash(
            (
                self._attrs["name"],
                tuple(self._attrs["values"]),
                self._attrs["symbolic_value"],
            )
        )

    def __add__(self, other: Union[Any, IntVar]) -> IntVar:
        self_values = self._attrs["values"]
        new_sym = self._attrs["symbolic_value"]
        if isinstance(other, IntVar):
            other_values = other._attrs["values"]
            new_sym = new_sym + other._attrs["symbolic_value"]
        elif isinstance(other, Number):
            other_values = [other]
            new_sym = new_sym + other
        else:
            raise NotImplementedError(f"Unable to do addition on {self} and {other}")

        new_values = [
            self_values[0] + other_values[0],
            self_values[-1] + other_values[-1],
        ]
        if new_values[0] == new_values[1]:
            return IntImm(new_values[0])

        return IntVar(values=new_values, symbolic_value=new_sym)

    def __radd__(self, other: Union[Any, IntVar]) -> IntVar:
        return self + other

    def __sub__(self, other: Union[Any, IntVar]) -> IntVar:
        self_values = self._attrs["values"]
        new_sym = self._attrs["symbolic_value"]
        if isinstance(other, IntVar):
            other_values = other._attrs["values"]
            new_sym = new_sym - other._attrs["symbolic_value"]
        elif isinstance(other, Number):
            other_values = [other]
            new_sym = new_sym - other
        else:
            raise NotImplementedError(f"Unable to do subtraction on {self} and {other}")

        new_values = [
            max(0, self_values[0] - other_values[-1]),
            max(0, self_values[-1] - other_values[0]),
        ]
        if new_values[0] == new_values[1]:
            return IntImm(new_values[0])

        return IntVar(values=new_values, symbolic_value=new_sym)

    def __rsub__(self, other: Union[Any, IntVar]) -> IntVar:
        self_values = self._attrs["values"]
        new_sym = self._attrs["symbolic_value"]
        if isinstance(other, IntVar):
            other_values = other._attrs["values"]
            new_sym = other._attrs["symbolic_value"] - new_sym
        elif isinstance(other, Number):
            other_values = [other]
            new_sym = other - new_sym
        else:
            raise NotImplementedError(
                f"Unable to do r-subtraction on {self} and {other}"
            )

        new_values = [
            max(0, other_values[0] - self_values[-1]),
            max(0, other_values[-1] - self_values[0]),
        ]
        if new_values[0] == new_values[1]:
            return IntImm(value=new_values[0])

        return IntVar(values=new_values, symbolic_value=new_sym)

    def __mul__(self, other: Union[Any, IntVar]) -> IntVar:
        self_values = self._attrs["values"]
        new_sym = self._attrs["symbolic_value"]
        if isinstance(other, IntVar):
            other_values = other._attrs["values"]
            new_sym = new_sym * other._attrs["symbolic_value"]
        elif isinstance(other, Number):
            other_values = [other]
            new_sym = new_sym * other
        else:
            raise NotImplementedError(
                f"Unable to do multiplication on {self} and {other}"
            )

        new_values = [
            self_values[0] * other_values[0],
            self_values[-1] * other_values[-1],
        ]
        if new_values[0] == new_values[1]:
            return IntImm(value=new_values[0])

        return IntVar(values=new_values, symbolic_value=new_sym)

    def __rmul__(self, other: Union[Any, IntVar]) -> IntVar:
        return self * other

    def __truediv__(self, other: Union[Any, IntVar]) -> IntVar:
        self_values = self._attrs["values"]
        new_sym = self._attrs["symbolic_value"]
        if isinstance(other, IntVar):
            other_values = other._attrs["values"]
            new_sym = new_sym / other._attrs["symbolic_value"]
        elif isinstance(other, Number):
            other_values = [other]
            new_sym = new_sym / other
        else:
            raise NotImplementedError(f"Unable to do division on {self} and {other}")

        new_values = [
            math.floor(self_values[0] / max(1, other_values[-1])),
            math.ceil(self_values[-1] / max(1, other_values[0])),
        ]
        if new_values[0] == new_values[1]:
            return IntImm(value=new_values[0])

        return IntVar(values=new_values, symbolic_value=new_sym)

    def __rtruediv__(self, other: Union[Any, IntVar]) -> IntVar:
        self_values = self._attrs["values"]
        new_sym = self._attrs["symbolic_value"]
        if isinstance(other, IntVar):
            other_values = other._attrs["values"]
            new_sym = other._attrs["symbolic_value"] / new_sym
        elif isinstance(other, Number):
            other_values = [other]
            new_sym = other / new_sym
        else:
            raise NotImplementedError(f"Unable to do r-division on {self} and {other}")

        new_values = [
            math.floor(other_values[0] / max(1, self_values[-1])),
            math.ceil(other_values[-1] / max(1, self_values[0])),
        ]
        if new_values[0] == new_values[1]:
            return IntImm(value=new_values[0])

        return IntVar(values=new_values, symbolic_value=new_sym)

    def lower_bound(self) -> int:
        """Returns lower bound of this dynamic dim."""
        return self._attrs["values"][0]

    def upper_bound(self) -> int:
        """Returns upper bound of this dynamic dim."""
        return self._attrs["values"][-1]

    def symbolic_value(self):
        """Returns the symbolic value of this dynamic dim."""
        return self._attrs["symbolic_value"]

    def pseudo_code(self, with_shape=False) -> str:
        return (
            self._attrs["name"]
            if self._attrs["name"] is not None
            else f"IntVar({str(self._attrs['values'])})"
        )


class IntImm(IntVar):
    """
    An IntImm represents a static dimension.
    IntVar (see above) and IntImm are used together to represent a Tensor's shape.
    """

    def __init__(
            self,
            value: int,
            name: str = None,
    ) -> None:
        """Initializes an IntImm.

        Parameters
        ----------
        value : int
            Value of this static dimension.

        name : str, optional
            Name of this dimension, by default None.
            This field must be set for dims which are used by input tensors.
        """

        if not isinstance(value, int):
            raise RuntimeError(
                "IntImm only takes an int value! Name: {}, current value: {}".format(
                    name, value
                )
            )

        Node.__init__(self)  # pylint: disable=W0233
        self._attrs["name"] = name
        self._attrs["values"] = [value]
        self._attrs["symbolic_value"] = value

    def __eq__(self, another: Union[int, IntVar]) -> bool:
        if isinstance(another, int):
            return self.value() == another

        return (
                isinstance(another, IntImm)
                and self._attrs["values"] == another._attrs["values"]
        )

    def value(self) -> int:
        """Returns value of this IntImm."""
        return self._attrs["values"][0]

    def pseudo_code(self, with_shape=False) -> str:
        return str(self.value())


def wrap_dim(idx, rank):
    """
    Wrap tensor index, idx, if it's negative.
    """
    assert isinstance(idx, int), "idx must be int, but got {}".format(type(idx))
    if idx < 0:
        idx = idx + rank
    assert idx < rank, "idx {} out of range; rank {}".format(idx, rank)
    return idx


class Tensor(Node):
    """
    A Tensor represents a piece of data, which is used as an input / output of an Operator.
    Both Tensor and Operator are used at model compilation stage.
    """

    def __init__(
            self,
            shape: List[IntVar],
            name: str = None,
            src_ops: Iterable[Node] = None,
            dst_ops: Iterable[Node] = None,
            dtype: str = "float16",
            is_input: bool = False,
            is_output: bool = False,
            value: Any = None,
            is_view_of: Any = None,
            is_internal_constant: bool = False,
            skip_constant_folding: bool = False,
            check_nan_and_inf: bool = False,
            check_outputs: bool = False,
    ) -> None:
        """Initializes a Tensor.

        Parameters
        ----------
        shape : List[IntVar]
            Shape of this Tensor.
        name : str, optional
            Name of this Tensor. By default, it's None.
        src_ops : Iterable[Node], optional
            Source operators of this Tensor which write to this Tensor.
            By default, it's an empty set.
        dst_ops : Iterable[Node], optional
            Destination operators of this Tensor which take this Tensor as
            one of their inputs.
            By default, it's an empty set.
        dtype : str, optional
            Date type of this Tensor. By default, it's "float16".
        is_input : bool, optional
            Whether this Tensor is an input Tensor of a graph.
            Note that constant Tensors (e.g. weights) are NOT input Tensors.
        is_output : bool, optional
            Whether this Tensor is an output Tensor of a graph.
        value : Any, optional
            The value of this Tensor. When value is set and shape is an
            empty list, this Tensor is used to represent a number.
        is_view_of : Any, optional
            Whether this Tensor is a view of another Tensor.
        is_internal_constant: bool, optional
            Whether this constant tensor could be modified.
        skip_constant_folding: bool, optional
            Whether this tensor participates in constant folding.
        check_nan_and_inf : bool, optional
            Whether or not to check this tensor is nan or inf during runtime.
        check_outputs : bool, optional
            Whether or not to print this tensor's value out during runtime.
        """
        super().__init__()
        self._attrs["shape"] = self._convert_shape(shape)
        self._attrs["name"] = name
        self._attrs["src_ops"] = set(src_ops)
        self._attrs["dst_ops"] = set(dst_ops)
        self._attrs["dtype"] = dtype
        self._attrs["is_output"] = is_output
        self._attrs["is_input"] = is_input
        self._attrs["is_internal_constant"] = is_internal_constant
        self._attrs["skip_constant_folding"] = skip_constant_folding

        # True if this is an internal tensor that aliases an output through
        # a view. Set up in mark_param_tensor
        self._attrs["has_output_aliases"] = False

        # For special views. When an output is a view of an input/constant/other
        # output, this attribute points to that view. Note that this is not the
        # same as is_view_of if the output is a view of a view. This is set up
        # in the mark_param_tensor graph pass.
        self._attrs["external_tensor"] = None

        # link to original tensor if this tensor is a view
        self._attrs["is_view_of"] = is_view_of

        if is_view_of:
            self._attrs["dtype"] = is_view_of._attrs["dtype"]

        self._attrs["value"] = value
        src_deps = [src_op._attrs["depth"] for src_op in self._attrs["src_ops"]]
        self._attrs["depth"] = max(src_deps) + 1 if len(src_deps) > 0 else 0

        # Offset into internal memory slab, set by memory planning
        self._attrs["offset"] = None

        # Data to be bound for constant folding. See _bind_data.
        self._attrs["data"] = None

        self._attrs["constant_folding_output_idx"] = None

        self._attrs["check_nan_and_inf"] = check_nan_and_inf
        self._attrs["check_outputs"] = check_outputs

    def __str__(self) -> str:
        output = {}
        for key in self._attrs.keys():
            if key in ("src_ops", "dst_ops") and self._attrs[key] is not None:
                output[key] = [x._attrs["name"] for x in self._attrs[key]]
            else:
                output[key] = self._attrs[key]
        return pformat(output, indent=2)

    def _convert_shape(self, shape: List[Union[int, IntVar]]) -> List[IntVar]:
        """
        Converts from a list of ints / IntVars to a list of IntVars.
        """
        ret = []
        for v in shape:
            if isinstance(v, int):
                ret.append(IntImm(v))
            elif isinstance(v, IntVar):
                ret.append(v)
            else:
                raise RuntimeError(f"Unsupported dim type: {type(v)}, dim: {v}")
        return ret

    def shape(self) -> List[IntVar]:
        """
        Returns the shape of the tensor.
        It should not be used directly in IR.
        """
        return self._attrs["shape"]

    def _rank(self) -> int:
        """
        Returns the rank of the tensor.
        It should not be used directly in IR.
        """
        return len(self._attrs["shape"])

    def _size(self, dim) -> IntVar:
        """
        Gets the size of tensor at dim=dim.
        dim must be between [-rank, rank - 1].
        It should not be used directly in IR, use ops.size(dim) instead.
        """
        return self._attrs["shape"][wrap_dim(dim, self._rank())]

    def dtype(self) -> str:
        """Returns Tensor's data type str."""
        return self._attrs["dtype"]

    def src_ops(self) -> Set[Operator]:
        """Returns a set of source operators which write to this Tensor."""
        return self._attrs["src_ops"]

    def dst_ops(self) -> Set[Operator]:
        """Returns a set of destination operators which read from this Tensor."""
        return self._attrs["dst_ops"]

    def is_a_const_num(self) -> bool:
        """Returns whether this Tensor represents a constant number."""
        return len(self._attrs["shape"]) == 0 and self._attrs["value"] is not None

    def pseudo_code(self, with_shape=True) -> str:
        name = self._attrs["name"]
        if name is None:
            name = "None"

        args = [f"name={name}"]

        if with_shape:
            shapes = ", ".join([dim.pseudo_code() for dim in self._attrs["shape"]])
            args.append(f"shape=[{shapes}]")

        data = self._attrs["data"]
        if data is not None:
            args.append(f"data=({data.size()} bytes)")

        if self.is_jagged():
            args.append("jagged=True")

        return f"Tensor({', '.join(args)})"

    def __deepcopy__(self, memo):
        result = Tensor(self.shape())
        memo[id(self)] = result
        result._attrs = copy.deepcopy(self._attrs, memo)
        return result

    def __add__(self, other: Any) -> Tensor:
        return OP_REGISTRY.get("ADD")(self, other)

    def __radd__(self, other: Any) -> Tensor:
        return OP_REGISTRY.get("ADD")(other, self)

    def __sub__(self, other: Any) -> Tensor:
        return OP_REGISTRY.get("SUB")(self, other)

    def __rsub__(self, other: Any) -> Tensor:
        return OP_REGISTRY.get("SUB")(other, self)

    def __mul__(self, other: Any) -> Tensor:
        return OP_REGISTRY.get("MUL")(self, other)

    def __rmul__(self, other: Any) -> Tensor:
        return OP_REGISTRY.get("MUL")(other, self)

    def __truediv__(self, other: Any) -> Tensor:
        return OP_REGISTRY.get("DIV")(self, other)

    def __rtruediv__(self, other: Any) -> Tensor:
        return OP_REGISTRY.get("DIV")(other, self)

    def __neg__(self) -> Tensor:
        return OP_REGISTRY.get("MUL")(-1, self)


class Operator(Node):
    """Base class for all operators"""

    def __init__(self) -> None:
        """Initializes the operator."""
        super().__init__()
        self._attrs["inputs"] = None

    def __call__(self, *args: List[Tensor]) -> List[Tensor]:
        """Performs offline shape inference and constructs the model graph.

        Parameters
        -------
        *args : List[Tensor]
            Input tensors.

        Returns
        -------
        List[Tensor]
            Output tensors.

        Raises
        ------
        NotImplementedError
        """
        raise NotImplementedError

    def __deepcopy__(self, memo):
        result = type(self)(**self._get_op_attributes())
        memo[id(self)] = result
        result._attrs = copy.deepcopy(self._attrs, memo)
        return result

    def __str__(self) -> str:
        """Generates a debug string."""
        output = {}
        for key in self._attrs.keys():
            if (
                    key in ("inputs", "args", "outputs", "original_inputs")
                    and self._attrs[key] is not None
            ):
                output[key] = [x._attrs["name"] for x in self._attrs[key]]
            else:
                output[key] = self._attrs[key]
        return pformat(output, indent=2)

    def gen_function(self) -> str:
        """Generates function source code string.

        Returns
        -------
        str : a string which contains C++ function implementation source code.

        Raises
        ------
        NotImplementedError
        """
        raise NotImplementedError("gen_function is not defined for {}".format(self))

    def _get_op_attributes(self) -> Dict[str, Any]:
        """
        Returns a dictionary of the core attributes of the op.
        The core attributes are attributes that are required to create an op, for
        example, the FuncEnum for a elementwise op.

        This is used when we need to copy the op with identical behaviour.

        Returns
        -------
        Dict of attributes
        """

        return {}

    # APIs below are for pseudo code generation.
    def _inputs_for_pseudo_code(self):
        return self._attrs["inputs"]

    def _outputs_for_pseudo_code(self):
        return self._attrs["outputs"]

    def _args_for_pseudo_code(self):
        return [f"{key}={value}" for key, value in self._get_op_attributes().items()]

    def _pseudo_code_helper(self, node: Any, with_shape: bool) -> str:
        if isinstance(node, list):
            if len(node) > 3 and isinstance(node[0], Tensor):
                return ",\n".join(self._pseudo_code_helper(n, with_shape) for n in node)
            else:
                return ", ".join(self._pseudo_code_helper(n, with_shape) for n in node)
        if isinstance(node, Node):
            return node.pseudo_code(with_shape)
        return str(node)

    def pseudo_code(self, with_shape=True):
        args = self._pseudo_code_helper(self._args_for_pseudo_code(), with_shape)
        inputs = self._pseudo_code_helper(self._inputs_for_pseudo_code(), with_shape)
        outputs = self._pseudo_code_helper(self._outputs_for_pseudo_code(), with_shape)
        name = self._attrs.get("name", None)
        return f"# {name}\n({outputs}) \n= {self._attrs['op']}({args})(\n{inputs})\n"
