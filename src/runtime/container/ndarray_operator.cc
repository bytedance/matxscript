// Copyright 2022 ByteDance Ltd. and/or its affiliates.
#include <matxscript/runtime/container/ndarray_helper.h>
#include "matxscript/runtime/runtime_port.h"

namespace matxscript {
namespace runtime {
namespace {

struct AddOP {
  template <typename LDType, typename RDType, typename DType>
  MATXSCRIPT_ALWAYS_INLINE static void Map(LDType x, RDType y, DType* t) {
    *t = ((DType)(x) + (DType)(y));
  }
};

struct MulOP {
  template <typename LDType, typename RDType, typename DType>
  MATXSCRIPT_ALWAYS_INLINE static void Map(LDType x, RDType y, DType* t) {
    *t = ((DType)(x) * (DType)(y));
  }
};

struct SubOP {
  template <typename LDType, typename RDType, typename DType>
  MATXSCRIPT_ALWAYS_INLINE static void Map(LDType x, RDType y, DType* t) {
    *t = ((DType)(x) - (DType)(y));
  }
};

struct RSubOP {
  template <typename LDType, typename RDType, typename DType>
  MATXSCRIPT_ALWAYS_INLINE static void Map(LDType x, RDType y, DType* t) {
    *t = ((DType)(y) - (DType)(x));
  }
};

struct DivOP {
  template <typename LDType, typename RDType, typename DType>
  MATXSCRIPT_ALWAYS_INLINE static void Map(LDType x, RDType y, DType* t) {
    *t = ((DType)(x) / (DType)(y));
  }
};

struct RDivOP {
  template <typename LDType, typename RDType, typename DType>
  MATXSCRIPT_ALWAYS_INLINE static void Map(LDType x, RDType y, DType* t) {
    *t = ((DType)(y) / (DType)(x));
  }
};

template <typename OP, typename DstDtype, typename LDType, typename RDType>
void ScalarAssign(DstDtype* dst_data,
                  const LDType* l_data,
                  const RDType r,
                  const int64_t* dst_strides,
                  const int64_t* l_strides,
                  const int64_t* shape,
                  int ndim) {
  if (ndim == 1) {
    DstDtype* t = dst_data;
    for (int64_t i = 0; i < shape[0]; ++i, t += dst_strides[0]) {
      OP::Map(l_data[i * l_strides[0]], r, t);
    }
    return;
  }
  for (int64_t i = 0; i < shape[0]; ++i) {
    ScalarAssign<OP, DstDtype, LDType, RDType>(dst_data + i * dst_strides[0],
                                               l_data + i * l_strides[0],
                                               r,
                                               dst_strides + 1,
                                               l_strides + 1,
                                               shape + 1,
                                               ndim - 1);
  }
}

template <typename OP, typename DstDtype, typename LDType, typename RDType>
void BinaryAssign(DstDtype* dst_data,
                  const LDType* l_data,
                  const RDType* r_data,
                  const int64_t* dst_strides,
                  const int64_t* l_strides,
                  const int64_t* r_strides,
                  const int64_t* shape,
                  int ndim) {
  if (ndim == 1) {
    DstDtype* t = dst_data;
    for (int64_t i = 0; i < shape[0]; ++i, t += dst_strides[0]) {
      OP::Map(l_data[i * l_strides[0]], r_data[i * r_strides[0]], t);
    }
    return;
  }
  for (int64_t i = 0; i < shape[0]; ++i) {
    BinaryAssign<OP, DstDtype, LDType, RDType>(dst_data + i * dst_strides[0],
                                               l_data + i * l_strides[0],
                                               r_data + i * r_strides[0],
                                               dst_strides + 1,
                                               l_strides + 1,
                                               r_strides + 1,
                                               shape + 1,
                                               ndim - 1);
  }
}

std::vector<int64_t> broadcast_stride(const std::vector<int64_t>& broadcast_shape,
                                      const int64_t* shape,
                                      const int64_t* strides,
                                      int dim) {
  int bdim = broadcast_shape.size();
  int delta = bdim - dim;
  std::vector<int64_t> ret(bdim, 0);
  for (int i = 0; i < dim; ++i) {
    if (shape[i] > 1) {
      ret[i + delta] = strides[i];
    } else {
      ret[i + delta] = 0;
    }
  }
  return ret;
}

template <typename OP>
NDArray broadcast_binary_nd(const NDArray& nd1, const NDArray& nd2, const DataType& data_type) {
  std::vector<int64_t> broadcast_shape;
  // TODO: nd.Shape create a new vector, which is not necessary here
  if (!NDArrayHelper::GetBroadcastShape(nd1.Shape(), nd2.Shape(), broadcast_shape)) {
    MXTHROW << "ndarray operator: shape not match";
  }
  std::vector<int64_t> nd1_strides =
      broadcast_stride(broadcast_shape, nd1.GetShapePtr(), nd1.GetStridesPtr(), nd1.GetDim());
  std::vector<int64_t> nd2_strides =
      broadcast_stride(broadcast_shape, nd2.GetShapePtr(), nd2.GetStridesPtr(), nd2.GetDim());
  NDArray ret = NDArray::Empty(broadcast_shape, data_type, nd1->ctx);
  MATX_NDARRAY_TYPE_SWITCH(ret.DataType(), DType, {
    MATX_NDARRAY_TYPE_SWITCH(nd1.DataType(), LDType, {
      MATX_NDARRAY_TYPE_SWITCH(nd2.DataType(), RDType, {
        LDType* l_data = (LDType*)((char*)nd1->data + nd1->byte_offset);
        RDType* r_data = (RDType*)((char*)nd2->data + nd2->byte_offset);
        DType* dst_data = (DType*)((char*)ret->data + ret->byte_offset);
        BinaryAssign<OP, DType, LDType, RDType>(dst_data,
                                                l_data,
                                                r_data,
                                                ret.GetStridesPtr(),
                                                nd1_strides.data(),
                                                nd2_strides.data(),
                                                broadcast_shape.data(),
                                                broadcast_shape.size());
      });
    });
  });
  return ret;
}

template <typename OP, typename SType>
void broadcast_binary_scalar(const NDArray& nd1, SType s, NDArray& ret) {
  auto shape = ret.Shape();
  MATX_NDARRAY_TYPE_SWITCH(ret.DataType(), DType, {
    MATX_NDARRAY_TYPE_SWITCH(nd1.DataType(), LDType, {
      LDType* l_data = (LDType*)((char*)nd1->data + nd1->byte_offset);
      DType* dst_data = (DType*)((char*)ret->data + ret->byte_offset);
      ScalarAssign<OP, DType, LDType, SType>(dst_data,
                                             l_data,
                                             s,
                                             ret.GetStridesPtr(),
                                             nd1.GetStridesPtr(),
                                             ret.Shape().data(),
                                             shape.size());
    });
  });
}

template <typename OP>
NDArray contiguous_binary_nd(const NDArray& nd1, const NDArray& nd2, const DataType& data_type) {
  NDArray ret = NDArray::Empty(nd1.Shape(), data_type, nd1->ctx);
  int64_t element_num = NDArrayHelper::GetItemNum(ret.GetShapePtr(), ret.GetDim());
  MATX_NDARRAY_TYPE_SWITCH(ret.DataType(), DType, {
    MATX_NDARRAY_TYPE_SWITCH(nd1.DataType(), LDType, {
      MATX_NDARRAY_TYPE_SWITCH(nd2.DataType(), RDType, {
        LDType* l_data = (LDType*)((char*)nd1->data + nd1->byte_offset);
        RDType* r_data = (RDType*)((char*)nd2->data + nd2->byte_offset);
        DType* dst_data = (DType*)((char*)ret->data + ret->byte_offset);
        for (int64_t i = 0; i < element_num; ++i) {
          OP::Map(*(l_data + i), *(r_data + i), dst_data + i);
        }
      });
    });
  });
  return ret;
}

template <typename OP, typename SType>
void contiguous_binary_scalar(const NDArray& nd1, SType s, NDArray& ret) {
  int64_t element_num = NDArrayHelper::GetItemNum(ret.GetShapePtr(), ret.GetDim());
  MATX_NDARRAY_TYPE_SWITCH(ret.DataType(), DType, {
    MATX_NDARRAY_TYPE_SWITCH(nd1.DataType(), LDType, {
      LDType* l_data = (LDType*)((char*)nd1->data + nd1->byte_offset);
      DType* dst_data = (DType*)((char*)ret->data + ret->byte_offset);
      for (int64_t i = 0; i < element_num; ++i) {
        OP::Map(*(l_data + i), s, dst_data + i);
      }
    });
  });
}

}  // namespace

NDArray NDArrayOperate::Add(const NDArray& lhs, const NDArray& rhs) {
  if (NDArrayHelper::IsSameShape(lhs, rhs) && lhs.IsContiguous() && rhs.IsContiguous()) {
    return contiguous_binary_nd<AddOP>(
        lhs, rhs, NDArrayHelper::DTypePromotion(lhs.DataType(), rhs.DataType()));
  }
  return broadcast_binary_nd<AddOP>(
      lhs, rhs, NDArrayHelper::DTypePromotion(lhs.DataType(), rhs.DataType()));
}

NDArray NDArrayOperate::Add(const NDArray& lhs, int64_t num) {
  NDArray ret = NDArray::Empty(lhs.Shape(), lhs.DataType(), NDArrayHelper::GetCPUDevice());
  if (lhs.IsContiguous()) {
    contiguous_binary_scalar<AddOP>(lhs, num, ret);
    return ret;
  }
  broadcast_binary_scalar<AddOP>(lhs, num, ret);
  return ret;
}

NDArray NDArrayOperate::Add(const NDArray& lhs, double num) {
  NDArray ret = NDArray::Empty(
      lhs.Shape(), NDArrayHelper::DTypeFromDouble(lhs.DataType()), NDArrayHelper::GetCPUDevice());
  if (lhs.IsContiguous()) {
    contiguous_binary_scalar<AddOP>(lhs, num, ret);
    return ret;
  }
  broadcast_binary_scalar<AddOP>(lhs, num, ret);
  return ret;
}

NDArray NDArrayOperate::Sub(const NDArray& lhs, const NDArray& rhs) {
  if (NDArrayHelper::IsSameShape(lhs, rhs) && lhs.IsContiguous() && rhs.IsContiguous()) {
    return contiguous_binary_nd<SubOP>(
        lhs, rhs, NDArrayHelper::DTypePromotion(lhs.DataType(), rhs.DataType()));
  }
  return broadcast_binary_nd<SubOP>(
      lhs, rhs, NDArrayHelper::DTypePromotion(lhs.DataType(), rhs.DataType()));
}

NDArray NDArrayOperate::Sub(int64_t num, const NDArray& rhs) {
  NDArray ret = NDArray::Empty(rhs.Shape(), rhs.DataType(), NDArrayHelper::GetCPUDevice());
  if (rhs.IsContiguous()) {
    contiguous_binary_scalar<RSubOP>(rhs, num, ret);
    return ret;
  }
  broadcast_binary_scalar<RSubOP>(rhs, num, ret);
  return ret;
}

NDArray NDArrayOperate::Sub(double num, const NDArray& rhs) {
  NDArray ret = NDArray::Empty(
      rhs.Shape(), NDArrayHelper::DTypeFromDouble(rhs.DataType()), NDArrayHelper::GetCPUDevice());
  if (rhs.IsContiguous()) {
    contiguous_binary_scalar<RSubOP>(rhs, num, ret);
    return ret;
  }
  broadcast_binary_scalar<RSubOP>(rhs, num, ret);
  return ret;
}

NDArray NDArrayOperate::Div(const NDArray& lhs, const NDArray& rhs) {
  auto target_dt = NDArrayHelper::DTypePromotion(lhs.DataType(), rhs.DataType());
  if (target_dt.is_int()) {
    target_dt = DataType(String2DLDataType("float32"));
  }
  if (NDArrayHelper::IsSameShape(lhs, rhs) && lhs.IsContiguous() && rhs.IsContiguous()) {
    return contiguous_binary_nd<DivOP>(lhs, rhs, target_dt);
  }
  return broadcast_binary_nd<DivOP>(lhs, rhs, target_dt);
}

NDArray NDArrayOperate::Div(double num, const NDArray& rhs) {
  NDArray ret = NDArray::Empty(
      rhs.Shape(), NDArrayHelper::DTypeFromDouble(rhs.DataType()), NDArrayHelper::GetCPUDevice());
  if (rhs.IsContiguous()) {
    contiguous_binary_scalar<RDivOP>(rhs, num, ret);
    return ret;
  }
  broadcast_binary_scalar<RDivOP>(rhs, num, ret);
  return ret;
}

NDArray NDArrayOperate::Div(const NDArray& lhs, double num) {
  NDArray ret = NDArray::Empty(
      lhs.Shape(), NDArrayHelper::DTypeFromDouble(lhs.DataType()), NDArrayHelper::GetCPUDevice());
  if (lhs.IsContiguous()) {
    contiguous_binary_scalar<DivOP>(lhs, num, ret);
    return ret;
  }
  broadcast_binary_scalar<DivOP>(lhs, num, ret);
  return ret;
}

NDArray NDArrayOperate::Mul(const NDArray& lhs, const NDArray& rhs) {
  if (NDArrayHelper::IsSameShape(lhs, rhs) && lhs.IsContiguous() && rhs.IsContiguous()) {
    return contiguous_binary_nd<MulOP>(
        lhs, rhs, NDArrayHelper::DTypePromotion(lhs.DataType(), rhs.DataType()));
  }
  return broadcast_binary_nd<MulOP>(
      lhs, rhs, NDArrayHelper::DTypePromotion(lhs.DataType(), rhs.DataType()));
}

NDArray NDArrayOperate::Mul(const NDArray& lhs, int64_t num) {
  NDArray ret = NDArray::Empty(lhs.Shape(), lhs.DataType(), NDArrayHelper::GetCPUDevice());
  if (lhs.IsContiguous()) {
    contiguous_binary_scalar<MulOP>(lhs, num, ret);
    return ret;
  }
  broadcast_binary_scalar<MulOP>(lhs, num, ret);
  return ret;
}

NDArray NDArrayOperate::Mul(const NDArray& lhs, double num) {
  NDArray ret = NDArray::Empty(
      lhs.Shape(), NDArrayHelper::DTypeFromDouble(lhs.DataType()), NDArrayHelper::GetCPUDevice());
  if (lhs.IsContiguous()) {
    contiguous_binary_scalar<MulOP>(lhs, num, ret);
    return ret;
  }
  broadcast_binary_scalar<MulOP>(lhs, num, ret);
  return ret;
}

}  // namespace runtime
}  // namespace matxscript