// Copyright 2022 ByteDance Ltd. and/or its affiliates.
/*
 * Acknowledgement:
 * The structure of the expressions is inspired by Halide/TVM IR.
 *
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * \file matx/ir/function.h
 * \brief Function nodes.
 */
#pragma once

#include <string>
#include <type_traits>

#include <matxscript/ir/_base/optional_ref.h>
#include <matxscript/ir/attrs.h>
#include <matxscript/ir/base.h>
#include <matxscript/ir/hlo_var.h>
#include <matxscript/ir/prim_var.h>
#include <matxscript/runtime/container.h>

namespace matxscript {
namespace ir {

/*!
 * \brief Possible Calling conventions.
 *
 *  NOTE: The calling convention also implies
 *  the way we implement the function during lowering.
 */
enum class CallingConv : int {
  /*!
   * \brief Default calling convetion.
   *
   * - Uses the native calling convention of the target.
   * - Implementation: specified by the native target.
   */
  kDefault = 0,
  /*!
   * \brief PackedFunc that exposes a CPackedFunc signature.
   *
   * - Calling by PackedFunc calling convention.
   * - Implementation: Expose a function with the CPackedFunc signature.
   */
  kCPackedFunc = 1,
  /*!
   * \brief Device kernel launch
   *
   * - Call by PackedFunc calling convention.
   * - Implementation: defined by device runtime(e.g. runtime/cuda)
   */
  kDeviceKernelLaunch = 2,
};

/*!
 * \brief Base node of all functions.
 *
 * We support several variants of functions throughout the stack.
 * All of the functions share the same type system(via checked_type)
 * to support cross variant calls.
 *
 * \sa BaseFunc
 */
class BaseFuncNode : public HLOExprNode {
 public:
  /*! \brief Additional attributes storing the meta-data */
  DictAttrs attrs;

  /*!
   * \brief Get a function attribute.
   *
   * \param attr_key The attribute key.
   * \param default_value The default value if the key does not exist, defaults to nullptr.
   *
   * \return The result
   *
   * \tparam TObjectRef the expected object type.
   * \throw Error if the key exists but the value does not match TObjectRef
   *
   * \code
   *
   *  void GetAttrExample(const BaseFunc& f) {
   *    auto value = f->GetAttr<Integer>("AttrKey", 0);
   *  }
   *
   * \endcode
   */
  template <typename TObjectRef>
  runtime::Optional<TObjectRef> GetAttr(
      const StringRef& attr_key,
      runtime::Optional<TObjectRef> default_value = runtime::Optional<TObjectRef>(nullptr)) const {
    static_assert(std::is_base_of<ObjectRef, TObjectRef>::value,
                  "Can only call GetAttr with ObjectRef types.");
    if (!attrs.defined())
      return default_value;
    auto it = attrs->dict.find(attr_key);
    if (it != attrs->dict.end()) {
      return runtime::Downcast<runtime::Optional<TObjectRef>>((*it).second);
    } else {
      return default_value;
    }
  }
  // variant that uses TObjectRef to enable implicit conversion to default value.
  template <typename TObjectRef>
  runtime::Optional<TObjectRef> GetAttr(const StringRef& attr_key, TObjectRef default_value) const {
    return GetAttr<TObjectRef>(attr_key, runtime::Optional<TObjectRef>(default_value));
  }
  /*!
   * \brief Check whether the function has an non-zero integer attr.
   *
   * This function can be used to check whether an optional
   * attribute mark(e.g. inline) exists.
   *
   * \param attr_key The key to the attribute.
   * \return The check result.
   *
   * \code
   *
   *  void HasNonzeroAttrExample(const BaseFunc& f) {
   *    if (f->HasNonzeroAttr(attr::kInline)) {
   *      // inline the function.
   *    }
   *  }
   *
   * \endcode
   */
  bool HasNonzeroAttr(const StringRef& attr_key) const {
    return GetAttr<Integer>(attr_key, 0) != 0;
  }

  MATX_DLL bool HasGlobalName() const;
  MATX_DLL StringRef GetGlobalName() const;

  MATX_DLL bool HasBoundName() const;
  MATX_DLL StringRef GetBoundName() const;

  MATX_DLL bool ExportSymbol() const;
  MATX_DLL bool CaptureSessionHandle() const;
  MATX_DLL bool IsClassConstructor() const;
  MATX_DLL bool IsClassMember() const;
  MATX_DLL StringRef GetBelongToClassName() const;

  MATX_DLL virtual runtime::Array<BaseExpr> GetParams() const = 0;
  MATX_DLL virtual runtime::Array<BaseExpr> GetDefaultParams() const = 0;
  MATX_DLL virtual Type GetReturnType() const = 0;
  MATX_DLL virtual StringRef GetReprName() const = 0;
  MATX_DLL virtual Stmt GetBody() const = 0;
  MATX_DLL virtual FuncType func_type_annotation() const = 0;

  static constexpr const char* _type_key = "BaseFunc";
  static constexpr const uint32_t _type_child_slots = 2;
  MATXSCRIPT_DECLARE_BASE_OBJECT_INFO(BaseFuncNode, HLOExprNode);
};

/*!
 * \brief Managed reference to BaseFuncNode.
 * \sa BaseFuncNode
 */
class BaseFunc : public HLOExpr {
 public:
  MATXSCRIPT_DEFINE_OBJECT_REF_METHODS(BaseFunc, HLOExpr, BaseFuncNode);
};

/*!
 * \brief Create a new function that copies func, but overrides
 *        the attribute value key with the value.
 *
 * \param func The input function.
 * \param attr_key The attribute key.
 * \param attr_value The value attribute value.
 *
 * \tparam TFunc The corresponding function type.
 *
 * \returns The new function with updated attributes.
 *
 * \note This function performs copy on write optimization for func.
 *       If we move a uniquely referenced func into WithAttr,
 *       then no additional copy will be performed.
 *
 *       This is also why we make it as a function instead of a member function
 *       and why we pass by value in the first argument.
 *
 * \code
 *
 *  // Recommended way to trigger copy on write
 *  func = WithAttr(std::move(func), "key1", value1);
 *  func = WithAttr(std::move(func), "key2", value2);
 *
 * \endcode
 */
template <typename TFunc,
          typename = typename std::enable_if<std::is_base_of<BaseFunc, TFunc>::value>::type>
inline TFunc WithAttr(TFunc func, StringRef attr_key, ObjectRef attr_value) {
  using TNode = typename TFunc::ContainerType;
  static_assert(TNode::_type_final, "Can only operate on the leaf nodes");
  TNode* node = func.CopyOnWrite();
  if (node->attrs.defined()) {
    node->attrs.CopyOnWrite()->dict.Set(attr_key, attr_value);
  } else {
    runtime::Map<StringRef, ObjectRef> dict = {{attr_key, attr_value}};
    node->attrs = DictAttrs(dict);
  }
  return func;
}

/*!
 * \brief Primitive functions that contains IR statements.
 *
 * The PrimFunc provides low-level code representation does not
 * automatically manage
 *
 * \sa PrimFunc
 */
class PrimFuncNode : public BaseFuncNode {
 public:
  /*! \brief Function parameters */
  runtime::Array<PrimVar> params;
  runtime::Array<PrimExpr> default_params;
  /*! \brief The body of the function */
  Stmt body;
  /*! \brief The return type of the function. */
  Type ret_type;

  void VisitAttrs(runtime::AttrVisitor* v) {
    v->Visit("params", &params);
    v->Visit("default_params", &default_params);
    v->Visit("body", &body);
    v->Visit("ret_type", &ret_type);
    v->Visit("attrs", &attrs);
    v->Visit("span", &span);
    v->Visit("_checked_type_", &checked_type_);
  }

  bool SEqualReduce(const PrimFuncNode* other, SEqualReducer equal) const {
    // visit params and buffer_map first as they contains defs.
    return equal.DefEqual(params, other->params) &&
           equal.DefEqual(default_params, other->default_params) &&
           equal(ret_type, other->ret_type) && equal(body, other->body) &&
           equal(attrs, other->attrs);
  }

  void SHashReduce(SHashReducer hash_reduce) const {
    hash_reduce.DefHash(params);
    hash_reduce.DefHash(default_params);
    hash_reduce(ret_type);
    hash_reduce(body);
    hash_reduce(attrs);
  }
  /*!
   * \brief Return the derived function annotation of this function.
   *
   * \return The function type annotation.
   * \note The function type annotation of PrimExpr is
   *       directly derived from the Vars without the need of type inference.
   */
  MATX_DLL FuncType func_type_annotation() const override;

  MATX_DLL runtime::Array<BaseExpr> GetParams() const override;
  MATX_DLL runtime::Array<BaseExpr> GetDefaultParams() const override;
  MATX_DLL Type GetReturnType() const override;
  MATX_DLL Stmt GetBody() const override;
  MATX_DLL StringRef GetReprName() const override;

  static constexpr const char* _type_key = "ir.PrimFunc";
  MATXSCRIPT_DECLARE_FINAL_OBJECT_INFO(PrimFuncNode, BaseFuncNode);
};

/*!
 * \brief Managed reference to PrimFuncNode.
 * \sa PrimFuncNode
 */
class PrimFunc : public BaseFunc {
 public:
  /*!
   * \brief Constructor
   * \param params The parameters of the function.
   * \param default_params The default parameters of the function.
   * \param body The body of the function.
   * \param ret_type The return type of the function.
   * \param buffer_map The buffer map for parameter buffer unpacking.
   * \param attrs Additional function attributes.
   */
  MATX_DLL PrimFunc(runtime::Array<PrimVar> params,
                    runtime::Array<PrimExpr> default_params,
                    Stmt body,
                    Type ret_type = VoidType(),
                    DictAttrs attrs = NullValue<DictAttrs>(),
                    Span span = {});

  MATXSCRIPT_DEFINE_OBJECT_REF_METHODS(PrimFunc, BaseFunc, PrimFuncNode);
  MATXSCRIPT_DEFINE_OBJECT_REF_COW_METHOD(PrimFuncNode);
};

/*!
 * \brief Function container
 * \sa Function
 */
class FunctionNode : public BaseFuncNode {
 public:
  /*! \brief Function parameters */
  runtime::Array<BaseExpr> params;
  runtime::Array<BaseExpr> default_params;
  /*!
   * \brief
   * The expression which represents the computation of the function,
   * the expression may reference the parameters, and the type of it
   * or sub-expressions may reference the type variables.
   */
  Stmt body;
  /*! \brief User annotated return type of the function. */
  Type ret_type;
  /*!
   * \brief Type parameters of the function.
   *  Enables the function to vary its type based on these.
   *  This corresponds to template paramaters in c++'s terminology.
   *
   * \note This can be usually empty for non-polymorphic functions.
   */
  runtime::Array<TypeVar> type_params;

  void VisitAttrs(runtime::AttrVisitor* v) {
    v->Visit("params", &params);
    v->Visit("default_params", &default_params);
    v->Visit("body", &body);
    v->Visit("ret_type", &ret_type);
    v->Visit("type_params", &type_params);
    v->Visit("attrs", &attrs);
    v->Visit("span", &span);
    v->Visit("_checked_type_", &checked_type_);
  }

  bool SEqualReduce(const FunctionNode* other, SEqualReducer equal) const {
    // Important to make def equal first.
    equal->MarkGraphNode();
    return equal.DefEqual(params, other->params) &&
           equal.DefEqual(default_params, other->default_params) &&
           equal.DefEqual(type_params, other->type_params) && equal(ret_type, other->ret_type) &&
           equal(attrs, other->attrs) && equal(body, other->body);
  }

  void SHashReduce(SHashReducer hash_reduce) const {
    hash_reduce->MarkGraphNode();
    hash_reduce.DefHash(params);
    hash_reduce.DefHash(default_params);
    hash_reduce.DefHash(type_params);
    hash_reduce(ret_type);
    hash_reduce(attrs);
    hash_reduce(body);
  }

  /*!
   * \brief Return the derived function annotation of this expression.
   *
   * \return The function type annotation.
   * \note The function type annotation can contain IncompleteType.
   */
  MATX_DLL FuncType func_type_annotation() const override;

  MATX_DLL runtime::Array<BaseExpr> GetParams() const override;
  MATX_DLL runtime::Array<BaseExpr> GetDefaultParams() const override;
  MATX_DLL Type GetReturnType() const override;
  MATX_DLL Stmt GetBody() const override;
  MATX_DLL StringRef GetReprName() const override;

  static constexpr const char* _type_key = "ir.Function";
  MATXSCRIPT_DECLARE_FINAL_OBJECT_INFO(FunctionNode, BaseFuncNode);
};

/*!
 * \brief Managed reference to FunctionNode.
 * \sa FunctionNode
 */
class Function : public BaseFunc {
 public:
  /*!
   * \brief Constructor
   * \param params The parameters of the function.
   * \param body The body of the function.
   * \param ret_type The return type of the function.
   * \param ty_params The type parameters.
   * \param attrs Additional function attributes.
   * \param span The span of the function.
   */
  MATX_DLL Function(runtime::Array<BaseExpr> params,
                    runtime::Array<BaseExpr> default_params,
                    Stmt body,
                    Type ret_type,
                    runtime::Array<TypeVar> ty_params,
                    DictAttrs attrs = NullValue<DictAttrs>(),
                    Span span = Span());

  MATXSCRIPT_DEFINE_OBJECT_REF_METHODS(Function, BaseFunc, FunctionNode);
  MATXSCRIPT_DEFINE_OBJECT_REF_COW_METHOD(FunctionNode);
};

/*!
 * \brief Lambda Function container
 * \sa LambdaFunction
 */
class LambdaFunctionNode : public BaseFuncNode {
 public:
  /*! \brief Function parameters */
  runtime::Array<BaseExpr> params;
  runtime::Array<BaseExpr> captures;
  /*!
   * \brief
   * The expression which represents the computation of the function,
   * the expression may reference the parameters, and the type of it
   * or sub-expressions may reference the type variables.
   */
  Stmt body;
  /*! \brief User annotated return type of the function. */
  Type ret_type;

  void VisitAttrs(runtime::AttrVisitor* v) {
    v->Visit("params", &params);
    v->Visit("captures", &captures);
    v->Visit("body", &body);
    v->Visit("ret_type", &ret_type);
    v->Visit("attrs", &attrs);
    v->Visit("span", &span);
    v->Visit("_checked_type_", &checked_type_);
  }

  bool SEqualReduce(const LambdaFunctionNode* other, SEqualReducer equal) const {
    // Important to make def equal first.
    equal->MarkGraphNode();
    return equal.DefEqual(params, other->params) && equal.DefEqual(captures, other->captures) &&
           equal(ret_type, other->ret_type) && equal(attrs, other->attrs) &&
           equal(body, other->body);
  }

  void SHashReduce(SHashReducer hash_reduce) const {
    hash_reduce->MarkGraphNode();
    hash_reduce.DefHash(params);
    hash_reduce.DefHash(captures);
    hash_reduce(ret_type);
    hash_reduce(attrs);
    hash_reduce(body);
  }

  /*!
   * \brief Return the derived function annotation of this expression.
   *
   * \return The function type annotation.
   * \note The function type annotation can contain IncompleteType.
   */
  MATX_DLL FuncType func_type_annotation() const override;

  MATX_DLL runtime::Array<BaseExpr> GetParams() const override;
  MATX_DLL runtime::Array<BaseExpr> GetDefaultParams() const override;
  MATX_DLL Type GetReturnType() const override;
  MATX_DLL Stmt GetBody() const override;
  MATX_DLL StringRef GetReprName() const override;

  static constexpr const char* _type_key = "ir.LambdaFunction";
  MATXSCRIPT_DECLARE_FINAL_OBJECT_INFO(LambdaFunctionNode, BaseFuncNode);
};

/*!
 * \brief Managed reference to LambdaFunctionNode.
 * \sa LambdaFunctionNode
 */
class LambdaFunction : public BaseFunc {
 public:
  /*!
   * \brief Constructor
   * \param captures The captures of the function.
   * \param params The parameters of the function.
   * \param body The body of the function.
   * \param ret_type The return type of the function.
   * \param attrs Additional function attributes.
   * \param span The span of the function.
   */
  MATX_DLL LambdaFunction(runtime::Array<BaseExpr> captures,
                          runtime::Array<BaseExpr> params,
                          Stmt body,
                          Type ret_type,
                          DictAttrs attrs = NullValue<DictAttrs>(),
                          Span span = Span());

  MATXSCRIPT_DEFINE_OBJECT_REF_METHODS(LambdaFunction, BaseFunc, LambdaFunctionNode);
  MATXSCRIPT_DEFINE_OBJECT_REF_COW_METHOD(LambdaFunctionNode);
};

/*!
 * \brief Generic attribute names that can be attached to any function.
 *
 * \sa ::matxscript::ir::attr
 */
namespace attr {
/*!
 * \brief Indicates the special calling convention.
 *
 * Type: Integer
 *
 * \sa matx::CallingConv
 */
constexpr const char* kCallingConv = "calling_conv";

/*!
 * \brief Global linker symbol of the function in generated code.
 *
 *  This option forces the code generator to name the
 *  function with the given.
 *
 *  For example, we could set a global_symbol of a function
 *  early to make sure that we can always refer to it by
 *  the symbol name in the generated DLL.
 *
 *  We should not set the attribute for local functions,
 *  so that the compiler can freely rename them.
 *
 *  A unique global symbol will be automatically assigned
 *  to each function in the module before the target code
 *  generation phase.
 *
 * Type: String
 */
constexpr const char* kGlobalSymbol = "global_symbol";
constexpr const char* kBoundSymbol = "bound_symbol";

constexpr const char* kExportSymbol = "export_symbol";
constexpr const char* kClassConstructor = "class_constructor";
constexpr const char* kClassNameBelongTo = "class_name_belong_to";
constexpr const char* kCaptureSessionHandle = "capture_session_handle";

/*!
 * \brief Whether to set noalias rule on the function arguments.
 *
 * Type: Integer
 */
constexpr const char* kNoAlias = "ir.noalias";

/*!
 * \brief Mark the function as the entry function of
 *        the final generated runtime module.
 *
 * Type: Integer
 *
 * \note There can only be one entry function per module.
 */
constexpr const char* kIsEntryFunc = "ir.is_entry_func";
}  // namespace attr

}  // namespace ir
}  // namespace matxscript
