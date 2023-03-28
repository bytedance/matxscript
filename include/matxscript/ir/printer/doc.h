// Copyright 2022 ByteDance Ltd. and/or its affiliates.
/*
 * Acknowledgement: This file originates from TVM.
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
#pragma once

#include <matxscript/runtime/data_type.h>
#include <matxscript/runtime/object.h>

#include <matxscript/ir/_base/cow_array_ref.h>
#include <matxscript/ir/_base/cow_map_ref.h>
#include <matxscript/ir/_base/object_path.h>
#include <matxscript/ir/_base/optional_ref.h>
#include <matxscript/ir/_base/reflection.h>
#include <matxscript/ir/_base/string_ref.h>
#include <matxscript/ir/expr.h>

namespace matxscript {
namespace ir {
namespace printer {

using runtime::DataType;
using runtime::Object;
using runtime::ObjectPtr;
using runtime::ObjectRef;

/*!
 * \brief The base class of all Doc.
 *
 * Doc is an intermediate representation between IR and the text.
 * During printing, IR graph is first translated into Doc tree,
 * then the Doc tree is translated to the target language in
 * text format.
 *
 * \sa Doc
 */
class DocNode : public Object {
 public:
  /*!
   * \brief The list of object paths of the source IR node.
   *
   * This is used to trace back to the IR node position where
   * this Doc is generated, in order to position the diagnostic
   * message.
   */
  mutable Array<ObjectPath> source_paths;

  void VisitAttrs(AttrVisitor* v) {
    v->Visit("source_paths", &source_paths);
  }

  static constexpr const char* _type_key = "ir.printer.Doc";
  MATXSCRIPT_DECLARE_BASE_OBJECT_INFO(DocNode, Object);

 public:
  virtual ~DocNode() = default;
};

/*!
 * \brief Reference type of DocNode.
 *
 * \sa DocNode
 */
class Doc : public ObjectRef {
 protected:
  Doc() = default;

 public:
  virtual ~Doc() = default;
  MATXSCRIPT_DEFINE_NOTNULLABLE_OBJECT_REF_METHODS(Doc, ObjectRef, DocNode);
};

class ExprDoc;

/*!
 * \brief The base class of expression doc.
 *
 * \sa ExprDoc
 */
class ExprDocNode : public DocNode {
 public:
  /*!
   * \brief Create a doc representing attribute access on the current ExprDoc
   * \param attr The attribute to access.
   */
  ExprDoc Attr(StringRef attr) const;

  /*!
   * \brief Create a doc representing index access on the current ExprDoc
   * \param indices The indices to access.
   */
  ExprDoc operator[](Array<Doc> indices) const;

  /*!
   * \brief Create a doc representing calling the current ExprDoc
   * \param args The positional arguments of the function call.
   */
  ExprDoc Call(Array<ExprDoc, void> args) const;

  /*!
   * \brief Create a doc representing attribute access on the current ExprDoc
   * \param args The positional arguments of the function call.
   * \param kwargs_keys Keys of keywords arguments of the function call.
   * \param kwargs_values Values of keywords arguments of the function call.
   */
  ExprDoc Call(Array<ExprDoc, void> args,     //
               Array<StringRef> kwargs_keys,  //
               Array<ExprDoc, void> kwargs_values) const;

  void VisitAttrs(AttrVisitor* v) {
    DocNode::VisitAttrs(v);
  }

  static constexpr const char* _type_key = "ir.printer.ExprDoc";
  MATXSCRIPT_DECLARE_BASE_OBJECT_INFO(ExprDocNode, DocNode);
};

/*!
 * \brief Reference type of ExprDocNode.
 *
 * \sa ExprDocNode
 */
class ExprDoc : public Doc {
 protected:
  ExprDoc() = default;

 public:
  /*!
   * \brief Create a doc representing index access on the current ExprDoc
   * \param indices The indices to access.
   */
  ExprDoc operator[](Array<Doc> indices) const;

  MATXSCRIPT_DEFINE_NOTNULLABLE_OBJECT_REF_METHODS(ExprDoc, Doc, ExprDocNode);
};

/*!
 * \brief The base class of statement doc.
 *
 * \sa StmtDoc
 */
class StmtDocNode : public DocNode {
 public:
  /*!
   * \brief The comment of this doc.
   *
   * The actual position of the comment depends on the type of Doc
   * and also the DocPrinter implementation. It could be on the same
   * line as the statement, or the line above, or inside the statement
   * if it spans over multiple lines.
   * */
  mutable Optional<StringRef> comment{NullOpt};

  void VisitAttrs(AttrVisitor* v) {
    DocNode::VisitAttrs(v);
    v->Visit("comment", &comment);
  }

  static constexpr const char* _type_key = "ir.printer.StmtDoc";
  MATXSCRIPT_DECLARE_BASE_OBJECT_INFO(StmtDocNode, DocNode);
};

/*!
 * \brief Reference type of StmtDocNode.
 *
 * \sa StmtDocNode
 */
class StmtDoc : public Doc {
 protected:
  StmtDoc() = default;

 public:
  MATXSCRIPT_DEFINE_NOTNULLABLE_OBJECT_REF_METHODS(StmtDoc, Doc, StmtDocNode);
};

/*!
 * \brief The container doc that holds a list of StmtDoc.
 * \note `StmtBlockDoc` is never used in the IR, but a temporary container that allows holding a
 * list of StmtDoc.
 * \sa StmtBlockDoc
 */
class StmtBlockDocNode : public DocNode {
 public:
  /*! \brief The list of statements. */
  Array<StmtDoc> stmts;

  void VisitAttrs(AttrVisitor* v) {
    DocNode::VisitAttrs(v);
    v->Visit("stmts", &stmts);
  }

  static constexpr const char* _type_key = "ir.printer.StmtBlockDoc";
  MATXSCRIPT_DECLARE_FINAL_OBJECT_INFO(StmtBlockDocNode, DocNode);
};

/*!
 * \brief Reference type of StmtBlockDocNode.
 * \sa StmtBlockDocNode
 */
class StmtBlockDoc : public Doc {
 public:
  /*!
   * \brief Constructor of StmtBlockDoc.
   * \param stmts The list of statements.
   */
  explicit StmtBlockDoc(Array<StmtDoc> stmts);
  MATXSCRIPT_DEFINE_NOTNULLABLE_OBJECT_REF_METHODS(StmtBlockDoc, Doc, StmtBlockDocNode);
};

/*!
 * \brief Doc that represents literal value.
 *
 * \sa LiteralDoc
 */
class LiteralDocNode : public ExprDocNode {
 public:
  /*!
   * \brief the internal representation of the literal value.
   *
   * Possible actual types:
   * - IntImm (integer or boolean)
   * - FloatImm
   * - StringRef
   * - null
   */
  ObjectRef value;

  void VisitAttrs(AttrVisitor* v) {
    ExprDocNode::VisitAttrs(v);
    v->Visit("value", &value);
  }

  static constexpr const char* _type_key = "ir.printer.LiteralDoc";
  MATXSCRIPT_DECLARE_FINAL_OBJECT_INFO(LiteralDocNode, ExprDocNode);
};

/*!
 * \brief Reference type of LiteralDocNode.
 *
 * \sa LiteralDocNode
 */
class LiteralDoc : public ExprDoc {
 protected:
  explicit LiteralDoc(ObjectRef value, const Optional<ObjectPath>& object_path);

 public:
  /*!
   * \brief Create a LiteralDoc to represent None/null/empty value.
   * \param p The object path
   */
  static LiteralDoc None(const Optional<ObjectPath>& p) {
    return LiteralDoc(ObjectRef(nullptr), p);
  }
  /*!
   * \brief Create a LiteralDoc to represent integer.
   * \param v The integer value.
   * \param p The object path
   */
  static LiteralDoc Int(int64_t v, const Optional<ObjectPath>& p) {
    return LiteralDoc(IntImm(DataType::Int(64), v), p);
  }
  /*!
   * \brief Create a LiteralDoc to represent boolean.
   * \param v The boolean value.
   * \param p The object path
   */
  static LiteralDoc Boolean(bool v, const Optional<ObjectPath>& p) {
    return LiteralDoc(IntImm(DataType::Bool(), v), p);
  }
  /*!
   * \brief Create a LiteralDoc to represent float.
   * \param v The float value.
   * \param p The object path
   */
  static LiteralDoc Float(double v, const Optional<ObjectPath>& p) {
    return LiteralDoc(FloatImm(DataType::Float(64), v), p);
  }
  /*!
   * \brief Create a LiteralDoc to represent string.
   * \param v The string value.
   * \param p The object path
   */
  static LiteralDoc Str(const StringRef& v, const Optional<ObjectPath>& p) {
    return LiteralDoc(v, p);
  }
  /*!
   * \brief Create a LiteralDoc to represent string.
   * \param v The string value.
   * \param p The object path
   */
  static LiteralDoc DataType(const runtime::DataType& v, const Optional<ObjectPath>& p) {
    StringRef dtype = v.is_void() ? "void" : runtime::DLDataType2String(v);
    return LiteralDoc::Str(dtype, p);
  }

  /*!
   * \brief Create a LiteralDoc to represent string.
   * \param v The string value.
   * \param p The object path
   */
  static LiteralDoc HLOType(const Type& v, const Optional<ObjectPath>& p) {
    if (auto const* pt = v.as<PrimTypeNode>()) {
      return LiteralDoc::DataType(pt->dtype, p);
    }
    StringRef dtype = v->GetPythonTypeName().encode();
    return LiteralDoc::Str(dtype, p);
  }

  MATXSCRIPT_DEFINE_NOTNULLABLE_OBJECT_REF_METHODS(LiteralDoc, ExprDoc, LiteralDocNode);
};

/*!
 * \brief Doc that represents identifier.
 *
 * \sa IdDoc
 */
class IdDocNode : public ExprDocNode {
 public:
  /*! \brief The name of the identifier */
  StringRef name;

  void VisitAttrs(AttrVisitor* v) {
    ExprDocNode::VisitAttrs(v);
    v->Visit("name", &name);
  }

  static constexpr const char* _type_key = "ir.printer.IdDoc";
  MATXSCRIPT_DECLARE_FINAL_OBJECT_INFO(IdDocNode, ExprDocNode);
};

/*!
 * \brief Reference type of IdDocNode.
 *
 * \sa IdDocNode
 */
class IdDoc : public ExprDoc {
 public:
  /*!
   * \brief Constructor of IdDoc.
   * \param name The name of identifier.
   */
  explicit IdDoc(StringRef name);
  explicit IdDoc(std::nullptr_t) : ExprDoc(nullptr) {
  }
  MATXSCRIPT_DEFINE_NOTNULLABLE_OBJECT_REF_METHODS(IdDoc, ExprDoc, IdDocNode);
};

/*!
 * \brief Doc that represents attribute access on another expression.
 *
 * \sa AttrAccessDoc
 */
class AttrAccessDocNode : public ExprDocNode {
 public:
  /*! \brief The target expression to be accessed */
  ExprDoc value{nullptr};
  /*! \brief The attribute to be accessed */
  StringRef name;

  void VisitAttrs(AttrVisitor* v) {
    ExprDocNode::VisitAttrs(v);
    v->Visit("value", &value);
    v->Visit("name", &name);
  }

  static constexpr const char* _type_key = "ir.printer.AttrAccessDoc";
  MATXSCRIPT_DECLARE_FINAL_OBJECT_INFO(AttrAccessDocNode, ExprDocNode);
};

/*!
 * \brief Reference type of AttrAccessDocNode.
 *
 * \sa AttrAccessDocNode
 */
class AttrAccessDoc : public ExprDoc {
 public:
  /*!
   * \brief Constructor of AttrAccessDoc
   * \param value The target expression of attribute access.
   * \param name The name of attribute to access.
   */
  explicit AttrAccessDoc(ExprDoc value, StringRef name);
  MATXSCRIPT_DEFINE_NOTNULLABLE_OBJECT_REF_METHODS(AttrAccessDoc, ExprDoc, AttrAccessDocNode);
};

/*!
 * \brief Doc that represents index access on another expression.
 *
 * \sa IndexDoc
 */
class IndexDocNode : public ExprDocNode {
 public:
  /*! \brief The container value to be accessed */
  ExprDoc value{nullptr};
  /*!
   * \brief The indices to access
   *
   * Possible actual types:
   * - ExprDoc (single point access like a[1, 2])
   * - SliceDoc (slice access like a[1:5, 2])
   */
  Array<Doc> indices;  // Each element is union of: Slice / ExprDoc

  void VisitAttrs(AttrVisitor* v) {
    ExprDocNode::VisitAttrs(v);
    v->Visit("value", &value);
    v->Visit("indices", &indices);
  }

  static constexpr const char* _type_key = "ir.printer.IndexDoc";
  MATXSCRIPT_DECLARE_FINAL_OBJECT_INFO(IndexDocNode, ExprDocNode);
};

/*!
 * \brief Reference type of IndexDocNode.
 *
 * \sa IndexDocNode
 */
class IndexDoc : public ExprDoc {
 public:
  /*!
   * \brief Constructor of IndexDoc
   * \param value The target expression of index access.
   * \param indices The indices to access.
   */
  explicit IndexDoc(ExprDoc value, Array<Doc> indices);
  MATXSCRIPT_DEFINE_NOTNULLABLE_OBJECT_REF_METHODS(IndexDoc, ExprDoc, IndexDocNode);
};

/*!
 * \brief Doc that represents function call.
 *
 * \sa CallDoc
 */
class CallDocNode : public ExprDocNode {
 public:
  /*! \brief The callee of this function call */
  ExprDoc callee{nullptr};
  /*! \brief The positional arguments */
  Array<ExprDoc> args;
  /*! \brief The keys of keyword arguments */
  Array<StringRef> kwargs_keys;
  /*!
   * \brief The values of keyword arguments.
   *
   * The i-th element is the value of the i-th key in `kwargs_keys`.
   * It must have the same length as `kwargs_keys`.
   */
  Array<ExprDoc> kwargs_values;

  void VisitAttrs(AttrVisitor* v) {
    ExprDocNode::VisitAttrs(v);
    v->Visit("callee", &callee);
    v->Visit("args", &args);
    v->Visit("kwargs_keys", &kwargs_keys);
    v->Visit("kwargs_values", &kwargs_values);
  }

  static constexpr const char* _type_key = "ir.printer.CallDoc";
  MATXSCRIPT_DECLARE_FINAL_OBJECT_INFO(CallDocNode, ExprDocNode);
};

/*!
 * \brief Reference type of CallDocNode.
 *
 * \sa CallDocNode
 */
class CallDoc : public ExprDoc {
 public:
  /*!
   * \brief Constructor of CallDoc
   * \param callee The callee of this function call.
   * \param args The positional arguments.
   * \param kwargs_keys Keys of keyword arguments.
   * \param kwargs_values Values of keyword arguments, must have the same length as `kwargs_keys.
   */
  CallDoc(ExprDoc callee,
          Array<ExprDoc> args,
          Array<StringRef> kwargs_keys,
          Array<ExprDoc> kwargs_values);
  MATXSCRIPT_DEFINE_NOTNULLABLE_OBJECT_REF_METHODS(CallDoc, ExprDoc, CallDocNode);
};

/*!
 * \brief Doc that represents operation.
 *
 * It can be unary, binary and other special operators (for example,
 * the if-then-else expression).
 *
 * \sa OperationDoc
 */
class OperationDocNode : public ExprDocNode {
 public:
  enum class Kind : int32_t {
    // Unary operators
    kUnaryStart = 0,
    kUSub = 1,    // -x
    kInvert = 2,  // ~x
    kNot = 3,     // not x
    kUnaryEnd = 4,

    // Binary operators
    kBinaryStart = 5,
    kAdd = 6,        // +
    kSub = 7,        // -
    kMult = 8,       // *
    kDiv = 9,        // /
    kFloorDiv = 10,  // // in Python
    kMod = 11,       // % in Python
    kPow = 12,       // ** in Python
    kLShift = 13,    // <<
    kRShift = 14,    // >>
    kBitAnd = 15,    // &
    kBitOr = 16,     // |
    kBitXor = 17,    // ^
    kLt = 18,        // <
    kLtE = 19,       // <=
    kEq = 20,        // ==
    kNotEq = 21,     // !=
    kGt = 22,        // >
    kGtE = 23,       // >=
    kAnd = 24,       // and
    kOr = 25,        // or
    kIn = 26,        // in
    kNotIn = 27,     // not in
    kBinaryEnd = 28,

    // Special
    kSpecialStart = 27,
    kIfThenElse = 28,  // <operands[1]> if <operands[0]> else <operands[2]>
    kSpecialEnd = 29
  };

  /*! \brief The kind of operation (operator) */
  Kind kind;
  /*! \brief Operands of this expression */
  Array<ExprDoc> operands;

  void VisitAttrs(AttrVisitor* v) {
    ExprDocNode::VisitAttrs(v);
    v->Visit("kind", &kind);
    v->Visit("operands", &operands);
  }

  static constexpr const char* _type_key = "ir.printer.OperationDoc";
  MATXSCRIPT_DECLARE_FINAL_OBJECT_INFO(OperationDocNode, ExprDocNode);
};

/*!
 * \brief Reference type of OperationDocNode.
 *
 * \sa OperationDocNode
 */
class OperationDoc : public ExprDoc {
 public:
  /*!
   * \brief Constructor of OperationDoc
   * \param kind The kind of operation.
   * \param operands Operands of this expression.
   */
  explicit OperationDoc(OperationDocNode::Kind kind, Array<ExprDoc> operands);
  MATXSCRIPT_DEFINE_NOTNULLABLE_OBJECT_REF_METHODS(OperationDoc, ExprDoc, OperationDocNode);
};

/*!
 * \brief Doc that represents anonymous function.
 *
 * LambdaDoc can only have positional arguments without type annotation,
 * and a single expression as body.
 *
 * \sa LambdaDoc
 */
class LambdaDocNode : public ExprDocNode {
 public:
  /*! \brief The arguments of this anonymous function */
  Array<IdDoc> args;
  /*! \brief The body of this anonymous function */
  ExprDoc body{nullptr};

  void VisitAttrs(AttrVisitor* v) {
    ExprDocNode::VisitAttrs(v);
    v->Visit("args", &args);
    v->Visit("body", &body);
  }

  static constexpr const char* _type_key = "ir.printer.LambdaDoc";
  MATXSCRIPT_DECLARE_FINAL_OBJECT_INFO(LambdaDocNode, ExprDocNode);
};

/*!
 * \brief Reference type of LambdaDocNode.
 *
 * \sa LambdaDocNode
 */
class LambdaDoc : public ExprDoc {
 public:
  /*!
   * \brief Constructor of LambdaDoc
   * \param args Arguments of this function.
   * \param body Body expression of this function.
   */
  explicit LambdaDoc(Array<IdDoc> args, ExprDoc body);
  MATXSCRIPT_DEFINE_NOTNULLABLE_OBJECT_REF_METHODS(LambdaDoc, ExprDoc, LambdaDocNode);
};

/*!
 * \brief Doc that represents tuple literal.
 *
 * \sa TupleDoc
 */
class TupleDocNode : public ExprDocNode {
 public:
  /*! \brief Elements of tuple */
  Array<ExprDoc> elements;

  void VisitAttrs(AttrVisitor* v) {
    ExprDocNode::VisitAttrs(v);
    v->Visit("elements", &elements);
  }

  static constexpr const char* _type_key = "ir.printer.TupleDoc";
  MATXSCRIPT_DECLARE_FINAL_OBJECT_INFO(TupleDocNode, ExprDocNode);
};

/*!
 * \brief Reference type of TupleDocNode.
 *
 * \sa TupleDocNode
 */
class TupleDoc : public ExprDoc {
 public:
  /*!
   * \brief Create an empty TupleDoc
   */
  TupleDoc() : TupleDoc(runtime::make_object<TupleDocNode>()) {
  }
  /*!
   * \brief Constructor of TupleDoc
   * \param elements Elements of tuple.
   */
  explicit TupleDoc(Array<ExprDoc> elements);
  MATXSCRIPT_DEFINE_NOTNULLABLE_OBJECT_REF_METHODS(TupleDoc, ExprDoc, TupleDocNode);
};

/*!
 * \brief Doc that represents list literal.
 *
 * \sa AttrAccessDoc
 */
class ListDocNode : public ExprDocNode {
 public:
  /*! \brief Elements of list */
  Array<ExprDoc> elements;

  void VisitAttrs(AttrVisitor* v) {
    ExprDocNode::VisitAttrs(v);
    v->Visit("elements", &elements);
  }

  static constexpr const char* _type_key = "ir.printer.ListDoc";
  MATXSCRIPT_DECLARE_FINAL_OBJECT_INFO(ListDocNode, ExprDocNode);
};

/*!
 * \brief Reference type of ListDocNode.
 *
 * \sa ListDocNode
 */
class ListDoc : public ExprDoc {
 public:
  /*!
   * \brief Create an empty ListDoc
   */
  ListDoc() : ListDoc(runtime::make_object<ListDocNode>()) {
  }
  /*!
   * \brief Constructor of ListDoc
   * \param elements Elements of list.
   */
  explicit ListDoc(Array<ExprDoc> elements);
  MATXSCRIPT_DEFINE_NOTNULLABLE_OBJECT_REF_METHODS(ListDoc, ExprDoc, ListDocNode);
};

/*!
 * \brief Doc that represents dictionary literal.
 *
 * \sa AttrAccessDoc
 */
class DictDocNode : public ExprDocNode {
 public:
  /*! \brief keys of dictionary */
  Array<ExprDoc> keys;
  /*!
   * \brief Values of dictionary
   *
   * The i-th element is the value of the i-th element of `keys`.
   * It must have the same length as `keys`.
   */
  Array<ExprDoc> values;

  void VisitAttrs(AttrVisitor* v) {
    ExprDocNode::VisitAttrs(v);
    v->Visit("keys", &keys);
    v->Visit("values", &values);
  }

  static constexpr const char* _type_key = "ir.printer.DictDoc";
  MATXSCRIPT_DECLARE_FINAL_OBJECT_INFO(DictDocNode, ExprDocNode);
};

/*!
 * \brief Reference type of DictDocNode.
 *
 * \sa DictDocNode
 */
class DictDoc : public ExprDoc {
 public:
  /*!
   * \brief Create an empty dictionary
   */
  DictDoc() : DictDoc(runtime::make_object<DictDocNode>()) {
  }
  /*!
   * \brief Constructor of DictDoc
   * \param keys Keys of dictionary.
   * \param values Values of dictionary, must have same length as `keys`.
   */
  explicit DictDoc(Array<ExprDoc> keys, Array<ExprDoc> values);
  MATXSCRIPT_DEFINE_NOTNULLABLE_OBJECT_REF_METHODS(DictDoc, ExprDoc, DictDocNode);
};

/*!
 * \brief Doc that represents set literal.
 *
 * \sa SetDoc
 */
class SetDocNode : public ExprDocNode {
 public:
  /*! \brief Elements of list */
  Array<ExprDoc> elements;

  void VisitAttrs(AttrVisitor* v) {
    ExprDocNode::VisitAttrs(v);
    v->Visit("elements", &elements);
  }

  static constexpr const char* _type_key = "ir.printer.SetDoc";
  MATXSCRIPT_DECLARE_FINAL_OBJECT_INFO(SetDocNode, ExprDocNode);
};

/*!
 * \brief Reference type of SetDocNode.
 *
 * \sa SetDocNode
 */
class SetDoc : public ExprDoc {
 public:
  /*!
   * \brief Create an empty SetDoc
   */
  SetDoc() : SetDoc(runtime::make_object<SetDocNode>()) {
  }
  /*!
   * \brief Constructor of SetDoc
   * \param elements Elements of list.
   */
  explicit SetDoc(Array<ExprDoc> elements);
  MATXSCRIPT_DEFINE_NOTNULLABLE_OBJECT_REF_METHODS(SetDoc, ExprDoc, SetDocNode);
};

/*!
 * \brief Representing the comprehension.
 */
class ComprehensionDocNode : public ExprDocNode {
 public:
  ExprDoc target{nullptr};
  ExprDoc iter{nullptr};
  Optional<Array<ExprDoc>> ifs{nullptr};

  void VisitAttrs(AttrVisitor* v) {
    v->Visit("target", &target);
    v->Visit("iter", &iter);
    v->Visit("ifs", &ifs);
  }

  static constexpr const char* _type_key = "ir.printer.ComprehensionDoc";
  MATXSCRIPT_DECLARE_FINAL_OBJECT_INFO(ComprehensionDocNode, ExprDocNode);
};

/*!
 * \brief Managed reference to ComprehensionNode.
 * \sa ComprehensionNode
 */
class ComprehensionDoc : public ExprDoc {
 public:
  MATX_DLL ComprehensionDoc(ExprDoc target, ExprDoc iter, Optional<Array<ExprDoc>> ifs);

  MATXSCRIPT_DEFINE_NOTNULLABLE_OBJECT_REF_METHODS(ComprehensionDoc, ExprDoc, ComprehensionDocNode);
};

/*!
 * \brief Representing the ListComp.
 */
class ListCompDocNode : public ExprDocNode {
 public:
  ExprDoc elt{nullptr};
  Array<ComprehensionDoc> generators;

  void VisitAttrs(AttrVisitor* v) {
    v->Visit("elt", &elt);
    v->Visit("generators", &generators);
  }

  static constexpr const char* _type_key = "ir.printer.ListCompDoc";
  MATXSCRIPT_DECLARE_FINAL_OBJECT_INFO(ListCompDocNode, ExprDocNode);
};

/*!
 * \brief Managed reference to ListCompDocNode.
 * \sa ListCompDocNode
 */
class ListCompDoc : public ExprDoc {
 public:
  MATX_DLL ListCompDoc(ExprDoc elt, Array<ComprehensionDoc> generators);

  MATXSCRIPT_DEFINE_NOTNULLABLE_OBJECT_REF_METHODS(ListCompDoc, ExprDoc, ListCompDocNode);
};

/*!
 * \brief Representing the SetComp.
 */
class SetCompDocNode : public ExprDocNode {
 public:
  ExprDoc elt{nullptr};
  Array<ComprehensionDoc> generators;

  void VisitAttrs(AttrVisitor* v) {
    v->Visit("elt", &elt);
    v->Visit("generators", &generators);
  }

  static constexpr const char* _type_key = "ir.printer.SetCompDoc";
  MATXSCRIPT_DECLARE_FINAL_OBJECT_INFO(SetCompDocNode, ExprDocNode);
};

/*!
 * \brief Managed reference to SetCompDocNode.
 * \sa SetCompDocNode
 */
class SetCompDoc : public ExprDoc {
 public:
  MATX_DLL SetCompDoc(ExprDoc elt, Array<ComprehensionDoc> generators);

  MATXSCRIPT_DEFINE_NOTNULLABLE_OBJECT_REF_METHODS(SetCompDoc, ExprDoc, SetCompDocNode);
};

/*!
 * \brief Representing the DictComp.
 */
class DictCompDocNode : public ExprDocNode {
 public:
  ExprDoc key{nullptr};
  ExprDoc value{nullptr};
  Array<ComprehensionDoc> generators;

  void VisitAttrs(AttrVisitor* v) {
    v->Visit("key", &key);
    v->Visit("value", &value);
    v->Visit("generators", &generators);
  }

  static constexpr const char* _type_key = "ir.printer.DictCompDoc";
  MATXSCRIPT_DECLARE_FINAL_OBJECT_INFO(DictCompDocNode, ExprDocNode);
};

/*!
 * \brief Managed reference to DictCompDocNode.
 * \sa DictCompDocNode
 */
class DictCompDoc : public ExprDoc {
 public:
  MATX_DLL DictCompDoc(ExprDoc key, ExprDoc value, Array<ComprehensionDoc> generators);

  MATXSCRIPT_DEFINE_NOTNULLABLE_OBJECT_REF_METHODS(DictCompDoc, ExprDoc, DictCompDocNode);
};

/*!
 * \brief Doc that represents slice in Index expression.
 *
 * This doc can only appear in IndexDoc::indices.
 *
 * \sa AttrAccessDoc
 */
class SliceDocNode : public DocNode {
 public:
  /*! \brief The start of slice */
  Optional<ExprDoc> start;
  /*! \brief The exclusive end of slice */
  Optional<ExprDoc> stop;
  /*! \brief The step of slice */
  Optional<ExprDoc> step;

  void VisitAttrs(AttrVisitor* v) {
    DocNode::VisitAttrs(v);
    v->Visit("start", &start);
    v->Visit("stop", &stop);
    v->Visit("step", &step);
  }

  static constexpr const char* _type_key = "ir.printer.SliceDoc";
  MATXSCRIPT_DECLARE_FINAL_OBJECT_INFO(SliceDocNode, DocNode);
};

/*!
 * \brief Reference type of SliceDocNode.
 *
 * \sa SliceDocNode
 */
class SliceDoc : public Doc {
 public:
  /*!
   * \brief Constructor of SliceDoc
   * \param start The start of slice.
   * \param stop The exclusive end of slice.
   * \param step The step of slice.
   */
  explicit SliceDoc(Optional<ExprDoc> start, Optional<ExprDoc> stop, Optional<ExprDoc> step);
  MATXSCRIPT_DEFINE_NOTNULLABLE_OBJECT_REF_METHODS(SliceDoc, Doc, SliceDocNode);
};

/*!
 * \brief Doc that represents assign statement.
 *
 * \sa AssignDoc
 */
class AssignDocNode : public StmtDocNode {
 public:
  /*! \brief The left hand side of the assignment */
  ExprDoc lhs{nullptr};
  /*!
   * \brief The right hand side of the assignment.
   *
   * If null, this doc represents declaration, e.g. `A: T.Buffer((1,2))`
   * */
  Optional<ExprDoc> rhs;
  /*! \brief The type annotation of this assignment. */
  Optional<ExprDoc> annotation;

  void VisitAttrs(AttrVisitor* v) {
    StmtDocNode::VisitAttrs(v);
    v->Visit("lhs", &lhs);
    v->Visit("rhs", &rhs);
    v->Visit("annotation", &annotation);
  }

  static constexpr const char* _type_key = "ir.printer.AssignDoc";
  MATXSCRIPT_DECLARE_FINAL_OBJECT_INFO(AssignDocNode, StmtDocNode);
};

/*!
 * \brief Reference type of AssignDocNode.
 *
 * \sa AssignDoc
 */
class AssignDoc : public StmtDoc {
 public:
  /*!
   * \brief Constructor of AssignDoc.
   * \param lhs The left hand side of the assignment.
   * \param rhs The right hand side of the assignment.
   * \param annotation The type annotation of this assignment.
   */
  explicit AssignDoc(ExprDoc lhs, Optional<ExprDoc> rhs, Optional<ExprDoc> annotation);
  MATXSCRIPT_DEFINE_NOTNULLABLE_OBJECT_REF_METHODS(AssignDoc, StmtDoc, AssignDocNode);
};

/*!
 * \brief Doc that represent if-then-else statement.
 *
 * \sa IfDoc
 */
class IfDocNode : public StmtDocNode {
 public:
  /*! \brief The predicate of the if-then-else statement. */
  ExprDoc predicate{nullptr};
  /*! \brief The then branch of the if-then-else statement. */
  Array<StmtDoc> then_branch;
  /*! \brief The else branch of the if-then-else statement. */
  Array<StmtDoc> else_branch;

  void VisitAttrs(AttrVisitor* v) {
    StmtDocNode::VisitAttrs(v);
    v->Visit("predicate", &predicate);
    v->Visit("then_branch", &then_branch);
    v->Visit("else_branch", &else_branch);
  }

  static constexpr const char* _type_key = "ir.printer.IfDoc";
  MATXSCRIPT_DECLARE_FINAL_OBJECT_INFO(IfDocNode, StmtDocNode);
};

/*!
 * \brief Reference type of IfDocNode.
 *
 * \sa IfDocNode
 */
class IfDoc : public StmtDoc {
 public:
  /*!
   * \brief Constructor of IfDoc.
   * \param predicate The predicate of the if-then-else statement.
   * \param then_branch The then branch of the if-then-else statement.
   * \param else_branch The else branch of the if-then-else statement.
   */
  explicit IfDoc(ExprDoc predicate, Array<StmtDoc> then_branch, Array<StmtDoc> else_branch);
  MATXSCRIPT_DEFINE_NOTNULLABLE_OBJECT_REF_METHODS(IfDoc, StmtDoc, IfDocNode);
};

/*!
 * \brief Doc that represents while statement.
 *
 * \sa WhileDoc
 */
class WhileDocNode : public StmtDocNode {
 public:
  /*! \brief The predicate of the while statement. */
  ExprDoc predicate{nullptr};
  /*! \brief The body of the while statement. */
  Array<StmtDoc> body;

  void VisitAttrs(AttrVisitor* v) {
    StmtDocNode::VisitAttrs(v);
    v->Visit("predicate", &predicate);
    v->Visit("body", &body);
  }

  static constexpr const char* _type_key = "ir.printer.WhileDoc";
  MATXSCRIPT_DECLARE_FINAL_OBJECT_INFO(WhileDocNode, StmtDocNode);
};

/*!
 * \brief Reference type of WhileDocNode.
 *
 * \sa WhileDocNode
 */
class WhileDoc : public StmtDoc {
 public:
  /*!
   * \brief Constructor of WhileDoc.
   * \param predicate The predicate of the while statement.
   * \param body The body of the while statement.
   */
  explicit WhileDoc(ExprDoc predicate, Array<StmtDoc> body);
  MATXSCRIPT_DEFINE_NOTNULLABLE_OBJECT_REF_METHODS(WhileDoc, StmtDoc, WhileDocNode);
};

/*!
 * \brief Doc that represents for statement.
 *
 * Example:
 * for 'lhs' in 'rhs':
 *   'body...'
 *
 * \sa ForDoc
 */
class ForDocNode : public StmtDocNode {
 public:
  /*! \brief The left hand side of the assignment of iterating variable. */
  ExprDoc lhs{nullptr};
  /*! \brief The right hand side of the assignment of iterating variable. */
  ExprDoc rhs{nullptr};
  /*! \brief The body of the for statement. */
  Array<StmtDoc> body;

  void VisitAttrs(AttrVisitor* v) {
    StmtDocNode::VisitAttrs(v);
    v->Visit("lhs", &lhs);
    v->Visit("rhs", &rhs);
    v->Visit("body", &body);
  }

  static constexpr const char* _type_key = "ir.printer.ForDoc";
  MATXSCRIPT_DECLARE_FINAL_OBJECT_INFO(ForDocNode, StmtDocNode);
};

/*!
 * \brief Reference type of ForDocNode.
 *
 * \sa ForDocNode
 */
class ForDoc : public StmtDoc {
 public:
  /*!
   * \brief Constructor of ForDoc.
   * \param lhs The left hand side of the assignment of iterating variable.
   * \param rhs The right hand side of the assignment of iterating variable.
   * \param body The body of the for statement.
   */
  explicit ForDoc(ExprDoc lhs, ExprDoc rhs, Array<StmtDoc> body);
  MATXSCRIPT_DEFINE_NOTNULLABLE_OBJECT_REF_METHODS(ForDoc, StmtDoc, ForDocNode);
};

/*!
 * \brief Doc that represents break statement.
 *
 * \sa BreakDoc
 */
class BreakDocNode : public StmtDocNode {
 public:
  void VisitAttrs(AttrVisitor* v) {
    StmtDocNode::VisitAttrs(v);
  }

  static constexpr const char* _type_key = "ir.printer.BreakDoc";
  MATXSCRIPT_DECLARE_FINAL_OBJECT_INFO(BreakDocNode, StmtDocNode);
};

/*!
 * \brief Reference type of BreakDocNode.
 *
 * \sa BreakDocNode
 */
class BreakDoc : public StmtDoc {
 public:
  /*!
   * \brief Constructor of BreakDoc.
   */
  explicit BreakDoc();
  MATXSCRIPT_DEFINE_NOTNULLABLE_OBJECT_REF_METHODS(BreakDoc, StmtDoc, BreakDocNode);
};

/*!
 * \brief Doc that represents continue statement.
 *
 * \sa ContinueDoc
 */
class ContinueDocNode : public StmtDocNode {
 public:
  void VisitAttrs(AttrVisitor* v) {
    StmtDocNode::VisitAttrs(v);
  }

  static constexpr const char* _type_key = "ir.printer.ContinueDoc";
  MATXSCRIPT_DECLARE_FINAL_OBJECT_INFO(ContinueDocNode, StmtDocNode);
};

/*!
 * \brief Reference type of ContinueDocNode.
 *
 * \sa ContinueDocNode
 */
class ContinueDoc : public StmtDoc {
 public:
  /*!
   * \brief Constructor of ContinueDoc.
   */
  explicit ContinueDoc();
  MATXSCRIPT_DEFINE_NOTNULLABLE_OBJECT_REF_METHODS(ContinueDoc, StmtDoc, ContinueDocNode);
};

/*!
 * \brief Doc that represents ExceptionHandler statement.
 *
 * \sa ExceptionHandlerDoc
 */
class ExceptionHandlerDocNode : public StmtDocNode {
 public:
  /*! \brief The except type of the except statement. */
  Optional<ExprDoc> type{nullptr};
  /*! \brief The var name of the except statement. */
  Optional<ExprDoc> name{nullptr};
  /*! \brief The body of the except statement. */
  Array<StmtDoc> body;

 public:
  void VisitAttrs(AttrVisitor* v) {
    StmtDocNode::VisitAttrs(v);
    v->Visit("type", &type);
    v->Visit("name", &name);
    v->Visit("body", &body);
  }

  static constexpr const char* _type_key = "ir.printer.ExceptionHandlerDoc";
  MATXSCRIPT_DECLARE_FINAL_OBJECT_INFO(ExceptionHandlerDocNode, StmtDocNode);
};

/*!
 * \brief Reference type of ExceptionHandlerDocNode.
 *
 * \sa ExceptionHandlerDocNode
 */
class ExceptionHandlerDoc : public StmtDoc {
 public:
  /*!
   * \brief Constructor of ExceptionHandlerDoc.
   */
  explicit ExceptionHandlerDoc(Optional<ExprDoc> type, Optional<ExprDoc> name, Array<StmtDoc> body);
  MATXSCRIPT_DEFINE_NOTNULLABLE_OBJECT_REF_METHODS(ExceptionHandlerDoc,
                                                   StmtDoc,
                                                   ExceptionHandlerDocNode);
};

/*!
 * \brief Doc that represents TryExcept statement.
 *
 * \sa TryExceptDoc
 */
class TryExceptDocNode : public StmtDocNode {
 public:
  /*! \brief The body of the try statement. */
  Array<StmtDoc> body;
  /*! \brief The handlers of the try statement. */
  Optional<Array<ExceptionHandlerDoc>> handlers{nullptr};
  /*! \brief The orelse of the try statement. */
  Optional<Array<StmtDoc>> orelse{nullptr};
  /*! \brief The finalbody of the try statement. */
  Optional<Array<StmtDoc>> finalbody{nullptr};

 public:
  void VisitAttrs(AttrVisitor* v) {
    StmtDocNode::VisitAttrs(v);
    v->Visit("body", &body);
    v->Visit("handlers", &handlers);
    v->Visit("orelse", &orelse);
    v->Visit("finalbody", &finalbody);
  }

  static constexpr const char* _type_key = "ir.printer.TryExceptDoc";
  MATXSCRIPT_DECLARE_FINAL_OBJECT_INFO(TryExceptDocNode, StmtDocNode);
};

/*!
 * \brief Reference type of TryExceptDocNode.
 *
 * \sa TryExceptDocNode
 */
class TryExceptDoc : public StmtDoc {
 public:
  /*!
   * \brief Constructor of TryExceptDoc.
   */
  explicit TryExceptDoc(Array<StmtDoc> body,
                        Optional<Array<ExceptionHandlerDoc>> handlers,
                        Optional<Array<StmtDoc>> orelse,
                        Optional<Array<StmtDoc>> finalbody);
  MATXSCRIPT_DEFINE_NOTNULLABLE_OBJECT_REF_METHODS(TryExceptDoc, StmtDoc, TryExceptDocNode);
};

/*!
 * \brief Doc that represents raise statement.
 *
 * \sa RaiseDoc
 */
class RaiseDocNode : public StmtDocNode {
 public:
  /*! \brief The exc of the raise statement. */
  Optional<ExprDoc> exc{nullptr};
  /*! \brief The cause of the raise statement. */
  Optional<ExprDoc> cause{nullptr};

 public:
  void VisitAttrs(AttrVisitor* v) {
    StmtDocNode::VisitAttrs(v);
    v->Visit("exc", &exc);
    v->Visit("cause", &cause);
  }

  static constexpr const char* _type_key = "ir.printer.RaiseDoc";
  MATXSCRIPT_DECLARE_FINAL_OBJECT_INFO(RaiseDocNode, StmtDocNode);
};

/*!
 * \brief Reference type of RaiseDocNode.
 *
 * \sa RaiseDocNode
 */
class RaiseDoc : public StmtDoc {
 public:
  /*!
   * \brief Constructor of RaiseDoc.
   */
  explicit RaiseDoc(Optional<ExprDoc> exc, Optional<ExprDoc> cause);
  MATXSCRIPT_DEFINE_NOTNULLABLE_OBJECT_REF_METHODS(RaiseDoc, StmtDoc, RaiseDocNode);
};

/*!
 * \brief Doc that represents special scopes.
 *
 * Specifically, this means the with statement in Python:
 *
 * with 'rhs' as 'lhs':
 *   'body...'
 *
 * \sa ScopeDoc
 */
class ScopeDocNode : public StmtDocNode {
 public:
  /*! \brief The name of the scoped variable. */
  Optional<ExprDoc> lhs{NullOpt};
  /*! \brief The value of the scoped variable. */
  ExprDoc rhs{nullptr};
  /*! \brief The body of the scope doc. */
  Array<StmtDoc> body;

  void VisitAttrs(AttrVisitor* v) {
    StmtDocNode::VisitAttrs(v);
    v->Visit("lhs", &lhs);
    v->Visit("rhs", &rhs);
    v->Visit("body", &body);
  }

  static constexpr const char* _type_key = "ir.printer.ScopeDoc";
  MATXSCRIPT_DECLARE_FINAL_OBJECT_INFO(ScopeDocNode, StmtDocNode);
};

/*!
 * \brief Reference type of ScopeDocNode.
 *
 * \sa ScopeDocNode
 */
class ScopeDoc : public StmtDoc {
 public:
  /*!
   * \brief Constructor of ScopeDoc.
   * \param lhs The name of the scoped variable.
   * \param rhs The value of the scoped variable.
   * \param body The body of the scope doc.
   */
  explicit ScopeDoc(Optional<ExprDoc> lhs, ExprDoc rhs, Array<StmtDoc> body);

  /*!
   * \brief Constructor of ScopeDoc.
   * \param rhs The value of the scoped variable.
   * \param body The body of the scope doc.
   */
  explicit ScopeDoc(ExprDoc rhs, Array<StmtDoc> body);

  MATXSCRIPT_DEFINE_NOTNULLABLE_OBJECT_REF_METHODS(ScopeDoc, StmtDoc, ScopeDocNode);
};

/*!
 * \brief Doc that represents an expression as statement.
 *
 * \sa ExprStmtDoc
 */
class ExprStmtDocNode : public StmtDocNode {
 public:
  /*! \brief The expression represented by this doc. */
  ExprDoc expr{nullptr};

  void VisitAttrs(AttrVisitor* v) {
    StmtDocNode::VisitAttrs(v);
    v->Visit("expr", &expr);
  }

  static constexpr const char* _type_key = "ir.printer.ExprStmtDoc";
  MATXSCRIPT_DECLARE_FINAL_OBJECT_INFO(ExprStmtDocNode, StmtDocNode);
};

/*!
 * \brief Reference type of ExprStmtDocNode.
 *
 * \sa ExprStmtDocNode
 */
class ExprStmtDoc : public StmtDoc {
 public:
  /*!
   * \brief Constructor of ExprStmtDoc.
   * \param expr The expression represented by this doc.
   */
  explicit ExprStmtDoc(ExprDoc expr);
  MATXSCRIPT_DEFINE_NOTNULLABLE_OBJECT_REF_METHODS(ExprStmtDoc, StmtDoc, ExprStmtDocNode);
};

/*!
 * \brief Doc that represents assert statement.
 *
 * \sa AssertDoc
 */
class AssertDocNode : public StmtDocNode {
 public:
  /*! \brief The expression to test. */
  ExprDoc test{nullptr};
  /*! \brief The optional error message when assertion failed. */
  Optional<ExprDoc> msg{NullOpt};

  void VisitAttrs(AttrVisitor* v) {
    StmtDocNode::VisitAttrs(v);
    v->Visit("test", &test);
    v->Visit("msg", &msg);
  }

  static constexpr const char* _type_key = "ir.printer.AssertDoc";
  MATXSCRIPT_DECLARE_FINAL_OBJECT_INFO(AssertDocNode, StmtDocNode);
};

/*!
 * \brief Reference type of AssertDocNode.
 *
 * \sa AssertDocNode
 */
class AssertDoc : public StmtDoc {
 public:
  /*!
   * \brief Constructor of AssertDoc.
   * \param test The expression to test.
   * \param msg The optional error message when assertion failed.
   */
  explicit AssertDoc(ExprDoc test, Optional<ExprDoc> msg = NullOpt);
  MATXSCRIPT_DEFINE_NOTNULLABLE_OBJECT_REF_METHODS(AssertDoc, StmtDoc, AssertDocNode);
};

/*!
 * \brief Doc that represents return statement.
 *
 * \sa ReturnDoc
 */
class ReturnDocNode : public StmtDocNode {
 public:
  /*! \brief The value to return. */
  ExprDoc value{nullptr};

  void VisitAttrs(AttrVisitor* v) {
    StmtDocNode::VisitAttrs(v);
    v->Visit("value", &value);
  }

  static constexpr const char* _type_key = "ir.printer.ReturnDoc";
  MATXSCRIPT_DECLARE_FINAL_OBJECT_INFO(ReturnDocNode, StmtDocNode);
};

/*!
 * \brief Reference type of ReturnDocNode.
 *
 * \sa ReturnDocNode
 */
class ReturnDoc : public StmtDoc {
 public:
  /*!
   * \brief Constructor of ReturnDoc.
   * \param value The value to return.
   */
  explicit ReturnDoc(ExprDoc value);
  MATXSCRIPT_DEFINE_NOTNULLABLE_OBJECT_REF_METHODS(ReturnDoc, StmtDoc, ReturnDocNode);
};

/*!
 * \brief Doc that represents function definition.
 *
 * \sa FunctionDoc
 */
class FunctionDocNode : public StmtDocNode {
 public:
  /*! \brief The name of function. */
  IdDoc name{nullptr};
  /*!
   * \brief The arguments of function.
   *
   * The `lhs` means argument name,
   * `annotation` means argument type,
   * and `rhs` means default value.
   */
  Array<AssignDoc> args;
  /*! \brief Decorators of function. */
  Array<ExprDoc> decorators;
  /*! \brief The return type of function. */
  Optional<ExprDoc> return_type{NullOpt};
  /*! \brief The body of function. */
  Array<StmtDoc> body;

  void VisitAttrs(AttrVisitor* v) {
    StmtDocNode::VisitAttrs(v);
    v->Visit("name", &name);
    v->Visit("args", &args);
    v->Visit("decorators", &decorators);
    v->Visit("return_type", &return_type);
    v->Visit("body", &body);
  }

  static constexpr const char* _type_key = "ir.printer.FunctionDoc";
  MATXSCRIPT_DECLARE_FINAL_OBJECT_INFO(FunctionDocNode, StmtDocNode);
};

/*!
 * \brief Reference type of FunctionDocNode.
 *
 * \sa FunctionDocNode
 */
class FunctionDoc : public StmtDoc {
 public:
  /*!
   * \brief Constructor of FunctionDoc.
   * \param name The name of function..
   * \param args The arguments of function.
   * \param decorators The decorator of function.
   * \param return_type The return type of function.
   * \param body The body of function.
   */
  explicit FunctionDoc(IdDoc name,
                       Array<AssignDoc> args,
                       Array<ExprDoc> decorators,
                       Optional<ExprDoc> return_type,
                       Array<StmtDoc> body);
  MATXSCRIPT_DEFINE_NOTNULLABLE_OBJECT_REF_METHODS(FunctionDoc, StmtDoc, FunctionDocNode);
};

/*!
 * \brief Doc that represents class definition.
 *
 * \sa ClassDoc
 */
class ClassDocNode : public StmtDocNode {
 public:
  /*! \brief The name of class. */
  IdDoc name{nullptr};
  /*! \brief The name of class. */
  IdDoc base{nullptr};
  /*! \brief Decorators of class. */
  Array<ExprDoc> decorators;
  /*! \brief The body of class. */
  Array<StmtDoc> body;

  void VisitAttrs(AttrVisitor* v) {
    StmtDocNode::VisitAttrs(v);
    v->Visit("name", &name);
    v->Visit("base", &base);
    v->Visit("decorators", &decorators);
    v->Visit("body", &body);
  }

  static constexpr const char* _type_key = "ir.printer.ClassDoc";
  MATXSCRIPT_DECLARE_FINAL_OBJECT_INFO(ClassDocNode, StmtDocNode);
};

/*!
 * \brief Reference type of ClassDocNode.
 *
 * \sa ClassDocNode
 */
class ClassDoc : public StmtDoc {
 public:
  /*!
   * \brief Constructor of ClassDoc.
   * \param name The name of class.
   * \param decorators The decorator of class.
   * \param body The body of class.
   */
  explicit ClassDoc(IdDoc name, IdDoc base, Array<ExprDoc> decorators, Array<StmtDoc> body);
  MATXSCRIPT_DEFINE_NOTNULLABLE_OBJECT_REF_METHODS(ClassDoc, StmtDoc, ClassDocNode);
};

/*!
 * \brief Doc that represents comment.
 *
 * \sa CommentDoc
 */
class CommentDocNode : public StmtDocNode {
 public:
  static constexpr const char* _type_key = "ir.printer.CommentDoc";
  MATXSCRIPT_DECLARE_FINAL_OBJECT_INFO(CommentDocNode, StmtDocNode);
};

/*!
 * \brief Reference type of CommentDocNode.
 *
 * \sa CommentDocNode
 */
class CommentDoc : public StmtDoc {
 public:
  explicit CommentDoc(StringRef comment);
  MATXSCRIPT_DEFINE_NOTNULLABLE_OBJECT_REF_METHODS(CommentDoc, StmtDoc, CommentDocNode);
};

/*!
 * \brief Doc that represents docstring.
 *
 * \sa DocStringDoc
 */
class DocStringDocNode : public StmtDocNode {
 public:
  static constexpr const char* _type_key = "ir.printer.DocStringDoc";
  MATXSCRIPT_DECLARE_FINAL_OBJECT_INFO(DocStringDocNode, StmtDocNode);
};

/*!
 * \brief Reference type of DocStringDocNode.
 *
 * \sa DocStringDocNode
 */
class DocStringDoc : public StmtDoc {
 public:
  explicit DocStringDoc(StringRef docs);
  MATXSCRIPT_DEFINE_NOTNULLABLE_OBJECT_REF_METHODS(DocStringDoc, StmtDoc, DocStringDocNode);
};

/*!
 * \brief Doc that represents module file.
 *
 * \sa ModuleDoc
 */
class ModuleDocNode : public StmtDocNode {
 public:
  /*! \brief The body of class. */
  Array<StmtDoc> body;

  void VisitAttrs(AttrVisitor* v) {
    StmtDocNode::VisitAttrs(v);
    v->Visit("body", &body);
  }

  static constexpr const char* _type_key = "ir.printer.ModuleDoc";
  MATXSCRIPT_DECLARE_FINAL_OBJECT_INFO(ModuleDocNode, StmtDocNode);
};

/*!
 * \brief Reference type of ModuleDocNode.
 *
 * \sa ModuleDocNode
 */
class ModuleDoc : public StmtDoc {
 public:
  /*!
   * \brief Constructor of ModuleDoc.
   * \param body The body of class.
   */
  explicit ModuleDoc(Array<StmtDoc> body);
  MATXSCRIPT_DEFINE_NOTNULLABLE_OBJECT_REF_METHODS(ModuleDoc, StmtDoc, ModuleDocNode);
};

}  // namespace printer
}  // namespace ir
}  // namespace matxscript
