// Copyright 2022 ByteDance Ltd. and/or its affiliates.
/*
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
 * \file src/printer/doc.cc
 * \brief Doc ADT used for pretty printing.
 *
 *  Reference: Philip Wadler. A Prettier Printer. Journal of Functional Programming'98
 */
#include "doc.h"

#include <sstream>
#include <vector>

#include <matxscript/runtime/container/unicode.h>
#include <matxscript/runtime/memory.h>
#include <matxscript/runtime/str_escape.h>
#include <matxscript/runtime/utf8_util.h>

namespace matxscript {
namespace printer {

/*!
 * \brief Represent a piece of text in the doc.
 */
class DocTextNode : public DocAtomNode {
 public:
  /*! \brief The str content in the text. */
  runtime::String str;

  explicit DocTextNode(runtime::String str_val) : str(str_val) {
  }

  static constexpr const char* _type_key = "printer.DocText";
  MATXSCRIPT_DECLARE_FINAL_OBJECT_INFO(DocTextNode, DocAtomNode);
};

MATXSCRIPT_REGISTER_OBJECT_TYPE(DocTextNode);

class DocText : public DocAtom {
 public:
  explicit DocText(runtime::String str) {
    if (str.find_first_of("\t\n") != str.npos) {
      MXLOG(WARNING) << "text node: '" << str << "' should not has tab or newline.";
    }
    data_ = runtime::make_object<DocTextNode>(str);
  }

  MATXSCRIPT_DEFINE_OBJECT_REF_METHODS(DocText, DocAtom, DocTextNode);
};

/*!
 * \brief Represent a line breaker in the doc.
 */
class DocLineNode : public DocAtomNode {
 public:
  /*! \brief The amount of indent in newline. */
  int indent;

  explicit DocLineNode(int indent) : indent(indent) {
  }

  static constexpr const char* _type_key = "printer.DocLine";
  MATXSCRIPT_DECLARE_FINAL_OBJECT_INFO(DocLineNode, DocAtomNode);
};

MATXSCRIPT_REGISTER_OBJECT_TYPE(DocLineNode);

class DocLine : public DocAtom {
 public:
  explicit DocLine(int indent) {
    data_ = runtime::make_object<DocLineNode>(indent);
  }

  MATXSCRIPT_DEFINE_OBJECT_REF_METHODS(DocLine, DocAtom, DocLineNode);
};

// DSL function implementations
Doc& Doc::operator<<(const Doc& right) {
  MXCHECK(this != &right);
  this->stream_.insert(this->stream_.end(), right.stream_.begin(), right.stream_.end());
  return *this;
}

Doc& Doc::operator<<(runtime::String right) {
  return *this << DocText(right);
}

Doc& Doc::operator<<(const DocAtom& right) {
  this->stream_.push_back(right);
  return *this;
}

runtime::String Doc::str() {
  std::ostringstream os;
  for (auto atom : this->stream_) {
    if (auto* text = atom.as<DocTextNode>()) {
      os << text->str;
    } else if (auto* line = atom.as<DocLineNode>()) {
      os << "\n" << runtime::String(line->indent, ' ');
    } else {
      MXLOG(FATAL) << "do not expect type " << atom->GetTypeKey();
    }
  }
  return os.str();
}

Doc Doc::NewLine(int indent) {
  return Doc() << DocLine(indent);
}

Doc Doc::Text(runtime::String text) {
  return Doc() << DocText(text);
}

Doc Doc::RawText(runtime::String text) {
  return Doc() << DocAtom(runtime::make_object<DocTextNode>(text));
}

Doc Doc::Indent(int indent, Doc doc) {
  for (size_t i = 0; i < doc.stream_.size(); ++i) {
    if (auto* line = doc.stream_[i].as<DocLineNode>()) {
      doc.stream_[i] = DocLine(indent + line->indent);
    }
  }
  return doc;
}

Doc Doc::StrLiteral(const runtime::String& value, const runtime::String& quote) {
  Doc doc;
  return doc << quote << runtime::BytesEscape(value) << quote;
}

Doc Doc::PyBoolLiteral(bool value) {
  if (value) {
    return Doc::Text("True");
  } else {
    return Doc::Text("False");
  }
}

Doc Doc::Brace(runtime::String open, const Doc& body, runtime::String close, int indent) {
  Doc doc;
  doc << open;
  doc << Indent(indent, NewLine() << body) << NewLine();
  doc << close;
  return doc;
}

Doc Doc::Concat(const std::vector<Doc>& vec, const Doc& sep) {
  Doc seq;
  if (vec.size() != 0) {
    if (vec.size() == 1)
      return vec[0];
    seq << vec[0];
    for (size_t i = 1; i < vec.size(); ++i) {
      seq << sep << vec[i];
    }
  }
  return seq;
}

}  // namespace printer
}  // namespace matxscript
