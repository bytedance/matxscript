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
#include <matxscript/runtime/global_type_index.h>
#include <matxscript/runtime/object.h>
#include <matxscript/runtime/runtime_port.h>

namespace matxscript {
namespace runtime {

MATX_DLL const char* TypeIndex2Str(int32_t type_code) {
  switch (type_code) {
    case TypeIndex::kRuntimeInteger:
      return "int";
    case TypeIndex::kRuntimeFloat:
      return "float";
    case TypeIndex::kRuntimeOpaqueHandle:
      return "handle";
    case TypeIndex::kRuntimeNullptr:
      return "nullptr";
    case TypeIndex::kMATXByteArray:
      return "ByteArray";
    case TypeIndex::kRuntimeDataType:
      return "DLDataType";
    case TypeIndex::kRuntimeContext:
      return "MATXContext";
    case TypeIndex::kRuntimeDLTensorHandle:
      return "ArrayHandle";
    case TypeIndex::kRuntimePackedFuncHandle:
      return "MATXFunctionHandle";
    case TypeIndex::kRuntimeObjectRValueRefArg:
      return "ObjectRValueRefArg";
    case TypeIndex::kRoot:
      return "Object";
    case TypeIndex::kRuntimeString:
      return "String";
    case TypeIndex::kRuntimeUnicode:
      return "Unicode";
    case TypeIndex::kRuntimeList:
      return "List";
    case TypeIndex::kRuntimeDict:
      return "Dict";
    case TypeIndex::kRuntimeSet:
      return "Set";
    case TypeIndex::kRuntimeFTList:
      return "FTList";
    case TypeIndex::kRuntimeFTDict:
      return "FTDict";
    case TypeIndex::kRuntimeFTSet:
      return "FTSet";
    case TypeIndex::kRuntimeNDArray:
      return "NDArray";
    case TypeIndex::kRuntimeFile:
      return "File";
    case TypeIndex::kRuntimeTrie:
      return "Trie";
    case TypeIndex::kRuntimeTuple:
      return "Tuple";
    case TypeIndex::kRuntimeRegex:
      return "Regex";
    case TypeIndex::kRuntimeIterator:
      return "Iterator";
    case TypeIndex::kRuntimeUserData:
      return "UserData";
    case TypeIndex::kRuntimeOpaqueObject:
      return "OpaqueObject";
    case TypeIndex::kRuntimeUnknown:
      return "Unknown";
    default:
      if (type_code >= 0) {
        string_view name;
        if (Object::TryTypeIndex2Key(type_code, &name)) {
          return name.data();
        } else {
          return "Object";
        }
      } else {
        return "Unknown";
      }
  }
}

MATX_DLL int32_t Str2TypeIndex(string_view str) {
  if (str == "int") {
    return TypeIndex::kRuntimeInteger;
  } else if (str == "float") {
    return TypeIndex::kRuntimeFloat;
  } else if (str == "handle") {
    return TypeIndex::kRuntimeOpaqueHandle;
  } else if (str == "nullptr") {
    return TypeIndex::kRuntimeNullptr;
  } else if (str == "Object") {
    return TypeIndex::kRoot;
  } else if (str == "String") {
    return TypeIndex::kRuntimeString;
  } else if (str == "Unicode") {
    return TypeIndex::kRuntimeUnicode;
  } else if (str == "List") {
    return TypeIndex::kRuntimeList;
  } else if (str == "Dict") {
    return TypeIndex::kRuntimeDict;
  } else if (str == "Set") {
    return TypeIndex::kRuntimeSet;
  } else if (str == "NDArray") {
    return TypeIndex::kRuntimeNDArray;
  } else if (str == "File") {
    return TypeIndex::kRuntimeFile;
  } else if (str == "Trie") {
    return TypeIndex::kRuntimeTrie;
  } else if (str == "Tuple") {
    return TypeIndex::kRuntimeTuple;
  } else if (str == "Regex") {
    return TypeIndex::kRuntimeRegex;
  } else if (str == "Iterator") {
    return TypeIndex::kRuntimeIterator;
  } else if (str == "UserData") {
    return TypeIndex::kRuntimeUserData;
  } else if (str == "OpaqueObject") {
    return TypeIndex::kRuntimeOpaqueObject;
  } else {
    return Object::TypeKey2Index(str);
  }
}

}  // namespace runtime
}  // namespace matxscript
