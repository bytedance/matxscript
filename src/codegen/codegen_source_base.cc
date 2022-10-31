// Copyright 2022 ByteDance Ltd. and/or its affiliates.
/*
 * Acknowledgement: The structure of the codegen is inspired by TVM.
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
 * \file codegen_source_base.cc
 */
#include "codegen_source_base.h"

namespace matxscript {
namespace codegen {

void CodeGenSourceBase::ClearFuncState() {
  name_alloc_map_.clear();
  ssa_assign_map_.clear();
  var_idmap_.clear();
  scope_mark_.clear();
}

runtime::String CodeGenSourceBase::GetUniqueName(runtime::String prefix) {
  for (size_t i = 0; i < prefix.size(); ++i) {
    if (prefix[i] == '.')
      prefix[i] = '_';
  }
  auto it = name_alloc_map_.find(prefix);
  if (it != name_alloc_map_.end()) {
    while (true) {
      std::ostringstream os;
      os << prefix << (++it->second);
      runtime::String name = os.str();
      if (name_alloc_map_.count(name) == 0) {
        prefix = name;
        break;
      }
    }
  }
  name_alloc_map_[prefix] = 0;
  return prefix;
}

runtime::String CodeGenSourceBase::SSAGetID(runtime::String src, ir::Type t, std::ostream& os) {
  if (name_alloc_map_.count(src))
    return src;
  auto it = ssa_assign_map_.find(src);
  if (it != ssa_assign_map_.end()) {
    if (scope_mark_.at(it->second.scope_id)) {
      return it->second.vid;
    }
  }
  SSAEntry e;
  e.vid = GetUniqueName("_");
  e.scope_id = static_cast<int>(scope_mark_.size() - 1);
  ssa_assign_map_[src] = e;
  this->PrintIndent(os);
  PrintSSAAssign(e.vid, src, t, os);
  return e.vid;
}

runtime::String CodeGenSourceBase::AllocVarID(const ir::PrimVarNode* v) {
  MXCHECK(!var_idmap_.count(v)) << "Need input to be in SSA form dup " << v->name_hint;
  runtime::String key = v->name_hint;
  runtime::String vid = GetUniqueName(key);
  var_idmap_[v] = vid;
  return vid;
}

runtime::String CodeGenSourceBase::AllocVarID(const ir::HLOVarNode* v) {
  MXCHECK(!var_idmap_.count(v)) << "Need input to be in SSA form dup " << v->name_hint();
  runtime::String key = v->name_hint();
  runtime::String vid = GetUniqueName(key);
  var_idmap_[v] = vid;
  return vid;
}

runtime::String CodeGenSourceBase::GetVarID(const ir::PrimVarNode* v) const {
  auto it = var_idmap_.find(v);
  MXCHECK(it != var_idmap_.end()) << "Find undefined Variable " << v->name_hint;
  return it->second;
}

runtime::String CodeGenSourceBase::GetVarID(const ir::HLOVarNode* v) const {
  auto it = var_idmap_.find(v);
  MXCHECK(it != var_idmap_.end()) << "Find undefined Variable " << v->name_hint();
  return it->second;
}

void CodeGenSourceBase::PrintIndent(std::ostream& os) {
  for (int i = 0; i < indent_; ++i) {
    os << ' ';
  }
}

void CodeGenSourceBase::MarkConst(runtime::String vid) {
  auto it = ssa_assign_map_.find(vid);
  if (it == ssa_assign_map_.end()) {
    SSAEntry e;
    e.vid = vid;
    e.scope_id = 0;
    ssa_assign_map_[vid] = e;
  } else {
    MXCHECK_EQ(it->second.vid, vid);
  }
}

int CodeGenSourceBase::BeginScope() {
  int sid = static_cast<int>(scope_mark_.size());
  scope_mark_.push_back(true);
  indent_ += 2;
  return sid;
}

void CodeGenSourceBase::EndScope(int scope_id) {
  scope_mark_[scope_id] = false;
  indent_ -= 2;
}

}  // namespace codegen
}  // namespace matxscript
