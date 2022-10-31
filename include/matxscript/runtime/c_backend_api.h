// Copyright 2022 ByteDance Ltd. and/or its affiliates.
/*
 * Acknowledgement: The structure of the api is inspired by TVM.
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
 * \file matx/runtime/c_backend_api.h
 * \brief MATX runtime backend API.
 *
 *  The functions defined in this header are intended to be
 *  used by compiled operators, usually user do not need to use these
 *  function directly.
 */
#pragma once

#include <matxscript/runtime/c_runtime_api.h>

#ifdef __cplusplus
extern "C" {
#endif

/*!
 * \brief Signature for backend functions exported as DLL.
 *
 * \param args The arguments
 * \param num_args Number of arguments.
 * \param out_ret_value The output value of the the return value.
 * \param resource_handle Pointer to associated resource.
 *
 * \return 0 if success, -1 if failure happens, set error via MATXScriptAPISetLastError.
 */
typedef int (*MATXScriptBackendPackedCFunc)(MATXScriptAny* args,
                                            int num_args,
                                            MATXScriptAny* out_ret_value,
                                            void* resource_handle);
/*!
 * \brief Backend function to register system-wide library symbol.
 *
 * \param name The name of the symbol
 * \param ptr The symbol address.
 * \return 0 when no error is thrown, -1 when failure happens
 */
MATX_DLL int MATXScriptBackendRegisterSystemLibSymbol(const char* name, void* ptr);

/*!
 * \brief A data structure that facilitates function lookup by C-string name.
 */
typedef struct MATXScriptFuncRegistry {
  /*! \brief Names of registered functions, concatenated together and separated by \0.
   * An additional \0 is present at the end of the concatenated blob to mark the end.
   *
   * Byte 0 is the number of functions in `funcs`.
   */
  const char* names;

  /*! \brief Function pointers, in the same order as their names in `names`. */
  const MATXScriptBackendPackedCFunc* funcs;
} MATXScriptFuncRegistry;

#ifdef __cplusplus
}  // MATXSCRIPT_EXTERN_C
#endif
