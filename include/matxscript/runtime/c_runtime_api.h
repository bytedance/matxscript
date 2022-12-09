// Copyright 2022 ByteDance Ltd. and/or its affiliates.
/*
 * Acknowledgement:
 * The c api and DeviceAPIManager structure design originates from TVM.
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

/*
 * \file matx/runtime/c_runtime_api.h
 * \brief MATX runtime library.
 *
 *  The common flow is:
 *   - Use MATXFuncListGlobalNames to get global function name
 *   - Use MATXFuncCall to call these functions.
 */
#pragma once

#include <matxscript/runtime/global_type_index.h>
#include <matxscript/runtime/runtime_port.h>

#define MATXSCRIPT_RUNTIME_VERSION "4.0"

// MATXSCRIPT Runtime is DLPack compatible.
#include <matxscript/runtime/dlpack.h>

#ifdef __cplusplus
extern "C" {
#endif
#include <stddef.h>
#include <stdint.h>

/*! \brief type of array index. */
typedef int64_t matx_script_index_t;

/*!
 * \brief The Device information, abstract away common device types.
 */
typedef DLDevice MATXScriptDevice;

/*! \brief the array handle */
typedef DLTensor* MATXScriptTensorHandle;

/*!
 * \brief Union type of values
 *  being passed through API and function calls.
 */
typedef struct {
  union {
    unsigned char* bytes;
    char32_t* chars;
  };
  size_t size;
} MATXScriptStringMediumLarge;

typedef union {
  unsigned char v_small_bytes[sizeof(MATXScriptStringMediumLarge)];
  char32_t v_small_chars[sizeof(MATXScriptStringMediumLarge) / sizeof(char32_t)];
  MATXScriptStringMediumLarge v_ml;
} MATXScriptStringStorage;

typedef union {
  int64_t v_int64;
  double v_float64;
  void* v_handle;
  const char* v_str;
  DLDataType v_type;
  MATXScriptDevice v_device;
  MATXScriptStringStorage v_str_store;
} MATXScriptValue;

typedef struct {
  MATXScriptValue data;
  int32_t pad;  // category_or_len for String/Unicode
  int32_t code;
} MATXScriptAny;

/*! \brief Handle to MATXScript runtime modules. */
typedef void* MATXScriptModuleHandle;
/*! \brief Handle to packed function handle. */
typedef void* MATXScriptFunctionHandle;
/*! \brief Handle to hold return value. */
typedef void* MATXScriptValueHandle;
/*!
 * \brief The stream that is specific to device
 * can be NULL, which indicates the default one.
 */
typedef void* MATXScriptStreamHandle;
/*! \brief Handle to Object. */
typedef void* MATXScriptObjectHandle;

MATX_DLL int MATXScriptAPI_USE_CXX11_ABI();

/*!
 * \brief Used for implementing C API function.
 *  Set last error message before return.
 * \param msg The error message to be set.
 */
MATX_DLL void MATXScriptAPISetLastError(const char* msg);

/*!
 * \brief return str message of the last error
 *  all function in this file will return 0 when success
 *  and -1 when an error occurred,
 *  MATXScriptAPIGetLastError can be called to retrieve the error
 *
 *  this function is threadsafe and can be called by different thread
 *  \return error info
 */
MATX_DLL const char* MATXScriptAPIGetLastError(void);

/*!
 * \brief Load module from file.
 * \param file_name The file name to load the module from.
 * \param format The format of the module.
 * \param out The result module
 *
 * \return 0 when success, -1 when failure happens
 * \note The resulting module do not contain import relation.
 *  It can be reconstructed by MATXScriptModImport.
 */
MATX_DLL int MATXScriptModLoadFromFile(const char* file_name,
                                       const char* format,
                                       MATXScriptModuleHandle* out);

/*!
 * \brief Add dep to mod's dependency.
 *  This allows functions in this module to use modules.
 *
 * \param mod The module handle.
 * \param dep The dependent module to be imported.
 * \return 0 when success, -1 when failure happens
 */
MATX_DLL int MATXScriptModImport(MATXScriptModuleHandle mod, MATXScriptModuleHandle dep);

/*!
 * \brief Get function from the module.
 * \param mod The module handle.
 * \param func_name The name of the function.
 * \param query_imports Whether to query imported modules
 * \param out The result function, can be NULL if it is not available.
 * \return 0 when no error is thrown, -1 when failure happens
 */
MATX_DLL int MATXScriptModGetFunction(MATXScriptModuleHandle mod,
                                      const char* func_name,
                                      int query_imports,
                                      MATXScriptFunctionHandle* out);

/*!
 * \brief Free the Module
 * \param mod The module to be freed.
 *
 * \note This may not free up the module's resources.
 *  If there is active MATXFunctionHandle uses the module
 *  Or if this module is imported by another active module.
 *
 *  The all functions remains valid until MATXScriptFuncFree is called.
 * \return 0 when success, -1 when failure happens
 */
MATX_DLL int MATXScriptModFree(MATXScriptModuleHandle mod);

/*!
 * \brief Free the function when it is no longer needed.
 * \param func The function handle
 * \return 0 when success, -1 when failure happens
 */
MATX_DLL int MATXScriptFuncFree(MATXScriptFunctionHandle func);

/*!
 * \brief Call a Packed MATX Function.
 *
 * \param func node handle of the function.
 * \param arg_values The arguments
 * \param num_args Number of arguments.
 *
 * \param ret_val The return value.
 *
 * \return 0 when success, -1 when failure happens
 * \note MATX calls always exchanges with type bits=64, lanes=1
 *
 * \note API calls always exchanges with type bits=64, lanes=1
 *   If API call returns container handles (e.g. FunctionHandle)
 *   these handles should be managed by the front-end.
 *   The front-end need to call free function (e.g. MATXScriptFuncFree)
 *   to free these handles.
 */
MATX_DLL int MATXScriptFuncCall_PYTHON_C_API(MATXScriptFunctionHandle func,
                                             MATXScriptAny* arg_values,
                                             int num_args,
                                             MATXScriptAny* ret_val);

MATX_DLL int MATXScriptAPIDLDataTypeToString(DLDataType dtype, char* buffer, int* size);

/*!
 * \brief Increase the reference of an object.
 *
 * \param value The object handle.
 * \note Internally we increase the reference of the object.
 * \return 0 when success, -1 when failure happens
 */
MATX_DLL int MATXScriptRuntimeRetain(MATXScriptAny* value);

/**
 * \brief Free MATXScriptAny.
 *
 * \param values The arguments
 * \param num Number of arguments.
 *
 * \return 0 when success, -1 when failure happens
 */
MATX_DLL int MATXScriptRuntimeDestroyN(MATXScriptAny* values, int num);

/**
 * \brief Free MATXScriptAny.
 *
 * \param value The argument
 *
 * \return 0 when success, -1 when failure happens
 */
MATX_DLL int MATXScriptRuntimeDestroy(MATXScriptAny* value);

/**
 * \brief Call a TXSession.
 *
 * \param session_handle TXSession pointer
 * \param arg_keys The keys of arguments
 * \param arg_values The values of arguments
 * \param num_args Number of arguments.
 * \param move_mode Whether to move the arguments to native.
 * \param num_rets The Number of return value.
 * \param ret_val The return values.
 *
 * \return 0 when success, -1 when failure happens
 */
MATX_DLL int MATXScriptPipelineTXSessionRun(void* session_handle,
                                            const char** keys,
                                            MATXScriptAny* arg_values,
                                            int num_args,
                                            int move_mode,
                                            int* num_rets,
                                            MATXScriptAny* ret_val);

/**
 * \brief Call a OpKernel.
 *
 * \param op_handle OpKernel pointer
 * \param arg_values The arguments
 * \param num_args Number of arguments.
 * \param move_mode Whether to move the arguments to native.
 * \param ret_val The return value.
 *
 * \return 0 when success, -1 when failure happens
 */
MATX_DLL int MATXScriptPipelineOpKernelCall(void* op_handle,
                                            MATXScriptAny* arg_values,
                                            int num_args,
                                            int move_mode,
                                            MATXScriptAny* ret_val);

/**
 * \brief Make a Native Bytes
 *
 * \param buffer The argument
 * \param size The argument
 * \param ret_val The return value.
 *
 * \return 0 when success, -1 when failure happens
 */
MATX_DLL int MATXScriptRuntimeMakeString(const char* buffer, size_t size, MATXScriptAny* ret_val);

/**
 * \brief Make a Native Str
 *
 * \param buffer The argument
 * \param size The argument
 * \param ret_val The return value.
 *
 * \return 0 when success, -1 when failure happens
 */
MATX_DLL int MATXScriptRuntimeMakeUnicode(const char* buffer, size_t size, MATXScriptAny* ret_val);

/**
 * \brief
 *
 * \param arg_value The argument
 * \param ret_val The return value.
 *
 * @return 0 when success, -1 when failure happens
 */
MATX_DLL int MATXScriptRuntimeUnicodeEncode(MATXScriptAny* arg_value, MATXScriptAny* ret_val);

/**
 * \brief Make a Native List
 *
 * \param arg_values The arguments
 * \param num_args Number of arguments.
 * \param move_mode Whether to move the arguments to the list.
 * \param ret_val The return value.
 *
 * \return 0 when success, -1 when failure happens
 */
MATX_DLL int MATXScriptRuntimeMakeList(MATXScriptAny* arg_values,
                                       int num_args,
                                       int move_mode,
                                       MATXScriptAny* ret_val);

/**
 * \brief Get Size of a Native List
 *
 * \param arg_value The generic list
 * \param size The return value.
 *
 * \return 0 when success, -1 when failure happens
 */
MATX_DLL int MATXScriptRuntimeGetListSize(MATXScriptAny* arg_value, int64_t* size);

/**
 * \brief Get Items of a Native List
 *
 * \param arg_value The generic list
 * \param move_mode Whether move the argument
 * \param num_rets Size of return the of values.
 * \param ret_val The return values.
 *
 * \return 0 when success, -1 when failure happens
 */
MATX_DLL int MATXScriptRuntimeGetListItems(MATXScriptAny* arg_value,
                                           int move_mode,
                                           int64_t* num_rets,
                                           MATXScriptAny* ret_val);

/**
 * \brief Make a Native Dict
 *
 * \param arg_values The arguments
 * \param num_args Number of arguments.
 * \param move_mode Whether to move the arguments to the dict.
 * \param ret_val The return value.
 *
 * \return 0 when success, -1 when failure happens
 */
MATX_DLL int MATXScriptRuntimeMakeDict(MATXScriptAny* arg_values,
                                       int num_args,
                                       int move_mode,
                                       MATXScriptAny* ret_val);

/**
 * \brief Get Size of a Native Dict
 *
 * \param arg_value The generic dict
 * \param size The return value.
 *
 * \return 0 when success, -1 when failure happens
 */
MATX_DLL int MATXScriptRuntimeGetDictSize(MATXScriptAny* arg_value, int64_t* size);

/**
 * \brief Get Items of a Native Dict
 *
 * \param arg_value The generic dict
 * \param move_mode Whether move the argument
 * \param num_rets Size of return the of values.
 * \param ret_val The return values.
 *
 * \return 0 when success, -1 when failure happens
 */
MATX_DLL int MATXScriptRuntimeGetDictItems(MATXScriptAny* arg_value,
                                           int move_mode,
                                           int64_t* num_rets,
                                           MATXScriptAny* ret_val);

/**
 * \brief Make a Native Set
 *
 * \param arg_values The arguments
 * \param num_args Number of arguments.
 * \param move_mode Whether to move the arguments to the set.
 * \param ret_val The return value.
 *
 * \return 0 when success, -1 when failure happens
 */
MATX_DLL int MATXScriptRuntimeMakeSet(MATXScriptAny* arg_values,
                                      int num_args,
                                      int move_mode,
                                      MATXScriptAny* ret_val);

/**
 * \brief Get Size of a Native Set
 *
 * \param arg_value The generic set
 * \param size The return value.
 *
 * \return 0 when success, -1 when failure happens
 */
MATX_DLL int MATXScriptRuntimeGetSetSize(MATXScriptAny* arg_value, int64_t* size);

/**
 * \brief Get Items of a Native Set
 *
 * \param arg_value The generic set
 * \param move_mode Whether move the argument
 * \param num_rets Size of return the of values.
 * \param ret_val The return values.
 *
 * \return 0 when success, -1 when failure happens
 */
MATX_DLL int MATXScriptRuntimeGetSetItems(MATXScriptAny* arg_value,
                                          int move_mode,
                                          int64_t* num_rets,
                                          MATXScriptAny* ret_val);

/**
 * \brief Make a Native Tuple
 *
 * \param arg_values The arguments
 * \param num_args Number of arguments.
 * \param move_mode Whether to move the arguments to the tuple.
 * \param ret_val The return value.
 *
 * \return 0 when success, -1 when failure happens
 */
MATX_DLL int MATXScriptRuntimeMakeTuple(MATXScriptAny* arg_values,
                                        int num_args,
                                        int move_mode,
                                        MATXScriptAny* ret_val);

/**
 * \brief Get Size of a Native Tuple
 *
 * \param arg_value The generic tuple
 * \param size The return value.
 *
 * \return 0 when success, -1 when failure happens
 */
MATX_DLL int MATXScriptRuntimeGetTupleSize(MATXScriptAny* arg_value, int64_t* size);

/**
 * \brief Get Items of a Native Tuple
 *
 * \param arg_value The generic tuple
 * \param move_mode Whether move the argument
 * \param num_rets Size of return the of values.
 * \param ret_val The return values.
 *
 * \return 0 when success, -1 when failure happens
 */
MATX_DLL int MATXScriptRuntimeGetTupleItems(MATXScriptAny* arg_value,
                                            int move_mode,
                                            int64_t* num_rets,
                                            MATXScriptAny* ret_val);

/*!
 * \brief Set the return value of MATXPackedCFunc.
 *
 *  This function is called by MATXPackedCFunc to set the return value.
 *  When this function is not called, the function returns null by default.
 *
 * \param ret The return value handle, pass by ret in MATXPackedCFunc
 * \param value The value to be returned.
 * \param num_ret Number of return values, for now only 1 is supported.
 */
MATX_DLL int MATXScriptCFuncSetReturn(MATXScriptValueHandle ret, MATXScriptAny* value, int num_ret);

/*!
 * \brief C type of packed function.
 *
 * \param args The arguments
 * \param type_codes The type codes of the arguments
 * \param num_args Number of arguments.
 * \param ret The return value handle.
 * \param resource_handle The handle additional resouce handle from fron-end.
 * \return 0 if success, -1 if failure happens, set error via MATXScriptAPISetLastError.
 * \sa MATXScriptCFuncSetReturn
 */
typedef int (*MATXScriptPackedCFunc)(MATXScriptAny* args,
                                     int num_args,
                                     MATXScriptValueHandle ret,
                                     void* resource_handle);

/*!
 * \brief C callback to free the resource handle in C packed function.
 * \param resource_handle The handle additional resouce handle from fron-end.
 */
typedef void (*MATXScriptPackedCFuncFinalizer)(void* resource_handle);

/*!
 * \brief Wrap a MATXPackedCFunc to become a FunctionHandle.
 *
 * The resource_handle will be managed by MATX API, until the function is no longer used.
 *
 * \param func The packed C function.
 * \param resource_handle The resource handle from front-end, can be NULL.
 * \param fin The finalizer on resource handle when the FunctionHandle get freed, can be NULL
 * \param out the result function handle.
 * \param do_stack_trace_on_error
 * \return 0 when success, -1 when failure happens
 */
MATX_DLL int MATXScriptFuncCreateFromCFunc(MATXScriptPackedCFunc func,
                                           void* resource_handle,
                                           MATXScriptPackedCFuncFinalizer fin,
                                           MATXScriptFunctionHandle* out,
                                           int do_stack_trace_on_error);

/*!
 * \brief Register the function to runtime's global table.
 *
 * The registered function then can be pulled by the backend by the name.
 *
 * \param name The name of the function.
 * \param f The function to be registered.
 * \param override Whether allow override already registered function.
 */
MATX_DLL int MATXScriptFuncRegisterGlobal(const char* name,
                                          MATXScriptFunctionHandle f,
                                          int override);

/*!
 * \brief Get a global function.
 *
 * \param name The name of the function.
 * \param out the result function pointer, NULL if it does not exist.
 *
 * \note The function handle of global function is managed by MATX runtime,
 *  So MATXScriptFuncFree is should not be called when it get deleted.
 */
MATX_DLL int MATXScriptFuncGetGlobal(const char* name, MATXScriptFunctionHandle* out);

/*!
 * \brief List all the globally registered function name
 * \param out_size The number of functions
 * \param out_array The array of function names.
 * \return 0 when success, -1 when failure happens
 */
MATX_DLL int MATXScriptFuncListGlobalNames(int* out_size, const char*** out_array);

// Array related apis for quick proptyping
/*!
 * \brief Allocate a nd-array's memory,
 *  including space of shape, of given spec.
 *
 * \param shape The shape of the array, the data content will be copied to out
 * \param ndim The number of dimension of the array.
 * \param dtype_code The type code of the dtype
 * \param dtype_bits The number of bits of dtype
 * \param dtype_lanes The number of lanes in the dtype.
 * \param device_type The device type of context
 * \param device_id The device id of context.
 * \param out The output handle.
 * \return 0 when success, -1 when failure happens
 */
MATX_DLL int MATXScriptArrayAlloc(const matx_script_index_t* shape,
                                  int ndim,
                                  int dtype_code,
                                  int dtype_bits,
                                  int dtype_lanes,
                                  int device_type,
                                  int device_id,
                                  MATXScriptTensorHandle* out);

/*!
 * \brief Free the MATX Array.
 * \param handle The array handle to be freed.
 * \return 0 when success, -1 when failure happens
 */
MATX_DLL int MATXScriptArrayFree(MATXScriptTensorHandle handle);

/*!
 * \brief Copy array data from CPU byte array.
 * \param handle The array handle.
 * \param data the data pointer
 * \param nbytes The number of bytes to copy.
 * \return 0 when success, -1 when failure happens
 */
MATX_DLL int MATXScriptArrayCopyFromBytes(MATXScriptTensorHandle handle, void* data, size_t nbytes);

/*!
 * \brief Copy array data to CPU byte array.
 * \param handle The array handle.
 * \param data the data pointer
 * \param nbytes The number of bytes to copy.
 * \return 0 when success, -1 when failure happens
 */
MATX_DLL int MATXScriptArrayCopyToBytes(MATXScriptTensorHandle handle, void* data, size_t nbytes);

/*!
 * \brief Copy the array, both from and to must be valid during the copy.
 * \param from The array to be copied from.
 * \param to The target space.
 * \param stream The stream where the copy happens, can be NULL.
 * \return 0 when success, -1 when failure happens
 */
MATX_DLL int MATXScriptArrayCopyFromTo(MATXScriptTensorHandle from,
                                       MATXScriptTensorHandle to,
                                       MATXScriptStreamHandle stream);

/*!
 * \brief Produce an array from the DLManagedTensor that shares data memory
 * with the DLManagedTensor.
 * \param from The source DLManagedTensor.
 * \param out The output array handle.
 * \return 0 when success, -1 when failure happens
 */
MATX_DLL int MATXScriptArrayFromDLPack(DLManagedTensor* from, MATXScriptTensorHandle* out);

/*!
 * \brief Produce a DLMangedTensor from the array that shares data memory with
 * the array.
 * \param from The source array.
 * \param out The DLManagedTensor handle.
 * \return 0 when success, -1 when failure happens
 */
MATX_DLL int MATXScriptArrayToDLPack(MATXScriptTensorHandle from, DLManagedTensor** out);

/*!
 * \brief Delete (free) a DLManagedTensor's data.
 * \param dltensor Pointer to the DLManagedTensor.
 */
MATX_DLL void MATXScriptDLManagedTensorCallDeleter(DLManagedTensor* dltensor);

/*!
 * \brief Create a new runtime stream.
 *
 * \param device_type The device type of context
 * \param device_id The device id of context
 * \param out The new stream handle
 * \return 0 when success, -1 when failure happens
 */
MATX_DLL int MATXScriptStreamCreate(int device_type, int device_id, MATXScriptStreamHandle* out);

/*!
 * \brief Free a created stream handle.
 *
 * \param device_type The device type of context
 * \param device_id The device id of context
 * \param stream The stream to be freed
 * \return 0 when success, -1 when failure happens
 */
MATX_DLL int MATXScriptStreamFree(int device_type, int device_id, MATXScriptStreamHandle stream);

/*!
 * \brief Set the runtime stream of current thread to be stream.
 *  The subsequent calls to the same device_type
 *  will use the setted stream handle.
 *  The specific type of stream is runtime device dependent.
 *
 * \param device_type The device type of context
 * \param device_id The device id of context.
 * \param handle The stream handle.
 * \return 0 when success, -1 when failure happens
 */
MATX_DLL int MATXScriptSetCurrentThreadStream(int device_type,
                                              int device_id,
                                              MATXScriptStreamHandle handle);

/*!
 * \brief Wait until all computations on stream completes.
 *
 * \param device_type The device type of context
 * \param device_id The device id of context.
 * \param stream The stream to be synchronized.
 * \return 0 when success, -1 when failure happens
 */
MATX_DLL int MATXScriptSynchronize(int device_type, int device_id, MATXScriptStreamHandle stream);

/*!
 * \brief Synchronize two streams of execution.
 *
 * \param device_type The device type of context
 * \param device_id The device id of context
 * \param src The source stream to synchronize.
 * \param dst The destination stream to synchronize.
 * \return 0 when success, -1 when failure happens
 */
MATX_DLL int MATXScriptStreamStreamSynchronize(int device_type,
                                               int device_id,
                                               MATXScriptStreamHandle src,
                                               MATXScriptStreamHandle dst);

/*!
 * \brief Get the type_index from an object.
 *
 * \param obj The object handle.
 * \param out_tindex the output type index.
 * \return 0 when success, -1 when failure happens
 */
MATX_DLL int MATXScriptObjectGetTypeIndex(MATXScriptObjectHandle obj, unsigned* out_tindex);

/*!
 * \brief Convert type key to type index.
 * \param type_key The key of the type.
 * \param out_tindex the corresponding type index.
 * \return 0 when success, -1 when failure happens
 */
MATX_DLL int MATXScriptObjectTypeKey2Index(const char* type_key, unsigned* out_tindex);

/*!
 * \brief Increase the reference count of an object.
 *
 * \param obj The object handle.
 * \note Internally we increase the reference counter of the object.
 * \return 0 when success, -1 when failure happens
 */
MATX_DLL int MATXScriptObjectRetain(MATXScriptObjectHandle obj);

/*!
 * \brief Free the object.
 *
 * \param obj The object handle.
 * \note Internally we decrease the reference counter of the object.
 *       The object will be freed when every reference to the object are removed.
 * \return 0 when success, -1 when failure happens
 */
MATX_DLL int MATXScriptObjectFree(MATXScriptObjectHandle obj);

/*!
 * \brief Allocate a data space on device.
 * \param ctx The device context to perform operation.
 * \param nbytes The number of bytes in memory.
 * \param alignment The alignment of the memory.
 * \param type_hint The type of elements. Only needed by certain backends such
 *                   as nbytes & alignment are sufficient for most backends.
 * \param out_data The allocated device pointer.
 * \return 0 when success, -1 when failure happens
 */
MATX_DLL int MATXScriptDeviceAllocDataSpace(
    DLDevice device, size_t nbytes, size_t alignment, DLDataType type_hint, void** out_data);

/*!
 * \brief Free a data space on device.
 * \param ctx The device context to perform operation.
 * \param ptr The data space.
 * \return 0 when success, -1 when failure happens
 */
MATX_DLL int MATXScriptDeviceFreeDataSpace(DLDevice device, void* ptr);

/*!
 * \brief Check that an object is derived from another.
 * \param child_type_index The type index of the derived type.
 * \param parent_type_index The type index of the parent type.
 * \param is_derived A boolean representing whether this predicate holds.
 * \return 0 when success, -1 when failure happens.
 */
MATX_DLL int MATXScriptObjectDerivedFrom(uint32_t child_type_index,
                                         uint32_t parent_type_index,
                                         int* is_derived);

MATX_DLL int MATXScriptNDArrayToDLPack(MATXScriptAny* value, DLManagedTensor** dlpack);

MATX_DLL int MATXScriptNDArrayFromDLPack(void* dlm_tensor, MATXScriptAny* value);

MATX_DLL int MATXScriptSetDeviceDriverError(int device_type, const char* msg);

#ifdef __cplusplus
}  // MATXSCRIPT_EXTERN_C
#endif
