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
#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <structmember.h>

#include <matxscript/runtime/c_runtime_api.h>

#include <unordered_map>

static int PyObjectToMATXScriptAny(PyObject* arg_0, MATXScriptAny* value);
static int PyObjectToMATXScriptList(PyObject* arg_0, MATXScriptAny* value);
static int PyObjectToMATXScriptDict(PyObject* arg_0, MATXScriptAny* value);
static int PyObjectToMATXScriptSet(PyObject* arg_0, MATXScriptAny* value);
static int PyObjectToMATXScriptTuple(PyObject* arg_0, MATXScriptAny* value);

static PyObject* matx_script_api_return_switch_impl(MATXScriptAny* value);
static PyObject* matx_script_api__exe_input_instance_callback(PyObject* arg);

/******************************************************************************
 * traceback
 *****************************************************************************/
static PyObject* MATXScriptPythonTraceBackOnErr() {
  PyObject *ptype, *pvalue, *ptraceback;
  PyObject *module_name, *pyth_module, *pyth_func;
  PyObject* result_msg = NULL;

  PyObject* err = PyErr_Occurred();
  if (!err) {
    return NULL;
  }

  PyErr_Fetch(&ptype, &pvalue, &ptraceback);
  result_msg = PyObject_Str(pvalue);

  /* See if we can get a full traceback */
  module_name = PyBytes_FromString("traceback");
  pyth_module = PyImport_Import(module_name);
  Py_DECREF(module_name);

  if (pyth_module == NULL) {
    return result_msg;
  }

  pyth_func = PyObject_GetAttrString(pyth_module, "format_exception");
  if (pyth_func && PyCallable_Check(pyth_func)) {
    PyObject* pyth_val;
    pyth_val = PyObject_CallFunctionObjArgs(pyth_func, ptype, pvalue, ptraceback, NULL);
    Py_DECREF(result_msg);
    result_msg = PyObject_Str(pyth_val);
    Py_DECREF(pyth_val);
  }
  Py_DECREF(pyth_module);
  return result_msg;
}

static int MATXScriptAutoSetLastErrorByPythonTraceback() {
  PyObject* errmsg = MATXScriptPythonTraceBackOnErr();
  if (errmsg) {
    Py_ssize_t len;
    const char* bytes = PyUnicode_AsUTF8AndSize(errmsg, &len);
    MATXScriptAPISetLastError(bytes);
    return -1;
  }
  return 0;
}

/******************************************************************************
 * PyObjectMATXPackedFuncBase
 *****************************************************************************/
typedef struct PyObjectMATXScriptPackedFuncBase {
  PyObject_HEAD;
  void* handle;
  int is_global;
} PyObjectMATXScriptPackedFuncBase;

static PyMemberDef PyObjectMATXPackedFuncBase_Members[] = {
    {
        "handle",                                           /* name */
        T_ULONGLONG,                                        /* type */
        offsetof(PyObjectMATXScriptPackedFuncBase, handle), /* offset */
        0,                                                  /* flags */
        "the handle to the underlying function",            /* docstring */
    },
    {
        "is_global",                                           /* name */
        T_INT,                                                 /* type */
        offsetof(PyObjectMATXScriptPackedFuncBase, is_global), /* offset */
        0,                                                     /* flags */
        "Whether this is a global function in python",         /* docstring */
    },
    {NULL} /* Sentinel */
};

static PyObject* PyObjectMATXScriptPackedFuncBase_new(PyTypeObject* type,
                                                      PyObject* args,
                                                      PyObject* kwargs) {
  PyObjectMATXScriptPackedFuncBase* self;
  self = (PyObjectMATXScriptPackedFuncBase*)type->tp_alloc(type, 0);
  self->handle = NULL;
  self->is_global = 1;
  return (PyObject*)self;
}

static int PyObjectMATXScriptPackedFuncBase_init(PyObject* self0,
                                                 PyObject* args,
                                                 PyObject* kwargs) {
  PyObjectMATXScriptPackedFuncBase* self = (PyObjectMATXScriptPackedFuncBase*)(self0);
  uintptr_t handle = 0x0;
  int is_global = 1;
  if (!PyArg_ParseTuple(args, "Ki", &handle, &is_global)) {
    return -1;
  }
  self->is_global = is_global;
  self->handle = (void*)(handle);
  return 0;
}

static void PyObjectMATXScriptPackedFuncBase_finalize(PyObject* self0) {
  PyObjectMATXScriptPackedFuncBase* self = (PyObjectMATXScriptPackedFuncBase*)(self0);
  PyObject *error_type, *error_value, *error_traceback;

  /* Save the current exception, if any. */
  PyErr_Fetch(&error_type, &error_value, &error_traceback);

  if (!self->is_global) {
    MATXScriptFuncFree(self->handle);
  }

  /* Restore the saved exception. */
  PyErr_Restore(error_type, error_value, error_traceback);
}

static PyObject* PyObjectMATXScriptPackedFuncBase_repr(PyObject* self0) {
  PyObjectMATXScriptPackedFuncBase* self = (PyObjectMATXScriptPackedFuncBase*)(self0);
  return PyUnicode_FromFormat(
      "PackedFuncBase(handle: %p, is_global: %d)", self->handle, self->is_global);
}

PyObject* PyObjectMATXScriptPackedFuncBase_call(PyObject* self0, PyObject* args, PyObject* kwargs) {
  PyObjectMATXScriptPackedFuncBase* self = (PyObjectMATXScriptPackedFuncBase*)(self0);
  Py_ssize_t size = PyTuple_GET_SIZE(args);
  MATXScriptAny* item_buffer = new MATXScriptAny[size];
  PyObject* result = NULL;
  int success_args = 0;
  for (Py_ssize_t i = 0; i < size; ++i) {
    PyObject* item = PyTuple_GET_ITEM(args, i);
    if (0 != PyObjectToMATXScriptAny(item, item_buffer + i)) {
      goto FREE_ARGS;
    }
    ++success_args;
  }

  MATXScriptAny ret_val;
  if (0 != MATXScriptFuncCall_PYTHON_C_API(self->handle, item_buffer, success_args, &ret_val)) {
    PyErr_SetString(PyExc_TypeError, MATXScriptAPIGetLastError());
    goto FREE_ARGS;
  }
  result = matx_script_api_return_switch_impl(&ret_val);

FREE_ARGS:
  MATXScriptRuntimeDestroyN(item_buffer, success_args);
  delete[] item_buffer;
  return result;
}

static PyTypeObject PyType_MATXScriptPackedFuncBase = {
    PyVarObject_HEAD_INIT(&PyType_Type, 0)    /**/
    "matx_script_api.PackedFuncBase",         /* tp_name */
    sizeof(PyObjectMATXScriptPackedFuncBase), /* tp_basicsize */
    0,                                        /* tp_itemsize */
    0,                                        /* tp_dealloc */
    0,                                        /* tp_print */
    0,                                        /* tp_getattr */
    0,                                        /* tp_setattr */
    0,                                        /* tp_reserved */
    PyObjectMATXScriptPackedFuncBase_repr,    /* tp_repr */
    0,                                        /* tp_as_number */
    0,                                        /* tp_as_sequence */
    0,                                        /* tp_as_mapping */
    0,                                        /* tp_hash  */
    0,                                        /* tp_call */
    0,                                        /* tp_str */
    0,                                        /* tp_getattro */
    0,                                        /* tp_setattro */
    0,                                        /* tp_as_buffer */
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE, /* tp_flags */
    "Function base.",                         /* tp_doc */
};

/******************************************************************************
 * PyObjectMATXObjectBase
 *****************************************************************************/

typedef struct PyObjectMATXScriptObjectBase {
  PyObject_HEAD;
  void* handle;
  int32_t type_code;
} PyObjectMATXObjectBase;

static PyMemberDef PyObjectMATXScriptObjectBase_Members[] = {
    {
        "handle",                                       /* name */
        T_ULONGLONG,                                    /* type */
        offsetof(PyObjectMATXScriptObjectBase, handle), /* offset */
        0,                                              /* flags */
        "Object Pointer",                               /* docstring */
    },
    {
        "type_code",                                       /* name */
        T_INT,                                             /* type */
        offsetof(PyObjectMATXScriptObjectBase, type_code), /* offset */
        0,                                                 /* flags */
        "Object TypeCode",                                 /* docstring */
    },
    {NULL} /* Sentinel */
};

static void PyObjectMATXScriptObjectBase_finalize(PyObject* self0) {
  PyObjectMATXScriptObjectBase* self = (PyObjectMATXScriptObjectBase*)(self0);
  PyObject *error_type, *error_value, *error_traceback;

  /* Save the current exception, if any. */
  PyErr_Fetch(&error_type, &error_value, &error_traceback);

  MATXScriptObjectFree(self->handle);

  /* Restore the saved exception. */
  PyErr_Restore(error_type, error_value, error_traceback);
}

static PyObject* PyObjectMATXScriptObjectBase_repr(PyObject* self0) {
  PyObjectMATXScriptObjectBase* self = (PyObjectMATXScriptObjectBase*)(self0);
  return PyUnicode_FromFormat("ObjectBase(code:%d, handle: %p)", self->type_code, self->handle);
}

static PyTypeObject PyType_MATXScriptObjectBase = {
    PyVarObject_HEAD_INIT(&PyType_Type, 0)                               /**/
    "matx_script_api.ObjectBase",                                        /* tp_name */
    sizeof(PyObjectMATXScriptObjectBase),                                /* tp_basicsize */
    0,                                                                   /* tp_itemsize */
    0,                                                                   /* tp_dealloc */
    0,                                                                   /* tp_print */
    0,                                                                   /* tp_getattr */
    0,                                                                   /* tp_setattr */
    0,                                                                   /* tp_reserved */
    PyObjectMATXScriptObjectBase_repr,                                   /* tp_repr */
    0,                                                                   /* tp_as_number */
    0,                                                                   /* tp_as_sequence */
    0,                                                                   /* tp_as_mapping */
    0,                                                                   /* tp_hash  */
    0,                                                                   /* tp_call */
    0,                                                                   /* tp_str */
    0,                                                                   /* tp_getattro */
    0,                                                                   /* tp_setattro */
    0,                                                                   /* tp_as_buffer */
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE | Py_TPFLAGS_HAVE_FINALIZE, /* tp_flags */
    "Base object for all object types",                                  /* tp_doc */
};

PyObject* PyObjectMATXScriptObjectBase_same_as(PyObject* self0, PyObject* other0) {
  PyObjectMATXScriptObjectBase* self = (PyObjectMATXScriptObjectBase*)(self0);
  PyObjectMATXScriptObjectBase* other = (PyObjectMATXScriptObjectBase*)(other0);
  if (PyObject_IsInstance(other0, (PyObject*)&PyType_MATXScriptObjectBase)) {
    if (self->handle == other->handle) {
      Py_RETURN_TRUE;
    }
  }
  Py_RETURN_FALSE;
}

static const char* PyObjectMATXScriptObjectBase_same_as_doc =
    "Check object identity.\n"
    "\n"
    "Parameters\n"
    "----------\n"
    "other : object\n"
    "    The other object to compare against.\n"
    "\n"
    "Returns\n"
    "-------\n"
    "result : bool\n"
    "     The comparison result.";

static const char* PyObjectMATXScriptObjectBase___init_handle_by_constructor___doc =
    "Initialize the handle by calling constructor function.\n"
    "\n"
    "Parameters\n"
    "----------\n"
    "fconstructor : Function\n"
    "    Constructor function.\n"
    "\n"
    "args: list of objects\n"
    "    The arguments to the constructor\n"
    "\n"
    "Note\n"
    "----\n"
    "We have a special calling convention to call constructor functions.\n"
    "So the return handle is directly set into the Node object\n"
    "instead of creating a new Node.";

PyObject* PyObjectMATXScriptObjectBase___init_handle_by_constructor__(PyObject* self,
                                                                      PyObject* args) {
  PyObjectMATXScriptObjectBase* super = (PyObjectMATXScriptObjectBase*)self;
  Py_ssize_t size = PyTuple_GET_SIZE(args);
  MATXScriptAny* item_buffer = new MATXScriptAny[size];
  PyObject* item_0 = NULL;
  void* func_addr = NULL;
  int success_args = 0;
  if (size < 1) {
    PyErr_SetString(PyExc_TypeError, "need one or more args(0 given)");
    goto RETURN_FLAG;
  }
  item_0 = PyTuple_GET_ITEM(args, 0);
  if (!PyObject_IsInstance(item_0, (PyObject*)&PyType_MATXScriptPackedFuncBase)) {
    PyErr_SetString(PyExc_TypeError, "the first argument is not PackedFunc type");
    goto RETURN_FLAG;
  }
  func_addr = ((PyObjectMATXScriptPackedFuncBase*)(item_0))->handle;
  for (Py_ssize_t i = 1; i < size; ++i) {
    PyObject* item = PyTuple_GET_ITEM(args, i);
    if (0 != PyObjectToMATXScriptAny(item, item_buffer + i - 1)) {
      goto FREE_ARGS;
    }
    ++success_args;
  }

  MATXScriptAny ret_val;
  if (0 != MATXScriptFuncCall_PYTHON_C_API(func_addr, item_buffer, size - 1, &ret_val)) {
    PyErr_SetString(PyExc_TypeError, MATXScriptAPIGetLastError());
    goto FREE_ARGS;
  }
  if (ret_val.code < 0) {
    PyErr_SetString(PyExc_TypeError, "the return value is not ObjectBase Type");
    goto FREE_ARGS;
  }
  super->handle = ret_val.data.v_handle;
  super->type_code = ret_val.code;

FREE_ARGS:
  MATXScriptRuntimeDestroyN(item_buffer, success_args);

RETURN_FLAG:
  delete[] item_buffer;
  Py_RETURN_NONE;
}

static PyMethodDef PyObjectMATXScriptObjectBase_Methods[] = {
    {
        "same_as",                               /* name */
        PyObjectMATXScriptObjectBase_same_as,    /* meth */
        METH_O,                                  /* flags */
        PyObjectMATXScriptObjectBase_same_as_doc /* docstring */
    },
    {
        "__init_handle_by_constructor__",                               /* name */
        PyObjectMATXScriptObjectBase___init_handle_by_constructor__,    /* meth */
        METH_VARARGS,                                                   /* flags */
        PyObjectMATXScriptObjectBase___init_handle_by_constructor___doc /* docstring */
    },
    {NULL} /* Sentinel */
};

static PyObject* PyObjectMATXScriptObjectBase_new(PyTypeObject* type,
                                                  PyObject* args,
                                                  PyObject* kwargs) {
  PyObjectMATXScriptObjectBase* self;
  self = (PyObjectMATXScriptObjectBase*)type->tp_alloc(type, 0);
  self->handle = NULL;
  self->type_code = ::matxscript::runtime::TypeIndex::kRuntimeNullptr;
  return (PyObject*)self;
}

/******************************************************************************
 * PyObjectMATXScriptAny
 *****************************************************************************/

typedef struct PyObjectMATXScriptAny {
  PyObject_HEAD;
  MATXScriptAny value;
} PyObjectMATXScriptAny;

static PyObject* PyObjectMATXScriptAny_new(PyTypeObject* type, PyObject* args, PyObject* kwargs) {
  PyObjectMATXScriptAny* self;
  self = (PyObjectMATXScriptAny*)type->tp_alloc(type, 0);
  self->value.data.v_handle = NULL;
  self->value.pad = 0;
  self->value.code = ::matxscript::runtime::TypeIndex::kRuntimeNullptr;
  return (PyObject*)self;
}

static int PyObjectMATXScriptAny_init(PyObject* self0, PyObject* args, PyObject* kwargs) {
  PyObjectMATXScriptAny* self = (PyObjectMATXScriptAny*)(self0);
  PyObject* arg_0 = NULL;
  if (!PyArg_ParseTuple(args, "O", &arg_0)) {
    return -1;
  }
  return PyObjectToMATXScriptAny(arg_0, &self->value);
}

static PyObject* PyObjectMATXScriptAny_repr(PyObject* self0) {
  PyObjectMATXScriptAny* self = (PyObjectMATXScriptAny*)(self0);
  switch (self->value.code) {
    case ::matxscript::runtime::TypeIndex::kRuntimeNullptr: {
      return PyUnicode_FromFormat(
          "Any(code: %d, pad: %d, value: nullptr)", self->value.code, self->value.pad);
    } break;
    case ::matxscript::runtime::TypeIndex::kRuntimeInteger: {
      return PyUnicode_FromFormat("Any(code: %d, pad: %d, value: %lld)",
                                  self->value.code,
                                  self->value.pad,
                                  self->value.data.v_int64);
    } break;
    case ::matxscript::runtime::TypeIndex::kRuntimeFloat: {
      PyObject* d_obj = PyFloat_FromDouble(self->value.data.v_float64);
      PyObject* result = PyUnicode_FromFormat(
          "Any(code: %d, pad: %d, value: %R)", self->value.code, self->value.pad, d_obj);
      Py_DecRef(d_obj);
      return result;
    } break;
    case ::matxscript::runtime::TypeIndex::kRuntimeOpaqueHandle: {
      return PyUnicode_FromFormat("Any(code: %d, pad: %d, value: OpaqueHandle(addr: %p))",
                                  self->value.code,
                                  self->value.pad,
                                  self->value.data.v_handle);
    } break;
    case ::matxscript::runtime::TypeIndex::kMATXByteArray: {
      return PyUnicode_FromFormat("Any(code: %d, pad: %d, value: ByteArray(addr: %p))",
                                  self->value.code,
                                  self->value.pad,
                                  self->value.data.v_handle);
    } break;
    case ::matxscript::runtime::TypeIndex::kRuntimeDataType: {
      return PyUnicode_FromFormat(
          "Any(code: %d, pad: %d, value: DataType(code: %d, bits: %d, lanes: %d))",
          self->value.code,
          self->value.pad,
          self->value.data.v_type.code,
          self->value.data.v_type.bits,
          self->value.data.v_type.lanes);
    } break;
    case ::matxscript::runtime::TypeIndex::kRuntimeContext: {
      return PyUnicode_FromFormat(
          "Any(code: %d, pad: %d, value: MATXScriptDevice(device_type: %d, device_id: %d))",
          self->value.code,
          self->value.pad,
          self->value.data.v_device.device_type,
          self->value.data.v_device.device_id);
    } break;
    case ::matxscript::runtime::TypeIndex::kRuntimeDLTensorHandle: {
      return PyUnicode_FromFormat("Any(code: %d, pad: %d, value: DLTensor(addr: %p))",
                                  self->value.code,
                                  self->value.pad,
                                  self->value.data.v_handle);
    } break;
    case ::matxscript::runtime::TypeIndex::kRuntimePackedFuncHandle: {
      return PyUnicode_FromFormat("Any(code: %d, pad: %d, value: PackedFunc(addr: %p))",
                                  self->value.code,
                                  self->value.pad,
                                  self->value.data.v_handle);
    } break;
    case ::matxscript::runtime::TypeIndex::kRuntimeObjectRValueRefArg: {
      return PyUnicode_FromFormat("Any(code: %d, pad: %d, value: ObjectRValueRefArg(addr: %p))",
                                  self->value.code,
                                  self->value.pad,
                                  self->value.data.v_handle);
    } break;
    case ::matxscript::runtime::TypeIndex::kRuntimeString: {
      PyObject* bytes = NULL;
      if (self->value.pad >= 0) {
        bytes = PyBytes_FromStringAndSize((const char*)self->value.data.v_str_store.v_small_bytes,
                                          self->value.pad);
      } else {
        bytes = PyBytes_FromStringAndSize((const char*)self->value.data.v_str_store.v_ml.bytes,
                                          self->value.data.v_str_store.v_ml.size);
      }
      PyObject* result = PyUnicode_FromFormat(
          "Any(code: %d, pad: %d, value: %R)", self->value.code, self->value.pad, bytes);
      Py_DecRef(bytes);
      return result;
    } break;
    case ::matxscript::runtime::TypeIndex::kRuntimeUnicode: {
      PyObject* str = NULL;
      if (self->value.pad >= 0) {
        str = PyUnicode_FromKindAndData(
            PyUnicode_4BYTE_KIND, self->value.data.v_str_store.v_small_chars, self->value.pad);
      } else {
        str = PyUnicode_FromKindAndData(PyUnicode_4BYTE_KIND,
                                        self->value.data.v_str_store.v_ml.chars,
                                        self->value.data.v_str_store.v_ml.size);
      }
      PyObject* result = PyUnicode_FromFormat(
          "Any(code: %d, pad: %d, value: %R)", self->value.code, self->value.pad, str);
      Py_DecRef(str);
      return result;
    } break;
    default: {
      // repr as object
      return PyUnicode_FromFormat("Any(code: %d, pad: %d, value: Object(addr: %p))",
                                  self->value.code,
                                  self->value.pad,
                                  self->value.data.v_handle);
    } break;
  }
}

static PyTypeObject PyType_MATXScriptAny = {
    PyVarObject_HEAD_INIT(&PyType_Type, 0) /**/
    "matx_script_api.Any",                 /* tp_name */
    sizeof(PyObjectMATXScriptAny),         /* tp_basicsize */
    0,                                     /* tp_itemsize */
    0,                                     /* tp_dealloc */
    0,                                     /* tp_print */
    0,                                     /* tp_getattr */
    0,                                     /* tp_setattr */
    0,                                     /* tp_reserved */
    PyObjectMATXScriptAny_repr,            /* tp_repr */
    0,                                     /* tp_as_number */
    0,                                     /* tp_as_sequence */
    0,                                     /* tp_as_mapping */
    0,                                     /* tp_hash  */
    0,                                     /* tp_call */
    0,                                     /* tp_str */
    0,                                     /* tp_getattro */
    0,                                     /* tp_setattro */
    0,                                     /* tp_as_buffer */
    Py_TPFLAGS_DEFAULT,                    /* tp_flags */
    "MATXScriptAny",                       /* tp_doc */
};

/******************************************************************************
 * PyObjectToMATXScriptAny
 *****************************************************************************/

static int PyObjectToMATXScriptList(PyObject* arg_0, MATXScriptAny* value) {
  Py_ssize_t size = PyList_GET_SIZE(arg_0);
  MATXScriptAny* item_buffer = new MATXScriptAny[size];
  for (Py_ssize_t i = 0; i < size; ++i) {
    PyObject* item = PyList_GET_ITEM(arg_0, i);
    if (0 != PyObjectToMATXScriptAny(item, item_buffer + i)) {
      MATXScriptRuntimeDestroyN(item_buffer, i);
      goto ERROR_FLAG;
    }
  }
  if (0 != MATXScriptRuntimeMakeList(item_buffer, size, 1, value)) {
    PyErr_SetString(PyExc_TypeError, "failed to convert python list to matx");
    goto ERROR_FLAG;
  }

  delete[] item_buffer;
  return 0;

ERROR_FLAG:
  delete[] item_buffer;
  return -1;
}

static int PyObjectToMATXScriptDict(PyObject* arg_0, MATXScriptAny* tx_value) {
  Py_ssize_t size = PyDict_Size(arg_0);
  size_t kv_sum = 2 * size;
  MATXScriptAny* item_buffer = new MATXScriptAny[kv_sum];

  PyObject* key = NULL;
  PyObject* value = NULL;
  Py_ssize_t pos = 0;
  size_t i = 0;

  while (PyDict_Next(arg_0, &pos, &key, &value)) {
    if (0 != PyObjectToMATXScriptAny(key, item_buffer + i)) {
      MATXScriptRuntimeDestroyN(item_buffer, i);
      goto ERROR_FLAG;
    }
    ++i;
    if (0 != PyObjectToMATXScriptAny(value, item_buffer + i)) {
      MATXScriptRuntimeDestroyN(item_buffer, i);
      goto ERROR_FLAG;
    }
    ++i;
  }

  if (0 != MATXScriptRuntimeMakeDict(item_buffer, kv_sum, 1, tx_value)) {
    PyErr_SetString(PyExc_TypeError, "failed to convert python dict to matx");
    goto ERROR_FLAG;
  }

  delete[] item_buffer;
  return 0;

ERROR_FLAG:
  delete[] item_buffer;
  return -1;
}

static int PyObjectToMATXScriptSet(PyObject* arg_0, MATXScriptAny* value) {
  Py_ssize_t size = PySet_GET_SIZE(arg_0);
  MATXScriptAny* item_buffer = new MATXScriptAny[size];
  PyObject* iterator = PyObject_GetIter(arg_0);
  PyObject* item = NULL;
  size_t i = 0;

  if (iterator == NULL) {
    PyErr_SetString(PyExc_TypeError, "failed to convert python set to matx");
    goto ERROR_FLAG;
  }
  while ((item = PyIter_Next(iterator))) {
    /* do something with item */
    int status = PyObjectToMATXScriptAny(item, item_buffer + i);
    /* release reference when done */
    Py_DECREF(item);
    if (0 != status) {
      MATXScriptRuntimeDestroyN(item_buffer, i);
      Py_DECREF(iterator);
      goto ERROR_FLAG;
    }
    ++i;
  }
  Py_DECREF(iterator);

  if (0 != MATXScriptRuntimeMakeSet(item_buffer, size, 1, value)) {
    PyErr_SetString(PyExc_TypeError, "failed to convert python set to matx");
    goto ERROR_FLAG;
  }

  delete[] item_buffer;
  return 0;

ERROR_FLAG:
  delete[] item_buffer;
  return -1;
}

static int PyObjectToMATXScriptTuple(PyObject* arg_0, MATXScriptAny* value) {
  Py_ssize_t size = PyTuple_GET_SIZE(arg_0);
  MATXScriptAny* item_buffer = new MATXScriptAny[size];
  for (Py_ssize_t i = 0; i < size; ++i) {
    PyObject* item = PyTuple_GET_ITEM(arg_0, i);
    if (0 != PyObjectToMATXScriptAny(item, item_buffer + i)) {
      MATXScriptRuntimeDestroyN(item_buffer, i);
      goto ERROR_FLAG;
    }
  }
  if (0 != MATXScriptRuntimeMakeTuple(item_buffer, size, 1, value)) {
    PyErr_SetString(PyExc_TypeError, "failed to convert python tuple to matx");
    goto ERROR_FLAG;
  }

  delete[] item_buffer;
  return 0;

ERROR_FLAG:
  delete[] item_buffer;
  return -1;
}

static int PyObjectToMATXScriptAny(PyObject* arg_0, MATXScriptAny* value) {
  if (PyFloat_Check(arg_0)) {
    value->code = ::matxscript::runtime::TypeIndex::kRuntimeFloat;
    value->data.v_float64 = PyFloat_AsDouble(arg_0);
  } else if (PyLong_Check(arg_0)) {
    value->code = ::matxscript::runtime::TypeIndex::kRuntimeInteger;
    value->data.v_int64 = PyLong_AsLongLong(arg_0);
  } else if (PyBool_Check(arg_0)) {
    value->code = ::matxscript::runtime::TypeIndex::kRuntimeInteger;
    value->data.v_int64 = (arg_0 == Py_True);
  } else if (Py_None == arg_0) {
    value->code = ::matxscript::runtime::TypeIndex::kRuntimeNullptr;
    value->data.v_handle = NULL;
  } else if (PyBytes_Check(arg_0)) {
    const char* bytes = PyBytes_AsString(arg_0);
    Py_ssize_t len = PyBytes_GET_SIZE(arg_0);
    if (MATXScriptRuntimeMakeString(bytes, size_t(len), value)) {
      PyErr_SetString(PyExc_TypeError, "failed to convert python bytes to matx runtime bytes");
      return -1;
    }
  } else if (PyUnicode_Check(arg_0)) {
    Py_ssize_t len;
    const char* bytes = PyUnicode_AsUTF8AndSize(arg_0, &len);
    if (MATXScriptRuntimeMakeUnicode(bytes, size_t(len), value)) {
      PyErr_SetString(PyExc_TypeError, "failed to convert python str to matx runtime str");
      return -1;
    }
  } else if (PyList_Check(arg_0)) {
    return PyObjectToMATXScriptList(arg_0, value);
  } else if (PyDict_Check(arg_0)) {
    return PyObjectToMATXScriptDict(arg_0, value);
  } else if (PySet_Check(arg_0)) {
    return PyObjectToMATXScriptSet(arg_0, value);
  } else if (PyTuple_Check(arg_0)) {
    return PyObjectToMATXScriptTuple(arg_0, value);
  } else if (PyObject_IsInstance(arg_0, (PyObject*)&PyType_MATXScriptObjectBase)) {
    PyObjectMATXScriptObjectBase* super = (PyObjectMATXScriptObjectBase*)arg_0;
    if (0 != MATXScriptObjectRetain(super->handle)) {
      PyErr_SetString(PyExc_TypeError, "internal error: failed to add ref count");
      return -1;
    }
    value->code = super->type_code;
    value->data.v_handle = super->handle;
    value->pad = 0;
  } else if (PyObject_IsInstance(arg_0, (PyObject*)&PyType_MATXScriptPackedFuncBase)) {
    // PackedFuncBase
    PyObjectMATXScriptPackedFuncBase* arg = (PyObjectMATXScriptPackedFuncBase*)arg_0;
    value->code = ::matxscript::runtime::TypeIndex::kRuntimePackedFuncHandle;
    value->data.v_handle = arg->handle;
    value->pad = 0;
    if (0 != MATXScriptRuntimeRetain(value)) {
      PyErr_SetString(PyExc_TypeError, "internal error: failed to add ref count");
      return -1;
    }
  } else if (PyObject_IsInstance(arg_0, (PyObject*)&PyType_MATXScriptAny)) {
    PyObjectMATXScriptAny* tx_pod_value = (PyObjectMATXScriptAny*)(arg_0);
    *value = tx_pod_value->value;
  } else {
    if (PyObject* result = matx_script_api__exe_input_instance_callback(arg_0)) {
      if (PyObject_IsInstance(result, (PyObject*)&PyType_MATXScriptAny)) {
        PyObjectMATXScriptAny* tx_pod_value = (PyObjectMATXScriptAny*)(result);
        *value = tx_pod_value->value;
        Py_DECREF(result);
      } else {
        Py_DECREF(result);
        PyErr_SetString(PyExc_TypeError, "The return type of callback is not MATXScriptAny");
        return -1;
      }
    } else {
      PyObject* errmsg = PyUnicode_FromFormat("unsupported type '%s'", Py_TYPE(arg_0)->tp_name);
      PyErr_SetObject(PyExc_TypeError, errmsg);
      Py_DECREF(errmsg);
      return -1;
    }
  }
  return 0;
}

/******************************************************************************
 * RETURN SWITCH
 *****************************************************************************/

typedef struct OBJECT_CALLBACK_PAIR_ {
  long long index;
  PyObject* callback;
} OBJECT_CALLBACK_PAIR;

static constexpr int MAX_OBJECT_CALLBACK_NUM = 4;
static int OBJECT_CALLBACK_CUR_IDX = 0;
static OBJECT_CALLBACK_PAIR OBJECT_CALLBACK_TABLE[MAX_OBJECT_CALLBACK_NUM];
static PyObject* DEFAULT_CLASS_OBJECT = NULL;
static PyObject* RETURN_SWITCH = NULL;

static PyObject* PACKEDFUNC_CREATOR = NULL;

static PyObject* HANDLE_CREATOR = NULL;

static PyObject* matx_script_api_set_class_object(PyObject* self, PyObject* args) {
  PyObject* callable;

  if (!PyArg_ParseTuple(args, "O", &callable)) {
    return NULL;
  }
  if (!PyCallable_Check(callable)) {
    PyErr_SetString(PyExc_TypeError, "the arg is not a callable object");
    return NULL;
  }
  if (DEFAULT_CLASS_OBJECT) {
    Py_DECREF(DEFAULT_CLASS_OBJECT);
  }
  Py_INCREF(callable);
  DEFAULT_CLASS_OBJECT = callable;
  Py_RETURN_NONE;
}

static PyObject* matx_script_api_set_packedfunc_creator(PyObject* self, PyObject* args) {
  PyObject* func;

  if (!PyArg_ParseTuple(args, "O", &func)) {
    return NULL;
  }
  if (!PyCallable_Check(func)) {
    PyErr_SetString(PyExc_TypeError, "the arg is not a callable object");
    return NULL;
  }
  if (PACKEDFUNC_CREATOR) {
    Py_DECREF(PACKEDFUNC_CREATOR);
  }
  Py_INCREF(func);
  PACKEDFUNC_CREATOR = func;
  Py_RETURN_NONE;
}

static PyObject* matx_script_api_set_handle_creator(PyObject* self, PyObject* args) {
  PyObject* func;

  if (!PyArg_ParseTuple(args, "O", &func)) {
    return NULL;
  }
  if (!PyCallable_Check(func)) {
    PyErr_SetString(PyExc_TypeError, "the arg is not a callable object");
    return NULL;
  }
  if (HANDLE_CREATOR) {
    Py_DECREF(HANDLE_CREATOR);
  }
  Py_INCREF(func);
  HANDLE_CREATOR = func;
  Py_RETURN_NONE;
}

static PyObject* MATXScriptAnySwitchToPackedFunc(MATXScriptAny* value) {
  if (!PACKEDFUNC_CREATOR) {
    PyErr_SetString(PyExc_TypeError, "PackedFunc type_code is not registered");
    return NULL;
  }
  PyObject* handle = PyLong_FromVoidPtr(value->data.v_handle);
  PyObject* func_args = PyTuple_Pack(1, handle);
  Py_DECREF(handle);
  PyObject* result = PyObject_Call(PACKEDFUNC_CREATOR, func_args, NULL);
  Py_DECREF(func_args);
  return result;
}

static PyObject* MATXScriptAnySwitchToHandle(MATXScriptAny* value) {
  if (!HANDLE_CREATOR) {
    PyErr_SetString(PyExc_TypeError, "PackedFunc type_code is not registered");
    return NULL;
  }
  PyObject* handle = PyLong_FromVoidPtr(value->data.v_handle);
  PyObject* func_args = PyTuple_Pack(1, handle);
  Py_DECREF(handle);
  PyObject* result = PyObject_Call(HANDLE_CREATOR, func_args, NULL);
  Py_DECREF(func_args);
  return result;
}

static PyObject* matx_script_api_get_global_func(PyObject* self, PyObject* args) {
  const char* name = NULL;
  PyObject* allow_missing = NULL;

  if (!PyArg_ParseTuple(args, "sO", &name, &allow_missing)) {
    return NULL;
  }
  if (!PyBool_Check(allow_missing)) {
    PyErr_SetString(PyExc_TypeError, "allow_missing is not bool type");
    return NULL;
  }

  MATXScriptFunctionHandle handle;
  if (MATXScriptFuncGetGlobal(name, &handle)) {
    PyErr_SetString(PyExc_RuntimeError, "failed to call MATXScriptFuncGetGlobal");
    return NULL;
  }

  if (handle) {
    MATXScriptAny pod_v;
    pod_v.code = ::matxscript::runtime::TypeIndex::kRuntimePackedFuncHandle;
    pod_v.data.v_handle = handle;
    return MATXScriptAnySwitchToPackedFunc(&pod_v);
  }

  Py_RETURN_NONE;
}

static PyObject* matx_script_api_register_object(PyObject* self, PyObject* args) {
  long long index = 0;
  PyObject* creator;

  if (!PyArg_ParseTuple(args, "LO", &index, &creator)) {
    return NULL;
  }
  if (!PyCallable_Check(creator)) {
    PyErr_SetString(PyExc_TypeError,
                    "the second arg is not a PyType object or a callable function");
    return NULL;
  }
  Py_INCREF(creator);
  PyObject* index_obj = PyLong_FromLongLong(index);
  if (0 != PyDict_SetItem(RETURN_SWITCH, index_obj, creator)) {
    Py_DECREF(index_obj);
    Py_DECREF(creator);
    return NULL;
  }
  Py_RETURN_NONE;
}

static PyObject* matx_script_api_register_object_callback(PyObject* self, PyObject* args) {
  long long index = 0;
  PyObject* callback;

  if (!PyArg_ParseTuple(args, "LO", &index, &callback)) {
    return NULL;
  }
  if (!PyCallable_Check(callback)) {
    PyErr_SetString(PyExc_TypeError, "the second arg is not a callable object");
    return NULL;
  }
  if (OBJECT_CALLBACK_CUR_IDX >= MAX_OBJECT_CALLBACK_NUM) {
    PyErr_SetString(PyExc_TypeError, "callback register overflow");
    return NULL;
  }
  Py_INCREF(callback);

  if (OBJECT_CALLBACK_TABLE[OBJECT_CALLBACK_CUR_IDX].callback) {
    Py_DECREF(OBJECT_CALLBACK_TABLE[OBJECT_CALLBACK_CUR_IDX].callback);
  }
  OBJECT_CALLBACK_TABLE[OBJECT_CALLBACK_CUR_IDX].index = index;
  OBJECT_CALLBACK_TABLE[OBJECT_CALLBACK_CUR_IDX].callback = callback;
  ++OBJECT_CALLBACK_CUR_IDX;

  Py_RETURN_NONE;
}

static PyObject* MATXScriptAnySwitchToObject(MATXScriptAny* value) {
  if (value->code < 0) {
    PyErr_SetString(PyExc_TypeError, "the first argument is not Object pointer");
    return NULL;
  }
  PyObject* index = PyLong_FromLongLong(value->code);
  PyObject* creator = PyDict_GetItem(RETURN_SWITCH, index);
  Py_DECREF(index);

  if (!creator) {
    if (DEFAULT_CLASS_OBJECT) {
      creator = DEFAULT_CLASS_OBJECT;
    } else {
      PyErr_SetString(PyExc_TypeError, "type_code is not registered");
      return NULL;
    }
  }

  if (value->code == ::matxscript::runtime::TypeIndex::kRuntimeModule) {
    PyObject* handle = PyLong_FromVoidPtr(value->data.v_handle);
    PyObject* func_args = PyTuple_Pack(1, handle);
    Py_DECREF(handle);
    PyObject* result = PyObject_Call(creator, func_args, NULL);
    Py_DECREF(func_args);
    return result;
  } else {
    PyObject* func_args = PyTuple_Pack(0);
    PyObject* result = PyObject_Call(creator, func_args, NULL);
    Py_DECREF(func_args);
    PyObjectMATXScriptObjectBase* super = (PyObjectMATXScriptObjectBase*)(result);
    super->handle = value->data.v_handle;
    super->type_code = value->code;

    for (int i = 0; i < OBJECT_CALLBACK_CUR_IDX; ++i) {
      if (OBJECT_CALLBACK_TABLE[i].index == value->code) {
        PyObject* func_args = PyTuple_Pack(1, result);
        PyObject* ret = PyObject_Call(OBJECT_CALLBACK_TABLE[i].callback, func_args, NULL);
        Py_DECREF(func_args);
        Py_DECREF(ret);
        break;
      }
    }
    return result;
  }
}

static PyObject* matx_script_api_return_switch_impl(MATXScriptAny* value) {
  switch (value->code) {
    case ::matxscript::runtime::TypeIndex::kRuntimeNullptr: {
      Py_RETURN_NONE;
    } break;
    case ::matxscript::runtime::TypeIndex::kRuntimeInteger: {
      return PyLong_FromLongLong(value->data.v_int64);
    } break;
    case ::matxscript::runtime::TypeIndex::kRuntimeFloat: {
      return PyFloat_FromDouble(value->data.v_float64);
    } break;
    case ::matxscript::runtime::TypeIndex::kRuntimeOpaqueHandle: {
      return MATXScriptAnySwitchToHandle(value);
    } break;
    case ::matxscript::runtime::TypeIndex::kMATXByteArray: {
      PyErr_SetString(PyExc_TypeError, "kMATXByteArray is not supported");
      return NULL;
    } break;
    case ::matxscript::runtime::TypeIndex::kRuntimeDataType: {
      int size = 64;
      char buffer[64] = {0};
      if (0 != MATXScriptAPIDLDataTypeToString(value->data.v_type, buffer, &size)) {
        PyErr_SetString(PyExc_TypeError, "kRuntimeDataType is not supported");
        return NULL;
      }
      return PyUnicode_FromKindAndData(PyUnicode_1BYTE_KIND, buffer, size);
    } break;
    case ::matxscript::runtime::TypeIndex::kRuntimeContext: {
      PyErr_SetString(PyExc_TypeError, "kRuntimeContext is not supported");
      return NULL;
    } break;
    case ::matxscript::runtime::TypeIndex::kRuntimeDLTensorHandle: {
      PyErr_SetString(PyExc_TypeError, "kRuntimeDLTensorHandle is not supported");
      return NULL;
    } break;
    case ::matxscript::runtime::TypeIndex::kRuntimePackedFuncHandle: {
      return MATXScriptAnySwitchToPackedFunc(value);
      /* PyObject* obj = PyObjectMATXPackedFuncBase_new(&PyType_MATXPackedFuncBase, NULL,
      NULL); PyObjectMATXPackedFuncBase* pf_obj = (PyObjectMATXPackedFuncBase*)obj;
      pf_obj->is_global = 0;
      pf_obj->handle = (void*)(value->data.v_handle);
      return obj; */
    } break;
    case ::matxscript::runtime::TypeIndex::kRuntimeObjectRValueRefArg: {
      PyErr_SetString(PyExc_TypeError, "kRuntimeObjectRValueRefArg is not supported");
      return NULL;
    } break;
    case ::matxscript::runtime::TypeIndex::kRuntimeString: {
      PyObject* bytes = NULL;
      if (value->pad >= 0) {
        bytes = PyBytes_FromStringAndSize((const char*)value->data.v_str_store.v_small_bytes,
                                          value->pad);
      } else {
        bytes = PyBytes_FromStringAndSize((const char*)value->data.v_str_store.v_ml.bytes,
                                          value->data.v_str_store.v_ml.size);
      }
      MATXScriptRuntimeDestroy(value);
      return bytes;
    } break;
    case ::matxscript::runtime::TypeIndex::kRuntimeUnicode: {
      PyObject* str = NULL;
      if (value->pad >= 0) {
        str = PyUnicode_FromKindAndData(
            PyUnicode_4BYTE_KIND, value->data.v_str_store.v_small_chars, value->pad);
      } else {
        str = PyUnicode_FromKindAndData(PyUnicode_4BYTE_KIND,
                                        value->data.v_str_store.v_ml.chars,
                                        value->data.v_str_store.v_ml.size);
      }
      MATXScriptRuntimeDestroy(value);
      return str;
    } break;
    default: {
      if (value->code < 0) {
        MATXScriptRuntimeDestroy(value);
        PyErr_SetString(PyExc_TypeError, "return value type is not supported");
        return NULL;
      } else {
        return MATXScriptAnySwitchToObject(value);
      }
    } break;
  }
}

static PyObject* matx_script_api_return_switch(PyObject* self, PyObject* args) {
  PyObject* tx_object;

  if (!PyArg_ParseTuple(args, "O", &tx_object)) {
    return NULL;
  }
  if (Py_TYPE(tx_object) != &PyType_MATXScriptAny) {
    PyErr_SetString(PyExc_TypeError, "the first argument is not MATXScriptAnyType");
    return NULL;
  }

  PyObjectMATXScriptAny* tx_pod_value = (PyObjectMATXScriptAny*)(tx_object);
  return matx_script_api_return_switch_impl(&tx_pod_value->value);
}

/******************************************************************************
 * Input Instance Callback
 *****************************************************************************/
static const int MAX_INPUT_INSTANCE_CALLBACK_NUM = 100;
static PyObject* INPUT_INSTANCE_CALLBACK[MAX_INPUT_INSTANCE_CALLBACK_NUM][2];

static int INPUT_INSTANCE_CALLBACK_CUR = 0;

static PyObject* matx_script_api_register_input_instance_callback(PyObject* self, PyObject* args) {
  PyObject* user_type_object;
  PyObject* user_callback;

  if (!PyArg_ParseTuple(args, "OO", &user_type_object, &user_callback)) {
    return NULL;
  }
  if (!PyCallable_Check(user_callback)) {
    PyErr_SetString(PyExc_TypeError, "the second argument is not callable type");
    return NULL;
  }

  for (int i = 0; i < INPUT_INSTANCE_CALLBACK_CUR; ++i) {
    if (user_type_object == INPUT_INSTANCE_CALLBACK[i][0]) {
      Py_DECREF(INPUT_INSTANCE_CALLBACK[i][1]);
      Py_INCREF(user_callback);
      INPUT_INSTANCE_CALLBACK[i][1] = user_callback;
      Py_RETURN_NONE;
    }
  }

  if (INPUT_INSTANCE_CALLBACK_CUR >= MAX_INPUT_INSTANCE_CALLBACK_NUM) {
    PyErr_SetString(PyExc_TypeError, "too manny instance callback");
    return NULL;
  }

  Py_INCREF(user_type_object);
  Py_INCREF(user_callback);
  INPUT_INSTANCE_CALLBACK[INPUT_INSTANCE_CALLBACK_CUR][0] = user_type_object;
  INPUT_INSTANCE_CALLBACK[INPUT_INSTANCE_CALLBACK_CUR][1] = user_callback;
  ++INPUT_INSTANCE_CALLBACK_CUR;

  Py_RETURN_NONE;
}

static PyObject* matx_script_api__exe_input_instance_callback(PyObject* arg) {
  for (int i = 0; i < INPUT_INSTANCE_CALLBACK_CUR; ++i) {
    if (PyObject_IsInstance(arg, INPUT_INSTANCE_CALLBACK[i][0])) {
      PyObject* func_args = PyTuple_Pack(1, arg);
      PyObject* result = PyObject_Call(INPUT_INSTANCE_CALLBACK[i][1], func_args, NULL);
      Py_DECREF(func_args);
      return result;
    }
  }
  return NULL;
}

/******************************************************************************
 * op_kernel_call
 *****************************************************************************/

static PyObject* matx_script_api_op_kernel_call(PyObject* self, PyObject* args) {
  Py_ssize_t size = PyTuple_GET_SIZE(args);
  MATXScriptAny* item_buffer = new MATXScriptAny[size];
  PyObject* item_0 = PyTuple_GET_ITEM(args, 0);
  void* op_kernel_ptr = NULL;
  PyObject* result = NULL;
  PyObject* result_obj = NULL;
  PyObject* result_type_code = NULL;
  int success_args = 0;
  int32_t type_code = -1;
  if (size < 1) {
    PyErr_SetString(PyExc_TypeError, "need one or more args(0 given)");
    goto RETURN_FLAG;
  }
  if (!PyLong_Check(item_0)) {
    PyErr_SetString(PyExc_TypeError, "the first argument is not int type (aka OpKernel Pointer)");
    goto RETURN_FLAG;
  }
  op_kernel_ptr = PyLong_AsVoidPtr(item_0);
  for (Py_ssize_t i = 1; i < size; ++i) {
    PyObject* item = PyTuple_GET_ITEM(args, i);
    if (0 != PyObjectToMATXScriptAny(item, item_buffer + i - 1)) {
      MATXScriptRuntimeDestroyN(item_buffer, success_args);
      goto RETURN_FLAG;
    }
    ++success_args;
  }

  MATXScriptAny ret_val;
  if (0 != MATXScriptPipelineOpKernelCall(op_kernel_ptr, item_buffer, success_args, 1, &ret_val)) {
    PyErr_SetString(PyExc_TypeError, MATXScriptAPIGetLastError());
    goto RETURN_FLAG;
  }
  type_code = ret_val.code;
  result_obj = matx_script_api_return_switch_impl(&ret_val);
  result_type_code = PyLong_FromLong(type_code);
  result = PyTuple_Pack(2, result_obj, result_type_code);
  Py_DECREF(result_obj);
  Py_DECREF(result_type_code);

RETURN_FLAG:
  delete[] item_buffer;
  return result;
}

/******************************************************************************
 * Make MATXScriptAny by type_code and pointer
 *****************************************************************************/

static PyObject* matx_script_api_steal_object_handle(PyObject* self, PyObject* object_base) {
  if (!PyObject_IsInstance(object_base, (PyObject*)&PyType_MATXScriptObjectBase)) {
    PyErr_SetString(PyExc_TypeError, "the arg is not a ObjectBaseType");
    return NULL;
  }
  PyObjectMATXScriptObjectBase* super = (PyObjectMATXScriptObjectBase*)(object_base);
  PyObject* handle = PyLong_FromVoidPtr(super->handle);
  PyObject* type_code = PyLong_FromLong(super->type_code);
  PyObject* result = PyTuple_Pack(2, handle, type_code);
  super->handle = NULL;
  super->type_code = ::matxscript::runtime::TypeIndex::kRuntimeNullptr;
  Py_DECREF(handle);
  Py_DECREF(type_code);
  return result;
}

static PyObject* matx_script_api_release_object_handle(PyObject* self, PyObject* object_base) {
  if (!PyObject_IsInstance(object_base, (PyObject*)&PyType_MATXScriptObjectBase)) {
    PyErr_SetString(PyExc_TypeError, "the arg is not a ObjectBaseType");
    return NULL;
  }
  PyObjectMATXScriptObjectBase* super = (PyObjectMATXScriptObjectBase*)(object_base);
  MATXScriptObjectFree(super->handle);
  super->handle = NULL;
  super->type_code = ::matxscript::runtime::TypeIndex::kRuntimeNullptr;
  Py_RETURN_NONE;
}

static PyObject* matx_script_api_make_any(PyObject* self, PyObject* args) {
  int32_t type_code;
  int32_t pad;
  uintptr_t handle;
  int32_t move_mode;

  if (!PyArg_ParseTuple(args, "iiKi", &type_code, &pad, &handle, &move_mode)) {
    return NULL;
  }
  PyObject* obj = PyObjectMATXScriptAny_new(&PyType_MATXScriptAny, NULL, NULL);
  PyObjectMATXScriptAny* tx_pod_value = (PyObjectMATXScriptAny*)(obj);
  tx_pod_value->value.code = type_code;
  tx_pod_value->value.pad = pad;
  tx_pod_value->value.data.v_handle = (void*)(handle);
  if (!move_mode) {
    MATXScriptObjectRetain(tx_pod_value->value.data.v_handle);
  }
  return obj;
}

static int PythonClosureMATXScriptPackedCFunc(MATXScriptAny* args,
                                              int num_args,
                                              MATXScriptValueHandle ret,
                                              void* resource_handle) {
  PyObject* py_func = (PyObject*)resource_handle;
  PyObject* py_args = PyTuple_New(num_args);
  if (!py_args) {
    MATXScriptAutoSetLastErrorByPythonTraceback();
    return -1;
  }

  for (int i = 0; i < num_args; ++i) {
    // TODO: optimize str and bytes
    if (0 != MATXScriptRuntimeRetain(args + i)) {
      Py_DECREF(py_args);
      MATXScriptAutoSetLastErrorByPythonTraceback();
      return -1;
    }
    PyObject* arg_i = matx_script_api_return_switch_impl(args + i);
    if (NULL == arg_i) {
      Py_DECREF(py_args);
      MATXScriptAutoSetLastErrorByPythonTraceback();
      return -1;
    }
    PyTuple_SET_ITEM(py_args, i, arg_i);
  }

  PyObject* py_ret = PyObject_Call(py_func, py_args, NULL);
  Py_DECREF(py_args);
  if (!py_ret) {
    MATXScriptAutoSetLastErrorByPythonTraceback();
    return -1;
  }

  MATXScriptAny c_ret;
  if (0 != PyObjectToMATXScriptAny(py_ret, &c_ret)) {
    MATXScriptAutoSetLastErrorByPythonTraceback();
    Py_DECREF(py_ret);
    return -1;
  }
  Py_DECREF(py_ret);
  return MATXScriptCFuncSetReturn(ret, &c_ret, 1);
}

static void PythonClosureMATXScriptPackedCFuncFinalizer(void* resource_handle) {
  PyObject* py_func = (PyObject*)resource_handle;
  Py_DECREF(py_func);
}

static PyObject* matx_script_api_convert_to_packed_func(PyObject* self, PyObject* py_func) {
  if (!PyCallable_Check(py_func)) {
    PyErr_SetString(PyExc_TypeError, "the arg is not a Callable object");
    return NULL;
  }
  Py_INCREF(py_func);
  MATXScriptFunctionHandle handle;
  if (0 != MATXScriptFuncCreateFromCFunc(PythonClosureMATXScriptPackedCFunc,
                                         py_func,
                                         PythonClosureMATXScriptPackedCFuncFinalizer,
                                         &handle,
                                         0)) {
    PyErr_SetString(PyExc_TypeError, MATXScriptAPIGetLastError());
    return NULL;
  }
  MATXScriptAny value;
  value.code = ::matxscript::runtime::TypeIndex::kRuntimePackedFuncHandle;
  value.data.v_handle = handle;
  PyObject* func = MATXScriptAnySwitchToPackedFunc(&value);
  return func;
}

static PyObject* matx_script_api_to_runtime_object(PyObject* self, PyObject* py_obj) {
  if (PyFloat_Check(py_obj) || PyLong_Check(py_obj) || PyBool_Check(py_obj) || Py_None == py_obj ||
      PyBytes_Check(py_obj) || PyByteArray_Check(py_obj) || PyUnicode_Check(py_obj) ||
      PyObject_IsInstance(py_obj, (PyObject*)&PyType_MATXScriptObjectBase) ||
      PyObject_IsInstance(py_obj, (PyObject*)&PyType_MATXScriptPackedFuncBase)) {
    // fast check
    Py_INCREF(py_obj);
    return py_obj;
  }
  MATXScriptAny c_ret;
  if (0 != PyObjectToMATXScriptAny(py_obj, &c_ret)) {
    return NULL;
  }
  return matx_script_api_return_switch_impl(&c_ret);
}

static void dlpack_capsule_destructor(PyObject* data) {
  DLManagedTensor* dlm_tensor = (DLManagedTensor*)PyCapsule_GetPointer(data, "dltensor");
  if (dlm_tensor) {
    // the dlMTensor has not been consumed, call deleter ourselves
    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
    dlm_tensor->deleter(const_cast<DLManagedTensor*>(dlm_tensor));
  } else {
    // the dlMTensor has been consumed
    // PyCapsule_GetPointer has set an error indicator
    PyErr_Clear();
  }
}

// DLPack conversion in python mode, following data layout of PyTorch.
// https://github.com/pytorch/pytorch/blob/75955e4ef8a941d72db20c5098371325bd83ffd1/torch/csrc/Module.cpp#L367
static PyObject* matx_script_api_to_dlpack(PyObject* self, PyObject* py_obj) {
  MATXScriptAny c_ret;
  if (0 != PyObjectToMATXScriptAny(py_obj, &c_ret)) {
    PyErr_SetString(PyExc_TypeError, "failed to convert pyobj to pod");
    return NULL;
  }
  DLManagedTensor* dlm_tensor = nullptr;
  if (0 != MATXScriptNDArrayToDLPack(&c_ret, &dlm_tensor)) {
    PyErr_SetString(PyExc_TypeError, "failed to convert ndarray to dlpack.");
    return NULL;
  }
  return PyCapsule_New(dlm_tensor, "dltensor", dlpack_capsule_destructor);
}

static PyObject* matx_script_api_from_dlpack(PyObject* self, PyObject* py_obj) {
  void* dlm_tensor = PyCapsule_GetPointer(py_obj, "dltensor");
  if (!dlm_tensor) {
    PyErr_SetString(PyExc_RuntimeError,
                    "input is not a dlpack pycapsule, or a used dlpack pycapsule.");
    return NULL;
  }

  MATXScriptAny c_ret;
  if (0 != MATXScriptNDArrayFromDLPack(dlm_tensor, &c_ret)) {
    PyErr_SetString(PyExc_RuntimeError, "failed to convert dlpack to ndarray.");
    return NULL;
  }
  PyCapsule_SetName(py_obj, "used_dltensor");
  PyCapsule_SetDestructor(py_obj, [](PyObject*) {});
  return matx_script_api_return_switch_impl(&c_ret);
}

/******************************************************************************
 * API Module
 *****************************************************************************/

static PyMethodDef MATXScriptAPIMethods[] = {
    {"op_kernel_call",
     matx_script_api_op_kernel_call,
     METH_VARARGS,
     "call op kernel process function"},
    {"make_any", matx_script_api_make_any, METH_VARARGS, "make any by type_code and pointer"},
    {"register_input_callback",
     matx_script_api_register_input_instance_callback,
     METH_VARARGS,
     "register callback"},
    {"register_object", matx_script_api_register_object, METH_VARARGS, "register object class"},
    {"register_object_callback",
     matx_script_api_register_object_callback,
     METH_VARARGS,
     "register object callback"},
    {"return_switch", matx_script_api_return_switch, METH_VARARGS, "convert to python object"},
    {"set_class_object", matx_script_api_set_class_object, METH_VARARGS, ""},
    {"set_packedfunc_creator", matx_script_api_set_packedfunc_creator, METH_VARARGS, ""},
    {"set_handle_creator", matx_script_api_set_handle_creator, METH_VARARGS, ""},
    {"get_global_func", matx_script_api_get_global_func, METH_VARARGS, ""},
    {"steal_object_handle", matx_script_api_steal_object_handle, METH_O, ""},
    {"release_object_handle", matx_script_api_release_object_handle, METH_O, ""},
    {"convert_to_packed_func", matx_script_api_convert_to_packed_func, METH_O, ""},
    {"to_runtime_object", matx_script_api_to_runtime_object, METH_O, ""},
    {"_to_dlpack", matx_script_api_to_dlpack, METH_O, ""},
    {"_from_dlpack", matx_script_api_from_dlpack, METH_O, ""},
    {NULL, NULL, 0, NULL} /* Sentinel */
};

static struct PyModuleDef MATXScriptAPIModule = {
    PyModuleDef_HEAD_INIT,
    "matx_script_api",      /* name of module */
    "matx script fast ffi", /* module documentation, may be NULL */
    -1,                     /* size of per-interpreter state of the module,
                               or -1 if the module keeps state in global variables. */
    MATXScriptAPIMethods};

PyMODINIT_FUNC PyInit_matx_script_api(void) {
  PyObject* m;

  // init MATXScriptAnyType
  PyType_MATXScriptAny.tp_new = PyObjectMATXScriptAny_new;
  PyType_MATXScriptAny.tp_init = PyObjectMATXScriptAny_init;
  if (PyType_Ready(&PyType_MATXScriptAny) < 0) {
    return NULL;
  }

  // init MATXObjectBaseType
  PyType_MATXScriptObjectBase.tp_new = PyObjectMATXScriptObjectBase_new;
  PyType_MATXScriptObjectBase.tp_members = PyObjectMATXScriptObjectBase_Members;
  PyType_MATXScriptObjectBase.tp_methods = PyObjectMATXScriptObjectBase_Methods;
  PyType_MATXScriptObjectBase.tp_finalize = PyObjectMATXScriptObjectBase_finalize;
  if (PyType_Ready(&PyType_MATXScriptObjectBase) < 0) {
    return NULL;
  }

  // init MATXPackedFuncBaseType
  PyType_MATXScriptPackedFuncBase.tp_new = PyObjectMATXScriptPackedFuncBase_new;
  PyType_MATXScriptPackedFuncBase.tp_init = PyObjectMATXScriptPackedFuncBase_init;
  PyType_MATXScriptPackedFuncBase.tp_finalize = PyObjectMATXScriptPackedFuncBase_finalize;
  PyType_MATXScriptPackedFuncBase.tp_call = PyObjectMATXScriptPackedFuncBase_call;
  PyType_MATXScriptPackedFuncBase.tp_members = PyObjectMATXPackedFuncBase_Members;
  if (PyType_Ready(&PyType_MATXScriptPackedFuncBase) < 0) {
    return NULL;
  }

  // init module
  m = PyModule_Create(&MATXScriptAPIModule);
  if (m == NULL) {
    return NULL;
  }

  for (int i = 0; i < MAX_OBJECT_CALLBACK_NUM; ++i) {
    OBJECT_CALLBACK_TABLE[i].index = 0;
    OBJECT_CALLBACK_TABLE[i].callback = NULL;
  }

  for (int i = 0; i < MAX_INPUT_INSTANCE_CALLBACK_NUM; ++i) {
    INPUT_INSTANCE_CALLBACK[i][0] = NULL;
    INPUT_INSTANCE_CALLBACK[i][1] = NULL;
  }

  RETURN_SWITCH = PyDict_New();
  if (PyModule_AddObject(m, "RETURN_SWITCH", RETURN_SWITCH) < 0) {
    Py_DECREF(m);
    Py_DECREF(RETURN_SWITCH);
    return NULL;
  }

  Py_INCREF(&PyType_MATXScriptAny);
  Py_INCREF(&PyType_MATXScriptObjectBase);
  Py_INCREF(&PyType_MATXScriptPackedFuncBase);

  if (PyModule_AddObject(m, "Any", (PyObject*)&PyType_MATXScriptAny) < 0) {
    goto Failed;
  }
  if (PyModule_AddObject(m, "ObjectBase", (PyObject*)&PyType_MATXScriptObjectBase) < 0) {
    goto Failed;
  }
  if (PyModule_AddObject(m, "PackedFuncBase", (PyObject*)&PyType_MATXScriptPackedFuncBase) < 0) {
    goto Failed;
  }
  return m;

Failed:
  Py_DECREF(m);
  Py_DECREF(&PyType_MATXScriptAny);
  Py_DECREF(&PyType_MATXScriptObjectBase);
  Py_DECREF(&PyType_MATXScriptPackedFuncBase);
  return NULL;
}
