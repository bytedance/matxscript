# Copyright 2022 ByteDance Ltd. and/or its affiliates.
#
# Acknowledgement: The structure of the Module is inspired by incubator-tvm.
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

# pylint: disable=invalid-name, unused-import, import-outside-toplevel
"""Runtime Module namespace."""
import ctypes

from .._ffi.base import _LIB, check_call, c_str, string_types, _RUNTIME_ONLY
from .._ffi.libinfo import find_include_path
from .._ffi._selector import _set_class_module
from .packed_func import PackedFunc, PackedFuncHandle

from . import _ffi_api


class Module(object):
    """Runtime Module."""

    __slots__ = ["handle"]

    def __init__(self, handle):
        self.handle = handle

    def __del__(self):
        handle = self.handle
        if not isinstance(handle, ctypes.c_void_p):
            handle = ctypes.c_void_p(handle)
        check_call(_LIB.MATXScriptModFree(handle))

    def __hash__(self):
        if isinstance(self.handle, ctypes.c_void_p):
            return ctypes.cast(self.handle, ctypes.c_void_p).value
        else:
            return self.handle

    def get_function(self, name, query_imports=False):
        """Get function from the module.

        Parameters
        ----------
        name : str
            The name of the function

        query_imports : bool
            Whether also query modules imported by this module.

        Returns
        -------
        f : matx.runtime.PackedFunc
            The result function.
        """
        ret_handle = PackedFuncHandle()
        handle = self.handle
        if not isinstance(handle, ctypes.c_void_p):
            handle = ctypes.c_void_p(handle)
        check_call(
            _LIB.MATXScriptModGetFunction(
                handle, c_str(name), ctypes.c_int(query_imports), ctypes.byref(ret_handle)
            )
        )
        if not ret_handle.value:
            raise AttributeError("Module has no function '%s'" % name)
        return PackedFunc(ret_handle.value, False)

    def import_module(self, module):
        """Add module to the import list of current one.

        Parameters
        ----------
        module : matx.runtime.Module
            The other module.
        """
        handle = self.handle
        if not isinstance(handle, ctypes.c_void_p):
            handle = ctypes.c_void_p(handle)
        module_handle = module.handle
        if not isinstance(module_handle, ctypes.c_void_p):
            module_handle = ctypes.c_void_p(module_handle)

        check_call(_LIB.MATXScriptModImport(handle, module_handle))

    def __getitem__(self, name):
        if not isinstance(name, string_types):
            raise ValueError("Can only take string as function name")
        return self.get_function(name)

    def __repr__(self):
        handle = self.handle
        if not isinstance(handle, ctypes.c_void_p):
            handle = ctypes.c_void_p(handle)
        return "Module(%s, %x)" % (self.type_key, handle.value)

    @property
    def type_key(self):
        """Get type key of the module."""
        return _ffi_api.ModuleGetTypeKey(self)

    def get_source(self, fmt=""):
        """Get source code from module, if available.

        Parameters
        ----------
        fmt : str, optional
            The specified format.

        Returns
        -------
        source : str
            The result source code.
        """
        return _ffi_api.ModuleGetSource(self, fmt)

    @property
    def imported_modules(self):
        """Get imported modules

        Returns
        ----------
        modules : list of Module
            The module
        """
        nmod = _ffi_api.ModuleImportsSize(self)
        return [_ffi_api.ModuleGetImport(self, i) for i in range(nmod)]

    def save(self, file_name, fmt=""):
        """Save the module to file.

        This do not save the dependent device modules.
        See also export_shared

        Parameters
        ----------
        file_name : str
            The name of the file.
        fmt : str
            The format of the file.

        See Also
        --------
        runtime.Module.export_library : export the module to shared library.
        """
        _ffi_api.ModuleSaveToFile(self, file_name, fmt)

    def _collect_dso_modules(self):
        """Helper function to collect dso modules, then return it."""
        visited, stack, dso_modules = set(), [], []
        # append root module
        visited.add(self)
        stack.append(self)
        while stack:
            module = stack.pop()
            if module._dso_exportable():
                dso_modules.append(module)
            for m in module.imported_modules:
                if m not in visited:
                    visited.add(m)
                    stack.append(m)
        return dso_modules

    def _dso_exportable(self):
        return self.type_key == "llvm" or self.type_key == "c"

    def export_library(self, file_name, fcompile=None, addons=None, **kwargs):
        """Export the module and its imported device code one library.

        This function only works on host llvm modules.
        It will pack all the imported modules

        Parameters
        ----------
        file_name : str
            The name of the shared library.

        fcompile : function(target, file_list, kwargs), optional
            Compilation function to use create dynamic library.
            If fcompile has attribute object_format, will compile host library
            to that format. Otherwise, will use default format "o".

        kwargs : dict, optional
            Additional arguments passed to fcompile
        """
        # NOTE: this function depends on contrib library features
        # which are only available in when TVM function is available.
        if _RUNTIME_ONLY:
            raise RuntimeError("Cannot call export_library in runtime only mode")
        # Extra dependencies during runtime.
        from pathlib import Path
        from ..contrib import cc as _cc, tar as _tar, util as _util

        if isinstance(file_name, Path):
            file_name = str(file_name)

        assert self.type_key == "c"

        modules = self._collect_dso_modules()
        files = addons if addons else []
        is_system_lib = False
        has_c_module = False
        llvm_target_triple = None
        for index, module in enumerate(modules):
            if fcompile is not None and hasattr(fcompile, "object_format"):
                object_format = fcompile.object_format
            else:
                assert module.type_key == "c"
                object_format = "cc"
                has_c_module = True
            # We use the same name to save the generated cpp file
            # dso/libXXX.cc
            path_obj = file_name[:-2] + 'cc'
            module.save(path_obj)
            files.append(path_obj)
        if not fcompile:
            if file_name.endswith(".tar"):
                fcompile = _tar.tar
            else:
                fcompile = _cc.create_shared

        if getattr(fcompile, "need_system_lib", False) and not is_system_lib:
            raise ValueError("%s need --system-lib option" % str(fcompile))

        if self.imported_modules:
            raise RuntimeError("submodule is not supported!!!")
            path_cc = temp.relpath("devc.cc")
            with open(path_cc, "w") as f:
                f.write(_ffi_api.ModulePackImportsToC(self, is_system_lib))
            files.append(path_cc)

        if has_c_module:
            options = []
            if "options" in kwargs:
                opts = kwargs["options"]
                options = opts if isinstance(opts, (list, tuple)) else [opts]
            else:
                options = ["-std=c++14", "-O3"]
                if not _ffi_api.USE_CXX11_ABI():
                    options.append("-D_GLIBCXX_USE_CXX11_ABI=0")
            opts = options + ["-I" + path for path in find_include_path()]
            kwargs.update({"options": opts})

        # print(file_name, files, kwargs)
        fcompile(file_name, files, **kwargs)


def system_lib():
    """Get system-wide library module singleton.

    System lib is a global module that contains self register functions in startup.
    Unlike normal dso modules which need to be loaded explicitly.
    It is useful in environments where dynamic loading api like dlopen is banned.

    To build system lib function, simply specify target option ```llvm --system-lib```
    The system lib will be available as long as the result code is linked by the program.

    The system lib is intended to be linked and loaded during the entire life-cyle of the program.
    If you want dynamic loading features, use dso modules instead.

    Returns
    -------
    module : runtime.Module
        The system-wide library module.
    """
    return _ffi_api.SystemLib()


def load_module(path, fmt=""):
    """Load module from file.

    Parameters
    ----------
    path : str
        The path to the module file.

    fmt : str, optional
        The format of the file, if not specified
        it will be inferred from suffix of the file.

    Returns
    -------
    module : runtime.Module
        The loaded module
    """
    # Redirect to the load API
    return _ffi_api.ModuleLoadFromFile(path, fmt)


_set_class_module(Module)
