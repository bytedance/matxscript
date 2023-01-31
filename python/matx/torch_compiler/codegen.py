import copy
import logging
from typing import List

import torch
import torch._inductor.compile_fx as compile_fx
from torch import fx
from torch._inductor.debug import DebugContext
from torch._inductor.virtualized import V

from .utils import cpp_parse

log = logging.getLogger(__name__)

MATX_INCLUDE = '''
#include "matxscript/runtime/codegen_all_includes.h"
#include <math.h>

using namespace ::matxscript::runtime;
extern "C" void* __matxscript_module_ctx = NULL;

extern "C" MATX_DLL MATXScriptFuncRegistry __matxscript_func_registry__;
'''

SESSION_HANLDER = cpp_parse.CPPArg(name='handle_2_71828182846',
                                   type=cpp_parse.CPPType(name='void', is_pointer=True))
SESSION_HANLDER_WITH_DEAFULT = cpp_parse.CPPArg(name='handle_2_71828182846',
                                                type=cpp_parse.CPPType(name='void', is_pointer=True),
                                                default_val='((void*)(int64_t)0)')


def generate_ndarray_arg_cast(arg_name, arg_index, dtype, message='TODO'):
    return f'({dtype}*)internal::TypeAsHelper<NDArray>::run(({arg_name}[{arg_index}]), __FILE__, __LINE__, "{message}", "{message}").Data<{dtype}>()'


def get_c_api(kernel_name: str, args: List[cpp_parse.CPPArg], has_return_value) -> str:
    template_with_return = '''
int kernel__c_api(MATXScriptAny* args, int num_args, MATXScriptAny* out_ret_value, void* resource_handle = nullptr)
{{
  TArgs args_t(args, num_args);

  if (num_args > 0 && args[num_args - 1].code == TypeIndex::kRuntimeKwargs) {{
    string_view arg_names[{}] {{{}}};
    KwargsUnpackHelper helper("{}", arg_names, {}, nullptr, 0);
    RTView pos_args[{}];
    helper.unpack(pos_args, args, num_args);  // /Users/bytedance/Developer/open_source_library/matxscript/examples/simple_function.py:5

    auto ret = {}({}, 
                {}resource_handle);
    RTValue(std::move(ret)).MoveToCHost(out_ret_value);
  }} else {{
    switch(num_args) {{
      case {}: {{
        auto ret = {}({}, 
                    {}resource_handle);  // /Users/bytedance/Developer/open_source_library/matxscript/examples/simple_function.py:5
        RTValue(std::move(ret)).MoveToCHost(out_ret_value);
      }} break;
      default: {{THROW_PY_TypeError("TODO");}} break;  // /Users/bytedance/Developer/open_source_library/matxscript/examples/simple_function.py:5
    }}
  }}

  return 0;
}}
'''
    template_without_return = '''
int kernel__c_api(MATXScriptAny* args, int num_args, MATXScriptAny* out_ret_value, void* resource_handle = nullptr)
{{
  TArgs args_t(args, num_args);

  if (num_args > 0 && args[num_args - 1].code == TypeIndex::kRuntimeKwargs) {{
    string_view arg_names[{}] {{{}}};
    KwargsUnpackHelper helper("{}", arg_names, {}, nullptr, 0);
    RTView pos_args[{}];
    helper.unpack(pos_args, args, num_args);  // /Users/bytedance/Developer/open_source_library/matxscript/examples/simple_function.py:5

    {}({}, 
     {}resource_handle);
  }} else {{
    switch(num_args) {{
      case {}: {{
        {}({}, 
         {}resource_handle);  // /Users/bytedance/Developer/open_source_library/matxscript/examples/simple_function.py:5
        int ret = 1;
        RTValue(std::move(ret)).MoveToCHost(out_ret_value);
      }} break;
      default: {{THROW_PY_TypeError("TODO");}} break;  // /Users/bytedance/Developer/open_source_library/matxscript/examples/simple_function.py:5
    }}
  }}

  return 0;
}}
'''
    if has_return_value:
        template = template_with_return
    else:
        template = template_without_return

    num_args = len(args)
    arg_names_concat_str = ', '.join([f'"{arg.name}"' for arg in args])
    args_dtype = [arg.type.name for arg in args]

    pos_arg_cast_lst = []
    args_t_cast_lst = []
    for arg_index in range(num_args):
        pos_arg_cast_lst.append(generate_ndarray_arg_cast('pos_args', arg_index, args_dtype[arg_index]))
        args_t_cast_lst.append(generate_ndarray_arg_cast('args_t', arg_index, args_dtype[arg_index]))

    kernel_name_indentation = len(kernel_name) * ' '
    if has_return_value:
        return_name_indentation = ' ' * 11
    else:
        return_name_indentation = ''
    pos_arg_cast_indentation = '\n     ' + kernel_name_indentation + return_name_indentation
    args_t_cast_indentation = '\n         ' + kernel_name_indentation + return_name_indentation
    pos_arg_cast = (',' + pos_arg_cast_indentation).join(pos_arg_cast_lst)
    args_t_cast = (',' + args_t_cast_indentation).join(args_t_cast_lst)

    return template.format(num_args, arg_names_concat_str, kernel_name, num_args, num_args, kernel_name,
                           pos_arg_cast, kernel_name_indentation, num_args, kernel_name,
                           args_t_cast, kernel_name_indentation)


def get_registration_str(kernel_name):
    # TODO: currently, only 1 function is here.
    template = '''
extern "C" {{

MATX_DLL MATXScriptBackendPackedCFunc __matxscript_func_array__[] = {{
    (MATXScriptBackendPackedCFunc){}__c_api,
}};
MATX_DLL MATXScriptFuncRegistry __matxscript_func_registry__ = {{
    "1\\000{}\\000",    __matxscript_func_array__,
}};

}} // extern C

extern "C" {{

MATX_DLL const char* __matxscript_closures_names__ = "1\\000{}\\000";

}} // extern C

    '''
    return template.format(kernel_name, kernel_name, kernel_name)


def get_c_api_declare(kernel_name):
    return f'int {kernel_name}__c_api(MATXScriptAny*, int, MATXScriptAny*, void*);'


def extract_cpp_code(code: str):
    return code.split("'''")[1][1:-1]


def matx_cpp_code_format(code: str) -> str:
    code = extract_cpp_code(code)
    # split include and kernel code
    first_newline_idx = code.find('\n')
    include_code_str = code[:first_newline_idx]
    kernel_code_str = code[first_newline_idx + 1:]

    # add matx include
    include_code_str += MATX_INCLUDE

    # extract kernel declaration
    first_open_bracket = kernel_code_str.find('{')
    kernel_declaration_str = kernel_code_str[:first_open_bracket]
    kernel_body_str = kernel_code_str[first_open_bracket:]

    kernel_declaration = cpp_parse.parse_cpp_declaration(kernel_declaration_str)

    kernel_declaration_without_default = copy.deepcopy(kernel_declaration)
    kernel_declaration_without_default.append_arg(SESSION_HANLDER)
    kernel_declaration_with_default = copy.deepcopy(kernel_declaration)
    kernel_declaration_with_default.append_arg(SESSION_HANLDER_WITH_DEAFULT)

    # add kernel declaration and c-api
    function_declaration_str = str(kernel_declaration_with_default) + ';' + '\n\n' + \
                               get_c_api_declare(kernel_declaration_with_default.func_name) + '\n'

    # add kernel
    kernel_impl_str = str(kernel_declaration_without_default) + '\n' + kernel_body_str

    # add kernel c-api

    kernel_c_api_impl_str = get_c_api(kernel_name=kernel_declaration.func_name,
                                      args=kernel_declaration.args,
                                      has_return_value=kernel_declaration.return_type.name != 'void')

    # add namespace
    kernel_code_str = ['namespace {', function_declaration_str, kernel_impl_str,
                       kernel_c_api_impl_str, '} // namespace']
    kernel_code_str = '\n\n'.join(kernel_code_str)

    # registration str
    registration_code_str = get_registration_str(kernel_name=kernel_declaration.func_name)

    # final code
    final_code = [include_code_str, kernel_code_str, registration_code_str]

    final_code = '\n\n'.join(final_code)

    return final_code


"""
Use a global variable to hack the compile_fx_inner and record the compiled code.
This works in single process problem, but requires careful review in multi-processing
"""


class FakeCallableWithCode():
    code = None

    def __call__(self, *args, **kwargs):
        raise NotImplementedError

    def set_code(self, code):
        self.code = code


fake_callable = FakeCallableWithCode()


@DebugContext.wrap
@torch.utils._python_dispatch._disable_current_modes()
def compile_fx_inner_cpu(
        gm: torch.fx.GraphModule,
        example_inputs: List[torch.Tensor],
        cudagraphs=None,
        num_fixed=0,
        is_backward=False,
        graph_id=None,
):
    # lift the maximum depth of the Python interpreter stack
    # to adapt large/deep models
    compile_fx.sys.setrecursionlimit(max(compile_fx.sys.getrecursionlimit(), 2000))
    V.debug.fx_graph(gm, example_inputs)
    shape_env = compile_fx._shape_env_from_inputs(example_inputs)
    fake_mode = compile_fx.fake_mode_from_tensors(example_inputs)
    graph = compile_fx.GraphLowering(
        gm,
        shape_env=shape_env,
        num_static_inputs=num_fixed,
        graph_id=graph_id,
        fake_mode=fake_mode,
    )
    with V.set_graph_handler(graph):
        graph.run(*example_inputs)
        code = graph.codegen()
        fake_callable.set_code(code)

    return fake_callable


def extract_inductor_code(kernel, example_inputs):
    model = fx.symbolic_trace(kernel)
    compile_fx.compile_fx(model, example_inputs_=example_inputs, inner_compile=compile_fx_inner_cpu)

    code = fake_callable.code
    return code
