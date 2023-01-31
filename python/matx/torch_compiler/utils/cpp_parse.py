import dataclasses
from typing import List, Union


@dataclasses.dataclass
class CPPType(object):
    name: str = None
    is_pointer: bool = False

    def __str__(self):
        result = self.name
        if self.is_pointer:
            result += '*'

        return result


@dataclasses.dataclass
class CPPArg(object):
    name: str = None
    type: CPPType = CPPType()
    is_const: bool = False
    is_restricted: bool = False
    default_val: Union[str, None] = None

    def __str__(self):
        result = []
        if self.is_const:
            result.append('const')
        result.append(str(self.type))
        if self.is_restricted:
            result.append('__restrict__')
        result.append(self.name)

        if self.default_val is not None:
            result.append(f'= {self.default_val}')

        return ' '.join(result)


def parse_cpp_arg(cpp_arg_str: str) -> CPPArg:
    """Parse the C++ arg from a string such as const float* __restrict__ a = null_ptr

    :param cpp_arg_str: the string of the argument
    :return: a CPPArg dataclass
    """

    cpp_arg = CPPArg()

    # find if there is a default value
    if '=' in cpp_arg_str:
        cpp_arg_str, default_val = cpp_arg_str.split('=')
        default_val = default_val.replace(' ', '')
        cpp_arg.default_val = default_val

    word = cpp_arg_str.split()

    cpp_arg.name = word[-1]

    for w in word[:-1]:
        if w == 'const':
            cpp_arg.is_const = True
        elif w == '*':
            cpp_arg.type.is_pointer = True
        elif w == '__restrict__':
            cpp_arg.is_restricted = True
        else:
            # type
            if w[-1] == '*':
                cpp_arg.type.is_pointer = True
                w = w[:-1]  # remove *
            cpp_arg.type.name = w

    return cpp_arg


@dataclasses.dataclass
class CPPDeclaration(object):
    func_name: str = None
    return_type: CPPType = CPPType()
    args: List[CPPArg] = dataclasses.field(default_factory=list)
    is_extern_c: bool = False

    def append_arg(self, arg: CPPArg):
        self.args.append(arg)

    def __str__(self):
        result = []
        if self.is_extern_c:
            result.append('extern "C"')
        result.append(str(self.return_type))
        result.append(self.func_name)

        front = ' '.join(result)
        num_spaces = len(front) + 1
        interval = ',\n' + ' ' * num_spaces

        args_str = interval.join([str(arg) for arg in self.args])

        return front + '(' + args_str + ')'


def parse_cpp_declaration(cpp_declaration_str: str) -> CPPDeclaration:
    """Parse the CPP declaration in string and return a CPPDeclaration.

    :param cpp_declaration_str:
    :return:
    """
    cpp_declaration = CPPDeclaration()

    identifier_return_name, cpp_arg_str = cpp_declaration_str.split('(')
    cpp_arg_str = cpp_arg_str.split(')')[0]
    cpp_arg_str_lst = cpp_arg_str.split(',')
    # arguments
    for cpp_arg_str in cpp_arg_str_lst:
        cpp_declaration.args.append(parse_cpp_arg(cpp_arg_str))

    # process return type and function name
    identifier_return_name_lst = identifier_return_name.split()
    if identifier_return_name_lst[0] == 'extern' and identifier_return_name_lst[1] == '"C"':
        cpp_declaration.is_extern_c = True
        identifier_return_name_lst = identifier_return_name_lst[2:]

    cpp_declaration.func_name = identifier_return_name_lst[-1]
    # remove func_name
    return_type_str_lst = identifier_return_name_lst[:-1]

    if len(return_type_str_lst) == 1:
        return_type_str = return_type_str_lst[0]
        if return_type_str[-1] == '*':
            cpp_declaration.return_type.name = return_type_str[:-1]
            cpp_declaration.return_type.is_pointer = True
        else:
            cpp_declaration.return_type.name = return_type_str
    else:
        assert len(return_type_str_lst) == 2
        assert return_type_str_lst[-1] == '*'
        cpp_declaration.return_type.name = return_type_str_lst[0]
        cpp_declaration.return_type.is_pointer = True

    return cpp_declaration
