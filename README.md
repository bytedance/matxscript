MatxScript
===============================================================

[Documentation](https://bytedance.github.io/matxscript/) |
[文档](https://bytedance.github.io/matxscript/zh-CN/index.html) |
[Contributors](CONTRIBUTORS.md) |
[Release Notes](NEWS.md)


MatxScript is an ahead of time compiler for a subset of the Python language.

Installation
-------
```bash
pip install matxscript
```


A Quick Example
-------
```python
import matx
import timeit

def fib(n: int) -> int:
    if n <= 1:
        return n
    else:
        return fib(n - 1) + fib(n - 2)


if __name__ == '__main__':
    fib_script = matx.script(fib)
    
    # test on Macbook with m1 chip
    print(f'Python execution time: {timeit.timeit(lambda: fib(30), number=10)}s')  # 1.59s
    print(f'Matx execution time: {timeit.timeit(lambda: fib_script(30), number=10)}s') # 0.03s
```


License
-------
© Bytedance Inc. Licensed under an [Apache-2.0](LICENSE) license.

Contribute to MatxScript
------------------------
Everyone is welcomed to contribute. We value all forms of contributions, including, but not limited to:

   - Code reviewing of the existing patches.
   - Documentation and usage examples
   - Community participation in issues.
   - Code readability and developer guide
      - We welcome contributions that add code comments to improve readability
      - We also welcome contributions to docs to explain the design choices of the internal.
   - Test cases to make the codebase more robust
   - Tutorials, blog posts, talks that promote the project.

Acknowledgement
---------------
We learned a lot from the following projects when building MatxScript.
- [TVM](https://github.com/apache/tvm): Part of MatxScript's IR and Runtime
  originates from TVM. We also learned and adapted some part of codegen pipeline from TVM.

- [Python](https://github.com/python/cpython/tree/3.8): Part of the runtime code comes from cpython for align the semantics

- [Folly](https://github.com/facebook/folly): We adopted the idea of FBString to design our runtime::string_core

- [abseil-cpp](https://github.com/abseil/abseil-cpp): The string_view structure and ByteHash algorithm originates from abseil

- [rapidjson](https://github.com/Tencent/rapidjson): The json module is implemented based on rapidjson
