.. quick start

#################################
Quick Start
#################################


This section provides an example to walk through the basic usage of Matx. The complete code can be found `here <https://github.com/bytedance/matxscript/blob/main/examples/text2ids/text2ids.py>`_.


*********************************
1. Import dependencies
*********************************
.. code-block:: python3

    from typing import List, Dict, Callable, Any, AnyStr
    import matx


Note that we import various types from Python typing module as Matx enforces type annotations.

*********************************
2. Constructing Op
*********************************
An operation (Op) can be a class method or a function.

.. code-block:: python

    class Text2Ids:
        def __init__(self, texts: List[str]) -> None:
            self.table: Dict[str, int] = {}
            for i in range(len(texts)):
                self.table[texts[i]] = i

        def lookup(self, words: List[str]) -> List[int]:
            return [self.table.get(word, -1) for word in words]

.. code-block:: python

    op = Text2Ids(["hello", "world"])
    examples = "hello world unknown".split()
    ret = op.lookup(examples)
    print(ret)
    # should print out [0, 1, -1]

*********************************
3. Script
*********************************

.. code-block:: python

    cpp_text2id = matx.script(Text2Ids)(["hello", "world"])
    ret = cpp_text2id.lookup(examples)
    print(ret)
    # should print out [0, 1, -1]

*********************************
4. Trace
*********************************

.. code-block:: python

    def wrapper(inputs):
        return cpp_text2id.lookup(inputs)

    # trace and save
    traced = matx.trace(wrapper, examples)
    traced.save("demo_text2id")

    # load and run
    # for matx.load, the first argument is the stored trace path
    # the second argument indicates the device for further running the code
    # -1 means cpu, if for gpu, just pass in the device id
    loaded = matx.load("demo_text2id", -1)
    # we call 'run' interface here to actually run the traced op
    # note that the argument is a dict, where the key is the arg name of the traced function
    # and the value is the actual input data
    ret = loaded.run({"inputs": examples})
    print(ret)
    # should print out [0, 1, -1]

*********************************
5. C++ deployment
*********************************

.. code-block:: c++

    #include <iostream>
    #include <map>
    #include <string>
    #include <vector>

    #include <matxscript/pipeline/tx_session.h>

    using namespace ::matxscript::runtime;

    int main(int argc, char* argv[]) {
    // test case
    std::unordered_map<std::string, RTValue> feed_dict;
    feed_dict.emplace("inputs", List{Unicode(U"hello"), Unicode(U"world"), Unicode(U"unknown")});
    std::vector<std::pair<std::string, RTValue>> result;
    const char* module_path = argv[1];
    const char* module_name = "model.spec.json";
    {
        auto sess = TXSession::Load(module_path, module_name);
        auto result = sess->Run(feed_dict);
        for (auto& r : result) {
        std::cout << "result: " << r.second << std::endl;
        }
    }
    return 0;
    }


.. code-block:: bash

    MX_CFLAGS=$(python3 -c 'import matx; print( " ".join(matx.get_cflags()) ) ' )
    MX_LINK_FLAGS=$(python3 -c 'import matx; print( " ".join(matx.get_link_flags()) ) ' )
    RUNTIME_PATHS=$(python3 -c 'import matx; print( " ".join(["-Wl,-rpath," + p for p in matx.cpp_extension.library_paths()]) )')
    g++ -O2 -fPIC -std=c++14 $MX_CFLAGS $MX_LINK_FLAGS ${RUNTIME_PATHS} text2ids.cc -o text2ids


.. code-block:: bash

    ./text2ids demo_text2id
    # should print out 
    # result: [0, 1, -1]
