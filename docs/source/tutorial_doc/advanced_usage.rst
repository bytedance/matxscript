***************************************
Multithreading
***************************************
Due to Gobal Interpreter Lock (GIL) implemented in Python interpreter, multithreading in Python can‘t be truly executed in parallel. In matx, we implement matx.pmap to support multithreading in C++.

.. code-block:: python3 

    import matx
    from typing import Any, List

    def lower_impl(s: str) -> str:
        return s.lower()

    def MyParallelLower(inputs: List[str]) -> List[str]:
        return matx.pmap(lower_impl, inputs)

    p_lower = matx.script(MyParallelLower)
    print(p_lower(["Hello"]))

***************************************
Regular Expression
***************************************
Matx implements a builtin regular expression engine based on `PCRE <https://github.com/PCRE2Project/pcre2>`_.

.. code-block:: python3 

    import matx
    from typing import Any, List

    class Spliter:
        def __init__(self) -> None:
            self.regex: Any = matx.Regex(
                r'(([\v\f\r\n]+)|(?<=[^？。；，！!?][？。；，！!?])(?=[^？。；，！!?]))')

        def __call__(self, ss: str) -> List[str]:
            return self.regex.split(ss)

    spliter = matx.script(Spliter)()
    print(spliter("hELLO \v WORLD"))
