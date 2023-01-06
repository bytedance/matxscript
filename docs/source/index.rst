.. Matxscript documentation master file, created by
   sphinx-quickstart on Fri Dec  2 04:44:00 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to MATXScript's documentation!
======================================

Introduction
--------------------------------------
MATXScript(Matx) is a high-performance, extensible Python AOT compiler that compiles Python class/function to C++ without any runtime overhead.
Typical speedups over Python are on the order of 10-100x. Matx supports pmap which can lead to speedups many times higher still.


Currently matx is widely used in Bytedance. Including:

1) Unify the training and inference for deep learning.
2) Accelerate some offline MapReduce tasks.
3) Provide flexibility for some C++ engines, etc.


A Quick Example
--------------------------------------

.. code:: python

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


.. toctree::
   :maxdepth: 1
   :caption: Design Philosophy
   :hidden:

   tutorial_doc/design_philosophy


.. toctree::
   :maxdepth: 1
   :caption: Quick Start
   :hidden:

   tutorial_doc/installation
   tutorial_doc/quick_start


.. toctree::
   :maxdepth: 1
   :caption: Basics
   :hidden:

   tutorial_doc/ndarray
   tutorial_doc/script
   tutorial_doc/trace

.. toctree:: 
   :maxdepth: 1
   :caption: Advanced
   :hidden:

   tutorial_doc/advanced_usage

.. toctree::
   :maxdepth: 1
   :caption: Benchmarks
   :hidden:

   tutorial_doc/benchmarks

.. toctree::
   :maxdepth: 1
   :caption: Examples
   :hidden:

   tutorial_doc/e2e_example


.. toctree::
   :maxdepth: 1
   :caption: API
   :hidden:
   :titlesonly:

   apidoc/modules

.. toctree::
   :maxdepth: 1
   :caption: Appendix
   :hidden:

   tutorial_doc/appendix


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

.. raw:: html

    <embed>
        <p>Languages: <a href="/matxscript/index.html">English</a> <a href="/matxscript/zh-CN/index.html">中文</a></p>
    </embed>

