��          |               �      �   b   �   �   R     D  �   X     �     �  5   �     0  �   7  F   �  }  0     �  9   �  �   �     �  Y   �     Y     e  -   l     �  u   �  .       Design Philosophy Existing Python code that performs data preprocessing and postprocessing need not to be rewritten. Given them, Matx is an ahead-of-time compiler that transforms Python into C++. Besides machine learning applications, we expect Matx to be used in a wider range of applications that involve translating Python into C++ for better performance. High-level Concepts Matx is originally designed to integrate training and inference in machine learning production system. It comes with the following goals: Operation (Op) Overview Python interpreter is not required during deployment. Script Script is used to compile a Python function/class into a C++ function/class with Python wrapper. The Python wrapper has identical behavior as the original Python function/class. The generated code should run faster than the original Python version. Project-Id-Version: Matxscript 
Report-Msgid-Bugs-To: 
POT-Creation-Date: 2022-12-10 03:03+0800
PO-Revision-Date: YEAR-MO-DA HO:MI+ZONE
Last-Translator: FULL NAME <EMAIL@ADDRESS>
Language: zh_CN
Language-Team: zh_CN <LL@li.org>
Plural-Forms: nplurals=1; plural=0;
MIME-Version: 1.0
Content-Type: text/plain; charset=utf-8
Content-Transfer-Encoding: 8bit
Generated-By: Babel 2.11.0
 设计哲学 已有的前后处理代码可直接使用在推理中。 为实现这些目标，我们将 matx 设计为一个 Python 代码的超前编译器，将Python 代码自动编译成C++。目前 Matx 主要支持机器学习相关的应用，但我们预期将 matx 推广到更多对性能敏感的python应用。 顶层设计 Matx 旨在将机器学习生产系统中的训练和推理一体化，具体目标为： 算子 (Op) 总览 在部署模型时不需要Python编译器。 编译 (Script) Script 可将 Python 函数/类编译为一个经过编译和包装并可在Python直接调用的的c++ 函数/类。 自动生成比Python 代码更快的代码。 