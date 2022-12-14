.. design philosophy

Design Philosophy
##################################

Overview
**********************************
| Matx is originally designed to integrate training and inference in machine learning production system. It comes with the following goals:

* Python interpreter is not required during deployment. 
* Existing Python code that performs data preprocessing and postprocessing need not to be rewritten. 
* The generated code should run faster than the original Python version. 

| Given them, Matx is an ahead-of-time compiler that transforms Python into C++. Besides machine learning applications, we expect Matx to be used in a wider range of applications that involve translating Python into C++ for better performance.

High-level Concepts
**********************************

Script
==================================
Script is used to compile a Python function/class into a C++ function/class with Python wrapper. The Python wrapper has identical behavior as the original Python function/class. 

Operation (Op)
==================================
An operation (Op) is a compiled C++ function or a method of the compiled C++ class. It is the basic component used to construct a computational graph via tracing.

Trace
==================================
Trace is used to generate a computational graph composed of operations. Note that the computational graph must be a directly asyclic graph. The computational graph can be directly executed in Python. It can also be saved/loaded to/from disk files.

Python First
**********************************
Matx is an ahead-of-time compiler that compiles Python into C++. We choose Python as it is the dominate programming language in deep learning and has a large existing codebase. Note that we only support a subset of Python due to performance consideration.

Fast Execution
**********************************
As compiled code from Matx  is written in C++, it naturally achieves orders of speedup compared with the original Python code executed by Python Interpreter. To further improve the performance, we enforce type annotation when writing Python code.

Extensions Without Pain
**********************************
Writing custom C++ Operations can be easily integrated into Matx.

Machine Learning Support
**********************************

Preprocessing and Postprocessing
==================================
Data preprocessing and postprocessing are two key components in machine learning production systems. In NLP, data preprocessing includes text cleaning, text augmentation and tokenizing. A large portion of existing codebase is written in Python. With Matx, Developers can integrate data preprocessing code into neural network inference code easily.

Pytorch/TensorFlow Integration
==================================
Matx enables easy integration of existing Pytorch/TensorFlow models into preprocessing and postprocessing pipelines. 