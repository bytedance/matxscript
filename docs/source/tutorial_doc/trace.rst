.. trace
###############################################
Trace
###############################################

************************************************
Objective
************************************************

Trace can conviencelly pack training code written in Python to a format that matx c++ interface can recognize and use for online inference. It records the execution process of given Python code and saves it as a graph on disk. 

************************************************
What does trace record
************************************************

| Trace depends on the execution of the code, so only code executed during the trace process is recoreded.

* For now, only the Op Call Graph is saved.
* Only the if-branch executed during trace is recorded.
* For/While is unrolled based on the given input during trace. Usually, this causes bugs.
* Code after return is discarded.

************************************************
Restrictions on design pattern
************************************************
To use trace without potential bugs, users need to follow the restrctions below:

Modular design the pipeline. Each module is an operator.
====================================================================
Take text classification task as an example. Tokenization can be designed  as an operator. One-hot encoding can be another operator. 

Organize operators in your pipeline into a DAG
====================================================================
Every rectangle in the following chart is considered an Op.

Operators can be implemented in C++ or Python
====================================================================
For operators implemented in Python. It needs matx.script to be able to use in trace. Please refer to Script chapter.

************************************************
Third party library support.
************************************************
When the code being traced contains calls to a third party library such as requests, then returns from the calls are saved as constant and used when executing the trace result. In other words, third party libraries are only called during the generation of the trace file but not called anymore during loading and running the trace file.
 
************************************************
Integration Examples
************************************************

.. toctree::
   :maxdepth: 1
   
   trace.pytorch
   trace.tensorflow