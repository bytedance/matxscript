.. trace
###############################################
Trace
###############################################

************************************************
Objective
************************************************

Trace can conveniently record the execution process of Python code
and then generate a Graph which can be saved on disk.
In this way, it is very convenient to distribute and deploy to any place for execution.

************************************************
What does trace record
************************************************

| Trace depends on the execution of the Python, so only the code executed during the trace process is recorded.

* For now, only the Matx-Op Calls are recorded normally and other statements will be as constants.
* Only the if-branch executed during trace is recorded.
* For/While is unrolled based on the given input during trace. Usually, this causes bugs.
* Code after return is discarded.

************************************************
Restrictions on design pattern
************************************************
To use trace without potential bugs, users need to follow the restrictions below:

- **Modular design the pipeline. Each module is an operator.**

   Take text classification task as an example. Tokenization can be designed  as an operator. One-hot encoding can be another operator.

- **Organize operators in your pipeline into a DAG**

   Every rectangle in the following chart is considered an Op.

- **Operators can be implemented in C++ or Python**

   For operators implemented in Python. It needs matx.script to be able to use in trace. Please refer to Script chapter.

************************************************
Third party library support.
************************************************
When the code being traced contains calls to a third party library such as requests, then returns from the calls are saved as constant and used when executing the trace result. In other words, third party libraries are only called during the generation of the trace file but not called anymore during loading and running the trace file.

************************************************
How do we capture the local files?
************************************************
Matxscript can dump the local files used to initialize the ops into the model directory during tracing, and modify the corresponding path parameters, so that you can successfully load the matxscript model under any path without manually packaging and deploying these files. This brings great convenience for deploying models on server clusters.

- **Which files will be dumped?**


We only trace the files appeared in the initializing parameters. For example,


.. code-block:: python

   import matx
   from typing import Any

   CONFIG_FN = "configs/my_config.json"

   class MyOperator:
      def __init__(self, vocab_fn: str, model_name: str) -> None:
         self.vocab = self.load_vocab(vocab_fn)
         self.config = self.load_config(CONFIG_FN)
         self.sub_model = MyModel("models/" + model_name)

      def load_vocab(self, fn: str) -> Any:
         ...

      def load_config(self, fn: str) -> Any:
         ...

      def __call__(self, input: Any) -> Any:
         ...

   op = matx.script(MyOperator)("assets/vocab.txt", "bert1225")

   def pipeline(input):
      return op(input)

   mod = matx.trace(pipeline, "query")
   mod.save(...)

In this example, assets/vocab.txt will be dumped, but configs/my_config.json or models/bert1225 will not.

It is because that we have only one op in pipeline, which is MyOperator, and only vocab_fn, which is "assets/vocab.txt" is a valid file path. CONFIG_FN is a global variable, we havn't detected such paths used in __init__() function. As for model_name, we build a path in the __init__() function, but the value of the parameter is only "bert1225", which is not a valid path(the valid path should be "models/bert1225").

In a word, we will detect the parameters with type of str, list and dict(recursively). The parameters should be in the __init__() function of ops. If the parameter is a valid file path, we'll dump it.


************************************************
Integration Examples
************************************************

.. toctree::
   :maxdepth: 1
   
   trace.pytorch
   trace.tensorflow