.. Pytorch Integration

Pytorch Integration
#####################################################

Torch support
*****************************************************
Matx provides support for pytorch models. You can simply call matx.script() to convert a nn.Module or jit.ScriptModule to an InferenceOp and use it in trace pipeline.

InferenceOp
*****************************************************

Construction
=====================================================
There are two ways to construct InferenceOp 

From ScriptModule(ScriptFunction)
-----------------------------------------------------

| From a given ScriptModule and a device id, we can pack a ScriptModule into a matx InferenceOp.

| 1. Define a nn.Module and call torch.jit.trace

.. code-block:: python3 

    import torch

    class MyCell(torch.nn.Module):
    def __init__(self):
        super(MyCell, self).__init__()
        self.linear = torch.nn.Linear(4, 4)

    def forward(self, x, h):
        new_h = torch.tanh(self.linear(x) + h)
        return new_h, new_h

    device = torch.device("cuda:0")
    my_cell = MyCell().to(device)
    script_model = torch.jit.trace(my_cell, (torch.rand(3, 4, device=device), torch.rand(3, 4, device=device)))

| 2. Construct InferenceOp

.. code-block:: python3 

    import matx

    infer_op = matx.script(script_model)

| 3. Now we can use infer_op as a normal matx op or call it in pipeline for trace. Notice that the inputs for calling infer_op are the same as ScriptModule, but users have to substitute torch.tensor with matx.NDArray.

.. code-block:: python3 

    x = matx.array.rand([3, 4])
    h = matx.array.rand([3, 4])

    def process(x, h):
        return infer_op(x, h)

    r = process(x, h)
    print(r)
    mod = matx.trace(process, x, h)
    r = mod.run({'x': x, 'h': h})
    print(r)

From nn.Module 
-----------------------------------------------------

Using the same model above, we can skip torch.jit.trace as below.
.. code-block:: python3 

    infer_op = matx.script(my_cell, device=0)

This will call torch.jit.trace to convert nn.Module to ScriotModule during trace. So, there is no essential difference between this method and the one above. However, notice that users have to make sure that their nn.Module can be converted to ScriptModule by torch,jit.trace.

Remarks
=====================================================
#. InferenceOp needs a device id. Loading trace also needs a device id. Their relationship is:
    #. When InferenceOp device is cpu, matx will ignore device id given to trace, and InferenceOp runs on cpu.
    #. When InferenceOp device is gpu, and the trace is loaded to GPU, then InferenceOp will run on the gpu given to trace.
    #. When InferenceOp device isgpu, loading trace on CPU leads to undefined behaviors.
#. It is mandatory that the output tensor from Pytorch model is contiguous. If not, please call tensor.contiguous() before output.

