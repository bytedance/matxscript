.. Tensorflow Integration

Tensorflow Integration
###############################################

Save model
***********************************************

.. code-block:: python3 

    import tensorflow as tf

    class Adder(tf.Module):
    @tf.function
    def add(self, x):
        return x + x

    model = Adder()

    tf.saved_model.save(model, "./", signatures=model.add.get_concrete_function(tf.TensorSpec([], tf.float32)))

Use matxscript load SavedModel
***********************************************

.. code-block:: python3 

    import matx

    tf_op = matx.script("./", backend='TensorFlow', device=-1, use_xla=0, allow_growth=False)


Trace and inference 
***********************************************

.. code-block:: python3 

    ix = matx.NDArray([1], [1], 'float32')

    def process(x):
        return tf_op({"x":x})

    ret = process(ix)
    print(ret)

    s = matx.trace(process, ix)
    ret = s.run({"x":ix})
    print(ret)