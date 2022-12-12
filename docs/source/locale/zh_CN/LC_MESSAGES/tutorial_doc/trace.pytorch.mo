��          �               l  .   m     �     �     �  �   �  !   �  ^   �     ?  [   N     �  �   *     �     �     �    �       A     g   S  t   �  Q   0  }  �  B    	     C	     X	     r	    �	  '   �
  b   �
       i   0  �   �  �   M     �     �     �  �       �  >   �  ]   �  �   D  L   �   1. Define a nn.Module and call torch.jit.trace 2. Construct InferenceOp 2.1 From a existing instance 2.2 From local file 3. Now we can use infer_op as a normal matx op or call it in pipeline for trace. Notice that the inputs for calling infer_op are the same as ScriptModule, but users have to substitute torch.tensor with matx.NDArray. From ScriptModule(ScriptFunction) From a given ScriptModule and a device id, we can pack a ScriptModule into a matx InferenceOp. From nn.Module InferenceOp needs a device id. Loading trace also needs a device id. Their relationship is: It is mandatory that the output tensor from Pytorch model is contiguous. If not, please call tensor.contiguous() before output. Matx provides support for pytorch models. You can simply call matx.script() to convert a nn.Module or jit.ScriptModule to an InferenceOp and use it in trace pipeline. Overview Pytorch Integration Remarks This will call torch.jit.trace to convert nn.Module to ScriotModule during trace. So, there is no essential difference between this method and the one above. However, notice that users have to make sure that their nn.Module can be converted to ScriptModule by torch,jit.trace. Usage Using the same model above, we can skip torch.jit.trace as below. When InferenceOp device is cpu, matx will ignore device id given to trace, and InferenceOp runs on cpu. When InferenceOp device is gpu, and the trace is loaded to GPU, then InferenceOp will run on the gpu given to trace. When InferenceOp device isgpu, loading trace on CPU leads to undefined behaviors. Project-Id-Version: Matxscript 
Report-Msgid-Bugs-To: 
POT-Creation-Date: 2022-12-10 08:10+0800
PO-Revision-Date: YEAR-MO-DA HO:MI+ZONE
Last-Translator: FULL NAME <EMAIL@ADDRESS>
Language: zh_CN
Language-Team: zh_CN <LL@li.org>
Plural-Forms: nplurals=1; plural=0;
MIME-Version: 1.0
Content-Type: text/plain; charset=utf-8
Content-Transfer-Encoding: 8bit
Generated-By: Babel 2.11.0
 1. 定义一个简单的nn.Module，并将其转换为ScriptModule 2. 构造InferenceOp 2.1 从已有实例构建 2.2 从本地文件构建 3. 构造出InferenceOp后，我们可以如普通matx op一样调用，或者在pipeline中进行trace。调用InferenceOp的输入参数和调用ScriptModule需要的参数一致，仅需要注意的是将tensor类型用相应的NDArray类型替换即可。 从 ScriptModule(ScriptFunction) 构建 通过给定ScriptModule和设备（device id)，我们可以将一个ScriptModule封装成matx op 从 nn.Module 构建 InferenceOp需要指定device id，通用在session加载时也需要指定device id，其关系如下： 目前要求pytorch model输出的tensor是contiguous的，对于pytorch model非contiguous的tensor，pytorch model内部调用tensor.contiguous在输出前对其进行转换。 matxscript内置了对pytorch的支持，通过matx.script()将一个Pytorch实例包装成一个InferenceOp，可以用于被trace的pipeline中。 简介 PyTorch 集成 注意事项 同样infer_op可以进行调用及pipeline trace，在pipeline trace的过程中，InferenceOp内部会调用torch.jit.trace将nn.Module转换为ScriptModule， 因此nn.Module构造InferenceOp在本质上和ScriptModule没有什么不同，需要值得注意的是，在使用nn.Module构造的InfereceOp进行trace时，需要用户保证该nn.Module可以使用torch.jit.trace转换为ScriptModule。 使用方式 沿用MyCell模型，我们将其直接封装为InferenceOp。 InferenceOp device为cpu，则不关心session加载的device，InferenceOp在cpu上执行。 InferenceOp device为gpu，session加载的device为gpu，则忽略InferenceOp的id号，InfereceOp在session加载的device上执行。 InferenceOp device为gpu，session加载的device为cpu，行为未定义。 