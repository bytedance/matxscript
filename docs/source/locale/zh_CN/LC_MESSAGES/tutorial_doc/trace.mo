��          �               �  <   �  1   
  2   <     o     �  ;   �  `   �  u   L  W   �  "     �   =  i        p  �  �  X  '  	   �  5   �     �  �   �     d	  Q   �	  �   �	  m   �
  M        `  P  w  }  �  J   F  <   �  <   �  4     7   @  q   x  Z   �  j   E  S   �       �     p   �           �   %       7   #     [  n   n     �  @   �  �   1  w   �  Z   0  !   �  �   �   **Modular design the pipeline. Each module is an operator.** **Operators can be implemented in C++ or Python** **Organize operators in your pipeline into a DAG** **Which files will be dumped?** Code after return is discarded. Every rectangle in the following chart is considered an Op. For now, only the Matx-Op Calls are recorded normally and other statements will be as constants. For operators implemented in Python. It needs matx.script to be able to use in trace. Please refer to Script chapter. For/While is unrolled based on the given input during trace. Usually, this causes bugs. How do we capture the local files? In a word, we will detect the parameters with type of str, list and dict(recursively). The parameters should be in the __init__() function of ops. If the parameter is a valid file path, we'll dump it. In this example, assets/vocab.txt will be dumped, but configs/my_config.json or models/bert1225 will not. Integration Examples It is because that we have only one op in pipeline, which is MyOperator, and only vocab_fn, which is "assets/vocab.txt" is a valid file path. CONFIG_FN is a global variable, we havn't detected such paths used in __init__() function. As for model_name, we build a path in the __init__() function, but the value of the parameter is only "bert1225", which is not a valid path(the valid path should be "models/bert1225"). Matxscript can dump the local files used to initialize the ops into the model directory during tracing, and modify the corresponding path parameters, so that you can successfully load the matxscript model under any path without manually packaging and deploying these files. This brings great convenience for deploying models on server clusters. Objective Only the if-branch executed during trace is recorded. Restrictions on design pattern Take text classification task as an example. Tokenization can be designed  as an operator. One-hot encoding can be another operator. Third party library support. To use trace without potential bugs, users need to follow the restrictions below: Trace can conveniently record the execution process of Python code and then generate a Graph which can be saved on disk. In this way, it is very convenient to distribute and deploy to any place for execution. Trace depends on the execution of the Python, so only the code executed during the trace process is recorded. We only trace the files appeared in the initializing parameters. For example, What does trace record When the code being traced contains calls to a third party library such as requests, then returns from the calls are saved as constant and used when executing the trace result. In other words, third party libraries are only called during the generation of the trace file but not called anymore during loading and running the trace file. Project-Id-Version: Matxscript 
Report-Msgid-Bugs-To: 
POT-Creation-Date: 2023-01-06 18:44+0800
PO-Revision-Date: YEAR-MO-DA HO:MI+ZONE
Last-Translator: FULL NAME <EMAIL@ADDRESS>
Language: zh_CN
Language-Team: zh_CN <LL@li.org>
Plural-Forms: nplurals=1; plural=0;
MIME-Version: 1.0
Content-Type: text/plain; charset=utf-8
Content-Transfer-Encoding: 8bit
Generated-By: Babel 2.11.0
 **业务需要进行模块设计，每个模块视为一个 Operator(Op)** **Op 可以由 C++ 或 Python 开发，不受 Trace 约束** **由 Op 集合组成一个 DAG，即为业务的 Pipeline** **哪些文件和目录会被一起打包保存？** Return 之后的逻辑会被丢弃，需要灵活把握 比如下图的文本分类的 pipeline，每个框均是一个 Op，这些 Op 串联起来是一个简单的 DAG 当前只能正常记录 Matx Op 的调用，其他语句的执行结果会被当成常量 使用 Python 编写的 类或函数 需要使用 matx.script 编译成 Op，具体请参照 Script 章节 For/While 循环会根据当时的输入条件强制展开，通常这意味着 BUG 自动打包本地文件 简而言之，我们会分析 Op.__init__ 的参数，如果是字符串类型，并且是一个有效的路径，则会打包。 在这个例子中，assets/vocab.txt 会被打包，但是 configs/my_config.json 和 models/bert1225 不会。 集成样例 这主要是因为在 Trace 时 MyOperator.__init__ 参数中，只有 vocab_fn 被赋值为一个有效的文件路径，CONFIG_FN 是一个全局变量，并不会分析它, model_name 初始值并不是一个有效路径，因此这两个都不会被打包。 MATXScript 预期可以打包用于初始化 OP 的文件或者目录，并且会自动修改成相对路径。因此，我们可以将打包后的目录移动到任意地方都可以加载执行，这为服务热更新提供了较大的便利。 目标 IF 语句只能记录一个 branch，需要灵活把握 约束设计模式 比如一个文本分类的 pipeline，分词可以认为是一个 Op，文本 ID 化也是一个 Op 等等。 第三方库支持 使用 Trace 机制是有代价的，需要遵守以下约定： Trace 可以追踪记录 Python 代码执行过程，生成 Graph，并支持一键打包，方便分发部署到任意地方执行。 Trace 机制是依赖 Python 执行才能记录，换句话说，能够记录的逻辑需要在运行时被执行到！ MATXScript 仅打包 class.__init__ 函数中参数对应的本地文件，举例如下， 能够记录哪些执行过程？ 在被 trace 的函数中如果调用了第三方库比如 requests，那么本次的结果将被当做常量保存并且会在之后执行时使用。换句话说，第三方库执行的结果被固化成常量了。 