class KernelBaseOp:
    opname = None
    operator = None

    def __init__(self):
        if self.opname is None or self.operator is None:
            raise SyntaxError("opname and/or operator has to be defined")
        self.result_dtype = None
        self.result_shape = None
        self.result_type = None
