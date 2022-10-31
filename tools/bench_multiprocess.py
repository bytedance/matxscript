# Copyright 2022 ByteDance Ltd. and/or its affiliates.
#
import os
import time
import random
import multiprocessing
import matx
from typing import List, Any


def sub_thread_entry(query: str, max_ngram_size: int) -> Any:
    query = query.strip()
    query_terms = query.split(' ')
    ngram_list = []

    for l in range(1, max_ngram_size + 1):
        for j in range(0, len(query_terms) - l + 1):
            ngram_list.append(" ".join(query_terms[j: j + l]))
    return ngram_list


def bench_entry(query: str) -> Any:
    k = random.random()
    t = time.time()
    ngram_list = []
    for i in range(10000):
        ngram_list = sub_thread_entry(query, 4)
    return ngram_list, k, t


class MyWorker:

    def __init__(self):
        self.pool: Any = matx.make_native_object('ThreadPoolExecutor', 4, True)

    def __call__(self, query: str) -> Any:
        queries: List[str] = [query, query, query]
        return self.pool.ParallelFor(bench_entry, queries)


def prepare_model():
    test_make_ngram = matx.script(MyWorker)()

    def workflow(query):
        r1 = test_make_ngram(query)
        r2 = test_make_ngram(query)
        r3 = test_make_ngram(query)
        return r1, r2, r3

    jit_mod = matx.pipeline.Trace(workflow, "hello world")
    ret = jit_mod.run({"query": "hello world"})
    jit_mod.set_op_parallelism_threads(2)
    print(jit_mod.run({"query": "hello world"}))

    jit_mod.save("./my_pa_test")
    jit_mod = matx.pipeline.Load("./my_pa_test", -1)
    print(jit_mod.run({"query": "hello world"}))
    return jit_mod


def process():
    print("begin pid: ", os.getpid())
    jit_mod = matx.pipeline.Load("./my_pa_test", -1)
    ret = jit_mod.run({"query": "hello world"})
    print("end pid: ", os.getpid(), " result: ", ret)


if __name__ == '__main__':
    mod = prepare_model()
    process()
    p1 = multiprocessing.Process(target=process)
    p2 = multiprocessing.Process(target=process)
    p3 = multiprocessing.Process(target=process)
    p4 = multiprocessing.Process(target=process)
    p1.start()
    p2.start()
    p3.start()
    p4.start()
    p1.join()
    p2.join()
    p3.join()
    p4.join()
    process()
