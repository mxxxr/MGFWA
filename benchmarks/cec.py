import os
import sys
import numpy as np

import benchmarks.cec2013.cec13 as f13
import benchmarks.cec2017.cec17 as f17


def func_wrapper(func, func_id):
    def wrapped(x):
        
        origin_shape = x.shape
        dim = origin_shape[-1]
        
        if type(x) is np.ndarray:
            x = x.reshape((-1, dim)).tolist()
        if func == "cec13":
            tmp = f13.eval(x, func_id+1)
        elif func == "cec17":
            tmp = f17.eval(x, func_id+1)
        else:
            raise Exception("No such benchmark!")

        return np.array(tmp).reshape(origin_shape[:-1])

    return wrapped


class CEC13:
    def __init__(self) -> None:
        self.func_num = 28
        self.eval_num = 51
        self.name = 'cec2013'
        self.funcs = [func_wrapper("cec13", func_id) for func_id in range(self.func_num)]


class CEC17:
    def __init__(self) -> None:
        self.func_num = 30
        self.eval_num = 51
        self.name = 'cec2017'
        self.funcs = [func_wrapper("cec17", func_id) for func_id in range(self.func_num)]

