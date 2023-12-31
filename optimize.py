from argparse import ArgumentParser
from tqdm import tqdm
import numpy as np

from benchmarks.cec import CEC13, CEC17
from algs.MGFWA import MGFWA
from algs.LoTFWA import LoTFWA

parser = ArgumentParser()
parser.add_argument('--alg', type=str, required=True, choices=['MGFWA', 'LoTFWA'], help='which optimization algorithm you want to use')
parser.add_argument('--benchmark', type=str, required=True, choices=['cec2013', 'cec2017'], help='the benchmark you want to test on')
args = parser.parse_args()

alg = {
    "MGFWA": MGFWA,
    "LoTFWA": LoTFWA
}[args.alg]()

benchmark = {
    "cec2013": CEC13,
    "cec2017": CEC17
}[args.benchmark]()

filename = "results/MGFWA/13MGFWA_standard.txt"
for fun_id in range(benchmark.func_num):
    print("Function #{}, Optimizing...".format(fun_id+1))
    alg.load_prob(evaluator=benchmark.funcs[fun_id])
    bestFit, runTime = [], []
    for run_id in tqdm(range(benchmark.eval_num)):
        best_fit, run_time, _ = alg.run()
        if benchmark.name == 'cec2013':
            bestFit.append(best_fit - (fun_id - 13 - (fun_id <= 13)) * 100)
        elif benchmark.name == 'cec2017':
            bestFit.append(best_fit - (fun_id + 1) * 100)
        else:
            raise ValueError('No such benchmark!')
        runTime.append(run_time)
    print("MAX:", np.max(bestFit))
    print("MIN:", np.min(bestFit))
    print("MEAN:", np.mean(bestFit))
    print("MEDIAN:", np.median(bestFit))
    print("STD:", np.std(bestFit))
    print("Average runtime of a run: {} \n".format(np.mean(runTime)))