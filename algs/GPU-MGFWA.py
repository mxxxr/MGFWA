import os
import time
import torch

EPS = 1e-8

class GPUMGFWA(object):

    def __init__(self):
        # Parameters

        # params of method
        self.fw_size = None  # num of fireworks
        self.sp_size = None  # total spark size
        self.init_amp = None  # initial dynamic amplitude
        self.gm_ratio = None  # ratio for top sparks in guided mutation

        # params of problem
        self.evaluator = None
        self.dim = None
        self.upper_bound = None
        self.lower_bound = None
        self.max_eval = None

        self.parameter_N = None
        self.parameter_b = None

        self.name = None


    def load_prob(self,
                  # params for prob
                  batch_size=64,
                  evaluator=None,
                  dim=4,
                  upper_bound=100.0,
                  lower_bound=-100.0,
                  max_eval=3000,
                  # params for method
                  fw_size=25,
                  sp_size=300,
                  init_amp=200,
                  gm_ratio=0.2, 
                  parameter_N=10,
                  parameter_b=1.5,
                  name='GPUMGFWA'):

        # load params
        self.batch_size = batch_size
        self.evaluator = evaluator
        self.dim = dim
        self.upper_bound = upper_bound
        self.lower_bound = lower_bound
        self.max_eval = max_eval

        self.fw_size = fw_size
        self.sp_size = sp_size
        self.init_amp = init_amp
        self.gm_ratio = gm_ratio

        self.parameter_N = parameter_N
        self.parameter_b = parameter_b

        self.name = name

    def run(self, dim, duration, evaluator):
        with torch.no_grad():
            self.dim = dim
            self.evaluator = evaluator
            # running time
            begin_time = time.perf_counter()
            batch_size = self.batch_size

            fireworks = torch.rand(batch_size, self.fw_size, 1, self.dim, device='cuda') * (self.upper_bound - self.lower_bound) + self.lower_bound
            fits = self.evaluator(fireworks).squeeze(-1)
            amps = torch.ones(batch_size, self.fw_size, 1, device='cuda') * self.init_amp

            num_sparks = int(self.sp_size / self.fw_size)
            guiding_num = int(num_sparks * self.gm_ratio)
            num_eval = 0

            results = []
            times = []

            # Start main loop
            while time.perf_counter() - begin_time < duration:

                # explode
                bias = torch.rand(batch_size, self.fw_size, num_sparks, self.dim, device='cuda') * (self.upper_bound - self.lower_bound) + self.lower_bound
                bias = bias * amps.unsqueeze(-1)
                e_sparks = fireworks + bias
                e_sparks = self._map(e_sparks, fireworks, amps)
                e_fits = self.evaluator(e_sparks).squeeze(-1)

                # mutate
                sort_idx = torch.argsort(e_fits, dim=2)
                top_idx = sort_idx[:, :, -guiding_num:]
                btm_idx = sort_idx[:, :, :guiding_num]
                top_mean = torch.mean(e_sparks.gather(2, top_idx.unsqueeze(-1).expand(batch_size, self.fw_size, guiding_num, self.dim)), dim=2)
                btm_mean = torch.mean(e_sparks.gather(2, btm_idx.unsqueeze(-1).expand(batch_size, self.fw_size, guiding_num, self.dim)), dim=2)
                delta = top_mean - btm_mean
                weight = torch.rand(batch_size, self.fw_size, self.parameter_N, 1, device='cuda') * self.parameter_b
                m_sparks = btm_mean.unsqueeze(2) - delta.unsqueeze(2) * weight
                m_sparks = self._map(m_sparks, fireworks, amps)
                m_fits = self.evaluator(m_sparks).squeeze(-1)

                # select
                sparks = torch.cat([fireworks, e_sparks, m_sparks], dim=2)
                spark_fits = torch.cat([fits, e_fits, m_fits], dim=2)
                min_idx = torch.argmin(spark_fits, dim=2).unsqueeze(-1)
                n_fireworks = sparks.gather(2, min_idx.unsqueeze(-2).expand(batch_size, self.fw_size, 1, self.dim))
                n_fits = spark_fits.gather(2, min_idx)

                # new fireworks
                fireworks = n_fireworks
                fits = n_fits

                # dynamic amps
                amps = torch.where(n_fits < fits - EPS, amps * 1.2, amps * 0.9) 

                min_fw_idx = torch.argmin(n_fits, dim=1).squeeze(-1)
                best_idv = n_fireworks[torch.arange(batch_size), min_fw_idx, :]
                best_fit = n_fits[torch.arange(batch_size), min_fw_idx]

                min_idx = torch.argmin(best_fit)
                best_best_idv = best_idv[min_idx]
                best_best_fit = best_fit[min_idx]

                results.append(best_best_fit.item())
                times.append(time.perf_counter() - begin_time)
                print(best_best_fit, time.perf_counter() - begin_time)
   
            min_fw_idx = torch.argmin(n_fits, dim=1).squeeze(-1)
            best_idv = n_fireworks[torch.arange(batch_size), min_fw_idx, :]
            best_fit = n_fits[torch.arange(batch_size), min_fw_idx]

            min_idx = torch.argmin(best_fit)
            best_best_fit = best_fit[min_idx]

            return results, times

    
    def _map(self, samples, firework, amp):
        in_bound = (samples > self.lower_bound) & (samples < self.upper_bound)
        lb = torch.max(firework - amp.unsqueeze(-1), torch.ones_like(firework, device='cuda') * self.lower_bound)
        ub = torch.min(firework + amp.unsqueeze(-1), torch.ones_like(firework, device='cuda') * self.upper_bound)
        rand_samples = torch.rand_like(samples, device='cuda') * (ub - lb) + lb
        samples = torch.where(in_bound, samples, rand_samples)
        return samples

