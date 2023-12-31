import os
import time
import numpy as np

EPS = 1e-8


class MGFWA(object):

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

        self.max_iter = None
        self.max_eval = None


    def load_prob(self,
                  # params for prob
                  evaluator=None,
                  dim=30,
                  upper_bound=100,
                  lower_bound=-100,
                  max_eval=300000,
                  # params for method
                  fw_size=5,
                  sp_size=300,
                  init_amp=200,
                  gm_ratio=0.2, 
                  parameter_N=10,
                  parameter_b=1.5):

        # load params
        self.evaluator = evaluator
        self.dim = dim
        self.upper_bound = upper_bound
        self.lower_bound = lower_bound

        self.max_eval = max_eval

        self.fw_size = fw_size
        self.sp_size = sp_size
        self.gm_ratio = gm_ratio
        self.init_amp = init_amp
        self.parameter_N = parameter_N
        self.parameter_b = parameter_b

        # init random seed
        np.random.seed(int(os.getpid() * time.perf_counter()/10000))

    def run(self):
        # running time
        begin_time = time.perf_counter()

        # init fireworks
        fireworks = np.random.uniform(self.lower_bound,
                                      self.upper_bound,
                                      [self.fw_size, self.dim])
        fits = self.evaluator(fireworks)
        amps = np.ones(self.fw_size) * self.init_amp
        best_idx = np.argmin(fits)
        best_idv = fireworks[best_idx, :]
        best_fit = fits[best_idx]

        # Start main loop

        # max iteration number should be computed according to specific algorithm
        max_iter = int(self.max_eval / self.sp_size)
        num_iter = 0
        num_eval = self.fw_size

        while True:
            if num_eval >= self.max_eval:
                break

            # compute explode sparks
            num_sparks = [int(self.sp_size / self.fw_size)] * self.fw_size 
            sum_sparks = np.sum(num_sparks)

            # explode
            e_sparks = []
            e_fits = []
            for idx in range(self.fw_size):
                bias = np.random.uniform(-1, 1, [num_sparks[idx], self.dim])       
                sparks = fireworks[idx, :] + bias * amps[idx]
                sparks = self._map(sparks, fireworks[idx, :], amps[idx])
                spark_fits = self.evaluator(sparks)
                e_sparks.append(sparks)
                e_fits.append(spark_fits)

            # mutate
            m_sparks = []
            m_fits = []
            for idx in range(self.fw_size):
                sparks = []
                top_num = int(num_sparks[idx] * self.gm_ratio)
                sort_idx = np.argsort(e_fits[idx])
                top_idx = sort_idx[-top_num:]
                btm_idx = sort_idx[:top_num]

                top_mean = np.mean(e_sparks[idx][top_idx, :], axis=0)
                btm_mean = np.mean(e_sparks[idx][btm_idx, :], axis=0)
                delta = top_mean - btm_mean

                # multi-guiding spark
                weight = np.random.uniform(0, self.parameter_b, (self.parameter_N,1))
                sparks = btm_mean - delta * weight 
                sparks = self._map(sparks, fireworks[idx, :], amps[idx])
                m_fit = self.evaluator(sparks)
                m_sparks.append(sparks)
                m_fits.append(m_fit)

            # select
            n_fireworks = np.empty((self.fw_size, self.dim))
            n_fits = np.empty((self.fw_size))
            for idx in range(self.fw_size):
                sparks = np.concatenate([fireworks[idx, :][np.newaxis, :],
                                         e_sparks[idx],
                                         m_sparks[idx]], axis=0)
                spark_fits = np.concatenate([[fits[idx]],
                                             e_fits[idx],
                                             m_fits[idx]], axis=0)
                min_idx = np.argmin(spark_fits)

                n_fireworks[idx, :] = sparks[min_idx, :]
                n_fits[idx] = spark_fits[min_idx]


            # restart
            improves = fits - n_fits

            min_fit = min(n_fits)
            restart = (improves > EPS) * (improves * (max_iter - num_iter) < (n_fits - min_fit))
            replace = restart[:, np.newaxis].astype(np.int32)
            restart_num = sum(replace)

            if restart_num > 0:
                rand_sample = np.random.uniform(self.lower_bound,
                                                self.upper_bound,
                                                (self.fw_size, self.dim))
                n_fireworks = (1 - replace) * n_fireworks + replace * rand_sample
                n_fits[restart] = self.evaluator(n_fireworks[restart, :])
                amps[restart] = self.init_amp

            # update

            # dynamic amps
            for idx in range(self.fw_size):
                if n_fits[idx] < fits[idx] - EPS:
                    amps[idx] *= 1.2
                else:
                    amps[idx] *= 0.9

            # iter and eval num
            num_iter += 1
            num_eval += sum_sparks + restart_num + self.fw_size * self.parameter_N

            # record best results
            min_idx = np.argmin(n_fits)
            best_idv = n_fireworks[min_idx, :]
            best_fit = n_fits[min_idx]

            # new fireworks
            fireworks = n_fireworks
            fits = n_fits

        run_time = time.perf_counter() - begin_time

        return best_fit, run_time, best_idv


    # def _map(self, samples, firework, amp):
    #     in_bound = (samples > self.lower_bound) * (samples < self.upper_bound)
    #     rand_samples = np.random.uniform(max(firework[0] - amp, self.lower_bound),
    #                                      min(firework[0] + amp, self.upper_bound),
    #                                      (1, len(samples)))
    #     for i in range(1, len(firework)):
    #         temp_samples = np.random.uniform(max(firework[i] - amp, self.lower_bound),
    #                                          min(firework[i] + amp, self.upper_bound),
    #                                          (1, len(samples)))
    #         rand_samples = np.concatenate((rand_samples, temp_samples))
    #     rand_samples = rand_samples.transpose()
    #     samples = in_bound * samples + (1 - in_bound) * rand_samples
    #     return samples
    
    def _map(self, samples, firework, amp):
        in_bound = (samples > self.lower_bound) * (samples < self.upper_bound)
        rand_samples = np.random.uniform(max(firework[0] - amp, self.lower_bound),
                                         min(firework[0] + amp, self.upper_bound),
                                         (1, len(samples)))
        for i in range(1, len(firework)):
            temp_samples = np.random.uniform(max(firework[i] - amp, self.lower_bound),
                                             min(firework[i] + amp, self.upper_bound),
                                             (1, len(samples)))
            rand_samples = np.concatenate((rand_samples, temp_samples))
        rand_samples = rand_samples.transpose()
        samples = in_bound * samples + (1 - in_bound) * rand_samples
        return samples

