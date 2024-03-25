# -*- coding: utf-8 -*-

class Model:
    """
    the base class for inference model
    """
    LIKELIHOOD_DIFF_RELATIVE = 0.001

    def __init__(self):
        self.converge_rate = self.LIKELIHOOD_DIFF_RELATIVE
        pass

    def set_converge_rate(self, val):
        self.converge_rate = val

    def infer(self, dataset, soft=True):
        pass

    def sampling_infer(self, dataset, maxround, begin):
        pass
