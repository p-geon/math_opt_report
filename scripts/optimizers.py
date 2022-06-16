from typing import List, Callable, Tuple
import numpy as np


class SteepestDescent(object):
    def __init__(self,
            step_size: float, 
            df: Callable):
        self.eta = step_size # learning rate
        self.df = df # f()', not DataFrame


    def update(self, w, b) -> Tuple[np.ndarray, np.ndarray]:
        w = w - self.eta * self.df.dw(w, b)
        b = b - self.eta * self.df.db(b)
        return w, b


class Momentum(object):
    def __init__(self, 
            step_size: float, 
            df: Callable,
            momentum: float=0.9
            ):
        assert momentum < 1.0, "momentum value must be less than 1.0"
        assert momentum > 0.0, "momentum value must be greater than 1.0"

        self.eta = step_size # learning rate
        self.df = df # f()', not DataFrame
        self.momentum = momentum
        self.gw_prev = None
        self.gb_prev = None


    def update(self, w, b) -> Tuple[np.ndarray, np.ndarray]:
        gw = self.df.dw(w, b)
        gb = self.df.db(b)

        # insert zeros for the first time
        if(self.gw_prev is not None): 
            pass
        else:
            self.gw_prev = np.zeros_like(gw)
            self.gb_prev = np.zeros_like(gb)

        # gradient with momentum
        gw = gw + self.momentum * self.gw_prev
        gb = gb + self.momentum * self.gb_prev

        # GD
        w = w - self.eta * gw
        b = b - self.eta * gb

        # record momeutum
        self.gw_prev = gw
        self.gb_prev = gb
        return w, b
