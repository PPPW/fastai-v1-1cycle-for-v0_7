from fastai_v0_7.sgdr import LR_Updater
import numpy as np


class OneCycleLR(LR_Updater):
    def __init__(self, layer_opt, nb, div=25, pct=0.3, momentums=(0.95,0.85), on_cycle_end=None):
        self.nb,self.div,self.pct,self.on_cycle_end = nb,div,pct,on_cycle_end
        self.cycle_nb = int(nb * pct)
        if momentums is not None:
            self.moms = momentums            
        super().__init__(layer_opt, record_mom=(momentums is not None))

    def on_train_begin(self):
        self.cycle_iter,self.cycle_count=0,0
        super().on_train_begin()

    def calc_lr(self, lr_max):
        phase, pct = self._get_phase() 
        if phase == 1:
            res = self._linear(lr_max/self.div, lr_max, pct) 
        else:
            res = self._cosine(lr_max, lr_max/self.div/1e4, pct)
            
        self.cycle_iter += 1
        if self.cycle_iter == self.nb:
            self.cycle_iter = 0
            if self.on_cycle_end: self.on_cycle_end(self, self.cycle_count)
            self.cycle_count += 1
        return res

    def calc_mom(self):  
        phase, pct = self._get_phase() 
        if phase == 1:
            res = self._linear(self.moms[0], self.moms[1], pct) 
        else:
            res = self._cosine(self.moms[1], self.moms[0], pct)     
        return res
    
    def _get_phase(self):
        if self.cycle_iter < self.cycle_nb:
            return 1, self.cycle_iter/self.cycle_nb            
        else:
            return 2, (self.cycle_iter-self.cycle_nb)/(self.nb-self.cycle_nb)                  
    
    def _linear(self, start, end, pct):
        return start + pct * (end-start)    
    
    def _cosine(self, start, end, pct):
        cos_out = np.cos(np.pi * pct) + 1
        return end + (start-end)/2 * cos_out
