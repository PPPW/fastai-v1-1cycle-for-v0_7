from tqdm import trange


class LayerOptimizerMock():
    def __init__(self, lr):
        self.lrs = [lr]        

    @property
    def lr(self): return self.lrs[-1]

    def set_lrs(self, lrs): self.lrs = lrs

    @property
    def mom(self): return self.momentum

    def set_mom(self, momentum): self.momentum = momentum


def simulate_fit_v0_7(data_size, n_epochs, callbacks):    
    for cb in callbacks: 
        cb.on_train_begin()  
    
    for _ in trange(n_epochs, desc='Epoch'):
        for _ in range(data_size):                      
            for cb in callbacks: 
                cb.on_batch_end(0)


class LearnerMock():
    def __init__(self, data_size):
        self.data = DataBunchMock(data_size)
        self.opt = OptimizerMock()
    

class DataBunchMock():
    def __init__(self, data_size):
        self.train_dl = [0]*data_size


class OptimizerMock():
    def __init__(self):
        self.lr = 0.1
        self.mom = 0.9


class PBarMock():
    def write(self, msg, table):
        pass


def simulate_fit_v1(learner, n_epochs, callbacks, is_train=True):    
    ocs, rc = callbacks  
    
    ocs.on_train_begin(n_epochs=n_epochs)
    rc.on_train_begin(pbar=PBarMock(), metrics_names=[])  
    
    for _ in trange(n_epochs, desc='Epoch'):
        for _ in range(len(learner.data.train_dl)):                      
            rc.on_batch_begin(is_train)
            ocs.on_batch_end(is_train)
