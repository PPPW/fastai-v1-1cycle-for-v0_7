# Using the fastai v1.0 two-phase 1cycle policy in fastai v0.7

## Motivation

The "1cycle policy" technique for training neural networks is proposed in [this paper](https://arxiv.org/pdf/1803.09820.pdf). Sylvain Gugger has implemented this original version for fastai v0.7, and wrote a great [post](https://sgugger.github.io/the-1cycle-policy.html) to explain the idea. 

The original 1cycle policy has three phases: learning rate increasing, decreasing, and further decreasing. For all phases, a linear function is used to change the learning rate. In fastai v1.0, the 1cycle policy has only two phases: a linear increasing phase and a cosine decreasing phase. The [doc](https://docs.fast.ai/callbacks.one_cycle.html) claims better results using this method. 

In case you'd like to use this type of 1cycle policy in fastai v0.7 for whatever reason (e.g., you got a Windows and can't get PyTorch v1.0 to work at the moment), you can do it by using the `OneCycleLR` class defined in `one_cycle.py` here. 


## Usage

You can check [this notebook](example.ipynb) for a simple example to do the new 1cycle policy with more details. Basically, if you want to run a `learner` for N epochs, use:

```
cycle_len = N
layer_opt = learn.get_layer_opt(lr_max, None)
ocs = OneCycleLR(layer_opt, len(learn.data.trn_dl) * cycle_len)
learn.sched = ocs
learn.fit_gen(learn.model, learn.data, layer_opt, cycle_len)
```

In [this notebook](example.ipynb) I applied this policy to the fastai's tiny MNIST dataset, you can find more explanations there. 

## Comparison

In [this notebook](compare.ipynb), I compared my implementation for fastai v0.7 with the implementation in fastai v1.0, they agree well. For easier testing, I defined some mockup objects in `mocks.py`, which allows me to test without the need for a dataset or a learner. The mockups can also be used to test other learning rate schedulers. 
