# NeuralODE.pytorch

This is the simplest pytorch implement (100 lines) of **["Neural Ordinary Differential Equations"](https://arxiv.org/pdf/1806.07366.pdf) @ NeurIPS 2018 Best Paper**.


### Train on MNIST

* resnet baseline (0.60M #Params): accu = 99.42%
```
python train_mnist.py 
```

* ode network (0.22M #Params): accu = 99.31%
```
python train_mnist.py --ode
```

**Note:** This repo is not aim at reproducing the performace in the original paper, but to show the basic logics of how to do forward as well as backward in ode network in **100 lines** (`NeuralODE.py`). There are mutiple differences between my inplementation and the original implementation, including:

    1. ODESolver: Only implement `rk4` in this repo.
    2. Training strategy: inluding learning-rate schedule, optimizer and so on.
    3. ODE network: timestamp is hard-code in this repo.  


### Usage of NeuralODE
```
from NeuralODE import NeuralODE

# some kind of neural network.
model = NNet()

# [ADD SINLGE LINE] using NeuralODE to update this network.
model = NeuralODE(model)

# just train as usual, nothing need to change
optimizer = optim.Adadelta(model.parameters(), lr=args.lr)

scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
for epoch in range(1, args.epochs + 1):
    train(args, model, device, train_loader, optimizer, epoch)
    test(args, model, device, test_loader)
    scheduler.step()

```

**Note:** Currently, `NeuralODE` only support those models which don't change the resolution of input (input.shape == output.shape)


### Other Sources

[Author's Pytorch Implementation](https://github.com/rtqichen/torchdiffeq)
[TensorFlow Tutorial and Implementation](https://github.com/kmkolasinski/deep-learning-notes/tree/master/seminars/2019-03-Neural-Ordinary-Differential-Equations)