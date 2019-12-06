import torch
import torch.nn as nn

import numpy as np 

from NeuralODE import NeuralODE

class NNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.fc1 = nn.Linear(1, 50)
        self.fc2 = nn.Linear(50, 1)

    def forward(self, tk, yk):
        out = yk ** 3
        out = self.fc1(out)
        out = self.fc2(out)
        return out

def test_init():
    print ("===== init network func ====")
    nnet = NNet()
    for param in nnet.parameters():
        param.requires_grad_(True)
    
    neuralODE = NeuralODE(model=nnet)
    return neuralODE

def test_forward():
    neuralODE = test_init()
    print ("===== forward ====")
    input = torch.tensor(1.0).view(-1, 1).float()
    input.requires_grad_(True)
    output = neuralODE.forward(input)
    print (f"input: {input.data}; output: {output.data}")
    return output

def test_backward():
    output = test_forward()
    print ("===== backward ====")
    target = torch.tensor(1.2, dtype=torch.float32).view(-1, 1)
    loss = nn.MSELoss()(output, target)
    loss.backward()
    print (f"output: {output.data}; target: {target.data}; loss: {loss.data}")
    return loss

def test_training(device='cpu'):
    nnet = NNet().to(device)
    for param in nnet.parameters():
        param.requires_grad_(True)

    neuralODE = NeuralODE(model=nnet)

    input = torch.tensor(1.0).view(-1, 1).float().to(device)
    input.requires_grad_(True)
    target = torch.tensor(1.2, dtype=torch.float32).view(-1, 1).to(device)

    # updated_params = [input]
    # updated_params = [w for w in nnet.parameters()]
    updated_params = [input] + [w for w in nnet.parameters()]

    # TODO:RMSPropOptimizer
    optimizer = torch.optim.SGD(
        updated_params, lr=0.001
    )

    for i in range(100):
        output = neuralODE.forward(input)
        
        loss = nn.MSELoss()(output, target)
        loss.backward()
        
        optimizer.step()
        
        for params in updated_params:
            if params.grad is not None:
                params.grad.data.zero_()

        print (f"iter: {i}; input: {input.data}; output: {output.data}, loss: {loss.data}")

if __name__ == "__main__":
    test_training("cuda")
