
import torch
import torch.nn as nn

import numpy as np 

def zip_map(zipped, update_op):
    return [update_op(*elems) for elems in zipped]

def euler_update(h_list, dh_list, dt):
    return zip_map(zip(h_list, dh_list), lambda h, dh: h + dt * dh)

def euler_step(func, dt, state):
    return euler_update(state, func(state), dt)

def rk2_step(func, dt, state, **kwargs):
    k1 = func(state, **kwargs)
    k2 = func(euler_update(state, k1, dt), **kwargs)
    return zip_map(zip(state, k1, k2),
                   lambda h, dk1, dk2: h + dt * (dk1 + dk2) / 2)

def rk4_step(func, dt, state, **kwargs):
    k1 = func(state, **kwargs)
    k2 = func(euler_update(state, k1, dt / 2), **kwargs)
    k3 = func(euler_update(state, k2, dt / 2), **kwargs)
    k4 = func(euler_update(state, k3, dt), **kwargs)

    return zip_map(
        zip(state, k1, k2, k3, k4),
        lambda h, dk1, dk2, dk3, dk4: h + dt * (
                dk1 + 2 * dk2 + 2 * dk3 + dk4) / 6,
    )

class NeuralODE(nn.Module):
    def __init__(self, model, solver=rk4_step, t=np.linspace(0, 1, 40)):
        super().__init__()
        self.t = t
        self.model = model
        self.solver = solver
        self.delta_t = t[1:] - t[:-1]

    def forward_dynamics(self, state):
        t, y = state
        return [1.0, self.model(t, y)]

    def backward_dynamics(self, state):
        t, ht, at = state[0], state[1], -state[2]
        ht.requires_grad_(True)
        ht_new = self.model(t, ht)
        gradients = torch.autograd.grad(
            ht_new, [ht] + [w for w in self.model.parameters()], at,
            allow_unused=True, retain_graph=True
        )
        return [1.0, ht_new, *gradients]

    def forward(self, input):
        state = [0, input]
        for dt in self.delta_t:
            state = self.solver(func=self.forward_dynamics, dt=float(dt), state=state)
        return state[1]

    def backward(self, output, output_gradients=None):
        grad_weights = []
        for p in self.model.parameters():
            grad_weights.append(torch.zeros_like(p))

        if output_gradients is None:
            output_gradients = torch.zeros_like(output)

        state = [1, output, output_gradients, *grad_weights]

        for dt in self.delta_t:
            state = self.solver(func=self.backward_dynamics, dt=float(dt), state=state)
        
        inputs = state[1]
        dLdInputs = state[2]
        dLdWeights = state[3:]
        return inputs, dLdInputs, dLdWeights

def demo():
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

    ## init 
    print ("===== init network func ====")
    nnet = NNet()
    for param in nnet.parameters():
        param.requires_grad_(True)

    neuralODE = NeuralODE(model=nnet)

    ## forward
    print ("===== test forward ====")
    input = torch.tensor(1.0).view(-1, 1).float()
    input.requires_grad_(True)
    output = neuralODE.forward(input)
    print (output)

    ## backward
    print ("===== test backward ====")
    target = 1.2
    loss = "MSE"

    if loss == "MSE":
        output_gradients = 2 * (output - target).abs()
    inputs, dLdInputs, dLdWeights = neuralODE.backward(output, output_gradients)
    print (inputs, dLdInputs, dLdWeights)

    ## update input
    print ("===== update input ====")
    input = torch.tensor(1.0).view(-1, 1).float()
    input.requires_grad_(True)
    for i in range(50):
        output = neuralODE.forward(input)
        output_gradients = 2 * (output - 1.2).abs()
        _, dLdInputs, dLdWeights = neuralODE.backward(output, output_gradients)
        input = input + 0.05 * dLdInputs
        print (i, output.data)

if __name__ == "__main__":
    demo()
