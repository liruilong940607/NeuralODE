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

def forward_dynamics(state, nnet):
    t, y = state
    return [1.0, nnet(t, y)]

def backward_dynamics(state, nnet):
    with torch.set_grad_enabled(True):
        t, ht, at = state[0], state[1], state[2]
        ht = ht.detach()
        ht.requires_grad_(True)
        ht_new = nnet(t, ht)
        gradients = torch.autograd.grad(
            ht_new, [ht] + [w for w in nnet.parameters()], at,
            allow_unused=True, retain_graph=True
        )
    return [1.0, ht_new, *gradients]


class NeuralODEFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, nnet, solver, t, *params):
        delta_t = t[1:] - t[:-1]

        ctx.nnet = nnet
        ctx.solver = solver
        ctx.delta_t = delta_t

        state = [0, input]
        for dt in delta_t:
            state = solver(func=forward_dynamics, dt=float(dt), state=state, nnet=nnet)
        output = state[1]
        
        ctx.save_for_backward(input, output)
        return output

    @staticmethod
    def backward(ctx, output_gradients):
        input, output = ctx.saved_tensors
        nnet = ctx.nnet
        solver = ctx.solver
        delta_t = ctx.delta_t
        params = nnet.parameters()

        grad_weights = []
        for p in params:
            grad_weights.append(torch.zeros_like(p))

        state = [1, output, output_gradients, *grad_weights]

        for i, dt in enumerate(delta_t):
            state = solver(func=backward_dynamics, dt=float(dt), state=state, nnet=nnet)

        # input = state[1]
        grad_input = state[2]
        grad_weights = state[3:]
        return (grad_input, None, None, None, *grad_weights)


class NeuralODE(nn.Module):
    def __init__(self, model, solver=rk4_step, t=np.linspace(0, 1, 40)):
        super().__init__()
        self.t = t
        self.model = model
        self.solver = solver
        self.params = [w for w in model.parameters()]

    def forward(self, input):
        return NeuralODEFunction.apply(input, self.model, self.solver, self.t, *self.params)