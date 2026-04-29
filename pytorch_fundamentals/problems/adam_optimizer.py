import torch

def adam_optimizer(f, grad, x0, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8, num_iterations=10) -> torch.Tensor:
    """
    Implements Adam optimization algorithm using PyTorch's built-in optimizer.

    Args:
        f: The objective function to be optimized
        grad: A function that computes the gradient (unused; autograd is used instead)
        x0: Initial parameter values (torch.Tensor)
        learning_rate: The step size (default: 0.001)
        beta1: Exponential decay rate for the first moment estimates (default: 0.9)
        beta2: Exponential decay rate for the second moment estimates (default: 0.999)
        epsilon: A small constant for numerical stability (default: 1e-8)
        num_iterations: Number of iterations to run the optimizer (default: 10)

    Returns:
        torch.Tensor: Optimized parameters
    """
    # Your code here
    x = x0.clone().detach().requires_grad_(True)

    optimizer = torch.optim.Adam([x], lr=learning_rate, betas=(beta1, beta2), eps= epsilon)

    f_compiled = torch.compile(f)

    for _ in range(num_iterations):
        optimizer.zero_grad()

        loss = f_compiled(x)

        loss.backward()

        optimizer.step()

    return x.detach()
