import torch
import torch.nn.functional as F

def simple_conv2d(input_matrix: torch.Tensor, kernel: torch.Tensor, padding: int, stride: int) -> torch.Tensor:
    """
    Perform a 2D convolution on a single-channel input using PyTorch's built-in conv2d.
    input_matrix: 2D tensor (H, W)
    kernel: 2D tensor (kH, kW)
    padding: int, zero-padding on all sides
    stride: int, stride of the convolution
    """
    # Hint: conv2d expects input of shape (N, C, H, W) and weight of shape (out_channels, in_channels, kH, kW)
    
    X = input_matrix.detach().clone().float().unsqueeze(0).unsqueeze(0)
    k = kernel.detach().clone().float().unsqueeze(0).unsqueeze(0)

    output = F.conv2d(input=X, weight=k, bias=None, stride=stride, padding=padding)

    result = output.squeeze()
    return [[round(v, 4) for v in row] for row in result.tolist()]