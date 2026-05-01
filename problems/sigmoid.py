import torch

def sigmoid(z: float) -> float:
    """
    Compute the sigmoid activation function.
    Input:
      - z: float or torch scalar tensor
    Returns:
      - sigmoid(z) as Python float rounded to 4 decimals.
    """
    # Your implementation here
    if z == 0:
      return round(0.5, 4)
    
    return round(torch.sigmoid(torch.tensor(z, dtype=torch.float32)).item(), 4)

