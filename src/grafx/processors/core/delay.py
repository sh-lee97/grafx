import torch
import torch.nn as nn 


class NormalizedGradient(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        return input

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        return grad_input / (1e-7 + grad_input.abs())


class SurrogateDelay(nn.Module):
    """
    A module that represents a surrogate delay line.

    Args:
        N (int): The size of the delay line. Default is 2048.
        straight_through (bool): Whether to use straight-through gradient estimator during training. Default is True.
    """

    def __init__(self, N=2048, straight_through=True):
        super().__init__()

        self.straight_through = straight_through

        sin_N = N // 2 + 1
        arange_sin = torch.arange(sin_N)
        self.register_buffer("arange_sin", arange_sin[None, :])

    def forward(self, z):
        """
        Forward pass of the surrogate delay line.

        Args:
            z (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor representing the delay line.
            torch.Tensor: The radii loss.
        """
        assert z.dtype == torch.cfloat

        shape = z.shape
        z = z.view(-1)

        radii_loss = self.calculate_radii_loss(z)

        z = NormalizedGradient.apply(z)
        mag = torch.abs(z)
        z = z * torch.tanh(mag) / (mag + 1e-7)

        sins = z[:, None] ** self.arange_sin
        irs = torch.fft.irfft(sins)

        if self.straight_through:
            irs = self.apply_straight_through(irs)

        irs = irs.view(*shape, -1)
        return irs, radii_loss

    def calculate_radii_loss(self, z):
        mag = torch.abs(z)
        mag = torch.tanh(mag)
        radii_losses = (1 - mag).square()
        return radii_losses.sum()

    def apply_straight_through(self, irs):
        hard_irs = self.get_hard_irs(irs)
        irs = irs + (hard_irs - irs).detach()
        return irs

    @torch.no_grad()
    def get_hard_irs(self, irs):
        """
        Get the hard impulse responses.

        Args:
            irs (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The hard impulse responses.
        """
        hard_irs = torch.zeros_like(irs)
        onset = torch.argmax(irs, -1)
        arange_ir = torch.arange(len(hard_irs), device=irs.device)
        hard_irs[arange_ir, onset] = 1
        return hard_irs
