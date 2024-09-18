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
    r"""
    A surrogate FIR processor for a learnable delay line.

        A single delay can be represented as a FIR filter
        $
        h[n] = \delta[n-d]
        $
        where $0\leq d < N$ is a delay length we want to optimize and $\delta[n]$ denotes a unit impulse.
        We exploit the fact that each delay corresponds to a complex sinusoid in the frequency domain.
        Such a sinusoid's angular frequency $z \in \mathbb{C}$ can be optimized with the gradient descent
        if we allow it to be inside the unit disk, i.e., $|z| \leq 1$ :cite:`hayes2023sinusoidal`.
        We first start with an unconstrained complex parameter $\tilde{z} \in \mathbb{C}$
        and restrict it to be inside the unit disk (in the same way of restricting the poles :cite:`nercessian2021lightweight`)
        with the following activation function.
        $$
        z = \tilde{z}_k \cdot \frac{\tanh( | \tilde{z}_k | )}{ | \tilde{z}_k | + \epsilon}.
        $$

        Then, we compute a damped sinusoid with the normalized frequency $z$ then use its inverse FFT as a surrogate of the delay.
        $$
        \tilde{h}[n] = \frac{1}{N} \sum_{k=0}^{N-1} z^k z_N^{kn}.
        $$

        where $z_{N} = \exp(j\cdot 2\pi/N)$.
        Clearly, it is not a sparse delay line unless $z$ is an integer power of $z_N$ (on the unit circle with an exact angle).
        Instead it becomes a time-aliased and low-passed sinc kernel.
        We can use this soft delay as is, or we can use straight-through estimation (STE) :cite:`bengio2013estimating`
        so that the forward pass uses the hard delays $h[n]$ and the backward pass uses the soft delays $\smash{\tilde{h}[n]}$.
        $$
        \frac{\partial L}{\partial z^*} \leftarrow \sum_{n=0}^{N-1} \frac{\partial L}{\partial h[n]} \frac{\partial \tilde{h}[n]}{\partial z^*}
        $$

        For a stable and faster convergence, we provide two additional options.
        The first one is to normalize the gradients of the complex conjugate to have a unit norm.
        $$
        \frac{\partial L}{\partial z^*} \leftarrow \frac{\partial L}{\partial z^*}/ \left|\frac{\partial L}{\partial z^*} \right|.
        $$

        The second one is to use the radii loss
        $L_{\mathrm{radii}} = (1 - | z | )^2$
        to encourage complex angluar frequency $z$ to be near the unit circle, making the delays to be "sharp."
        We empirically found this regularization to be helpful especially when we use the STE as it alleviates the discrepancy between the hard and soft delays
        while still having the benefits of the soft FIR.




    Args:
        N (:python:`int`):
            The length surrogate FIR, which is also the largest delay length minus one.
        straight_through (:python:`bool`, *optional*):
            Use hard delays for the forward passes and surrogate soft delays for the backward passes
            with straight-through estimation
            (default: :python:`True`).
        normalize_gradients (:python:`bool`, *optional*):
            Normalize the complex conjugate gradients to unit norm
            (default: :python:`True`).
        radii_loss (:python:`bool`, *optional*):
            Use the radii loss to encourage the delays to be close to the unit circle
            (default: :python:`True`).
    """

    def __init__(
        self, N, straight_through=True, radii_loss=True, normalize_gradients=True
    ):
        super().__init__()

        self.straight_through = straight_through
        self.radii_loss = radii_loss
        self.normalize_gradients = normalize_gradients

        sin_N = N // 2 + 1
        arange_sin = torch.arange(sin_N)
        self.register_buffer("arange_sin", arange_sin[None, :])

    def forward(self, z):
        r"""
        Computes the surrogate delay FIRs from the complex angular frequencies.

        Args:
            z (:python:`ComplexTensor`, *any shape*): The unnormalized complex angular frequencies.

        Returns:
            :python:`FloatTensor` *or* :python:`Tuple[FloatTensor, FloatTensor]`: A batch of FIRs either hard (when using the straight-through estimation) of soft surrogate delays.
            The returned tensor has an additional dimension (last) for the FIR taps.
        """
        assert z.dtype == torch.cfloat

        shape = z.shape
        z = z.view(-1)

        loss = self.calculate_radii_loss(z)

        if self.normalize_gradients:
            z = NormalizedGradient.apply(z)

        mag = torch.abs(z)
        z = z * torch.tanh(mag) / (mag + 1e-7)

        sins = (z[:, None] + 1e-7) ** self.arange_sin
        irs = torch.fft.irfft(sins)

        if self.straight_through:
            irs = self.apply_straight_through(irs)

        irs = irs.view(*shape, -1)
        return irs, loss

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
        hard_irs = torch.zeros_like(irs)
        onset = torch.argmax(irs, -1)
        arange_ir = torch.arange(len(hard_irs), device=irs.device)
        hard_irs[arange_ir, onset] = 1
        return hard_irs
