import torch
import torch.fft
import torch.nn.functional as F
from dataclasses import dataclass

@dataclass
class GridSettings:
    size: int = 256
    full_width: int = 200


@dataclass
class WaterSettings:
    rain_unit_rate: float = 0.0008
    evaporation_rate: float = 0.0005


@dataclass
class SlopeSettings:
    min_height_delta: float = 0.0
    repose_slope: float = 0.0
    gravity: float = 9.8
    gradient_sigma: float = 0.0


@dataclass
class SedimentSettings:
    sediment_capacity_constant: float = 0.0
    dissolving_rate: float = 0.0
    deposition_rate: float = 0.0


@dataclass
class SimulationSettings:
    grid: GridSettings
    water: WaterSettings
    slope: SlopeSettings
    sediment: SedimentSettings

    @classmethod
    def from_dict(cls, settings_dict):
        return cls(
            grid=GridSettings(**settings_dict.get("grid", {})),
            water=WaterSettings(**settings_dict.get("water", {})),
            slope=SlopeSettings(**settings_dict.get("slope", {})),
            sediment=SedimentSettings(**settings_dict.get("sediment", {})),
        )

def normalize(tensor):
    min_val = tensor.min()
    max_val = tensor.max()
    return (tensor - min_val) / (max_val - min_val)


def lerp(x, y, a):
    return (1.0 - a) * x + a * y


def gaussian_kernel(kernel_size, sigma, dtype):
    """Create a 2D Gaussian kernel."""
    center = kernel_size // 2
    grid_range = torch.arange(kernel_size) - center
    grid_x, grid_y = torch.meshgrid(grid_range, grid_range, indexing="ij")
    gk = torch.exp(-0.5 * (torch.sqrt(grid_x**2 + grid_y**2) / sigma).pow(2))
    gk = gk / gk.sum()
    return gk.to(dtype)


def fbm(shape, p, lower=-float("inf"), upper=float("inf"), device="cpu", seed=None):
    # Set the seed if provided
    if seed is not None:
        torch.manual_seed(seed)

    # Compute the frequencies
    freqs = [
        torch.fft.fftfreq(n, d=1.0 / n, dtype=torch.float32, device=device)
        for n in shape
    ]
    freq_mesh = torch.meshgrid(*freqs, indexing="ij")
    freq_radial = torch.sqrt(sum(f**2 for f in freq_mesh))

    # Compute the envelope
    envelope = torch.pow(freq_radial, p) * (freq_radial > lower) * (freq_radial < upper)
    envelope[0, 0] = 0.0

    # Generate phase noise
    phase_noise = torch.exp(
        2j * torch.pi * torch.rand(*shape, dtype=torch.complex64, device=device)
    )

    # Apply the FFT and IFFT
    fft_phase_noise = torch.fft.fft2(phase_noise)
    result = torch.fft.ifft2(fft_phase_noise * envelope)

    # Normalize and return
    return normalize(result.real)


def gaussian_blur_2d(input_tensor, kernel_size=5, sigma=1.0):
    # Ensure kernel_size is odd
    if kernel_size % 2 == 0:
        raise ValueError("kernel_size should be an odd number")

    # Create the Gaussian kernel
    kernel = gaussian_kernel(kernel_size, sigma, input_tensor.dtype)
    kernel = kernel.unsqueeze(0).unsqueeze(0)  # Shape: (B, C, H, W)

    # Add batch and channel dimensions to input_tensor
    input_tensor = input_tensor.unsqueeze(0).unsqueeze(0)  # Shape: (B, C, H, W)
    input_tensor = F.pad(
        input_tensor,
        (kernel_size // 2, kernel_size // 2, kernel_size // 2, kernel_size // 2),
        mode="circular",
    )

    # Apply the Gaussian blur using convolution
    with torch.no_grad():
        blurred_tensor = F.conv2d(input_tensor, kernel, padding=0)

    return blurred_tensor.squeeze(0).squeeze(0)  # Remove batch and channel dimensions

def gaussian_blur_torch(a, sigma=1.0):
    # Compute frequencies in each dimension
    freqs = [torch.fft.fftfreq(n, d=1.0 / n, device=a.device) for n in a.shape]
    freq_mesh = torch.meshgrid(*freqs, indexing='ij')
    
    # Compute the radial frequency
    freq_radial = torch.sqrt(sum(f ** 2 for f in freq_mesh))
    
    # Gaussian function in frequency space
    sigma2 = sigma**2
    g = lambda x: ((2 * torch.pi * sigma2) ** -0.5) * torch.exp(-0.5 * (x / sigma) ** 2)
    
    # Create Gaussian kernel in frequency domain
    kernel = g(freq_radial)
    kernel /= kernel.sum()  # Normalize the kernel

    # Apply FFT to the input tensor and the kernel
    fft_a = torch.fft.fft2(a)
    fft_kernel = torch.fft.fft2(kernel, s=a.shape)  # Ensure kernel matches input shape

    # Perform element-wise multiplication in the frequency domain
    blurred_freq = fft_a * fft_kernel

    # Apply inverse FFT to get the blurred result
    blurred = torch.fft.ifft2(blurred_freq).real
    
    return blurred

def simple_gradient(a):
    dx = 0.5 * (torch.roll(a, shifts=1, dims=0) - torch.roll(a, shifts=-1, dims=0))
    dy = 0.5 * (torch.roll(a, shifts=1, dims=1) - torch.roll(a, shifts=-1, dims=1))
    return torch.stack([dy, dx])


# Smooths out slopes of `terrain` that are too steep. Rough approximation of the
# phenomenon described here: https://en.wikipedia.org/wiki/Angle_of_repose
def apply_slippage(terrain, repose_slope, cell_width):
    dx, dy = simple_gradient(terrain) / cell_width
    # smoothed = gaussian_blur_2d(terrain, kernel_size=63, sigma=5.0)
    smoothed = gaussian_blur_torch(terrain, sigma=1.5)    
    should_smooth = dx**2 + dy**2 > repose_slope**2
    result = torch.where(should_smooth, smoothed, terrain)
    return result


def sample(terrain, offset, device = None):
    if device is None:
        device = terrain.device

    # Create coordinate grid
    shape = torch.tensor(terrain.shape, device=device)
    coords = torch.meshgrid(
        [torch.arange(s, device=device) for s in shape], indexing="xy"
    )
    coords = torch.stack(coords) - offset

    # Floor and ceil coordinates
    lower_coords = torch.floor(coords).long()
    upper_coords = lower_coords + 1
    coord_offsets = coords - lower_coords.float()

    # Wrap coordinates using modulo
    lower_coords = lower_coords % shape.view(-1, 1, 1)
    upper_coords = upper_coords % shape.view(-1, 1, 1)

    # Perform bilinear interpolation
    lower_left = terrain[lower_coords[1], lower_coords[0]]
    lower_right = terrain[lower_coords[1], upper_coords[0]]
    upper_left = terrain[upper_coords[1], lower_coords[0]]
    upper_right = terrain[upper_coords[1], upper_coords[0]]

    interp_x1 = lerp(lower_left, lower_right, coord_offsets[0])
    interp_x2 = lerp(upper_left, upper_right, coord_offsets[0])
    result = lerp(interp_x1, interp_x2, coord_offsets[1])

    return result

def displace(a, gradient):
    result = torch.zeros_like(a)
    fns = {
        -1: lambda x: -x,
        0: lambda x: 1 - torch.abs(x),
        1: lambda x: x,
    }
    for dx in range(-1, 2):
        wx = torch.maximum(fns[dx](gradient[0]), torch.tensor(0.0))
        for dy in range(-1, 2):
            wy = torch.maximum(fns[dy](gradient[1]), torch.tensor(0.0))
            result += torch.roll(
                wx * wy * a,
                shifts=(dy, dx),  # Roll along both dimensions (dy along rows, dx along columns)
                dims=(0,1)
            )

    return result


def iterate_terrain_erosion(
    settings: SimulationSettings,
    terrain: torch.Tensor,
    sediment: torch.Tensor,
    water: torch.Tensor,
    velocity: torch.Tensor,
    device = None
) -> None:

    if device is None:
        device = terrain.device
    
    cell_width = settings.grid.full_width / settings.grid.size
    cell_area = cell_width**2
    rain_rate = settings.water.rain_unit_rate * cell_area

    # Add precipitation. This is done by via simple uniform random distribution,
    # although other models use a raindrop model
    output_water = water + torch.rand((settings.grid.size, settings.grid.size), device=device) * rain_rate

    # Compute the normalized gradient of the terrain height to determine where
    # water and sediment will be moving.

    angles = 2 * torch.pi * torch.rand((settings.grid.size, settings.grid.size), device=device)
    gradient = simple_gradient(terrain)

    gradient = torch.where(
        (gradient**2).sum(dim=0).sqrt() < 1e-10,
        torch.stack([torch.cos(angles), torch.sin(angles)]),
        gradient,
    )
    gradient /= torch.sqrt((gradient**2).sum(dim=0))

    # Compute the difference between teh current height the height offset by
    # `gradient`.
    neighbor_height = sample(terrain, -gradient)
    height_delta = terrain - neighbor_height

    # The sediment capacity represents how much sediment can be suspended in
    # water. If the sediment exceeds the quantity, then it is deposited,
    # otherwise terrain is eroded.
    sediment_capacity = (
        (
            torch.maximum(height_delta, torch.tensor(settings.slope.min_height_delta, device=device))
            / cell_width
        )
        * velocity
        * water
        * settings.sediment.sediment_capacity_constant
    )
    deposited_sediment = torch.where(
        height_delta < 0,
        torch.minimum(height_delta, sediment),
        torch.where(
            sediment > sediment_capacity,
            settings.sediment.deposition_rate * (sediment - sediment_capacity),
            settings.sediment.dissolving_rate * (sediment - sediment_capacity),
        ),
    )

    # Don't erode more sediment than the current terrain height.
    deposited_sediment = torch.maximum(-height_delta, deposited_sediment)

    # Update terrain and sediment quantities.
    output_sediment = sediment - deposited_sediment
    output_terrain = terrain + deposited_sediment
    
    output_sediment = displace(output_sediment, gradient)
    # sediment.copy_(displace(sediment, gradient))
    
    output_water = displace(output_water, gradient)
    # water.copy_(displace(water, gradient))

    # Smooth out steep slopes.
    output_terrain = apply_slippage(output_terrain, settings.slope.repose_slope, cell_width)
    # terrain.copy_(apply_slippage(terrain, settings.slope.repose_slope, cell_width))

    # Update velocity
    output_velocity = settings.slope.gravity * height_delta / cell_width
    # velocity.copy_(settings.slope.gravity * height_delta / cell_width)


    # Apply evaporation
    output_water *= 1 - settings.water.evaporation_rate
    return output_terrain, output_sediment, output_water, output_velocity
