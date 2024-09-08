import numpy as np

# Smooths out slopes of `terrain` that are too steep. Rough approximation of the
# phenomenon described here: https://en.wikipedia.org/wiki/Angle_of_repose
def apply_slippage(terrain, repose_slope, cell_width):
    delta = simple_gradient(terrain) / cell_width
    smoothed = gaussian_blur(terrain, sigma=1.5)
    should_smooth = np.abs(delta) > repose_slope
    result = np.select([np.abs(delta) > repose_slope], [smoothed], terrain)
    return result


# Simple gradient by taking the diff of each cell's horizontal and vertical
# neighbors.
def simple_gradient(a):
    dx = 0.5 * (np.roll(a, 1, axis=0) - np.roll(a, -1, axis=0))
    dy = 0.5 * (np.roll(a, 1, axis=1) - np.roll(a, -1, axis=1))
    return 1j * dx + dy


# Peforms a gaussian blur of `a`.
def gaussian_blur(a, sigma=1.0):
    freqs = tuple(np.fft.fftfreq(n, d=1.0 / n) for n in a.shape)
    freq_radial = np.hypot(*np.meshgrid(*freqs))
    sigma2 = sigma**2
    g = lambda x: ((2 * np.pi * sigma2) ** -0.5) * np.exp(-0.5 * (x / sigma) ** 2)
    kernel = g(freq_radial)
    kernel /= kernel.sum()
    return np.fft.ifft2(np.fft.fft2(a) * np.fft.fft2(kernel)).real


def lerp(x, y, a):
    return (1.0 - a) * x + a * y


def fbm(shape, p, lower=-np.inf, upper=np.inf):
    freqs = tuple(np.fft.fftfreq(n, d=1.0 / n) for n in shape)
    freq_radial = np.hypot(*np.meshgrid(*freqs))
    envelope = (
        np.power(freq_radial, p, where=freq_radial != 0)
        * (freq_radial > lower)
        * (freq_radial < upper)
    )
    envelope[0][0] = 0.0
    phase_noise = np.exp(2j * np.pi * np.random.rand(*shape))
    return normalize(np.real(np.fft.ifft2(np.fft.fft2(phase_noise) * envelope)))


# Renormalizes the values of `x` to `bounds`
def normalize(x, bounds=(0, 1)):
    return np.interp(x, (x.min(), x.max()), bounds)


def sample(a, offset):
    shape = np.array(a.shape)
    delta = np.array((offset.real, offset.imag))
    coords = np.array(np.meshgrid(*map(range, shape))) - delta

    lower_coords = np.floor(coords).astype(int)
    upper_coords = lower_coords + 1
    coord_offsets = coords - lower_coords
    lower_coords %= shape[:, np.newaxis, np.newaxis]
    upper_coords %= shape[:, np.newaxis, np.newaxis]

    result = lerp(
        lerp(
            a[lower_coords[1], lower_coords[0]],
            a[lower_coords[1], upper_coords[0]],
            coord_offsets[0],
        ),
        lerp(
            a[upper_coords[1], lower_coords[0]],
            a[upper_coords[1], upper_coords[0]],
            coord_offsets[0],
        ),
        coord_offsets[1],
    )
    return result


# Takes each value of `a` and offsets them by `delta`. Treats each grid point
# like a unit square.
def displace(a, gradient):
    fns = {
        -1: lambda x: -x,
        0: lambda x: 1 - np.abs(x),
        1: lambda x: x,
    }
    result = np.zeros_like(a)
    for dx in range(-1, 2):
        wx = np.maximum(fns[dx](gradient.real), 0.0)
        for dy in range(-1, 2):
            wy = np.maximum(fns[dy](gradient.imag), 0.0)
            result += np.roll(np.roll(wx * wy * a, dy, axis=0), dx, axis=1)

    return result