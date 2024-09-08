# Torch Terrainer

Functions to generate terrains and simulate erosion in PyTorch and use GPU acceleration if available.

Adapted to Pytorch from https://github.com/dandrino/terrain-erosion-3-ways/blob/master/simulation.py and included additional setting dataclass to ease invoking erosion iterations.

See https://github.com/dandrino/terrain-erosion-3-ways/tree/master for more info.

Sample Performance comparison generating erosion on a 1009x1009 terrain in 1412 iterations:
* Original Numpy code on CPU (AMD Ryzen 9 7900): 440.026s (3.30it/s)
* Pytorch code on CPU (AMD Ryzen 9 7900): 64.342s (22.00it/s)
* Pytorch code on GPU (AMD Ryzen 9 7900 + GeForce RTX 4080): 3.279s (440.21it/s)

See torch_terrainer_example.ipynb for usage.
Numpy code for performance comparison in /reference folder.


MIT License

Original Copyright (c) 2018 Daniel Andrino

Copyright (c) 2024 Cristiano Costantini - Modifications and enhancements
