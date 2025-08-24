from dataclasses import dataclass
from typing import Optional, Tuple

@dataclass
class Args:
    """Configuration for the drone simulation."""

    num_drones: int = 3
    """Number of drones (agents) in the swarm."""

    size: int = 100
    """Size of the square map (e.g., 100x100)."""

    emitter_pos: Optional[Tuple[int, int]] = None
    """(x, y) coordinates of the emitter. If None, a random position is chosen."""

    gpu: bool = False
    """If true, use CuPy for GPU acceleration. Otherwise, use NumPy for CPU."""

