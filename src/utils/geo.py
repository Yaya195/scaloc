import numpy as np


def latlon_to_xy(lat: np.ndarray, lon: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Minimal equirectangular projection placeholder."""
    lat0 = np.mean(lat)
    x = (lon - np.mean(lon)) * np.cos(np.deg2rad(lat0))
    y = lat - lat0
    return x, y
