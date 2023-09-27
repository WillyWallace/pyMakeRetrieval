import yaml
import numpy as np
from numpy import ma
from yaml.loader import SafeLoader


def read_yaml_config(_file) -> dict:
    """Reads config yaml files."""
    with open(_file) as f:
        config = yaml.load(f, Loader=SafeLoader)

    return config


def isscalar(array: any) -> bool:
    """Tests if input is scalar.
    By "scalar" we mean that array has a single value.
    Examples:
        >>> isscalar(1)
            True
        >>> isscalar([1])
            True
        >>> isscalar(np.array(1))
            True
        >>> isscalar(np.array([1]))
            True
    """

    arr = ma.array(array)
    if not hasattr(arr, "__len__") or arr.shape == () or len(arr) == 1:
        return True
    return False
