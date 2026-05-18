"""Utilility functions for the ensemble and ML capabilities.

Note: ``ml.ensemble`` makes use of ``np.random.default_rng``, which ignores the global
seed of ``numpy``. Make sure to set them locally for full determinism.

"""

from typing import Optional

import tensorflow as tf


def enable_determinism(seed: Optional[int] = None):
    """Set a seed for python, numpy, and tensorflow and enable deterministic behavior.

    Args:
        seed: (Optional[int]): Seed for the ``np.random.Generator``. Default is
            ``None``.

    """
    # ``tf.keras.utils.set_random_seed`` sets the python, numpy, and tensorflow seed
    # simultaneously.
    tf.keras.utils.set_random_seed(seed=seed)
    tf.config.experimental.enable_op_determinism()


def recursive_dict_update(d: dict, u: dict) -> dict:
    """Recursively update a dictionary with another dictionary.

    Args:
        d: The dictionary to be updated.
        u: The dictionary to update with.

    Returns:
        The updated dictionary.

    """
    for k, v in u.items():
        if isinstance(v, dict):
            d[k] = recursive_dict_update(d.get(k, {}), v)
        else:
            d[k] = v
    return d
