import numpy as np


def softmax(pred_logits: np.ndarray) -> np.ndarray:
    """Compute softmax activations from logits.

    Args:
        pred_logits (np.ndarray): Logits to be converted to softmax.

    Returns:
        np.ndarray: Softmax activations.
    """
    assert isinstance(pred_logits, np.ndarray), "pred_logits must be a NumPy array"
    return np.exp(pred_logits) / np.exp(pred_logits).sum(axis=-1, keepdims=True)


def ceiling_division(n: int, d: int) -> int:
    """Perform division with ceiling rounding.

    Args:
        n (int): Numerator.
        d (int): Denominator.

    Returns:
        int: Result of division with ceiling rounding.
    """
    assert isinstance(n, int), "n must be an integer"
    assert isinstance(d, int), "d must be an integer"
    assert d != 0, "d cannot be zero"
    return -(n // -d)
