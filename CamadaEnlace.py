import numpy as np
import numpy.typing as npt

def parity_insert(digitals: list[bool], *,
                odd: bool = False) -> list[bool]:
    """
    Inserts a parity bit at the end of a digital signal.
    Returns the new digital signal with parity added at its trail.

    :param digitals: The digital signal
    :type digitals: list[bool]
    :param odd: Odd parity
    :type odd: bool
    """

    # Count how many 1s (Trues) are in the signal
    ones: int = sum(digitals)

    # Append parity bit, respecting odd-ness choice
    digitals.append(bool(ones % 2) ^ odd)

    # Return new list
    return digitals

def parity_check(digitals: list[bool], *,
                odd: bool = False) -> bool:
    """
    Checks the parity of a given digital signal.
    Returns True if no error is detected, and False if it is.

    :param digitals: The digital signal
    :type digitals: list[bool]
    :param odd: Odd parity
    :type odd: bool
    """

    # Count how many 1s (Trues) are in the signal
    ones: int = sum(digitals)

    # Verify parity
    return not (bool(ones % 2) ^ odd)
