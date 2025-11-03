import numpy as np
import numpy.typing as npt

def string_to_bitstream(message: str) -> list[bool] | None:
    """
    Transforms a string into a stream True and False values,
    respectively representing the 1 and 0 bit values. 
    Returns None if the string is not ASCII encoded.
    :param message: Message to turn into a bitstream
    :type message: str
    """

    # Checks ASCII-ness
    if not message.isascii():
        return None
    
    # Converts each character into its ASCII int, then into its binary
    temp_array: bytearray = bytearray(message, 'ascii')
    result: list[bool] = []
    for c in temp_array:
        result += char_to_list8bit(c)
    return result

def char_to_list8bit(char: str) -> list[bool]:
    """
    Auxiliary funciton.
    Transforms a single character into its ASCII binary.
    Does not check if len(char) == 1. 
    :param char: String consisting of one ASCII character
    :type char: str
    """

    # Transforms the character into its binary string
    str8bit: str = '{0:08b}'.format(int(char))

    # Transforms the str into list[bool]
    return [c == '1' for c in str8bit]

def bitstream_to_string(array: list[bool]) -> str | None:
    """
    Transforms a sequence of bools (True-1 and False-0 bits)
    into its corresponding ASCII representation.
    If len(array) is not a multiple of 8, it returns None. 
    :param array: Bitstream
    :type array: list[bool]
    """

    # Checks if array has correct size
    if (len(array) % 8) != 0:
        return None

    # Transforms each 8-bool chunk into a character and appends
    result: str = ""
    for i in range(0, len(array), 8):
        result += list8bit_to_char(array[i:i+8])
    return result

def list8bit_to_char(array: list[bool]) -> str:
    """
    Auxiliary function.
    Transforms a sequence of 8 bools (True-1 and False-0 bits)
    into its corresponding ASCII character.
    Does not check if len(array) == 8.
    :param array: Bitstream of exactly 8 bits
    :type array: list[bool]
    """

    # Transforms list[bool] into a binary representation
    auxstr: str = "".join(['1' if b else '0' for b in array])

    # Converts binary -> decimal -> ASCII
    return chr(int(auxstr, 2))

def samples_addnoise(samples: npt.NDArray, *, average: float = 0, spread: float = 1) -> npt.NDArray:
    """
    Adds noise into a sample list. Noise is normal based, average
    and spread can be included as parameters.
    :param samples: Array of voltage samples
    :type samples: npt.NDArray
    :param average: The center of the normal distribution. Default 0
    :type average: float
    :param spread: The deviation of the normal distribution. Default 1
    :type spread: float 
    """

    # Uses numpy's list comprehension to add some noise to every value at once
    result: npt.NDArray = samples + np.random.default_rng().normal(average, spread, len(samples))
    return result

def logmsg(message: str) -> None:
    with open("log.txt", "a") as f:
        f.write(message)
        f.write('\n')