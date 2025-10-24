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

def ASK_modulation(digitals: list[bool], *, 
                    smplcnt: int = 220, 
                    f: float = 100, 
                    amp: float = 500,
                    lowwave: bool = False) -> npt.NDArray:
    """
    Modulation of a digital signal via Amplitude Shift Keying. In this case, it
    will modulate 1s as the wave's normal amplitude and 0s are silence.

    :param digitals: Digital signal (bit stream)
    :type digitals: list[bool]
    :param smplcnt: Amount of samples per byte of digital signal. Recommended at least 2*f but not a multiple of f
    :type smplcnt: int
    :param f: Frequency of the wave in Hz. In ASK, this is equivalent to the frequency of the channel
    :type f: float
    :param amp: Amplitude of the wave
    :type amp: float
    :param lowwave: Reverse behaviour (0 waves and 1 silence)
    :type lowwave: bool
    """

    # Basic variables
    result: npt.NDArray = np.empty(0)
    period: np.float64 = np.pi * 2 * f
    
    # Precalculations for faster performance
    # Creates a list of smplcnt values from 0 to 2πf then makes it a wave
    rOne: npt.NDArray = np.sin(np.arange(0, period, period / smplcnt)) * amp
    # Only zeroes
    rZero: npt.NDArray = np.zeros(smplcnt)

    # Signal creation
    for b in digitals:
        # Checks if b is 0/1, then reverses the option if such has been requested
        result = np.append(result, rOne if (b ^ lowwave) else rZero)
    
    return result

def ASK_demodulation(signal: npt.NDArray, *, 
                    smplcnt: int = 220, 
                    f: float = 100,
                    amp: float = 500,
                    lowwave: bool = False) -> list[bool]:
    """
    Demodulation of a digital signal modulated by Amplitude Shift Keying.
    It checks both possible outputs depending on the parameters and decides
    whichever one has least error as the correct value of the bit.

    :param signal: Analogic signal (voltage samples)
    :type signal: npt.NDArray
    :param smplcnt: Amount of samples per byte of digital signal
    :type smplcnt: int
    :param f: Frequency of the wave in Hz. In ASK, this is equivalent to the frequency of the channel
    :type f: float
    :param amp: Amplitude of the wave
    :type amp: float
    :param lowwave: Reverse behaviour (0 waves and 1 silence)
    :type lowwave: bool
    """

    # Precalculations for faster performance
    period: np.float64 = np.pi * 2 * f
    # Expected "1" wave
    eOne: npt.NDArray = np.sin(np.arange(0, period, period / smplcnt)) * amp
    # Expected "0" wave
    eZero: npt.NDArray = np.zeros(smplcnt)

    # Iterates through chunks of samples for each bit
    result: list[bool] = []
    for i in range(0, len(signal), smplcnt):
        # Checks the absolute difference between each sample of the passed
        # signal and the expected 0 and 1 waves.
        absdiffOne: np.float64 = np.sum(np.abs(signal[i:i+smplcnt] - eOne))
        absdiffZero: np.float64 = np.sum(np.abs(signal[i:i+smplcnt] - eZero))

        # Decides result based on which has least error
        result.append(bool(absdiffOne < absdiffZero) ^ lowwave)
    
    return result

def FSK_modulation(digitals: list[bool],
                    keys: tuple[float, float], *, 
                    smplcnt: int = 990,
                    amp: float = 500) -> npt.NDArray:
    """
    Modulation of a digital signal via Frequency Shift Keying. In this case, it
    will modulate based on the keys parameter: 0s will have waves of frequency
    keys[0] and 1s will have keys[1] as the frequency.

    :param digitals: Digital signal (bit stream)
    :type digitals: list[bool]
    :param keys: Frequency of wach wave in Hz
    :type keys: float
    :param smplcnt: Amount of samples per byte of digital signal. \
        Recommended at least two times the highest frequency, but not a multiple of either one.
    :type smplcnt: int
    :param amp: Amplitude of the wave
    :type amp: float
    """

    # Basic variables
    result: npt.NDArray = np.empty(0)
    periodZero: np.float64 = np.pi * 2 * keys[0]
    periodOne: np.float64 = np.pi * 2 * keys[1]
    
    # Precalculations for faster performance
    # Creates a list of smplcnt values from 0 to 2πf then makes it a wave
    rZero: npt.NDArray = np.sin(np.arange(0, periodZero, periodZero / smplcnt)) * amp
    rOne: npt.NDArray = np.sin(np.arange(0, periodOne, periodOne / smplcnt)) * amp

    # Signal creation
    for b in digitals:
        # Checks if b is 0/1, adds the correct wave to samples
        result = np.append(result, rOne if b else rZero)
    
    return result

def FSK_demodulation(signal: npt.NDArray,
                    keys: tuple[float, float], *, 
                    smplcnt: int = 220, 
                    amp: float = 500) -> list[bool]:
    """
    Demodulation of a digital signal modulated by Frequency Shift Keying.
    It checks both possible outputs depending on the parameters and decides
    whichever one has least error as the correct value of the bit.

    :param signal: Analogic signal (voltage samples)
    :type signal: npt.NDArray
    :param smplcnt: Amount of samples per byte of digital signal
    :type smplcnt: int
    :param keys: Frequency of wach wave in Hz
    :type keys: float
    :param amp: Amplitude of the wave
    :type amp: float
    """

    # Precalculations for faster performance
    periodZero: np.float64 = np.pi * 2 * keys[0]
    periodOne: np.float64 = np.pi * 2 * keys[1]
    # Expected "0" wave
    eZero: npt.NDArray = np.sin(np.arange(0, periodZero, periodZero / smplcnt)) * amp
    # Expected "1" wave
    eOne: npt.NDArray = np.sin(np.arange(0, periodOne, periodOne / smplcnt)) * amp

    # Iterates through chunks of samples for each bit
    result: list[bool] = []
    for i in range(0, len(signal), smplcnt):
        # Checks the absolute difference between each sample of the passed
        # signal and the expected 0 and 1 waves.
        absdiffOne: np.float64 = np.sum(np.abs(signal[i:i+smplcnt] - eOne))
        absdiffZero: np.float64 = np.sum(np.abs(signal[i:i+smplcnt] - eZero))

        # Decides result based on which has least error
        result.append(bool(absdiffOne < absdiffZero))
    
    return result


if __name__ == "__main__":
    message = "long message"
    print(message)
    signal = FSK_modulation(string_to_bitstream(message), (100, 433), smplcnt=990)
    signal = samples_addnoise(signal, spread=1500)
    print(bitstream_to_string(FSK_demodulation(signal, (100, 433), smplcnt=990)))