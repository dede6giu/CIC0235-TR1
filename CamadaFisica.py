import numpy as np
import numpy.typing as npt

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
                    smplcnt: int = 990, 
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


def BPSK_modulation(digitals: list[bool], *, 
                    f: float = 100,
                    smplcnt: int = 220,
                    amp: float = 500) -> npt.NDArray:

    # Basic variables
    result: npt.NDArray = np.empty(0)
    phases: tuple[float, float] = (0, np.pi)
    period: float = 2 * np.pi * f
    
    # Precalculations for faster performance
    # Creates a list of smplcnt values from 0 to 2πf then makes it a wave
    rZero: npt.NDArray = np.sin(np.arange(0, period, period / smplcnt) + phases[0]) * amp
    rOne: npt.NDArray = np.sin(np.arange(0, period, period / smplcnt) + phases[1]) * amp

    # Signal creation
    for b in digitals:
        # Checks if b is 0/1, adds the correct wave to samples
        result = np.append(result, rOne if b else rZero)
    
    return result

def BPSK_demodulation(signal: npt.NDArray, *,
                    f: float = 100,
                    smplcnt: int = 220,
                    amp: float = 500) -> list[bool]:
    
    # Basic variables
    phases: tuple[float, float] = (0, np.pi)
    period: float = 2 * np.pi * f

    # Precalculations for faster performance
    # Creates a list of each possible expected wave
    e0: npt.NDArray = np.sin(np.arange(0, period, period / smplcnt) + phases[0]) * amp
    e1: npt.NDArray = np.sin(np.arange(0, period, period / smplcnt) + phases[1]) * amp

    # Iterates through chunks of samples for each bit
    result: list[bool] = []
    for i in range(0, len(signal), smplcnt):
        # Checks the absolute difference between each sample of the passed
        # signal and the expected 0 and 1 waves.
        absdiff1: np.float64 = np.sum(np.abs(signal[i:i+smplcnt] - e1))
        absdiff0: np.float64 = np.sum(np.abs(signal[i:i+smplcnt] - e0))

        # Decides result based on which has least error
        result.append(bool(absdiff1 < absdiff0))

    return result

