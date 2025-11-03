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



def coder(bit_string: list[bool], tipo_modulacao: str) -> list[int]:
    """
    Transmission of a digital bit sequence using one of three modulation types:
    NRZ-Polar, Manchester, or Bipolar (AMI). Converts the binary sequence into
    an equivalent analog signal representation.

    :param bit_string: Input digital bit sequence (True/False for 1/0)
    :type bit_string: list[bool]
    :param tipo_modulacao: Type of digital modulation ("NRZ", "Manchester" or "Bipolar")
    :type tipo_modulacao: str
    :return: Encoded signal represented as a list of integer voltage levels
    :rtype: list[int]
    """

    if tipo_modulacao == "NRZ":
        return nrz_polar(bit_string)
      
    if tipo_modulacao == "Manchester":
        return manchester(bit_string)
      
    if tipo_modulacao == "Bipolar": 
        return bipolar(bit_string)
    
    print("Tipo de modulação não suportado")



def nrz_polar(bit_string: list[bool]) -> list[int]:
    """
    Encodes bits using Non-Return-to-Zero Polar (NRZ-Polar) modulation.
    Converts logical values to voltage levels:
      - True  → +1V
      - False → -1V

    :param bit_string: Input digital bit sequence (True/False for 1/0)
    :type bit_string: list[bool]
    :return: List of integer voltage levels (+1 or -1)
    :rtype: list[int]
    """

    sinal_codificado = []

    for bit in bit_string:
        if bit :
            sinal_codificado.append(1)

        else:
            sinal_codificado.append(-1)

    return sinal_codificado



def manchester(bit_string: list[bool]) -> list[int]:
    """
    Encodes bits using Manchester modulation.
    Each bit is represented by two voltage transitions:
      - False (0) → [1, 0]
      - True  (1) → [0, 1]
    The mid-bit transition provides clock synchronization.

    :param bit_string: Input digital bit sequence (True/False for 1/0)
    :type bit_string: list[bool]
    :return: List of integer voltage levels representing the Manchester signal
    :rtype: list[int]
    """

    sinal_codificado = []
    clock = [0,1]

    for bit in bit_string:
        sinal_codificado.append(clock[0]^bit)
        sinal_codificado.append(clock[1]^bit)

    return sinal_codificado


def bipolar(bit_string: list[bool]) -> list[int]:
    """
    Encodes bits using Bipolar (Alternate Mark Inversion - AMI) modulation.
    Logic 1 alternates between +1V and -1V to reduce DC component,
    while logic 0 remains at 0V.

    :param bit_string: Input digital bit sequence (True/False for 1/0)
    :type bit_string: list[bool]
    :return: List of integer voltage levels representing the Bipolar signal
    :rtype: list[int]
    """
  

    sinal_codificado = []
    aux = 1 

    for bit in bit_string:
        if not bit:
            sinal_codificado.append(0)
        else:
            sinal_codificado.append(aux)
            aux = -aux  # alternate polarity for each '1'

    return sinal_codificado




# --------------------------------------------------------------------
#                          DECODERS
# --------------------------------------------------------------------

def decoder(bit_string: list[int], tipo_modulacao: str) -> list[bool]:
    """
    Decodes a digital signal previously modulated using NRZ-Polar, Manchester, or Bipolar.
    Restores the original bit sequence from the received signal.

    :param bit_string: Encoded signal as a list of integer voltage levels
    :type bit_string: list[int]
    :param tipo_modulacao: Type of digital modulation ("NRZ", "Manchester" or "Bipolar")
    :type tipo_modulacao: str
    :return: Decoded bit sequence (list of True/False)
    :rtype: list[bool]
    """

    if tipo_modulacao == "NRZ":
        return nrz_polar_decoder(bit_string)
      
    if tipo_modulacao == "Manchester":
        return manchester_decoder(bit_string)
      
    if tipo_modulacao == "Bipolar":
        return bipolar_decoder(bit_string)

def nrz_polar_decoder(bit_string: list[int]) -> list[bool]:
    """
    Decodes a signal modulated with NRZ-Polar.
    Converts voltage levels back to logical values:
      - +1V → True (1)
      - -1V → False (0)

    :param bit_string: Encoded NRZ-Polar signal (+1/-1)
    :type bit_string: list[int]
    :return: Decoded bit sequence (True/False)
    :rtype: list[bool]
    """

    sinal_decodificado = []

    for bit in bit_string:
        if bit == 1:
            sinal_decodificado.append(True)

        else:
            sinal_decodificado.append(False)

    return sinal_decodificado


  
def manchester_decoder(bit_string: list[int]) -> list[bool]:
    """
    Decodes a signal modulated with Manchester encoding.
    Each pair of samples corresponds to one bit:
      - [0, 1] → False (0)
      - [1, 0] → True  (1)

    :param bit_string: Encoded Manchester signal (pairs of 0s and 1s)
    :type bit_string: list[int]
    :return: Decoded bit sequence (True/False)
    :rtype: list[bool]
    """

    i = 0
    sinal_decodificado = []
    #clock = [0,1]

    while i < len(bit_string):
        if bit_string[i] == 0 and bit_string[i+1] == 1:
            sinal_decodificado.append(False)

        elif bit_string[i] == 1 and bit_string[i+1] == 0:
            sinal_decodificado.append(True)

        else :
            print("Erro na decodificação")

        i += 2

    return sinal_decodificado


def bipolar_decoder(bit_string: list[int]) -> list[bool]:
    """
    Decodes a signal modulated with Bipolar (AMI) encoding.
    Converts voltage levels back to logical values:
      - 0V → False (0)
      - ±1V → True (1)

    :param bit_string: Encoded Bipolar signal (values 0, +1, -1)
    :type bit_string: list[int]
    :return: Decoded bit sequence (True/False)
    :rtype: list[bool]
    """
    
    sinal_decodificado = []
  
    for bit in bit_string:
        if bit == 0:
            sinal_decodificado.append(False)

        else:
            sinal_decodificado.append(True)

    return sinal_decodificado