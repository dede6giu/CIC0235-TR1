import numpy as np
import numpy.typing as npt
from typing import List


def add_bit_flags_to_frame(payload_bits: List[bool]) -> List[bool]:
    """
    Performs bit-oriented framing with bit stuffing.
    
    Adds a flag sequence (01111110) to the beginning and end of the frame.
    Whenever five consecutive '1' bits occur in the payload, a '0' bit is inserted
    to prevent accidental occurrence of the flag pattern within the data.
    
    Args:
        payload_bits (List[bool]): The original frame payload represented as bits (True = 1, False = 0).
    
    Returns:
        List[bool]: The framed bit sequence with bit stuffing and boundary flags.
    """

    FLAG_PATTERN = [False, True, True, True, True, True, True, False]  # 01111110 (0x7E)
    stuffed_payload: List[bool] = []  # Output list for the stuffed bits
    consecutive_ones = 0  # Counter for consecutive '1' bits
    index = 0  # Current position in the payload

    # Iterate through all bits in the input payload
    while index < len(payload_bits):
        bit = payload_bits[index]

        if bit:
            consecutive_ones += 1
            stuffed_payload.append(True)
            index += 1

            # After five consecutive '1's, insert a '0' bit (stuffing)
            if consecutive_ones == 5:
                stuffed_payload.append(False)
                consecutive_ones = 0

        else:
            stuffed_payload.append(False)
            consecutive_ones = 0
            index += 1

    # Return the final framed sequence with flags at both ends
    return FLAG_PATTERN + stuffed_payload + FLAG_PATTERN


def add_bit_flags_to_bitstream(frames_payloads: List[List[bool]]) -> List[bool]:
    """
    Performs bit-oriented framing with bit stuffing for multiple frames.

    For each frame payload in the list, applies bit stuffing and adds 
    start/end flag sequences (01111110). The resulting framed sequences 
    are concatenated into a single bitstream.

    Args:
        frames_payloads (List[List[bool]]): 
            A list of frame payloads, where each payload is a list of bits 
            (True = 1, False = 0).

    Returns:
        List[bool]: 
            The concatenated bitstream containing all framed payloads 
            with bit stuffing and flags.
    """

    bitstream: List[bool] = []  # Final concatenated output bitstream

    for payload_bits in frames_payloads:
        framed_bits = add_bit_flags_to_frame(payload_bits)  # Apply bit-oriented framing
        bitstream.extend(framed_bits)  # Append framed bits to the overall bitstream

    return bitstream


def remove_bit_oriented_flags_from_bitstream(bitstream: List[bool]) -> List[List[bool]]:
    """
    Removes framing flags (01111110) and stuffed bits (bit-stuffing) from a bit-oriented bitstream.
    Returns a list of extracted frame payloads, each represented as a list of booleans.
    """

    extracted_frames: List[List[bool]] = [[]]  # List of decoded frame payloads
    buffer: List[bool] = []                   # Temporary buffer for bit accumulation
    FLAG_SEQUENCE = [False, True, True, True, True, True, True, False]  # 0x7E → 01111110
    STUFFING_PATTERN = [True, True, True, True, True, False]            # 111110 pattern (5 ones + stuffed zero)

    in_frame = False  # Tracks whether we are currently inside a frame
    i: int = 0        # Main iteration index

    while i < len(bitstream):

        # Detect a complete flag sequence at the end of the buffer
        if buffer[-8:] == FLAG_SEQUENCE:
            # Store everything before the flag as frame data
            extracted_frames[-1].extend(buffer[:-8])
            buffer.clear()

            # Prepare for the next frame
            if in_frame:
                extracted_frames.append([])  
            # Toggle in/out of frame state
            in_frame = not in_frame

            continue

        # Detect and remove a stuffed zero after five consecutive ones
        if buffer[-6:] == STUFFING_PATTERN:
            buffer.pop()  # Remove the stuffed bit
            extracted_frames[-1].extend(buffer)
            buffer.clear()

            continue

        # Normal case: add current bit to buffer
        buffer.append(bitstream[i])
        i += 1

    # Handle possible remaining data if bitstream ends with a flag
    if buffer[-8:] == FLAG_SEQUENCE:
        payload = buffer[:-8]
        if payload:
            extracted_frames[-1].extend(payload)

    return extracted_frames



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











def separacao_byte(bit_string: list[int]) -> list[int]:
  """
    Pads the end of a bit string with zero bits to ensure its length is a 
    multiple of 8 (byte-alignment). Prepends an 8-bit header indicating 
    the number of zero bits added.

    :param bit_string: The list of bits (0s and 1s) to be padded.
    :type bit_string: list[int]
    :return: The byte-aligned list of bits, including the 8-bit padding count header.
    :rtype: list[int]
    """
  resto = len(bit_string)%8 #Verifica se o tamanho da string é multiplo de 8
  zeros = 0
  if resto != 0: #Se não for, adiciona zeros até ser multiplo de 8
    zeros = 8-resto
    for i in range(zeros):
      bit_string.extend([0])

  numero_de_zeros = [int(b) for b in format(zeros, "08b")] #Transformando decimal em binario
  bit_string = numero_de_zeros + bit_string #Adicionando a contagem de zeros no inicio da string

  return bit_string




def desseparacao_byte(bit_string: list[int]) -> list[int]:
  """
    Removes the 8-bit padding count header and the trailing zero padding 
    from a byte-aligned bit string, restoring the original data.

    :param bit_string: The padded bit string containing the 8-bit header.
    :type bit_string: list[int]
    :return: The original bit string data without the header and padding.
    :rtype: list[int]
    """
  byte_tamanho = bit_string[:8] #Pegando os 8 primeiros bits que representam a contagem de zeros
  zeros = int("".join(str(b) for b in byte_tamanho), 2) #Transformando binario em decimal
  bit_string = bit_string[8:] #Removendo os 8 primeiros bits da string
  
  if zeros != 0 : 
    bit_string = bit_string[:-zeros] #Removendo os zeros adicionados
    
  return bit_string




def character_count_framing(bit_string:list[bool]):
  """
    Implements the Character Count framing method. 
    Divides the bit string into fixed-size frames (maximum 8*4 bits) 
    and prepends each frame with a byte indicating the length of its payload.

    :param bit_string: The raw digital signal (list of bits) to be framed.
    :type bit_string: list[bool]
    :return: The list of bits containing all framed segments, each preceded by its length count.
    :rtype: list[int]
    """

  bit_string = separacao_byte(bit_string) #Adiciona zeros para completar o byte
  tamanho_max = 8*4 #Definindo o tamanho maximo de cada frame
  
  frames = []
  i = 0
  while i < len(bit_string):
    
    payload = bit_string[i:i + tamanho_max] #Informação util
    i += tamanho_max
    
    tamanho_payload = len(payload) #Defindo "contagem de caracteres"
    byte_tamanho = [int(b) for b in format(tamanho_payload, "08b")] #Transformando decimal em binario

    frame =  byte_tamanho + payload
    frames.extend(frame)

  
  return frames





def character_count_deframing(bit_string:list[bool]):
    """
    Decodes a bit string that was framed using the Character Count method.
    It reads the length header of each frame, extracts the payload, and finally 
    removes the zero padding that was added to the original data.

    :param bit_string: The framed bit string containing multiple segments, each preceded by a length count.
    :type bit_string: list[bool]
    :return: The original, contiguous bit string data after deframing and depadding.
    :rtype: list[int]
    """

    payout = []
    i = 0
    while i < len(bit_string):
        byte_tamanho = bit_string[i:i + 8]
    
        tamanho_payload = int("".join(str(b) for b in byte_tamanho), 2)# Transforma binario em decimal
        i += 8
    
        
        payload = bit_string[i:i + tamanho_payload]
        i += tamanho_payload
        payout.extend(payload)

    payout = desseparacao_byte(payout)#Remove os zeros adicionados

    return payout



def add_flag(bit_string: list[int]) -> list[int]:
    """
    Adds the standard HDLC Flag sequence (01111110) to the beginning 
    and end of the frame payload to define frame boundaries.

    :param bit_string: The frame payload data.
    :type bit_string: list[int]
    :return: The frame with the start and end Flags.
    :rtype: list[int]
    """
    flag = [0, 1, 1, 1, 1, 1, 1, 0]
    return flag + bit_string + flag




def payload_flagORescape(bit_string: list[int]) -> list[int]:
    """
    Performs Byte Stuffing on the frame payload. If a sequence matching 
    the Flag (01111110) or the Escape byte (01111101) is found within the data, 
    the Escape byte is inserted immediately before it.

    :param bit_string: The raw payload data before the final Flags are added.
    :type bit_string: list[int]
    :return: The payload with all necessary Escape bytes inserted.
    :rtype: list[int]
    """
  
    flag =   [0, 1, 1, 1, 1, 1, 1, 0]
    escape = [0, 1, 1, 1, 1, 1, 0, 1]
    i = 0
  
    while i < len(bit_string):
      # Verifica se a sequência de 8 bits é a Flag ou o Escape
      if bit_string[i:i + 8] == flag or bit_string[i:i + 8] == escape :   
        bit_string = bit_string[:i] + escape + bit_string[i:] #Inserir o escape antes da sequência

        i += 16 # ESCAPE inserida (8 bits)+ FLAG/ESCAPE original (8 bits).

      else:
          i += 8

    return bit_string


def flag_bit_stuffing_framing(bit_string: list[int]) -> list[int]:
  """
    Performs complete framing using the Flag and Byte Stuffing method (similar to HDLC protocol).
    
    The process involves three main steps for each segment:
    1. Byte-aligning the data (using separacao_byte).
    2. Applying Byte Stuffing to the payload to prevent accidental Flag/Escape sequences (using payload_flagORescape).
    3. Adding the start and end Flag sequences (using add_flag).

    :param bit_string: The raw digital signal (list of bits) to be framed.
    :type bit_string: list[int]
    :return: The list of bits containing all framed segments ready for transmission.
    :rtype: list[int]
    """
  
  bit_string = separacao_byte(bit_string) #Adiciona zeros para completar o byte
  tamanho_max = 8*4 #Definindo o tamanho maximo de cada frame
  frames = []
  i = 0
  
  while i < len(bit_string):
    payload = bit_string[i:i + tamanho_max] #Informação util
    i += tamanho_max
    payload_ESC = payload_flagORescape(payload) #Inserindo o escape antes da flag ou escape
    
    frame = add_flag(payload_ESC) #Inserindo a flag no inicio e no final do frame
    frames.extend(frame) 
  
  return frames


 


def flag_bit_stuffing_deframing(bit_string: list[int]) -> list[int]:
    """
    Performs deframing of a bit string segmented using the Flag and Byte Stuffing method. 
    It removes the start/end Flags, detects and removes the Escape bytes (de-stuffing), 
    and handles frame synchronization by searching for the Flag byte-by-byte when 
    outside a payload. Finally, it removes the zero padding.

    :param bit_string: The framed bit string containing all segments and protocol overhead.
    :type bit_string: list[int]
    :return: The original, contiguous bit string data after deframing and depadding.
    :rtype: list[int]
    """
    # Definições
    escape = [0, 1, 1, 1, 1, 1, 0, 1]
    flag = [0, 1, 1, 1, 1, 1, 1, 0]

    decoded_payload = [] 
    in_payload = False  
    i = 0

    
    while i + 8 <= len(bit_string):# Enquanto houver pelo menos 8 bits para ler

        current_byte = bit_string[i:i + 8]

        bytes_to_advance = 8

       
        if current_byte == flag:  # Byte é uma FLAG

            if not in_payload:# Flag de Início: Entra no payload  
                in_payload = True
              
            else:# Flag de Fim: Sai do payload
                in_payload = False
                
        elif in_payload and current_byte == escape:# Payload e byte é um ESCAPE

            if i + 16 <= len(bit_string):
                # Adiciona o byte escapado (que está na posição i + 8)
                decoded_payload.extend(bit_string[i + 8:i + 16])
                bytes_to_advance = 16 # Avança 8 do ESCAPE + 8 do byte escapado
            else:
                # Caso de erro: ESCAPE incompleto. Finaliza a leitura.
                bytes_to_advance = len(bit_string) # Avance o máximo para sair

       
        elif in_payload:
            decoded_payload.extend(current_byte)
            


        else: 
          # Não é a Flag: avança apenas 1 bit e tenta encontrar a Flag na próxima iteração
          bytes_to_advance = 1

        # Avança o índice i
        i += bytes_to_advance

    # Remove os zeros de padding
    decoded_payload = desseparacao_byte(decoded_payload) 
    return decoded_payload
  





