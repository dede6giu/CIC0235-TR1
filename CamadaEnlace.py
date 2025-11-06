import numpy as np
import numpy.typing as npt
from typing import List
from math import floor,ceil, log2
from enum import Enum

class EDP(Enum): #ErrorDetectionProtocol
    PARITY_BIT = 1
    CHECKSUM = 2
    CRC32 = 3
    HAMMING = 4

class FP(Enum): #FramingProtocol
    CHAR_COUNT = 1
    BIT_ORIENTED_FLAGGING = 2
    BYTE_ORIENTED_FLAGGING = 3



def int_to_bool_list(num: int, size: int):
    binary_num: str = bin(num)[2:].zfill(size)[-size:]
    bool_list: List[bool] = [c == "1" for c in binary_num]
    return bool_list

def bool_list_to_int(bool_list: List[bool]):
    num_str: str = "".join(["1" if b else "0" for b in bool_list])
    num = int(num_str, 2)
    return num


def split_bitstream_into_payloads(bitstream: List[bool], payload_size: int) -> List[List[bool]]:
    """
    Splits a continuous bitstream into multiple payloads (frame bodies) of fixed size.

    Each payload represents a portion of the bitstream that will later be framed
    with flags and possibly bit-stuffed.

    Args:
        bitstream (List[bool]): Continuous sequence of bits to be divided.
        payload_size (int): Number of bits per payload.

    Returns:
        List[List[bool]]: A list of payloads (each one is a list of booleans).
                          The last payload may be shorter if the bitstream
                          length is not divisible by payload_size.
    """

    payloads: List[List[bool]] = []  # List that will hold all generated payloads
    num_payloads: int = ceil(len(bitstream) / payload_size)  # Total number of chunks

    payload_index: int = 0  # Index of the current payload being processed

    # Split all full-size payloads except the last one
    while payload_index <= num_payloads - 2:
        start = payload_index * payload_size
        end = start + payload_size
        payloads.append(bitstream[start:end])
        payload_index += 1

    # Append the remaining bits as the final (possibly shorter) payload
    payloads.append(bitstream[(num_payloads - 1) * payload_size:])

    return payloads

def add_padding_and_padding_size(payloads: List[List[bool]], payload_size: int) -> List[List[bool]]:
    """
    Adds zero-padding to the last payload so that it reaches the desired payload size,
    then appends a new payload containing the padding size (encoded in bits).

    Args:
        payloads (List[List[bool]]): List of payloads (each a list of bits).
        payload_size (int): Desired fixed size for each payload.

    Returns:
        List[List[bool]]: Updated list of payloads including padding and padding size info.
    """

    # Compute how many bits are missing in the last payload to reach the target size
    padding_size: int = payload_size - len(payloads[-1])

    # Append 'False' (0) bits as padding to the last payload
    for _ in range(padding_size):
        payloads[-1].append(False)

    # Convert padding size to a list of bits (same size as a payload)
    padding_size_bits: List[bool] = int_to_bool_list(num=padding_size, size=payload_size)

    # Append this new payload (representing the padding size) to the list
    payloads.append(padding_size_bits)

    return payloads


def add_EDC(payloads: List[List[bool]], edp: EDP, odd: bool = False, k: int = 2) -> List[List[bool]]:
    """
    Appends an Error Detection Code (EDC) to each payload in a list, based on the selected protocol.

    Depending on the specified EDP (Error Detection Protocol), the function applies one of:
    - Parity bit (even or odd)
    - Checksum
    - CRC-32
    - Hamming code

    Args:
        payloads (List[List[bool]]): List of payloads, where each payload is a list of bits.
        edp (EDP): The chosen error detection protocol (e.g., EDP.PARITY_BIT, EDP.CHECKSUM, etc.).
        odd (bool, optional): If True, uses odd parity instead of even parity. Default is False.
        k (int, optional): Number of data segments used in checksum computation. Default is 2.

    Returns:
        List[List[bool]]: List of payloads with their respective EDC bits appended.
    """

    # Apply the selected error detection protocol to each payload
    match edp:
        case EDP.PARITY_BIT:
            # Add a single parity bit (even or odd)
            for i, payload in enumerate(payloads):
                payloads[i] = parity_insert(payload, odd=odd)

        case EDP.CHECKSUM:
            # Add checksum bits based on 'k' data segments
            for i, payload in enumerate(payloads):
                payloads[i] = checksum_insert(payload, k=k)

        case EDP.CRC32:
            # Compute and append a 32-bit CRC to each payload
            for i, payload in enumerate(payloads):
                payloads[i] = crc32_insert(payload)

        case EDP.HAMMING:
            # Insert Hamming code parity bits into each payload
            for i, payload in enumerate(payloads):
                payloads[i] = hamming_insert(payload)

        case _:
            # Default case: do nothing if no valid protocol is selected
            pass

    return payloads

def add_framing_protocol(payloads_with_edc: List[List[bool]], fp: FP) -> List[List[bool]]:
    """
    Applies the selected framing protocol to each payload in the list.

    Each payload in the input list already includes its Error Detection Code (EDC)
    appended at the end. This function encapsulates those EDC-augmented payloads
    with framing information according to the selected framing protocol.

    Currently supports:
    - Bit-oriented flagging (e.g., HDLC-style framing with start/end flags).

    Args:
        payloads_with_edc (List[List[bool]]): List of payloads (each already includes EDC) represented as bit lists.
        fp (FP): Selected framing protocol (e.g., FP.BIT_ORIENTED_FLAGGING).

    Returns:
        List[List[bool]]: List of framed payloads, each now wrapped with framing information.
    """

    # Apply the chosen framing protocol to each payload (which already contains its EDC)
    match fp:
        case FP.BIT_ORIENTED_FLAGGING:
            # Add flag sequences (bit-oriented framing) around each payload + EDC
            for i, payload_with_edc in enumerate(payloads_with_edc):
                payloads_with_edc[i] = add_bit_oriented_flags_to_frame(payload_with_edc)

        case _:
            # Default case: no framing applied
            pass

    return payloads_with_edc

def remove_framing_protocol(bitstream: List[bool], fp: FP) -> List[List[bool]]:
    """
    Removes the framing information from a bitstream according to the selected framing protocol.

    This function takes a raw received bitstream and extracts the individual frame bodies
    (each containing the payload + EDC). It performs the inverse operation of 
    `add_framing_protocol`.

    Currently supports:
    - Bit-oriented flagging (e.g., HDLC-style framing).

    Args:
        bitstream (List[bool]): The complete received bitstream including framing bits.
        fp (FP): The framing protocol used to encapsulate the frames.

    Returns:
        List[List[bool]]: A list of frame bodies (each corresponding to one original payload with EDC).
    """

    frame_bodies: List[List[bool]] = []

    # Apply the proper framing removal method based on the selected protocol
    match fp:
        case FP.BIT_ORIENTED_FLAGGING:
            # Remove start/end flag sequences from bit-oriented framed bitstream
            frame_bodies = remove_bit_oriented_flags_from_bitstream(bitstream)
        case _:
            # No framing removal performed if protocol not recognized
            pass

    return frame_bodies


def find_corrupted_frames(frame_bodies: List[List[bool]], edp: EDP, odd: bool = False, k: int = 2) -> List[int]:
    """
    Identifies corrupted frames based on the selected Error Detection Protocol (EDP).

    This function checks each frame body (payload + EDC) to determine whether
    it contains transmission errors. The verification method depends on the
    error detection protocol that was used to generate the EDC.

    Must be called **after** `remove_framing_protocol()` and **before**
    `remove_EDC()`, since the EDC is required to detect errors.

    Args:
        frame_bodies (List[List[bool]]): A list of frame bodies, each containing payload + EDC.
        edp (EDP): The error detection protocol used to compute and verify the EDC.
        odd (bool, optional): Whether odd parity was used (only for parity bit). Defaults to False.
        k (int, optional): Number of segments used in the checksum method. Defaults to 2.

    Returns:
        List[int]: A list of indices corresponding to corrupted frames.
    """

    corrupted_frames: List[int] = []

    match edp:
        case EDP.PARITY_BIT:
            # Verify parity for each frame
            for i, frame_body in enumerate(frame_bodies):
                is_valid = parity_check(frame_body, odd=odd)
                if not is_valid:
                    corrupted_frames.append(i)

        case EDP.CHECKSUM:
            # Verify checksum for each frame
            for i, frame_body in enumerate(frame_bodies):
                is_valid = checksum_check(frame_body, k=k)
                if not is_valid:
                    corrupted_frames.append(i)

        case EDP.CRC32:
            # Verify CRC-32 remainder for each frame
            for i, frame_body in enumerate(frame_bodies):
                is_valid = crc32_check(frame_body)
                if not is_valid:
                    corrupted_frames.append(i)

        case EDP.HAMMING:
            # Check Hamming syndrome — if nonzero, a bit is corrupted
            for i, frame_body in enumerate(frame_bodies):
                corrupted_bit = hamming_find_corrupted_bit(frame_body)
                if corrupted_bit:
                    corrupted_frames.append(i)

        case _:
            # No check performed if the protocol is not recognized
            pass

    return corrupted_frames


def remove_EDC(frame_bodies: List[List[bool]], edp: EDP, k: int = 2) -> List[List[bool]]:
    """
    Removes the Error Detection Code (EDC) from each frame body.

    This function must be called **after** `remove_framing_protocol()`,
    since it expects the input to be a list of frame bodies — each one
    containing a payload followed by its EDC.

    Args:
        frame_bodies (List[List[bool]]): A list of frame bodies,
            each containing payload + EDC.
        edp (EDP): The error detection protocol used to generate the EDC.
        k (int, optional): Number of segments used in the checksum method. Defaults to 2.

    Returns:
        List[List[bool]]: A list of frame bodies with the EDC removed,
        leaving only the original payload bits.
    """

    match edp:
        case EDP.PARITY_BIT:
            for i, frame_body in enumerate(frame_bodies):
                frame_bodies[i] = parity_remove_EDC(frame_body)
        case EDP.CHECKSUM:
            for i, frame_body in enumerate(frame_bodies):
                frame_bodies[i] = checksum_remove_EDC(frame_body, k=k)
        case EDP.CRC32:
            for i, frame_body in enumerate(frame_bodies):
                frame_bodies[i] = crc32_remove_EDC(frame_body)
        case EDP.HAMMING:
            for i, frame_body in enumerate(frame_bodies):
                frame_bodies[i] = hamming_remove_EDC(frame_body)

    return frame_bodies

def remove_paddings(payloads: List[List[bool]]) -> List[List[bool]]:
    """
    Removes zero-padding from the last payload in a list of payloads.

    The last element in 'payloads' contains the padding size (encoded in bits).
    This function decodes that value, removes the padding bits from the
    preceding payload, and returns the cleaned list of payloads.

    Args:
        payloads (List[List[bool]]): List of payloads (each is a list of bits).
                                    The last payload contains the padding size.

    Returns:
        List[List[bool]]: List of payloads with the padding removed.
    """

    # Retrieve and decode the last payload, which stores the padding size
    padding_size: int = bool_list_to_int(payloads.pop())

    # Remove 'padding_size' bits from the end of the last real payload
    payloads[-1] = payloads[-1][:-padding_size]

    # Return the list without the padding information payload
    return payloads



def add_bit_oriented_flags_to_frame(payload_bits: List[bool]) -> List[bool]:
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
        # Debugging (optional)
        print("i =", i)
        print("buffer =", buffer)
        print("buffer[-8:] =", buffer[-8:])
        print("buffer[-8:] == FLAG_SEQUENCE?", buffer[-8:] == FLAG_SEQUENCE)

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

            print("extracted_frames =", extracted_frames)
            print("continue (flag detected)")
            continue

        # Detect and remove a stuffed zero after five consecutive ones
        if buffer[-6:] == STUFFING_PATTERN:
            buffer.pop()  # Remove the stuffed bit
            extracted_frames[-1].extend(buffer)
            buffer.clear()

            print("extracted_frames =", extracted_frames)
            print("continue (stuffed bit removed)")
            continue

        # Normal case: add current bit to buffer
        buffer.append(bitstream[i])
        i += 1

    # Handle possible remaining data if bitstream ends with a flag
    if buffer[-8:] == FLAG_SEQUENCE:
        print("buffer[-8:] == FLAG_SEQUENCE?", buffer[-8:] == FLAG_SEQUENCE)
        payload = buffer[:-8]
        if payload:
            extracted_frames[-1].extend(payload)
            print("extracted_frames =", extracted_frames)

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

def parity_remove_EDC(frame_bits: List[bool]) -> List[bool]:
    """
    Removes the parity bit (Error Detection Code) from the received frame.

    Args:
        frame_bits (List[bool]): The received frame containing both
                                 the data bits and one parity bit at the end.

    Returns:
        List[bool]: The original data bits after removing the parity bit.
    """

    # Remove the last bit (the parity bit) from the frame
    frame_bits.pop()

    # Return the remaining bits (the original data)
    return frame_bits

def checksum_insert(data_bits: List[bool], k: int) -> List[bool]:
    """
    Computes and appends a simple one's complement checksum to a bit sequence.

    Args:
        data_bits (List[bool]): The original payload bits.
        k (int): The number of equal-sized segments to split the payload into.
                 Must divide the payload length exactly.

    Returns:
        List[bool]: The payload with the checksum bits appended.
    """

    # Check if payload can be evenly divided into k segments
    if len(data_bits) % k != 0:
        print("Error: k must evenly divide the payload length to compute checksum.")
        return data_bits

    # Compute the segment size (number of bits per segment)
    segment_size: int = len(data_bits) // k

    # Initialize checksum accumulator
    checksum_value: int = 0

    # Process each segment as an integer and accumulate their sum
    for i in range(k):
        start_index = i * segment_size
        end_index = start_index + segment_size
        segment_value = bool_list_to_int(data_bits[start_index:end_index])
        checksum_value += segment_value

    # Convert the accumulated sum into binary (truncate if needed)
    checksum_bits: List[bool] = int_to_bool_list(num=checksum_value, size=segment_size)

    # Apply one's complement (invert all bits)
    checksum_complement: List[bool] = [not bit for bit in checksum_bits]

    # Append the checksum bits to the original data
    data_bits.extend(checksum_complement)

    return data_bits
    
def checksum_check(frame_bits: List[bool], k: int) -> bool:
    """
    Verifies the integrity of a frame using one's complement checksum.

    Args:
        frame_bits (List[bool]): The received frame (payload + checksum).
        k (int): The number of data segments used when the checksum was created.

    Returns:
        bool: True if the checksum is valid (no bit errors), False otherwise.
    """

    # The total frame must be divisible into k data segments + 1 checksum segment
    if len(frame_bits) % (k + 1) != 0:
        print("Error: Frame size must be divisible by (k + 1).")
        return False

    # Compute the size (in bits) of each segment
    segment_size: int = len(frame_bits) // (k + 1)

    # Initialize checksum accumulator
    checksum_value: int = 0

    # Sum all k data segments as integers
    for i in range(k):
        start = i * segment_size
        end = start + segment_size
        segment_value = bool_list_to_int(frame_bits[start:end])
        checksum_value += segment_value

    # Include the checksum segment in the sum
    received_checksum = bool_list_to_int(frame_bits[-segment_size:])
    checksum_value += received_checksum

    # Convert the sum to binary and take the one's complement
    result_bits: List[bool] = int_to_bool_list(num=checksum_value, size=segment_size)
    complement_bits: List[bool] = [not bit for bit in result_bits]

    # If all bits are 0, the frame is error-free
    return not any(complement_bits)


def checksum_remove_EDC(frame_bits: List[bool], k: int) -> List[bool]:
    """
    Removes the checksum (EDC) from the received frame and returns only the payload bits.

    Args:
        frame_bits (List[bool]): The complete frame (payload + checksum).
        k (int): The number of data segments used when computing the checksum.

    Returns:
        List[bool]: The payload bits without the checksum.
    """
    # Each frame is divided into (k + 1) equal segments: k data segments + 1 checksum
    segment_size: int = len(frame_bits) // (k + 1)

    # Return only the data portion (excluding the checksum bits)
    return frame_bits[:-segment_size]


def crc32_insert(payload: List[bool]) -> List[bool]:
    """
    Appends a CRC-32 checksum to the given bit sequence.

    Args:
        payload (List[bool]): The original data bits to be transmitted.

    Returns:
        List[bool]: The data bits followed by the computed 32-bit CRC.
    """

    # CRC-32 (IEEE 802.3) generator polynomial:
    # x^32 + x^26 + x^23 + x^22 + x^16 + x^12 +
    # x^11 + x^10 + x^8 + x^7 + x^5 + x^4 + x^2 + x + 1
    GENERATOR_POLYNOMIAL: int = 0x104C11DB7

    # Append 32 zero bits (CRC placeholder)
    padding: List[bool] = [False] * 32
    dividend: List[bool] = payload + padding

    # Convert generator polynomial to bit list (33 bits, MSB first)
    divisor: List[bool] = int_to_bool_list(num=GENERATOR_POLYNOMIAL, size=33)

    # Initialize remainder with the first 33 bits of the dividend
    remainder: List[bool] = dividend[:33]

    # Perform polynomial long division (bitwise)
    for i in range(len(divisor), len(dividend)):
        if remainder[0]:
            # XOR remainder with divisor if the MSB is 1
            remainder = [a ^ b for a, b in zip(remainder, divisor)]
        # Shift left by one bit and bring next dividend bit
        remainder = remainder[1:] + [dividend[i]]

    # Final XOR step after last iteration
    if remainder[0]:
        remainder = [a ^ b for a, b in zip(remainder, divisor)]
    remainder = remainder[1:]  # Drop the MSB (overflow)

    # Remaining 32 bits are the CRC value
    crc_bits: List[bool] = remainder

    # Append CRC to original payload
    payload.extend(crc_bits)

    return payload


def crc32_check(frame_bits: List[bool]) -> bool:
    """
    Verifies the integrity of a received frame using CRC-32.

    Args:
        frame_bits (List[bool]): The received frame (original payload + 32-bit CRC).

    Returns:
        bool: True if the CRC check passes (no error detected), False otherwise.
    """

    # CRC-32 (IEEE 802.3) generator polynomial:
    # x^32 + x^26 + x^23 + x^22 + x^16 + x^12 +
    # x^11 + x^10 + x^8 + x^7 + x^5 + x^4 + x^2 + x + 1
    GENERATOR_POLYNOMIAL: int = 0x104C11DB7

    # Convert generator polynomial to bit list (33 bits, MSB first)
    divisor: List[bool] = int_to_bool_list(num=GENERATOR_POLYNOMIAL, size=33)

    # Copy the received frame (payload + CRC)
    dividend: List[bool] = frame_bits[:]

    # Initialize remainder with the first 33 bits
    remainder: List[bool] = dividend[:33]

    # Perform polynomial long division (bitwise)
    for i in range(len(divisor), len(dividend)):
        if remainder[0]:
            # XOR remainder with divisor if MSB is 1
            remainder = [a ^ b for a, b in zip(remainder, divisor)]
        # Shift left by one bit and bring next bit from dividend
        remainder = remainder[1:] + [dividend[i]]

    # Final XOR step after last iteration
    if remainder[0]:
        remainder = [a ^ b for a, b in zip(remainder, divisor)]
    remainder = remainder[1:]  # Drop overflow bit

    # If remainder is all zeros, CRC check passed
    return not any(remainder)

def crc32_remove_EDC(frame_bits: List[bool]) -> List[bool]:
    """
    Removes the 32-bit CRC (Error Detection Code) from a received frame.

    Args:
        frame_bits (List[bool]): The received frame containing
                                 both the payload and the appended 32-bit CRC.

    Returns:
        List[bool]: The original payload bits (without the CRC).
    """

    return frame_bits[:-32]


#def crc32_manual(bitstream: list[bool], generator: int = 0x04C11DB7) -> list[bool]:
#    """
#    Calcula o CRC-32 (sem refletir bits) diretamente sobre uma lista de bools.
#    generator deve ser um inteiro de 33 bits (grau 32).
#    """
#    n = 32
#    data = bitstream[:] + [False] * n  # acrescenta 32 zeros
#
#    for i in range(len(bitstream)):
#        if data[i]:  # bit 1 encontrado → XOR com gerador
#            for j in range(33):
#                data[i + j] ^= bool((generator >> (32 - j)) & 1)
#
#    # resto são os últimos 32 bits
#    return data[-n:]



def hamming_insert(data_bits: List[bool]) -> List[bool]:
    """
    Inserts parity bits into a data bit sequence to form a Hamming codeword.

    Args:
        data_bits (List[bool]): The original data bits (without parity bits).

    Returns:
        List[bool]: The full Hamming codeword including data and calculated parity bits.
    """

    data_size: int = len(data_bits)
    parity_bits_count: int = floor(log2(data_size)) + 1

    # Adjust the number of parity bits until condition 2^r >= m + r + 1 is met
    while log2(data_size + parity_bits_count) > parity_bits_count:
        parity_bits_count += 1

    # Insert placeholder parity bits (False) at positions 1, 2, 4, 8, ...
    position: int = 1
    for _ in range(parity_bits_count):
        data_bits.insert(position - 1, False)
        position *= 2

    # Compute actual parity bits
    step: int = 1
    for _ in range(parity_bits_count):
        parity: bool = False

        # include 'step' bits, skip 'step' bits, include 'step' bits, and so on.
        for j in range(step, len(data_bits) + 1, step * 2):
            # XOR all bits that this parity bit covers
            for bit in data_bits[j - 1 : j - 1 + step]:
                parity ^= bit

        # Store the computed parity at its correct position (1-based index)
        data_bits[step - 1] = parity

        # Move to the next parity bit position (2, 4, 8, ...)
        step *= 2

    return data_bits



def hamming_find_corrupted_bit(frame_body: List[bool]) -> int:
    """
    Finds the position of a corrupted bit in a Hamming codeword.

    Args:
        frame_body (List[bool]): The received Hamming code bits, including both
                                 data and parity bits. The first bit corresponds
                                 to position 1 (not index 0).

    Returns:
        int: The 1-based index of the corrupted bit.
             Returns 0 if no errors were detected (i.e., all parities are correct).
    """

    # Number of parity bits (r), estimated from codeword length
    parity_bits_count: int = floor(log2(len(frame_body))) + 1
    syndrome_bits: List[bool] = []

    # Each parity bit covers a block of 'step' bits, skipping every next block
    step: int = 1
    for _ in range(parity_bits_count):
        parity_sum: bool = False
        # Check bits covered by the current parity position
        for j in range(step, len(frame_body) + 1, step * 2):
            for bit in frame_body[j - 1 : j - 1 + step]:
                parity_sum ^= bit
        syndrome_bits.append(parity_sum)
        step *= 2

    # Syndrome bits form a binary number indicating the error position
    corrupted_bit_position: int = int("".join(["1" if b else "0" for b in syndrome_bits[::-1]]), 2)

    return corrupted_bit_position

def hamming_fix_corrupted_bit(frame_body: List[bool], corrupted_bit_position: int) -> List[bool]:
    if corrupted_bit_position != 0:
        frame_body[corrupted_bit_position - 1] = not(frame_body[corrupted_bit_position - 1])
    return frame_body

def hamming_remove_EDC(frame_bits: List[bool]) -> List[bool]:
    """
    Removes the error detection/correction bits from a Hamming-encoded frame.

    A Hamming code places parity bits at positions that are powers of two (1, 2, 4, 8, ...).
    This function removes those bits, returning only the data bits.

    Args:
        frame_bits (List[bool]): The encoded bit sequence (with parity bits included).

    Returns:
        List[bool]: The list of data bits (without parity bits).
    """
    data_bits: List[bool] = []

    for position, bit in enumerate(frame_bits, start=1):  # Positions start at 1 in Hamming code
        # Check if 'position' is NOT a power of two (i.e., not a parity bit)
        if log2(position).is_integer() is False:
            data_bits.append(bit)

    return data_bits





# ----------------------------- BYTE ORIENTED FLAGGING --------------------------------------- #



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
  





