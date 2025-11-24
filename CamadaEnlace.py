import numpy as np
import numpy.typing as npt
from typing import List
from math import floor,ceil, log2
from enum import Enum

"""
============= ORDER OF USAGE =============

Framing
1. split_bitstream_into_payloads
2. add_padding_and_padding_size
3. add_EDC
4. add_ECC
5. add_framing_protocol
6. list_linearize
7. add_padding_for_4bit_alignment

Deframing
1. Remove_padding_for_4bit_alignment
2. remove_framing_protocol
3. ECC_fix_corrupted_bits
4. remove_ECC
5. find_corrupted_frames
6. remove_EDC
7. remove_paddings
8. list_linearize

"""


class EDP(Enum): #ErrorDetectionProtocol
    PARITY_BIT = 1
    CHECKSUM = 2
    CRC32 = 3

class FP(Enum): #FramingProtocol
    CHAR_COUNT = 1
    BIT_ORIENTED_FLAGGING = 2
    BYTE_ORIENTED_FLAGGING = 3



def int_to_bool_list(num: int, size: int) -> List[bool]:
    """
    Converts an integer into a fixed-size list of booleans representing its
    binary form. If the binary representation has fewer than `size` bits,
    the function adds zeros on the left to reach the required number of elements.
    If it has more, only the least significant `size` bits are kept.

    Args:
        num (int):
            The integer to be converted into a bit list.
        size (int):
            The required number of bits in the output list.

    Returns:
        List[bool]:
            A list of booleans where True represents bit 1 and False represents bit 0.
            The list contains exactly `size` elements.
    """

    # Convert the integer into a binary string (without the '0b' prefix),
    # add leading zeros to reach the required length,
    # and keep exactly the last `size` bits.
    binary_num: str = bin(num)[2:].zfill(size)[-size:]

    # Convert each character ('0' or '1') into a boolean value.
    bool_list: List[bool] = [c == "1" for c in binary_num]

    return bool_list


def bool_list_to_int(bool_list: List[bool]) -> int:
    """
    Converts a list of boolean values into an integer by interpreting the list
    as a binary number. Each element is treated as a bit, where True is 1 and
    False is 0. The first element corresponds to the most significant bit.

    Args:
        bool_list (List[bool]):
            A list of booleans representing a binary number.

    Returns:
        int:
            The integer value obtained from interpreting the list as a binary number.
    """

    # Convert each boolean into its corresponding character ('1' or '0'),
    # then join them into a single binary string.
    num_str: str = "".join("1" if b else "0" for b in bool_list)

    # Convert the binary string into an integer using base 2.
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

def add_padding_and_padding_size(payloads: List[List[bool]]) -> List[List[bool]]:
    """
    Adds zero-padding to the last payload so that its length is a multiple of 2 (checksum requirement) 
    and then appends a new payload containing the padding size (encoded in bits).

    Args:
        payloads (List[List[bool]]): List of payloads (each a list of bits).
        payload_size (int): Maximum size for each payload.

    Returns:
        List[List[bool]]: Updated list of payloads including padding and padding size info.
    """

    # Compute how many bits are missing in the last payload to reach the target size
    padding_size: int = len(payloads[-1]) % 2

    # Append 'False' (0) bit as padding to the last payload
    if padding_size:
        payloads[-1].append(False)

    # Convert padding size to a list of bits (same size as a payload)
    padding_size_bits: List[bool] = int_to_bool_list(num=padding_size, size=2)

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

        case _:
            # Default case: do nothing if no valid protocol is selected
            pass

    return payloads

def add_ECC(payloads: List[List[bool]]) -> List[List[bool]]:
    """
    Applies Hamming-code based Error-Correcting Code (ECC) to a list of payloads.
    For each payload (a list of bits), parity bits are inserted using
    the `hamming_insert` function.

    Args:
        payloads (List[List[bool]]):
            A list of bitstreams, where each bitstream represents a payload
            to which ECC parity bits will be added.

    Returns:
        List[List[bool]]:
            A list of bitstreams where each payload now contains the
            corresponding Hamming parity bits.
    """

    for i, payload in enumerate(payloads):
        # Replace each payload with its ECC-extended version.
        payloads[i] = hamming_insert(payload)
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
        case FP.CHAR_COUNT:
            for i, payload_with_edc in enumerate(payloads_with_edc):
                payloads_with_edc[i] = add_char_count_flag_to_frame(payload_with_edc)
        case FP.BIT_ORIENTED_FLAGGING:
            for i, payload_with_edc in enumerate(payloads_with_edc):
                payloads_with_edc[i] = add_bit_oriented_flags_to_frame(payload_with_edc)

        case FP.BYTE_ORIENTED_FLAGGING:
            for i, payload_with_edc in enumerate(payloads_with_edc):
                payloads_with_edc[i] = add_byte_oriented_flags_to_frame(payload_with_edc)
        case _:
            # Default case: no framing applied
            pass
    return payloads_with_edc

def list_linearize(ilist: list[list[bool]]) -> list[bool]:
    """
    Linearizes a 2D list.
    
    Args:
        ilist (list[list[bool]]): Input list

    Returns:
        list[bool]: Linearized list
    """
    result = []
    for i in ilist:
        for j in i:
            result.append(j)
    return result


def add_padding_for_4bit_alignment(bitstream: List[bool]) -> List[bool]:
    """
    Adds padding bits to the input bitstream so that its length becomes a
    multiple of 4. Additionally, appends a 4-bit field at the end of the
    stream indicating how many padding bits were added.

    This is useful for digital modulation schemes such as 16QAM, where the
    number of bits must always be divisible by 4.

    Args:
        bitstream (List[bool]):
            The original list of bits (True = 1, False = 0).

    Returns:
        List[bool]:
            The bitstream with the necessary padding and a 4-bit padding-size field appended.
    """

    # Compute how many padding bits are needed.
    # If the bitstream length is already a multiple of 4,
    # we want padding_size = 0 rather than 4, so we apply an extra modulo.
    padding_size: int = (4 - (len(bitstream) % 4)) % 4

    # Create the padding sequence (False == 0) of the required length.
    padding_bits: List[bool] = [False] * padding_size

    # Convert the padding size (0–3) into a 4-bit boolean list.
    # Example: padding_size = 2  ->  [0, 0, 1, 0]
    padding_size_bits: List[bool] = int_to_bool_list(padding_size, 4)

    # Append the padding bits to the end of the bitstream.
    bitstream.extend(padding_bits)

    # Append the 4-bit representation of the padding size.
    bitstream.extend(padding_size_bits)
    return bitstream

def remove_padding_for_4bit_alignment(bitstream: List[bool]) -> List[bool]:
    """
    Removes the padding previously added by `add_padding_for_4bit_alignment`.
    The last 4 bits of the bitstream encode how many padding bits were appended.

    Args:
        bitstream (List[bool]):
            The bitstream containing the original data, the padding bits,
            and the final 4-bit padding-size field.

    Returns:
        List[bool]:
            The bitstream with the padding bits and the 4-bit padding-size field removed.
    """
    # Read the last 4 bits and convert them into an integer.
    # These bits specify how many padding bits were added.
    padding_size = bool_list_to_int(bitstream[-4:])
    # If padding_size is larger than 3, we know the bits were corrupted
    padding_size = 0 if padding_size > 3 else padding_size

    # Remove the 4-bit padding-size field.
    del bitstream[-4:]

    # Remove the padding bits themselves.
    for _ in range(padding_size):
        if bitstream:
            bitstream.pop()

    return bitstream


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
        case FP.CHAR_COUNT:
            frame_bodies = remove_char_count_flag_from_bitstream(bitstream)
        case FP.BIT_ORIENTED_FLAGGING:
            frame_bodies = remove_bit_oriented_flags_from_bitstream(bitstream)
        case FP.BYTE_ORIENTED_FLAGGING:
            frame_bodies = remove_byte_oriented_flags_from_bitstream(bitstream)
        case _:
            # No framing removal performed if protocol not recognized
            pass
    return frame_bodies

def ECC_fix_corrupted_bits(payloads: List[List[bool]]) -> List[List[bool]]:
    """
    Detects and corrects single-bit errors in each payload using the Hamming ECC
    algorithm. For each payload, the function identifies the position of the
    corrupted bit (if any) and then corrects it.

    Args:
        payloads (List[List[bool]]):
            A list of bitstreams, where each bitstream contains data bits, EDC
            and Hamming ECC parity bits.

    Returns:
        List[List[bool]]:
            A list of bitstreams where any single-bit error present in each
            payload has been corrected.
    """

    for i, payload in enumerate(payloads):

        # Determine the index of the corrupted bit using the Hamming syndrome.
        # If no error is present, this function returns 0.
        corrupted_bit_position = hamming_find_corrupted_bit(payload)

        # Correct the corrupted bit in the payload (if the position is non-zero)
        # and store the corrected payload back into the list.
        payloads[i] = hamming_fix_corrupted_bit(payload, corrupted_bit_position)

    return payloads
    

def remove_ECC(payloads: List[List[bool]]) -> List[List[bool]]:
    """
    Removes the Hamming-code ECC (Error-Correcting Code) parity bits from
    each payload.

    Args:
        payloads (List[List[bool]]):
            A list of bitstreams, where each bitstream includes ECC parity bits
            that were previously inserted using a Hamming code.

    Returns:
        List[List[bool]]:
            A list of bitstreams where the ECC parity bits have been removed
            and only the corrected data bits remain.
    """

    for i, payload in enumerate(payloads):
        # Remove Hamming parity bits from each payload
        payloads[i] = hamming_remove_ECC(payload)

    return payloads

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
                    corrupted_frames.append(i + 1)

        case EDP.CHECKSUM:
            # Verify checksum for each frame
            for i, frame_body in enumerate(frame_bodies):
                is_valid = checksum_check(frame_body, k=k)
                if not is_valid:
                    corrupted_frames.append(i + 1)

        case EDP.CRC32:
            # Verify CRC-32 remainder for each frame
            for i, frame_body in enumerate(frame_bodies):
                is_valid = crc32_check(frame_body)
                if not is_valid:
                    corrupted_frames.append(i + 1)

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

    return frame_bodies

def remove_paddings(payloads: List[List[bool]], last_frame_corrupted: bool) -> List[List[bool]]:
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
    padding_size = 0
    if len(payloads) > 1 and len(payloads[-1]) == 2 and not(last_frame_corrupted):
        padding_size: int = 1 if payloads.pop()[1] else 0

    # Remove 'padding_size' bits from the end of the last real payload
    if padding_size:
        if payloads:
            if payloads[-1]:
                payloads[-1].pop()
    # Return the list without the padding information payload
    return payloads


def parity_insert(payload: list[bool], *,
                odd: bool = False) -> list[bool]:
    """
    Inserts a parity bit at the end of a digital signal.
    Returns the new digital signal with parity added at its trail.

    :param payload: The digital signal
    :type payload: list[bool]
    :param odd: Odd parity
    :type odd: bool
    """

    # Count how many 1s (Trues) are in the signal
    ones: int = sum(payload)

    # Append parity bit, respecting odd-ness choice
    payload.append(bool(ones % 2) ^ odd)

    # Return new list
    return payload

def parity_check(frame_body: list[bool], *,
                odd: bool = False) -> bool:
    """
    Checks the parity of a given digital signal.
    Returns True if no error is detected, and False if it is.

    :param frame_body: payload + parity bit
    :type frame_body: list[bool]
    :param odd: Odd parity
    :type odd: bool
    """
    
    # If frame_body is empty, it means it must have been corrupted
    if len(frame_body) == 0:
        return False

    # Count how many 1s (Trues) are in the signal
    ones: int = sum(frame_body)

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
    if frame_bits:
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
    # If this condition is violated, it means part of the frame was lost or corrupted
    if len(frame_bits) % (k + 1) != 0:
        return False
    # If frame_bits is empty, it means the frame must have been corrupted
    if len(frame_bits) == 0:
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

    # Ensure the frame contains more than 32 bits.
    # A valid CRC-32 frame must contain: payload (>=1 bit) + 32 CRC bits.
    # If the frame has 32 bits or less, it means part of the message was lost,
    # therefore the frame is certainly corrupted and we return False immediately.
    if len(frame_bits) <= 32:
        return False
    
    # CRC-32 (IEEE 802.3) generator polynomial:
    # x^32 + x^26 + x^23 + x^22 + x^16 + x^12 +
    # x^11 + x^10 + x^8 + x^7 + x^5 + x^4 + x^2 + x + 1
    GENERATOR_POLYNOMIAL: int = 0x104C11DB7

    # Convert generator polynomial to bit list (33 bits, MSB first)
    divisor: List[bool] = int_to_bool_list(num=GENERATOR_POLYNOMIAL, size=33)

    # Copy the received frame (payload + CRC)
    dividend: List[bool] = frame_bits

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
            # XOR all bits that this parity bit covers (xor represents binary sum)
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
    if len(frame_body):
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
    else:
        return 0

def hamming_fix_corrupted_bit(frame_body: List[bool], corrupted_bit_position: int) -> List[bool]:
    """
    Corrects a single-bit error in a Hamming-encoded frame body.

    Args:
        frame_body: List of bits representing the frame body.
        corrupted_bit_position: 1-based index of the corrupted bit; 0 means no error.

    Returns:
        The frame body with the corrected bit (if an error was detected).
    """

    # Flip the corrupted bit if its 1-based position is nonzero
    if corrupted_bit_position != 0 and corrupted_bit_position <= len(frame_body):
        frame_body[corrupted_bit_position - 1] = not(frame_body[corrupted_bit_position - 1])
    return frame_body

def hamming_remove_ECC(frame_bits: List[bool]) -> List[bool]:
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



# ----------------------------- FRAMING PROTOCOLS --------------------------------------- #

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

     Args:
        bitstream (List[bool]):
            The raw bitstream containing HDLC-style frames, including flag
            sequences and stuffed bits.

    Returns:
        List[List[bool]]:
            A list where each element is a list of booleans representing the
            decoded payload of a single frame (with flags and stuffed bits removed).
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



def apply_8bit_padding(frame_body: List[bool]) -> List[bool]:
    """
    Ensures the frame_body length is aligned to an 8-bit (1 byte) boundary.
    If the number of bits is not divisible by 8, the function adds '0' bits
    to reach the next byte boundary, and then appends an 8-bit value 
    indicating how many padding bits were added.

    Args:
        frame_body (List[bool]): The input bit sequence to be byte-aligned.

    Returns:
        List[bool]: A new frame_body whose length is a multiple of 8,
                    ending with an 8-bit field representing the number
                    of added alignment bits.
    """

    # Number of zero bits that must be added
    # If the frame body length is already a multiple of 8,
    # we want padding_size = 0 rather than 8, so we apply an extra modulo.
    padding_size:int = (8 - len(frame_body) % 8) % 8
    
    # Extend frame_body with alignment bits (all zeros)
    frame_body += [False] * padding_size

    # Encode padding size as an 8-bit boolean list
    padding_bits: List[bool] = int_to_bool_list(padding_size, 8)

    # Append this information to the end of the frame_body
    frame_body += padding_bits

    return frame_body

def remove_8bit_padding(frame_body: List[bool]) -> List[bool]:
    """
    Removes 8-bit alignment padding added by apply_8bit_padding.

    Args:
        frame_body: Bit sequence ending with an 8-bit field indicating 
                    how many padding bits were added.

    Returns:
        The original bit sequence with padding removed.
    """

    # Read the last 8 bits to obtain the padding length

    if len(frame_body) <= 8: return frame_body

    padding_size = bool_list_to_int(frame_body[-8:])

    # Remove the padding-length field
    del frame_body[-8:]

    # padding_size must be at most 7
    # otherwise, we know it has been corrupted
    if padding_size >= 8: return frame_body

    # Remove the actual padding bits
    for _ in range(padding_size):
        if frame_body:
            frame_body.pop()

    return frame_body


def add_char_count_flag_to_frame(frame_body: List[bool]) -> List[bool]:
    """
    Applies character-count framing by prefixing the frame with an 8-bit
    header indicating how many bytes the frame contains.

    The frame body is first aligned to an 8-bit boundary so that the
    byte count can be computed reliably.

    Args:
        frame_body: Bit sequence to be framed.

    Returns:
        A new bit sequence with an 8-bit length header prepended.
    """

    # Ensure the frame body is aligned to whole bytes
    frame_body = apply_8bit_padding(frame_body)

    # Compute how many bytes the padded frame occupies
    num_of_bytes: int = len(frame_body) // 8

    # Encode the byte count as an 8-bit header
    header: List[bool] = int_to_bool_list(num_of_bytes, 8)

    # Prepend header to the frame body
    frame_body = header + frame_body

    return frame_body


def remove_char_count_flag_from_bitstream(bitstream: List[bool]) -> List[List[bool]]:
    """
    Extracts frame bodies from a character-count-framed bitstream.

    Each frame starts with an 8-bit header indicating how many bytes the
    frame occupies. The function then reads the frame body and attempts to
    remove padding bits based on a padding-size field located at the end
    of the frame.

    Args:
        bitstream: A bit sequence containing one or more framed units.

    Returns:
        A list of frame bodies with padding removed.
    """

    frame_bodies: List[List[bool]] = []
    i = 0

    # Continue while there is enough space to read a header
    while i + 8 <= len(bitstream):
        # Start a new frame body
        frame_bodies.append([])

        # Read the 8-bit character count (in bytes)
        num_of_bytes = bool_list_to_int(bitstream[i : i + 8])

        # Copy the frame body bits (excluding the header)
        for j in range(i + 8, min(i + 8 * num_of_bytes, len(bitstream))):
            frame_bodies[-1].append(bitstream[j])

        # Attempt to read the padding-size byte
        if i + 8 * num_of_bytes <= len(bitstream):
            padding_size = 0
            if len(bitstream) > i + 8 * num_of_bytes:
                padding_size = bool_list_to_int(
                    bitstream[i + 8 * num_of_bytes :
                              min(i + 8 * num_of_bytes + 8, len(bitstream))]
                )

            # Remove padding bits if present
            for _ in range(padding_size):
                if frame_bodies:
                    if frame_bodies[-1]:
                        frame_bodies[-1].pop()

        # Move to the next framed block
        i += 8 + 8 * num_of_bytes

    return frame_bodies

def add_byte_oriented_flags_to_frame(frame_body: List[bool]) -> List[bool]:
    """
    Performs Byte Stuffing on the frame payload. If a sequence matching 
    the Flag (#0x7E) or the Escape byte (#0x7D) is found within the data, 
    the Escape byte is inserted immediately before it.

    :param frame_body: The payload encoded with the EDC before the final Flags are added.
    :type frame_body: List[bool]
    :return: The payload with all necessary Escape bytes inserted.
    :rtype: List[bool]
    """
    frame_body = apply_8bit_padding(frame_body)

    FLAG_PATTERN: List[bool] = [False, True, True, True, True, True, True, False] #0x7E
    ESCAPE_PATTERN: List[bool] = [False, True, True, True, True, True, False, True] #0x7D
    
    i = 0
    while i < len(frame_body):
      # Verifies if each 8-bit sequence is a FLAG_PATTERN or an ESCAPE_PATTERN
      if frame_body[i:i + 8] == FLAG_PATTERN or frame_body[i:i + 8] == ESCAPE_PATTERN :   
        frame_body = frame_body[:i] + ESCAPE_PATTERN + frame_body[i:] #Insert ESCAPE before sequence

        i += 16 # inserted ESCAPE (8 bits)+ original FLAG/ESCAPE (8 bits).
      else:
          i += 8

    frame_body = FLAG_PATTERN + frame_body + FLAG_PATTERN
    return frame_body


def remove_byte_oriented_flags_from_bitstream(bitstream: List[bool]) -> List[List[bool]]:
    """
    Performs complete deframing of a signal encoded using the Flag and Byte Stuffing method.    
    This function identifies frame boundaries using FLAG bytes, handles ESCAPE sequences,
    and reconstructs the original payloads by removing framing bytes and padding.    
    The process is the reverse of the framing function (add_byte_oriented_flags_to_frame).   
    Steps:
        1. Iterate through the bit stream in 8-bit segments.
        2. Detect FLAG bytes (start and end of frames).
        3. Handle ESCAPE sequences to correctly interpret escaped bytes.
        4. Collect payload bits between FLAG delimiters.
        5. Remove zero-padding using 'remove_8bit_padding'.   
    :param bitstream: Complete framed bit sequence (list of bits).
    :type bitstream: List[int]
    :return: List of deframed bit sequences (payloads) extracted from the signal.
    :rtype: List[List[int]]
    """
    # # HDLC-style control bytes (0x7E = FLAG, 0x7D = ESCAPE)
    FLAG_PATTERN: List[bool] = [False, True, True, True, True, True, True, False] #0x7E
    ESCAPE_PATTERN: List[bool] = [False, True, True, True, True, True, False, True] #0x7D

    in_payload = False  
    current_payload = []
    decoded_frames=[]
    
    i = 0
    while i + 8 <= len(bitstream): # Process while at least one byte remains
        current_byte = bitstream[i:i + 8] 

        bytes_to_advance = 8  

        # FLAG byte marks frame start or end
        if current_byte == FLAG_PATTERN:
            if not in_payload: # Start of frame
                in_payload = True 

            else:               # End of frame
                in_payload = False
                current_payload = remove_8bit_padding(current_payload) # Remove 8-bit padding
                decoded_frames.append(current_payload)
                current_payload = []  

        # ESCAPE sequence: next byte is literal
        elif in_payload and current_byte == ESCAPE_PATTERN:
            if i + 16 <= len(bitstream):
                current_payload.extend(bitstream[i + 8:i + 16]) # Add escaped byte (at index i + 8)
                bytes_to_advance = 16 # Advance 8 from ESCAPE + 8 from escaped byte   
            else:
                # Incomplete ESCAPE. Finish reading
                bytes_to_advance = len(bitstream) # Advance to exit 

        elif in_payload: 
            current_payload.extend(current_byte) 

        # Not a FLAG and not inside a frame: slide forward one bit
        else:  
            bytes_to_advance = 1    

        i += bytes_to_advance 

    return decoded_frames
