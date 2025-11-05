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
