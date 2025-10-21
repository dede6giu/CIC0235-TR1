

def string_to_bitstream(message: str) -> list[bool] | None:
    """
    Transforms a string into a stream True and False values,
    respectively representing the 1 and 0 bit values. 
    Returns None if the string is not ASCII encoded.
    """

    # Checks ASCII-ness
    if not message.isascii():
        return None
    
    # Converts each character into its ASCII int, then into its binary
    temp_array: bytearray = bytearray(message, 'ascii')
    result: list[int] = []
    for c in temp_array:
        result += char_to_list8bit(c)
    return result

def char_to_list8bit(char: str) -> list[bool]:
    """
    Auxiliary funciton.
    Transforms a single character into its ASCII binary.
    Does not check if len(char) == 1. 
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
    """

    # Transforms list[bool] into a binary representation
    auxstr: str = "".join(['1' if b else '0' for b in array])

    # Converts binary -> decimal -> ASCII
    return chr(int(auxstr, 2))



if __name__ == "__main__":
    test: str = "Test a message here."

    result: list[bool] | None = string_to_bitstream(test)
    if result:
        print(bitstream_to_string(result))
    else:
        print("Incorrect message encoding! Please use only ASCII characters.")