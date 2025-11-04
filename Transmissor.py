import socket
import time
import numpy as np
import numpy.typing as npt

def send_server(message: npt.NDArray, *, 
                host: str = '127.0.0.1',
                port: int = 60000,
                max_atts: int = 3):

    # Uses numpy's internal functionality to transform an array
    # of samples into bytes, the proper type for sockets.
    byte_array: bytes = message.tobytes()

    # Small protocol for variable data length
    # ! - network byte order
    # I - pack int as unsigned int
    # This protocol becomes problematic at sockets >4GB
    # Unsigned integers take up 4 bytes
    import struct
    msg_len = struct.pack('!I', len(byte_array))
    
    # Does max_atts attempts of sending message. On repeated fails, aborts.
    for attempt in range(max_atts):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.connect((host, port))                 # Sets up server connection
                # utils.logmsg("Connection established")
                s.sendall(msg_len + byte_array)         # Send message size + message
                # utils.logmsg("Data sent")
                return True
        except ConnectionRefusedError:
            # utils.logmsg(f"Attempt {attempt + 1} out of {max_atts} failed. Attempting again...")
            time.sleep(3)
    # utils.logmsg("Server inaccessible.")
    return False


if __name__ == "__main__":
    # note: this is for debug testing. never run this file standalone!!

    import utils
    import CamadaFisica as cf
    import CamadaEnlace as ce

    message = "my final message. goodbye."
    message = utils.string_to_bitstream(message)
    message = ce.parity_insert(message)
    message = cf.ASK_modulation(message)
    message = utils.samples_addnoise(message, spread=700)

    if send_server(message):
        utils.logmsg("Connected, message sent!")
    else:
        utils.logmsg("Failed connection!")
