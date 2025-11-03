import socket
import numpy as np
import numpy.typing as npt

def start_server(*,
                host: str = '127.0.0.1',
                port: int = 60000):
    
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        # Sets up server connection
        s.bind((host, port))
        s.listen()
        print(f"Awaiting connection on {host}:{port}...")
        # utils.logmsg(f"Awaiting connection on {host}:{port}...")
        conn, addr = s.accept()

        with conn:
            # utils.logmsg(f"Connected by {addr}")
            
            # Small protocol for message length size
            # Unsigned integers take up 4 bytes
            import struct
            length_buf = recvall(conn, 4)
            length_msg, = struct.unpack('!I', length_buf)
            # utils.logmsg(f"{length_msg}")
            received_data: bytes = recvall(conn, length_msg)
            
            # utils.logmsg(f"Received data: {received_data}")

            # Reconstructs the received bytes into an npt.NDArray using
            # internal numpy methods. This should result in a list[float]
            byte_to_bit: npt.NDArray = np.frombuffer(received_data)
            
            # Returns the ready-to-use NDArray downstream
            return byte_to_bit

def recvall(sock: socket, count: int) -> bytes:
    buf = b''
    while count:
        newbuf = sock.recv(count)           # Try to get whole buffer
        if not newbuf: return None          # Receiving the message failed
        buf += newbuf
        count -= len(newbuf)                # In case less than expected comes
    return buf



if __name__ == "__main__":
    # note: this is for debug testing. never run this file standalone!!

    import utils
    import CamadaFisica as cf
    import CamadaEnlace as ce

    message = start_server()
    message = cf.ASK_demodulation(message)
    error_found = ce.parity_check(message)
    message = utils.bitstream_to_string(message[:-1])

    with open("test.txt", "a") as f:
        f.write(str(not error_found))
        f.write("\n")
        f.write(message)
        f.write("\n")