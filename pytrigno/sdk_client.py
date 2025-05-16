import socket
import struct
import time

import numpy as np

BYTES_PER_CHANNEL = 4
CMD_TERM = '\r\n\r\n'

class TrignoSDKClient:
    def __init__(self, host='127.0.0.1', cmd_port=50040, data_port=5041, timeout=2.0, fast_mode=False):
        self.host = host
        self.cmd_port = cmd_port
        self.data_port = data_port
        self.timeout = timeout
        self._comm_socket = None
        self.fast_mode = fast_mode

        if self.fast_mode:
            print("Warning: Fast mode enabled. Responses will not be waited for.")

        self.connect()

    def connect(self):
        """Establish connection to Trigno SDK command port."""
        self._comm_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._comm_socket.settimeout(self.timeout)
        self._comm_socket.connect((self.host, self.cmd_port))

        # Try to clear buffer (flush)
        try:
            _ = self._comm_socket.recv(1024)
        except socket.timeout:
            pass

        # create the data socket
        self._data_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._data_socket.settimeout(self.timeout)
        self._data_socket.connect((self.host, self.data_port))


    def _bytes_per_sample(self, n_channels):
        """Return the number of bytes per sample for a given number of channels."""
        return n_channels * BYTES_PER_CHANNEL

    def buffer_size(self, n_channels, n_samples):
        """Return the size of the buffer required to store a given number of samples for a given number of channels."""
        return self._bytes_per_sample(n_channels) * n_samples

    def disconnect(self):
        """Close the socket connection."""
        if self._comm_socket:
            self._comm_socket.close()
            self._comm_socket = None

    def send_command(self, command: str) -> str:
        """
        Send a command or query to the Trigno system and return the response as a string.
        Command strings must already include any needed arguments.
        """
        if self._comm_socket is None:
            raise RuntimeError("Not connected. Call connect() first.")

        full_command = f"{command}\r\n\r"
        self._comm_socket.sendall(full_command.encode('ascii'))

        # If in fast mode, return immediately without waiting for a response
        if self.fast_mode:
            return ""
        
        # Give the server time to respond
        time.sleep(0.5)
        try:
            response = self._comm_socket.recv(1024)
            return response.decode('ascii').strip()
        except socket.timeout:
            return ""
        
    def read(self, buffer_size, n_channels):
        data = b''
        while len(data) < buffer_size:
            packet = self._data_socket.recv(buffer_size - len(data))
            if not packet:
                return None
            data += packet
        emg_values = struct.unpack('<' + 'f' * (buffer_size // 4), data)
        emg_array = np.array(emg_values).reshape(-1, n_channels)
        return emg_array

    # def read(self, num_samples):
    #     """
    #     Request a sample of data from the device.

    #     This is a blocking method, meaning it returns only once the requested
    #     number of samples are available.

    #     Parameters
    #     ----------
    #     num_samples : int
    #         Number of samples to read per channel.

    #     Returns
    #     -------
    #     data : ndarray, shape=(total_channels, num_samples)
    #         Data read from the device. Each channel is a row and each column
    #         is a point in time.
    #     """
    #     l_des = num_samples * self._min_recv_size
    #     l = 0
    #     packet = bytes()
    #     while l < l_des:
    #         packet += self.socket.recv(l_des-l)
    #         l = len(packet)
    #     data = numpy.asarray(struct.unpack('<'+'f'*self.total_channels*num_samples, packet))
    #     data = numpy.transpose(data.reshape((-1, self.total_channels)))

    #     return data

    # Example high-level wrappers for convenience
    def set_rate(self, hz=2000):
        return self.send_command(f"RATE {hz}")

    def get_rate(self):
        return self.send_command("RATE?")

    def start_streaming(self):
        return self.send_command("START")

    def stop_streaming(self):
        return self.send_command("STOP")

    def get_trigger_state(self):
        return self.send_command("TRIGGER?")

    def set_trigger(self, which='START', state='ON'):
        return self.send_command(f"TRIGGER {which} {state}")

    def get_backwards_compatibility(self):
        return self.send_command("BACKWARDS COMPATIBILITY?")

    def set_backwards_compatibility(self, state='ON'):
        return self.send_command(f"BACKWARDS COMPATIBILITY {state}")

    def get_upsampling(self):
        return self.send_command("UPSAMPLING?")

    def set_upsampling(self, state='ON'):
        return self.send_command(f"UPSAMPLE {state}")

    def get_sensor_info(self, n, info='TYPE'):
        return self.send_command(f"SENSOR {n} {info}?")

    def pair_sensor(self, n):
        return self.send_command(f"SENSOR {n} PAIR")

    def set_sensor_mode(self, n, mode):
        return self.send_command(f"SENSOR {n} SETMODE {mode}")

    def get_endianness(self):
        return self.send_command("ENDIANNESS?")

    def set_endianness(self, mode="LITTLE"):
        return self.send_command(f"ENDIAN {mode}")

    def get_base_serial(self):
        return self.send_command("BASE SERIAL?")

    def get_base_firmware(self):
        return self.send_command("BASE FIRMWARE?")

