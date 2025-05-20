import socket
import struct
import threading
from queue import Queue
import time

import numpy as np
from .enums import AvantiSensor, LegacySensor, SensorType


BYTES_PER_CHANNEL = 4
CMD_TERM = '\r\n\r\n'
EMG_SAMPLE_RATE = 2000
AUX_SAMPLE_RATE = 148.148

class TrignoSDKClient:
    def __init__(self, host='127.0.0.1', cmd_port=50040, timeout=2.0, fast_mode=False):
        self.host = host
        self.cmd_port = cmd_port
        self.timeout = timeout
        self._comm_socket = None
        self.fast_mode = fast_mode
        self.avanti_emg_socket = None
        self.avanti_aux_socket = None
        self.legacy_emg_socket = None
        self.legacy_aux_socket = None
        self.all_socket = None
        self._last_data = {"avanti_emg": None, 
                          "avanti_aux": None,
                          "legacy_emg": None,
                          "legacy_aux": None,
                          }

        self.avanti_emg_queue = Queue()
        self.avanti_aux_queue = Queue()
        self.legacy_emg_queue = Queue()
        self.legacy_aux_queue = Queue()
        self.all_queue = {"avanti_emg": self.avanti_emg_queue, 
                          "avanti_aux": self.avanti_aux_queue,
                          "legacy_emg": self.legacy_emg_queue,
                          "legacy_aux": self.legacy_aux_queue,
                          }

        if self.fast_mode:
            print("Warning: Fast mode enabled. Responses will not be waited for.")

        self.connect()

    def connect(self):
        """Establish connection to Trigno SDK command port."""
        # self._comm_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        # self._comm_socket.settimeout(self.timeout)
        # self._comm_socket.connect((self.host, self.cmd_port))
        self._comm_socket = socket.create_connection(
            (self.host, self.cmd_port), self.timeout)
        self.send_command("BACKWARDS COMPATIBILITY OFF")
        self.initiate_data_connection()

        # Try to clear buffer (flush)
        try:
            _ = self._comm_socket.recv(1024)
        except socket.timeout:
            pass

    def initiate_data_connection(self):
        self.avanti_emg_socket = self._connect_to_socket(AvantiSensor().emg_port)
        self.avanti_aux_socket = self._connect_to_socket(AvantiSensor().aux_port)
        self.legacy_emg_socket = self._connect_to_socket(LegacySensor().emg_port)
        self.legacy_aux_socket = self._connect_to_socket(LegacySensor().aux_port)
        self.all_socket = {"avanti_emg": self.avanti_emg_socket, 
                           "avanti_aux": self.avanti_aux_socket,
                           "legacy_emg": self.legacy_emg_socket,
                           "legacy_aux": self.legacy_aux_socket,
                           }

    def _connect_to_socket(self, port):
        _data_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        _data_socket.connect((self.host, port))
        return _data_socket

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

        full_command = f"{command}\r\n\r\n"
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
            raise RuntimeError("Streaming not started")
        
    def read(self, connection, buffer_size, n_channels):
        l = 0
        packet = bytes()
        while l < buffer_size:
            packet += connection.recv(buffer_size-l)
            l = len(packet)
        data = np.asarray(struct.unpack('<'+'f'*(buffer_size // 4), packet))
        data = np.transpose(data.reshape((-1, n_channels)))
        return data

    def start_streaming(self, type: SensorType=SensorType.ALL):
        is_started = self.send_command("START") == "OK"
        if not is_started and not self.fast_mode:
            raise RuntimeError("Streaming not started.")
        self._launch_threads(type)

    def buffer_size_for_type(self, name):
        if "emg" in name:
            n_samples = self.get_max_emg_samples()
            n_channel = 16
            buffer_size = n_channel * n_samples * BYTES_PER_CHANNEL
        elif "aux" in name:
            n_samples = self.get_max_aux_samples()
            n_channel = 144
            buffer_size = n_channel * n_samples * BYTES_PER_CHANNEL
        else:
            raise RuntimeError("Invalid sensor type.")
        return buffer_size, n_channel, n_samples

    def _launch_one_thread(self, socket_tmp, name, data_queue):
        buffer_size, n_chanels, n_samples = self.buffer_size_for_type(name)
        def _thread_func():
            count = 0
            while True:
                data = self.read(socket_tmp, buffer_size, n_chanels)
                data_queue.queue.clear()
                data_queue.put_nowait((data, count))
                count += n_samples
        thread = threading.Thread(target=_thread_func, name=name)
        thread.start()

    def _launch_threads(self, type):
        if type == SensorType.ALL:
            _ = [self._launch_one_thread(self.all_socket[name], name, self.all_queue[name]) for name in self.all_socket.keys()]
        elif type == SensorType.Avanti_all:
            _ = [self._launch_one_thread(self.all_socket[name], name, self.all_queue[name]) for name in self.all_socket.keys() if "avanti" in name]
        elif type == SensorType.Avanti_emg:
            _ = [self._launch_one_thread(self.all_socket[name], name, self.all_queue[name]) for name in self.all_socket.keys() if "avanti" in name and "emg" in name]
        elif type == SensorType.Avanti_aux:
            _ = [self._launch_one_thread(self.all_socket[name], name, self.all_queue[name]) for name in self.all_socket.keys() if "avanti" in name and "aux" in name]
        elif type == SensorType.Legacy_all:
            _ = [self._launch_one_thread(self.all_socket[name], name, self.all_queue[name]) for name in self.all_socket.keys() if "legacy" in name]
        elif type == SensorType.Legacy_emg:
            _ = [self._launch_one_thread(self.all_socket[name], name, self.all_queue[name]) for name in self.all_socket.keys() if "legacy" in name and "emg" in name]
        elif type == SensorType.Legacy_aux:
            _ = [self._launch_one_thread(self.all_socket[name], name, self.all_queue[name]) for name in self.all_socket.keys() if "legacy" in name and "aux" in name]
            
    def stop_streaming(self):
        return self.send_command("STOP")
    
    def disconnect(self):
        self.stop_streaming()
        _ = [socket.close() for socket in self.all_socket.values()]
        self._comm_socket.close()
    
    def get_emg_streaming_rate(self):
        return int(self.send_command("MAX SAMPLES EMG")) * 0.0135
    
    def get_aux_streaming_rate(self):
        return int(self.send_command("MAX SAMPLES AUX")) * 0.0135

    def get_max_emg_samples(self):
        return int(self.send_command("MAX SAMPLES EMG"))
    
    def get_max_aux_samples(self):
        return int(self.send_command("MAX SAMPLES AUX"))
    
    def get_aux_streaming_rate(self):
        return int(self.send_command("MAX SAMPLES AUX")) * 0.0135

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
    
    def get_sensor_emgchannel(self, n):
        if not self.is_sensor_paired(n):
            return 0
        return int(self.send_command(f"SENSOR {n} EMGCHANNELCOUNT?"))
    
    def get_sensor_auxchannel(self, n):
        if not self.is_sensor_paired(n):
            return 0
        return int(self.send_command(f"SENSOR {n} AUXCHANNELCOUNT?"))
    
    def is_sensor_paired(self, n):
        return self.send_command(f"SENSOR {n} PAIRED?") == "YES"

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
    
    def get_number_emgchannel(self):
        nb_channel = 0
        for i in range(1, 16):
            nb_channel += self.get_sensor_emgchannel(i)
        return nb_channel
    
    def get_number_auxchannel(self):
        nb_channel = 0
        for i in range(1, 16):
            nb_channel += self.get_sensor_auxchannel(i)
        return nb_channel

    def _set_avanti_emg_data(self):
        try:
            self._last_data["avanti_emg"] = self.avanti_emg_queue.get_nowait()
        except:
            return
    
    def _set_avanti_aux_data(self):
        try:
            self._last_data["avanti_aux"] = self.avanti_aux_queue.get_nowait()
        except:
            return
        
    def _set_legacy_emg_data(self):
        try:
            self._last_data["legacy_emg"] = self.legacy_emg_queue.get_nowait()
        except:
            return
        
    def _set_legacy_aux_data(self):
        try:
            self._last_data["legacy_aux"] = self.legacy_aux_queue.get_nowait()
        except:
            return
    
    def _set_all_data(self):
        self._set_avanti_emg_data(),
        self._set_avanti_aux_data(),
        self._set_legacy_emg_data(),
        self._set_legacy_aux_data(),
        

    def get_avanti_emg_data(self):
        self._set_avanti_emg_data()
        return self._last_data["avanti_emg"]

    
    def get_avanti_aux_data(self):
        self._set_avanti_aux_data()
        return self._last_data["avanti_aux"]

        
    def get_legacy_emg_data(self):
        self._set_legacy_emg_data()
        return self._last_data["legacy_emg"]

    def get_legacy_aux_data(self):
        self._set_legacy_aux_data()
        return self._last_data["legacy_aux"]

    def get_all_data(self):
        return {
            "avanti_emg": self.get_avanti_emg_data(),
            "avanti_aux": self.get_avanti_aux_data(),
            "legacy_emg": self.get_legacy_emg_data(),
            "legacy_aux": self.get_legacy_aux_data(),
        }
            


