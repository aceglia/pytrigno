import socket
import struct
import threading
from queue import Queue
import time

import numpy as np
from .enums import AvantiSensor, LegacySensor
from .sensor import Sensor, Type


BYTES_PER_CHANNEL = 4
CMD_TERM = '\r\n\r\n'
EMG_SAMPLE_RATE = 2000
AUX_SAMPLE_RATE = 148.148

class TrignoSDKClient:
    def __init__(self, host='127.0.0.1', cmd_port=50040, timeout=2.0, fast_mode=False, buffer_size=1000):
        self.buffer_size = buffer_size
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
        self._comm_socket = socket.create_connection(
            (self.host, self.cmd_port), self.timeout)
        self.send_command("BACKWARDS COMPATIBILITY OFF")
        self.initiate_data_connection()
        self.initialize_sensors()

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
        self.all_events = {"avanti_emg": threading.Event(), 
                           "avanti_aux": threading.Event(),
                           "legacy_emg": threading.Event(),
                           "legacy_aux": threading.Event(),
                           }

    def initialize_sensors(self):
        """Initialize all sensors."""
        self.sensors = [Sensor(i, self, self.buffer_size) for i in range(1, 16)]
        self._get_which_thread_to_run()
        
    def _get_which_thread_to_run(self):
        is_avanti = [False, False] # EMG and AUX
        is_legacy = [False, False] # EMG and AUX
        for sensor in self.sensors:
            if not sensor.is_paired:
                continue
            avanti = True if sensor.type == Type.Avanti or sensor.type == Type.AvantiGoniometer else False
            is_avanti[0] = avanti and sensor.nb_emg_channels if not is_avanti[0] else True
            is_avanti[1] = avanti and sensor.nb_aux_channels if not is_avanti[1] else True
            is_legacy[0] = sensor.type == Type.Legacy and sensor.nb_emg_channels if not is_legacy[0] else True
            is_legacy[1] = sensor.type == Type.Legacy and sensor.nb_aux_channels if not is_legacy[1] else True
        self._threads_to_run = {"avanti_emg": is_avanti[0], 
                                "avanti_aux": is_avanti[1], 
                                "legacy_emg": is_legacy[0], 
                                "legacy_aux": is_legacy[1]}

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
        time.sleep(0.3)
        try:
            response = self._comm_socket.recv(1024)
            return response.decode('ascii').strip()
        except socket.timeout:
            return None
        
    def read(self, connection, buffer_size, n_channels):
        l = 0
        packet = bytes()
        while l < buffer_size:
            packet += connection.recv(buffer_size-l)
            l = len(packet)
        data = np.asarray(struct.unpack('<'+'f'*(buffer_size // 4), packet))
        data = np.transpose(data.reshape((-1, n_channels)))
        return data

    def start_streaming(self):
        is_started = self.send_command("START") == "OK"
        if not is_started and not self.fast_mode:
            raise RuntimeError("Streaming not started.")
        self._launch_threads()

    def buffer_size_for_type(self, name):
        if "emg" in name:
            n_samples = self.get_max_emg_samples()
            n_channel = 16
            buffer_size = n_channel * n_samples * BYTES_PER_CHANNEL
        elif "aux" in name:
            n_samples = self.get_max_aux_samples()
            n_channel = 144 if "avanti" in name.lower() else 48
            buffer_size = n_channel * n_samples * BYTES_PER_CHANNEL
        else:
            raise RuntimeError("Invalid sensor type.")
        return buffer_size, n_channel, n_samples

    def _launch_one_thread(self, socket_tmp, name, data_queue, event):
        buffer_size, n_chanels, n_samples = self.buffer_size_for_type(name)
        def _thread_func():
            count = 0
            while True:
                data = self.read(socket_tmp, buffer_size, n_chanels)
                data_queue.queue.clear()
                data_queue.put_nowait((data, count))
                event.set()
                count += n_samples
        thread = threading.Thread(target=_thread_func, name=name)
        thread.start()

    def _launch_threads(self):
        all_soc, all_q, all_ev = self.all_socket, self.all_queue, self.all_events
        _ = [self._launch_one_thread(all_soc[n], n, all_q[n], all_ev[n]) for n in self._threads_to_run.keys() if self._threads_to_run[n]]
   
        def _main_thread_func():
            while True:
                for name in self.all_socket.keys():
                    self.all_events[name].wait()
                    self.all_events[name].clear()
                self._set_all_data()

        main_thread = threading.Thread(target=_main_thread_func, name='main')
        main_thread.start()
        
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

    def get_sensor_idx(self, n):
        return int(self.send_command(f"SENSOR {n} STARTINDEX?"))
    
    def get_list_sensors_and_idx(self):
        sensors = []
        for i in range(1, 16):
            if self.is_sensor_paired(i):
                sensors.append([self.get_sensor_info(i, 'TYPE'), self.get_sensor_idx(i)])
            else:
                sensors.append([None, None])
        return sensors
    
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
            data = self.avanti_emg_queue.get_nowait()
            for sensor in self.sensors:
                if sensor.type == Type.Avanti or sensor.type == Type.AvantiGogniometer:
                    sensor.update_emg_buffer(data[0][sensor.emg_range[0]:sensor.emg_range[1], :], data[1])
        except:
            return
    
    def _set_avanti_aux_data(self):
        try:
            data = self.avanti_aux_queue.get_nowait()
            for sensor in self.sensors:
                if sensor.type == Type.Avanti or sensor.type == Type.AvantiGogniometer:
                    sensor.update_aux_buffer(data[0][sensor.aux_range[0]:sensor.aux_range[1], :], data[1])
        except:
            return
        
    def _set_legacy_emg_data(self):
        try:
            data = self.legacy_emg_queue.get_nowait()
            for sensor in self.sensors:
                if sensor.type == Type.Legacy:
                    sensor.update_emg_buffer(data[0][sensor.emg_range[0]:sensor.emg_range[1], :], data[1])
        except:
            return
        
    def _set_legacy_aux_data(self):
        try:
            data = self.legacy_aux_queue.get_nowait()
            for sensor in self.sensors:
                if sensor.type == Type.Legacy:
                    sensor.update_aux_buffer(data[0][sensor.aux_range[0]:sensor.aux_range[1], :], data[1])
        except:
            return
    
    def _set_all_data(self):
        self._set_avanti_emg_data(),
        self._set_avanti_aux_data(),
        self._set_legacy_emg_data(),
        self._set_legacy_aux_data(),
        
            


