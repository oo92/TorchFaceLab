import sys, ctypes, os, multiprocessing, json, time, torch
from pathlib import Path
from core.interact import interact as io

class Device(object):
    def __init__(self, index, pytorch_dev_type, name, total_mem, free_mem):
        self.index = index
        self.pytorch_dev_type = pytorch_dev_type
        self.name = name
        
        self.total_mem = total_mem
        self.total_mem_gb = total_mem / 1024**3
        self.free_mem = free_mem
        self.free_mem_gb = free_mem / 1024**3

    def __str__(self):
        return f"[{self.index}]:[{self.name}][{self.free_mem_gb:.3}/{self.total_mem_gb :.3}]"

class Devices(object):
    all_devices = None

    def __init__(self, devices):
        self.devices = devices

    def __len__(self):
        return len(self.devices)

    def __getitem__(self, key):
        result = self.devices[key]
        if isinstance(key, slice):
            return Devices(result)
        return result

    def __iter__(self):
        for device in self.devices:
            yield device

    def get_best_device(self):
        result = None
        idx_mem = 0
        for device in self.devices:
            mem = device.total_mem
            if mem > idx_mem:
                result = device
                idx_mem = mem
        return result

    def get_worst_device(self):
        result = None
        idx_mem = sys.maxsize
        for device in self.devices:
            mem = device.total_mem
            if mem < idx_mem:
                result = device
                idx_mem = mem
        return result

    def get_device_by_index(self, idx):
        for device in self.devices:
            if device.index == idx:
                return device
        return None

    def get_devices_from_index_list(self, idx_list):
        result = []
        for device in self.devices:
            if device.index in idx_list:
                result += [device]
        return Devices(result)

    def get_equal_devices(self, device):
        device_name = device.name
        result = []
        for device in self.devices:
            if device.name == device_name:
                result.append(device)
        return Devices(result)

    def get_devices_at_least_mem(self, totalmemsize_gb):
        result = []
        for device in self.devices:
            if device.total_mem >= totalmemsize_gb * (1024**3):
                result.append(device)
        return Devices(result)

    @staticmethod
    def initialize_main_env():
        if int(os.environ.get("NN_DEVICES_INITIALIZED", 0)) != 0:
            return
            
        visible_devices = []
        
        device_count = torch.cuda.device_count()
        for i in range(device_count):
            device_name = torch.cuda.get_device_name(i)
            device_type = 'CUDA'  # PyTorch primarily supports CUDA for GPUs
            total_mem = torch.cuda.get_device_properties(i).total_memory
            free_mem = total_mem  # PyTorch does not provide a straightforward way to get free memory.
            visible_devices.append((device_type, device_name, total_mem))
        
        os.environ['NN_DEVICES_INITIALIZED'] = '1'
        os.environ['NN_DEVICES_COUNT'] = str(len(visible_devices))
        
        for i, (dev_type, name, total_mem) in enumerate(visible_devices):
            os.environ[f'NN_DEVICE_{i}_PYTORCH_DEV_TYPE'] = dev_type
            os.environ[f'NN_DEVICE_{i}_NAME'] = name
            os.environ[f'NN_DEVICE_{i}_TOTAL_MEM'] = str(total_mem)
            os.environ[f'NN_DEVICE_{i}_FREE_MEM'] = str(total_mem)

    @staticmethod
    def getDevices():
        if Devices.all_devices is None:
            if int(os.environ.get("NN_DEVICES_INITIALIZED", 0)) != 1:
                raise Exception("nn devices are not initialized. Run initialize_main_env() in main process.")
            devices = []
            for i in range(int(os.environ['NN_DEVICES_COUNT'])):
                devices.append(Device(index=i,
                                      pytorch_dev_type=os.environ[f'NN_DEVICE_{i}_PYTORCH_DEV_TYPE'],
                                      name=os.environ[f'NN_DEVICE_{i}_NAME'],
                                      total_mem=int(os.environ[f'NN_DEVICE_{i}_TOTAL_MEM']),
                                      free_mem=int(os.environ[f'NN_DEVICE_{i}_FREE_MEM'])))
            Devices.all_devices = Devices(devices)

        return Devices.all_devices
