# from pathlib import Path
# import numpy as np
# import torch
# from core.interact import interact as io
# from .device import Devices
# import core.leras.ops
# import core.leras.layers
# import core.leras.initializers
# import core.leras.optimizers
# import core.leras.models
# import core.leras.archis

# class nn():
#     current_DeviceConfig = None
#     device = None
#     data_format = "NCHW"  # Default format for PyTorch
#     conv2d_ch_axis = 1
#     conv2d_spatial_axes = [2, 3]
#     floatx = None

#     @staticmethod
#     def initialize(device_config=None, floatx="float32"):

#         if nn.device is None:
#             if device_config is None:
#                 device_config = nn.getCurrentDeviceConfig()
#             nn.setCurrentDeviceConfig(device_config)

#             if torch.cuda.is_available():
#                 io.log_info(f"Using GPU: {torch.cuda.get_device_name(0)}")
#                 nn.device = torch.device('cuda')
#             else:
#                 io.log_info("Using CPU")
#                 nn.device = torch.device('cpu')

#         if floatx == "float32":
#             floatx = torch.float32
#         elif floatx == "float16":
#             floatx = torch.float16
#         else:
#             raise ValueError(f"unsupported floatx {floatx}")
#         nn.set_floatx(floatx)

#     @staticmethod
#     def initialize_main_env():
#         Devices.initialize_main_env()

#     @staticmethod
#     def set_floatx(torch_dtype):
#         """Set default float type for all layers when dtype is None for them"""
#         nn.floatx = torch_dtype

#     # Rest of the data format functions remain unchanged since PyTorch uses NCHW by default

#     @staticmethod
#     def get4Dshape(w, h, c):
#         """Returns 4D shape based on current data_format"""
#         if nn.data_format == "NHWC":
#             return (None, h, w, c)
#         else:
#             return (None, c, h, w)

#     @staticmethod
#     def to_data_format(x, to_data_format, from_data_format):
#         if to_data_format == from_data_format:
#             return x

#         if to_data_format == "NHWC":
#             return np.transpose(x, (0, 2, 3, 1))
#         elif to_data_format == "NCHW":
#             return np.transpose(x, (0, 3, 1, 2))
#         else:
#             raise ValueError(f"unsupported to_data_format {to_data_format}")

#     @staticmethod
#     def getCurrentDeviceConfig():
#         if nn.current_DeviceConfig is None:
#             nn.current_DeviceConfig = DeviceConfig.BestGPU()
#         return nn.current_DeviceConfig

#     @staticmethod
#     def setCurrentDeviceConfig(device_config):
#         nn.current_DeviceConfig = device_config

#     @staticmethod
#     def ask_choose_device_idxs(choose_only_one=False, allow_cpu=True, suggest_best_multi_gpu=False, suggest_all_gpu=False):
#         devices = Devices.getDevices()
#         if len(devices) == 0:
#             return []

#         all_devices_indexes = [device.index for device in devices]

#         if choose_only_one:
#             suggest_best_multi_gpu = False
#             suggest_all_gpu = False

#         if suggest_all_gpu:
#             best_device_indexes = all_devices_indexes
#         elif suggest_best_multi_gpu:
#             best_device_indexes = [device.index for device in devices.get_equal_devices(devices.get_best_device()) ]
#         else:
#             best_device_indexes = [ devices.get_best_device().index ]
#         best_device_indexes = ",".join([str(x) for x in best_device_indexes])

#         io.log_info ("")
#         if choose_only_one:
#             io.log_info ("Choose one GPU idx.")
#         else:
#             io.log_info ("Choose one or several GPU idxs (separated by comma).")
#         io.log_info ("")

#         if allow_cpu:
#             io.log_info ("[CPU] : CPU")
#         for device in devices:
#             io.log_info (f"  [{device.index}] : {device.name}")

#         io.log_info ("")

#         while True:
#             try:
#                 if choose_only_one:
#                     choosed_idxs = io.input_str("Which GPU index to choose?", best_device_indexes)
#                 else:
#                     choosed_idxs = io.input_str("Which GPU indexes to choose?", best_device_indexes)

#                 if allow_cpu and choosed_idxs.lower() == "cpu":
#                     choosed_idxs = []
#                     break

#                 choosed_idxs = [ int(x) for x in choosed_idxs.split(',') ]

#                 if choose_only_one:
#                     if len(choosed_idxs) == 1:
#                         break
#                 else:
#                     if all( [idx in all_devices_indexes for idx in choosed_idxs] ):
#                         break
#             except:
#                 pass
#         io.log_info ("")

#         return choosed_idxs

#     class DeviceConfig():
#         @staticmethod
#         def ask_choose_device(*args, **kwargs):
#             return nn.DeviceConfig.GPUIndexes( nn.ask_choose_device_idxs(*args,**kwargs) )
        
#         def __init__ (self, devices=None):
#             devices = devices or []

#             if not isinstance(devices, Devices):
#                 devices = Devices(devices)

#             self.devices = devices
#             self.cpu_only = len(devices) == 0

#         @staticmethod
#         def BestGPU():
#             devices = Devices.getDevices()
#             if len(devices) == 0:
#                 return nn.DeviceConfig.CPU()

#             return nn.DeviceConfig([devices.get_best_device()])

#         @staticmethod
#         def WorstGPU():
#             devices = Devices.getDevices()
#             if len(devices) == 0:
#                 return nn.DeviceConfig.CPU()

#             return nn.DeviceConfig([devices.get_worst_device()])

#         @staticmethod
#         def GPUIndexes(indexes):
#             if len(indexes) != 0:
#                 devices = Devices.getDevices().get_devices_from_index_list(indexes)
#             else:
#                 devices = []

#             return nn.DeviceConfig(devices)

#         @staticmethod
#         def CPU():
#             return nn.DeviceConfig([])
