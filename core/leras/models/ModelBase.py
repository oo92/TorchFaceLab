import torch
import numpy as np

from core.interact import interact as io
import torch.nn as nn
import core.leras.ops
import core.leras.layers.LayerBase as LayerBase
import core.leras.initializers
import core.leras.optimizers
import core.leras.models.ModelBase as ModelBase
import core.leras.archis

class ModelBase(nn.Module):
    def __init__(self, *args, name=None, **kwargs):
        super().__init__(name=name)
        self.layers = []
        self.layers_by_name = {}
        self.built = False
        self.args = args
        self.kwargs = kwargs
        self.run_placeholders = None

    def _build_sub(self, layer, name):
        # Simplified _build_sub for PyTorch since there's no direct equivalent of tf.variable_scope
        # Just handling naming and adding layers
        if isinstance(layer, (LayerBase, ModelBase)):
            layer.name = name
            self.layers.append(layer)
            self.layers_by_name[layer.name] = layer

    def xor_list(self, lst1, lst2):
        return [value for value in lst1 + lst2 if (value not in lst1) or (value not in lst2)]

    def build(self):
        generator = self.on_build(*self.args, **self.kwargs)
        if not isinstance(generator, type(iter([]))):
            generator = None

        if generator is not None:
            for _ in generator:
                pass

        current_vars = list(vars(self))
        for name in current_vars:
            self._build_sub(vars(self)[name], name)

        self.built = True

    def get_weights(self):
        if not self.built:
            self.build()
        return [layer.parameters() for layer in self.layers]

    def get_layer_by_name(self, name):
        return self.layers_by_name.get(name, None)

    def get_layers(self):
        if not self.built:
            self.build()
        return self.layers

    # Intended to be overridden by child classes
    def on_build(self, *args, **kwargs):
        pass

    # Intended to be overridden by child classes
    def forward(self, *args, **kwargs):
        pass

    def __call__(self, *args, **kwargs):
        if not self.built:
            self.build()
        return self.forward(*args, **kwargs)

    def build_for_run(self, shapes_list):
        if not isinstance(shapes_list, list):
            raise ValueError("shapes_list must be a list.")
        self.run_placeholders = [torch.zeros(*sh) for _, sh in shapes_list]
        self.run_output = self.__call__(self.run_placeholders)

    def run(self, inputs):
        if self.run_placeholders is None:
            raise Exception("Model didn't build for run.")

        if len(inputs) != len(self.run_placeholders):
            raise ValueError("len(inputs) != self.run_placeholders")

        # Direct forward pass in PyTorch
        with torch.no_grad():
            return self.forward(*inputs)

    # Similar summary function as in original code
    def summary(self):
        layers = self.get_layers()
        layers_names = []
        layers_params = []

        max_len_str = 0
        max_len_param_str = 0
        delim_str = "-"

        total_params = 0

        # Get layers names and str length for delim
        for layer in layers:
            name = str(layer)
            layers_names.append(name.capitalize())
            max_len_str = max(max_len_str, len(name))

        # Get params for each layer
        for layer in layers:
            params_count = sum(p.numel() for p in layer.parameters())
            layers_params.append(params_count)
            max_len_param_str = max(max_len_param_str, len(str(params_count)))

        total_params = sum(layers_params)

        # Set delim
        delim_str = delim_str * (max_len_str + max_len_param_str + 3)
        output = "\n" + delim_str + "\n"

        # Format model name str
        model_name_str = "| " + self.name.capitalize()
        while len(model_name_str) < len(delim_str) - 2:
            model_name_str += " "
        model_name_str += " |"

        output += model_name_str + "\n"
        output += delim_str + "\n"

        # Format layers table
        for layer_name, layer_param in zip(layers_names, layers_params):
            output += delim_str + "\n"
            formatted_name = layer_name.ljust(max_len_str)
            formatted_param = str(layer_param).rjust(max_len_param_str)
            output += f"| {formatted_name} | {formatted_param} |\n"

        output += delim_str + "\n"

        # Format sum of params
        total_params_str = "| Total params count: " + str(total_params)
        while len(total_params_str) < len(delim_str) - 2:
            total_params_str += " "
        total_params_str += " |"

        output += total_params_str + "\n"
        output += delim_str + "\n"

        io.log_info(output)


nn.ModelBase = ModelBase
