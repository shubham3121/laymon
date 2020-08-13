import torch.nn as nn
from .monitor import FeatureMapMonitor
from .observers import FeatureMapObserverFactory


class FeatureMapMonitoring(object):
    def __init__(self):
        self.observer_factory = FeatureMapObserverFactory()
        self.monitor = FeatureMapMonitor()

    def add_layer(self, layer, layer_name):

        if not layer_name:
            raise NameError("Specify the name of the layer to be monitored.")

        if not issubclass(layer.__class__, nn.Module):
            raise ReferenceError("Layer should be a subclass of nn.Module")
        return self._add_layer(layer=layer, layer_name=layer_name)

    def _add_layer(self, layer, layer_name):
        layer_observer = self.observer_factory.create(layer=layer, layer_name=layer_name)
        self.monitor.add_observer(layer_observer=layer_observer)
        return layer_observer

    def _remove_layer(self, layer_name):
        self.monitor.remove_observer(layer_name=layer_name)

    def remove_layer(self, layer_name):
        return self._remove_layer(layer_name=layer_name)

    def add_model(self, model):
        if not isinstance(model, nn.Module):
            raise AttributeError("Model should be an instance of nn.Module")
        for layer_name, layer in model.named_children():
            self.add_layer(layer=layer, layer_name=layer_name)

    def start(self):
        self.monitor.notify_observers()
