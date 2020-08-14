import torch.nn as nn
from .monitor import FeatureMapMonitor
from .observers import FeatureMapObserverFactory


class FeatureMapMonitoring(object):
    """
    A wrapper class for adding a model or model layers for monitoring the feature maps
    during training of the model.
    """

    def __init__(self):
        """
        Initialises:
        1. A observer factory for creating observer for a given layer.
        2. A monitor for displaying the feature maps for the monitored layer.
        """
        self.observer_factory = FeatureMapObserverFactory()
        self.monitor = FeatureMapMonitor()

    def add_layer(self, layer, layer_name):
        """
        Adds the layer whose feature maps are to be monitored.
        :param layer: pyTorch layer
        :param layer_name: (str) name of the layer
        :return: the observer object of the layer being monitored
        """
        if not layer_name:
            raise NameError("Specify the name of the layer to be monitored.")

        # the layer to be monitored should be a subclass of nn.Module
        if not issubclass(layer.__class__, nn.Module):
            raise ReferenceError("Layer should be a subclass of nn.Module")
        return self._add_layer(layer=layer, layer_name=layer_name)

    def _add_layer(self, layer, layer_name):
        """
        Create a observer class of the layer to be monitored and adds it to the list of
        observers being monitored.
        :param layer: pyTorch layer
        :param layer_name: (str) name of the layer
        :return: Observer object of the layer
        """
        layer_observer = self.observer_factory.create(layer=layer, layer_name=layer_name)
        self.monitor.add_observer(layer_observer=layer_observer)
        return layer_observer

    def _remove_layer(self, layer_name):
        """
        Removes an observer from the list of observers being monitored.
        :param layer_name: name of the observer
        :return: True if observer was deleted, else False
        """
        self.monitor.remove_observer(layer_name=layer_name)

    def remove_layer(self, layer_name):
        """
        Remove the layer from list of layer being monitored.
        :param layer_name: str (name of the layer)
        :return: True if the layer is deleted from the list of monitored objects, else False
        """
        return self._remove_layer(layer_name=layer_name)

    def add_model(self, model):
        """
        Registers all the layers a pyTorch model whose activations maps are to monitored.
        :param model:
        :return:
        """
        if not isinstance(model, nn.Module):
            raise AttributeError("Model should be an instance of nn.Module")
        for layer_name, layer in model.named_children():
            self.add_layer(layer=layer, layer_name=layer_name)

    def start(self):
        """Starts monitoring the feature maps of the registered layers/model."""
        self.monitor.notify_observers()
