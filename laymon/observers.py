from .interfaces import Observer, ObserverFactory
from .displays import FeatureMapDisplay


class FeatureMapObserver(Observer):
    def __init__(self, layer, layer_name, update_display):
        self._layer = layer
        self._layer_name = layer_name
        if not callable(update_display):
            raise TypeError("update display method should be callable.")
        self._update_display = update_display
        self._description = f"Observer -> {self._layer_name}"

    def update(self, parameters):
        activations = parameters.squeeze()
        for activation in activations:
            self._update_display(parameters=activation, display_title=self._layer_name)

    def get_layer_name(self):
        return self._layer_name

    def get_layer(self):
        return self._layer


class FeatureMapObserverFactory(ObserverFactory):
    display_object = FeatureMapDisplay

    def create(self, layer, layer_name):
        return FeatureMapObserver(
            layer=layer, layer_name=layer_name, update_display=self.display_object().update_display
        )
