from .interfaces import Observer, ObserverFactory
from .displays import FeatureMapDisplay


class FeatureMapObserver(Observer):
    """
    An class used to create observers that are used to monitor the feature maps of the given layer.
    """

    def __init__(self, layer, layer_name, update_display):
        """
        Initialises and creates a new observer object to track the feature
        maps of the specified PyTorch layer.
        :param layer: pyTorch layer
        :param layer_name: string
        :param update_display: method used to update the display of the observer object
        """
        self._layer = layer
        self._layer_name = layer_name

        # update_display needs to be a callable method.
        if not callable(update_display):
            raise TypeError("update display method should be callable.")
        self._update_display = update_display

        # Sets the description of the observer object.
        self._description = f"Observer -> {self._layer_name}"

    def update(self, parameters):
        """
        Update the display attached to the observer with the new parameters/activations.
        :param parameters: Tensor
        :return: None
        """
        # Update the display of the observer with the new parameters.
        activations = parameters.squeeze()
        for activation in activations:
            self._update_display(parameters=activation, display_title=self._layer_name)

    def get_layer_name(self):
        """Returns the layer name being observed."""
        return self._layer_name

    def get_layer(self):
        """Returns the layer object being observed."""
        return self._layer


class FeatureMapObserverFactory(ObserverFactory):
    """A factory type class to create a FeatureMapObserver for the given layer"""

    display_object = FeatureMapDisplay

    def create(self, layer, layer_name):
        """
        Create a FeatureMapObserver for the given layer and attaches the display function
        for the layer being monitored.
        :param layer:
        :param layer_name:
        :return:
        """
        return FeatureMapObserver(
            layer=layer, layer_name=layer_name, update_display=self.display_object().update_display
        )
