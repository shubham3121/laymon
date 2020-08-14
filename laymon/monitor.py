import warnings
from .interfaces import Monitor
from .exceptions import SingleDimensionalLayerWarning, LayerRegisterException


class ObserverHookObject:
    """
    AN object used to store:
        1. The observer object which is being hooked
        2. Parameters being monitored
        3. Handler of the hooked layer

    """

    def __init__(self, kwargs):
        self.__dict__.update(**kwargs)


class FeatureMapMonitor(Monitor):
    """
    A monitor type class for visualizing the feature maps of a neural network.
    """

    def __init__(self):
        self._layer_observers = dict()  # Maintains a mapping of layers/observers being monitored.

    def add_observer(self, layer_observer):
        """
        1. Creates a layer observer object.
        2. Hooks the layer to capture the activation map of the layer.
        3. Adds the observer object to the list of monitored observers.

        :param layer_observer: Observer object
        """

        if not (hasattr(layer_observer, "get_layer") and hasattr(layer_observer, "get_layer_name")):
            raise AttributeError(
                "Layer Observer objects must have a get_layer and get_layer_name function defined. "
            )

        # Get the pyTorch layer object and layer from the observer.
        layer = layer_observer.get_layer()
        layer_name = layer_observer.get_layer_name()

        # If layer is already being monitored then return.
        if layer_name in self._layer_observers:
            return

        # Creates an observer hook object and store it in the list of monitored observers.
        _observer_hook_object = ObserverHookObject(
            {"object": layer_observer, "parameters": None, "handler": None}
        )
        self._layer_observers[layer_name] = _observer_hook_object

        # Create a forward hook to capture the activation map for that layer.
        handler = layer.register_forward_hook(self._get_activation_map(layer_name))

        # Store the handler of the hook.
        self._layer_observers[layer_name].handler = handler

    def remove_observer(self, layer_observer=None, layer_name=None):
        """
        If the layer observer/layer name is in the list of monitored observers:
            1. Unhook the layer.
            2. Remove the layer observer from the list of monitored observers.

        :param layer_observer: Layer Observer.
        :param layer_name: Name of the layer being monitored.
        :return: True/False based on whether the observer by deleted or not.
        """
        # Either layer name or the layer observer must be provided.
        if not layer_name and not layer_observer:
            raise NameError("Either layer_name or observer_object must be provided.")
        # If layer name is not specified then extract the layer name from the observer object.
        elif not layer_name:
            layer_name = layer_observer.get_layer_name()

        # Get the hook object from the layer name.
        hook = self._layer_observers.get(layer_name, None)

        # If the hook was present for the layer name, then remove the unhook its handler and
        # delete the layer observer from the list of observers being monitored.
        if hook:
            hook.handler.remove()
            del self._layer_observers[layer_name]
            return True

        return False  # Return false if layer is not present

    @staticmethod
    def _is_layer_single_dim(layer):
        """Checks if the layer is a single dimensional layer"""
        return len(layer.shape[1:]) == 1

    def _get_activation_map(self, layer_name):
        """Hooks the layer to capture activation maps for the given layer and return the handler to the hook"""

        def hook(model, inp, out):
            try:
                observer = self._layer_observers[layer_name]
                observer.parameters = out.detach()
            except NameError:
                raise LayerRegisterException(
                    layer_name=layer_name
                )  # Raise an error if the layer fails to register

        return hook

    def notify_observers(self):
        """Updates all the observers being monitored with the new parameters"""

        for observer_name, observer in self._layer_observers.items():
            if self._is_layer_single_dim(observer.parameters):
                # If layer is a single dimensional layer, then raise a warning as an image needs
                # to be at least of two dimensions in order to be plotted on a graph.
                warnings.warn(SingleDimensionalLayerWarning(observer_name))
                continue
            # Retrieve the new parameters for an observer and
            # update the observers object with the new parameters.
            parameters = observer.parameters
            observer.object.update(parameters)

    def get_registered_observers(self):
        """Returns the list of observers being monitored."""
        return self._layer_observers
