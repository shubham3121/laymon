import warnings
from .interfaces import Monitor
from .exceptions import SingleDimensionalLayerWarning, LayerRegisterException


class ActivationMapHook:
    def __init__(self, kwargs):
        self.__dict__.update(**kwargs)


class FeatureMapMonitor(Monitor):
    def __init__(self):
        self._layer_observers = dict()

    def add_observer(self, layer_observer):
        layer = layer_observer.get_layer()
        layer_name = layer_observer.get_layer_name()
        _layer_observer = ActivationMapHook({"object": layer_observer, "parameters": None, "handler": None})
        if layer_name in self._layer_observers:
            return
        self._layer_observers[layer_name] = _layer_observer
        handler = layer.register_forward_hook(self._get_activation_map(layer_name))
        self._layer_observers[layer_name].handler = handler

    def remove_observer(self, layer_observer=None, layer_name=None):
        if not layer_name and not layer_observer:
            raise NameError("Either layer_name or observer_object must be provided.")
        elif not layer_name:
            layer_name = layer_observer.get_layer_name()
        hook = self._layer_observers.get(layer_name)
        if hook:
            hook.handler.remove()
            del self._layer_observers[layer_name]

    @staticmethod
    def _is_layer_single_dim(layer):
        return len(layer.shape[1:]) == 1

    def _get_activation_map(self, layer_name):
        def hook(model, inp, out):
            try:
                observer = self._layer_observers[layer_name]
                observer.parameters = out.detach()
            except NameError:
                raise LayerRegisterException(layer_name=layer_name)
        return hook

    def notify_observers(self):
        for observer_name, observer in self._layer_observers.items():
            if self._is_layer_single_dim(observer.parameters):
                warnings.warn(SingleDimensionalLayerWarning(observer_name))
                continue
            parameters = observer.parameters
            observer.object.update(parameters)

    def get_registered_observers(self):
        return self._layer_observers
