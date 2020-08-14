class LayerRegisterException(Exception):
    """
    Exception to indicate the observer that failed to register for generating
    feature maps.
    """

    default_message = "Failed to register the layer: "

    def __init__(self, layer_name, message=default_message):
        self.layer_name = layer_name
        self.message = message
        super(LayerRegisterException, self).__init__()

    def __str__(self):
        return f"{self.message} -> {self.layer_name}"


class SingleDimensionalLayerWarning(Warning):
    """
    Warning to indicate that the layer trying to be visualized is a single
    dimensional layer.
    """

    default_message = (
        "Layer is a single dimensional layer. Some of the displays might not with such a layer."
    )

    def __init__(self, layer_name, message=default_message):
        self.layer_name = layer_name
        self.message = message
        super(SingleDimensionalLayerWarning, self).__init__()

    def __str__(self):
        return f"{self.layer_name} -> {self.message}"
