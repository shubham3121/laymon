import matplotlib.pyplot as plt
from .interfaces import Display


class FeatureMapDisplay(Display):
    """
    A class for defining methods for displaying the parameters being monitored by a FeatureMapObserver.
    """

    def __init__(self):
        """
        Initialize the figures and subplots.
        """
        self._figure, self._subplots, self._parameters, self.title = (None, [], None, None)

    def display_params(self, activation, max_subplots=5):
        """Method for updating the subplots and figure with the new parameters (activation maps)."""

        num_of_subplots = min(activation.size(0), max_subplots)

        # If this method is called for the first time, then create a figure and respective subplots.
        if self._figure is None:
            self._figure, self._subplots = plt.subplots(num_of_subplots)
            self._figure.suptitle(self.title, fontsize=12)
            self._figure.show()

        # Overwrite and update each subplot against the respective activation parameters.
        for idx in range(num_of_subplots):
            self._subplots[idx].imshow(activation[idx])
        self._figure.canvas.draw()  # Overwrite the figure
        self._figure.show()  # Display the figure

    def _show(self):
        activations = self._parameters
        for activation in activations:
            self.display_params(activation=activation)

    def update_display(self, parameters, display_title):
        """
        Updates the display with the new parameters
        :param parameters: Tensor (activation map params)
        :param display_title: Title of the figure
        """

        # Update the parameters with the new values.
        self._parameters = parameters
        self.title = display_title
        self._show()  # Call the method to update the figures with the new parameters
