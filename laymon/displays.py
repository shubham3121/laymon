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

    def display_params(self):
        """Method for updating the subplots and figure with the new parameters (activation maps)."""

        # Get the new activation parameters
        activation = self._parameters

        # If this method is called for the first time, then create a figure and respective subplots.
        if self._figure is None:
            self._figure, self._subplots = plt.subplots(activation.size(0))
            self._figure.suptitle(self.title, fontsize=12)
            self._figure.show()

        # Overwrite and update each subplot against the respective activation parameters.
        for idx in range(activation.size(0)):
            self._subplots[idx].imshow(activation[idx], interpolation="None")
        self._figure.canvas.draw()  # Overwrite the figure
        self._figure.show()  # Display the figure

    def update_display(self, parameters, display_title):
        """
        Updates the display with the new parameters
        :param parameters: Tensor (activation map params)
        :param display_title: Title of the figure
        """

        # Update the parameters with the new values.
        self._parameters = parameters
        self.title = display_title
        self.display_params()  # Call the method to update figure with the new parameters
