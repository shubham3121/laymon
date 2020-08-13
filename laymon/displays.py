import matplotlib.pyplot as plt
from .interfaces import Display


class FeatureMapDisplay(Display):
    def __init__(self):
        self._figure, self._subplots, self._parameters, self.title = None, [], None, None

    def display(self):
        activation = self._parameters

        if self._figure is None:
            self._figure, self._subplots = plt.subplots(activation.size(0))
            self._figure.suptitle(self.title, fontsize=12)
            self._figure.show()

        for idx in range(activation.size(0)):
            self._subplots[idx].imshow(activation[idx], interpolation='None')
        self._figure.canvas.draw()
        self._figure.show()

    def update_display(self, parameters, display_title):
        self._parameters = parameters
        self.title = display_title
        self.display()
