"""
==============================================================================
Abstract Classes
==============================================================================

==============================================================================
Adding and removing hooks.
==============================================================================
model = ...
handle = model.register_forward_hook(...)
handle.remove()
"""
import abc


class Monitor:
    """
    Abstract class to monitors the observers. Methods to dynamically add or remove
    observers, get the list of registered observers at any given point of time and
    notify/ update the observers when there is a change in the monitored parameters.
    """

    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def add_observer(self, observer):
        raise NotImplementedError

    @abc.abstractmethod
    def remove_observer(self, observer):
        raise NotImplementedError

    @abc.abstractmethod
    def notify_observers(self):
        raise NotImplementedError

    @abc.abstractmethod
    def get_registered_observers(self):
        raise NotImplementedError


class Observer:
    """
    Abstract class to update the observer's state when there is a change in the parameters.
    """

    __metaclass__ = abc.ABCMeta

    _description = NotImplementedError

    @abc.abstractmethod
    def update(self, parameters):
        raise NotImplementedError

    def get_description(self):
        return self._description


class Display:
    """
    Abstract class to display the analysis derived from the various observers.
    """

    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def update_display(self, parameters, display_title):
        raise NotImplementedError


class ObserverFactory:
    """
    Abstract class for creating an observer factory
    """

    __metaclass__ = abc.ABCMeta

    display_object = None

    @abc.abstractmethod
    def create(self, observer, observer_name):
        raise NotImplementedError
