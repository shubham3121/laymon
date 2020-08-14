"""
===========================================
Abstract Classes
===========================================
"""

import abc


class Monitor:
    """
    Abstract class to monitor the observers.
    Methods to:
     1. Add observers
     2. Remove observers
     3. Get the list of registered observers
     4. Notify the observers when there is a change in the monitored parameters.
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
    """Abstract class to describe and update the state of an observer."""

    __metaclass__ = abc.ABCMeta

    _description = NotImplementedError

    @abc.abstractmethod
    def update(self, parameters):
        raise NotImplementedError

    def get_description(self):
        return self._description


class Display:
    """
    Abstract class to create and update the displays attached to an observer.
    """

    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def update_display(self, parameters, display_title):
        raise NotImplementedError


class ObserverFactory:
    """
    Abstract class for creating a factory to create new observers with a display attached to them.
    The `display_object` stores a display class used to visualize an observer's parameters/states.
    """

    __metaclass__ = abc.ABCMeta

    display_object = None

    @abc.abstractmethod
    def create(self, observer, observer_name):
        raise NotImplementedError
