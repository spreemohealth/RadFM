"""
by will.deng@coverahealth.com

The singleton metaclass for ensuring only one instance of a class.
"""
from abc import ABCMeta


class Singleton(ABCMeta):
    """
    Singleton class for ensuring only one instance of a class.
    usage:
        class MyClass(metaclass=Singleton):
            pass
    """

    _instances = {}

    def __call__(cls, *args, **kwargs):
        """Call method for the singleton metaclass."""
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]