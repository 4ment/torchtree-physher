import abc


class Interface:
    @abc.abstractmethod
    def update(self, index):
        ...
