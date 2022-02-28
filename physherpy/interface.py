import abc


class Interface(object):
    @abc.abstractmethod
    def update(self, index):
        ...
