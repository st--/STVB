import abc


class Q_xm:
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def sample(self):
        raise NotImplementedError("Subclass should implement this.")

    @abc.abstractmethod
    def evaluate_logprob(self):
        raise NotImplementedError("Subclass should implement this.")
