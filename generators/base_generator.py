from abc import abstractmethod
import numpy as np
import numpy.typing as npt


class BaseGenerator():
    @abstractmethod
    def generate(self) -> npt.NDArray[np.float64]:
        raise NotImplemented()
