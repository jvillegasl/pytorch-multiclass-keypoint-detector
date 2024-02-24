from typing import TypedDict
import numpy as np
from numpy._typing import NDArray
from generators.base_generator import BaseGenerator


class VSourceKeypoints(TypedDict):
    
    positive: tuple[float, float]
    negative: tuple[float, float]
    center: tuple[float, float]

class VSourceSettings()
    keypoints: VSourceKeypoints
    


class VSourceGenerator(BaseGenerator):
    def generate(self) -> NDArray[np.int64]:
        pass

    def __draw(self):
        pass

    def __get_settings(self):
        pass
