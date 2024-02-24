import math
from typing import TypedDict
import numpy as np
import numpy.typing as npt
from PIL import Image, ImageDraw

from generators.base_generator import BaseGenerator
from get_error import get_error


class DiodeKeypoints(TypedDict):
    anode_0: tuple[float, float]
    anode_1: tuple[float, float]
    anode_1_top: tuple[float, float]
    anode_1_bottom: tuple[float, float]
    catode_0: tuple[float, float]
    catode_0_top: tuple[float, float]
    catode_0_bottom: tuple[float, float]
    catode_1: tuple[float, float]


class DiodeSettings(TypedDict):
    keypoints: DiodeKeypoints
    width: int
    height: int


class DiodeGenerator(BaseGenerator):

    def generate(self, width: int) -> npt.NDArray[np.int64]:
        settings = self.__get_settings(width)
        img = self.__draw(settings)

        data = np.asarray(img)

        return data

    def __draw(self, settings: DiodeSettings) -> Image.Image:
        W = settings['width']
        H = settings['height']

        keypoints = settings['keypoints']
        anode_0 = keypoints['anode_0']
        anode_1 = keypoints['anode_1']
        anode_1_top = keypoints['anode_1_top']
        anode_1_bottom = keypoints['anode_1_bottom']
        catode_0 = keypoints['catode_0']
        catode_1 = keypoints['catode_1']

        img = Image.new("RGBA", (W, H))
        draw = ImageDraw.Draw(img)

        draw.line([anode_0, anode_1], fill="black")
        draw.line([anode_1, anode_1_top], fill="black")
        draw.line([anode_1, anode_1_bottom], fill="black")
        draw.line([anode_1_top, catode_0], fill="black")
        draw.line([anode_1_bottom, catode_0], fill="black")
        draw.line([catode_0, catode_1], fill="black")

        return img

    def __get_settings(self, width: int) -> DiodeSettings:
        W, H = 100, 100

        anode_width = 0.25*W
        anode_height = 0.5*H
        anode_catode_width = 0.5*W
        catode_width = 0.25*W

        anode_width_error = get_error(0.25)
        anode_top_height_error = get_error(0.5)
        anode_bottom_height_error = get_error(0.5)
        anode_catode_width_error = get_error(0.25)
        catode_width_error = get_error(0.25)

        anode_width = anode_width * (1 + anode_width_error)
        anode_top_height = 0.5 * anode_height * (1 + anode_top_height_error)
        anode_bottom_height = 0.5 * anode_height * \
            (1 + anode_bottom_height_error)
        anode_catode_width = anode_catode_width * \
            (1 + anode_catode_width_error)
        catode_width = catode_width * (1 + catode_width_error)

        anode_0 = np.array([0, anode_top_height])
        anode_1 = anode_0 + np.array([anode_width, 0])
        anode_1_top = anode_1 - np.array([0, anode_top_height])
        anode_1_bottom = anode_1 + np.array([0, anode_bottom_height])
        catode_0 = anode_1 + np.array([anode_catode_width, 0])
        catode_1 = catode_0 + np.array([catode_width, 0])

        anode_0 = tuple(anode_0)
        anode_1 = tuple(anode_1)
        anode_1_top = tuple(anode_1_top)
        anode_1_bottom = tuple(anode_1_bottom)
        catode_0 = tuple(catode_0)
        catode_1 = tuple(catode_1)

        nW = math.ceil(min(W, anode_width + anode_catode_width + catode_width))
        nH = math.ceil(min(H, anode_top_height + anode_bottom_height))

        keypoints: DiodeKeypoints = {
            'anode_0': anode_0,
            'anode_1': anode_1,
            'anode_1_bottom': anode_1_bottom,
            'anode_1_top': anode_1_top,
            'catode_0': catode_0,
            'catode_1': catode_1
        }

        scale_factor = width/nW
        for key in keypoints:
            value = keypoints[key]

            keypoints[key] = tuple([scale_factor * t for t in value])

        settings: DiodeSettings = {
            'keypoints': keypoints,
            'width': width,
            'height': math.ceil(scale_factor * nH)
        }

        return settings
