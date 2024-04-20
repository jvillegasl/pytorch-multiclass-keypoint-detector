import torch
from torch import Tensor
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from tkinter import Tk
from tkinter.filedialog import askopenfilename
from PIL import Image
from torchvision.transforms.functional import pil_to_tensor, resize
from matplotlib.patches import Circle


from models.model import MKDPred, MulticlassKptsDetector


def main():
    img = input_image()

    if img is None:
        return

    img = resize(img, [200, 200], antialias=True)

    model = MulticlassKptsDetector(list_num_kpts=[1, 2, 3], d_model=256)

    x = img.unsqueeze(0)

    t = torch.tensor([2])

    y: MKDPred = model(x, t)

    kpts = y.batch_kpts[0]

    ax: Axes
    fig, ax = plt.subplots()

    ax.imshow(img.permute(1, 2, 0))

    draw_kpts(kpts, ax)

    fig.show()

    plt.imshow(img.permute(1, 2, 0))
    plt.show()


def input_image() -> Tensor | None:
    Tk().withdraw()
    filename = askopenfilename(
        filetypes=[('image files', ('.png', '.jpg', '.jpeg'))]
    )

    if not filename:
        return None

    img = Image.open(filename)

    tensor_img = pil_to_tensor(img)
    tensor_img = tensor_img.float()
    tensor_img /= 255

    return tensor_img


def draw_kpts(kpts: Tensor, ax: Axes):
    """
    Shapes:
        - kpts: `(K, 2)`
    """

    CIRCLE_RADIUS = 2

    for idx, coords in enumerate(kpts):
        x = 255 * coords[0].item()
        y = 255 * coords[1].item()

        circle = Circle((x, y), CIRCLE_RADIUS, color='r')

        ax.add_patch(circle)
        ax.text(x + 2 * CIRCLE_RADIUS, y - 2 * CIRCLE_RADIUS, f'Kpt n°{idx}', fontsize=10, bbox={
                'facecolor': 'red', 'alpha': 0.75})


if __name__ == '__main__':
    main()
