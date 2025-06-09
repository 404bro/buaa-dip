from PIL import Image
from PySide6.QtGui import QImage, QPixmap
from PIL.ImageQt import ImageQt


def pillow_to_qimage(pil_img: Image.Image) -> QImage:
    if pil_img.mode != "RGBA":
        pil_img = pil_img.convert("RGBA")
    return ImageQt(pil_img)


def qimage_to_pillow(q_img: QImage) -> Image.Image:
    q_img = q_img.convertToFormat(QImage.Format.Format_RGBA8888)
    width = q_img.width()
    height = q_img.height()
    ptr = q_img.bits()
    pil_img = Image.frombytes("RGBA", (width, height), ptr.tobytes())
    return pil_img


def qpixmap_to_pillow(q_pixmap: QPixmap) -> Image.Image:
    return qimage_to_pillow(q_pixmap.toImage())
