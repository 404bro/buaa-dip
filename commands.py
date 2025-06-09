from PySide6.QtGui import QPixmap, QUndoCommand
from typing import Callable
from image_widget import ImageWidget
from utils import pillow_to_qimage, qpixmap_to_pillow


class EditCommand(QUndoCommand):
    def __init__(
        self,
        image_widget: "ImageWidget",
        operation_func: Callable,
        op_name: str,
        *args,
        **kwargs,
    ):
        super().__init__()
        self.widget = image_widget
        self.operation = operation_func
        self.op_name = op_name
        self.op_args = args
        self.op_kwargs = kwargs

        self.before_pixmap = self.widget.original_pixmap
        self.after_pixmap = None

        self.setText(f"{self.op_name}")

    def redo(self):
        if self.after_pixmap is None:
            pil_before = qpixmap_to_pillow(self.before_pixmap)
            pil_after = self.operation(pil_before, *self.op_args, **self.op_kwargs)
            qimage_after = pillow_to_qimage(pil_after)
            self.after_pixmap = QPixmap.fromImage(qimage_after)

        self.widget.set_pixmap(self.after_pixmap)
        self.widget.set_modified(True)

    def undo(self):
        self.widget.set_pixmap(self.before_pixmap)
        self.widget.set_modified(not self.widget.undo_stack.isClean())
