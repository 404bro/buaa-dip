from PySide6.QtWidgets import QWidget, QLabel, QVBoxLayout, QScrollArea
from PySide6.QtGui import QPixmap, QResizeEvent, QUndoStack, QShowEvent
from PySide6.QtCore import Qt, Signal, QFileInfo


class ImageWidget(QWidget):
    modification_changed = Signal(bool)

    def __init__(self, file_path: str, parent=None):
        super().__init__(parent)
        self.file_path = file_path
        self._is_modified = False

        self._original_pixmap: QPixmap | None = None
        self._initial_show = True

        self.undo_stack = QUndoStack(self)
        self.undo_stack.cleanChanged.connect(self.on_clean_changed)

        self.image_label = QLabel("正在加载图片...")
        self.image_label.setAlignment(Qt.AlignCenter)

        self.scroll_area = QScrollArea()
        self.scroll_area.setWidget(self.image_label)
        self.scroll_area.setWidgetResizable(True)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.scroll_area)

        self.load_image(file_path)

    def load_image(self, file_path: str):
        pixmap = QPixmap(file_path)
        if not pixmap.isNull():
            self.file_path = file_path
            self._original_pixmap = pixmap
            self.undo_stack.setClean()
        else:
            self._original_pixmap = None
            self.image_label.setPixmap(QPixmap())
            self.image_label.setText(f"错误：无法打开文件\n{file_path}")

    def showEvent(self, event: QShowEvent):
        super().showEvent(event)
        if self._initial_show:
            self.update_displayed_pixmap()
            self._initial_show = False

    def update_displayed_pixmap(self):
        if self._original_pixmap is None or self._original_pixmap.isNull():
            return

        viewport_size = self.scroll_area.viewport().size()

        scaled_pixmap = self._original_pixmap.scaled(
            viewport_size,
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation,
        )
        self.image_label.setPixmap(scaled_pixmap)

    def resizeEvent(self, event: QResizeEvent):
        super().resizeEvent(event)
        self.update_displayed_pixmap()

    @property
    def original_pixmap(self) -> QPixmap | None:
        return self._original_pixmap

    def set_pixmap(self, pixmap: QPixmap):
        self._original_pixmap = pixmap
        self.update_displayed_pixmap()

    def is_modified(self) -> bool:
        return self._is_modified

    def set_modified(self, modified: bool):
        if self._is_modified != modified:
            self._is_modified = modified
            self.modification_changed.emit(modified)

    def on_clean_changed(self, is_clean: bool):
        self.set_modified(not is_clean)

    def save_image(self, save_path: str) -> bool:
        if self._original_pixmap and self._original_pixmap.save(save_path):
            self.file_path = save_path
            self.undo_stack.setClean()
            return True
        return False

    @property
    def file_name(self) -> str:
        if self.file_path:
            return QFileInfo(self.file_path).fileName()
        return "未命名"
