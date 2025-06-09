from PySide6.QtWidgets import (
    QMainWindow,
    QTabWidget,
    QFileDialog,
    QMessageBox,
    QInputDialog,
    QDialog,
    QVBoxLayout,
    QLabel,
    QApplication,
    QScrollArea,
)
from PySide6.QtGui import (
    QAction,
    QKeySequence,
    QPixmap,
    QResizeEvent,
    QShowEvent,
)
from PySide6.QtCore import Qt
import numpy as np

from image_widget import ImageWidget
from commands import EditCommand
import image_processor

from utils import pillow_to_qimage, qpixmap_to_pillow
from PIL import Image


class ResultDialog(QDialog):
    def __init__(self, pixmap: QPixmap, parent=None):
        super().__init__(parent)
        self.setWindowTitle("SIFT匹配结果")

        self._original_pixmap = pixmap
        self._initial_show = True

        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)

        self.scroll_area = QScrollArea()
        self.scroll_area.setWidget(self.image_label)
        self.scroll_area.setWidgetResizable(True)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.scroll_area)

        screen_geometry = self.screen().availableGeometry()
        initial_width = min(
            self._original_pixmap.width(), int(screen_geometry.width() * 0.8)
        )
        initial_height = min(
            self._original_pixmap.height(), int(screen_geometry.height() * 0.8)
        )
        self.resize(initial_width, initial_height)

    def showEvent(self, event: QShowEvent):
        super().showEvent(event)
        if self._initial_show:
            self.update_displayed_pixmap()
            self._initial_show = False

    def update_displayed_pixmap(self):
        if self._original_pixmap.isNull():
            return

        scaled_pixmap = self._original_pixmap.scaled(
            self.scroll_area.viewport().size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation,
        )
        self.image_label.setPixmap(scaled_pixmap)

    def resizeEvent(self, event: QResizeEvent):
        super().resizeEvent(event)
        if not self._initial_show:
            self.update_displayed_pixmap()


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("BUAA DIP")
        self.setGeometry(100, 100, 1000, 700)

        self.tab_widget = QTabWidget()
        self.tab_widget.setTabsClosable(True)
        self.tab_widget.setMovable(True)
        self.tab_widget.tabCloseRequested.connect(self.close_tab)
        self.tab_widget.currentChanged.connect(self.update_actions_state)
        self.setCentralWidget(self.tab_widget)

        self.last_fft_result: np.ndarray | None = None

        self.create_actions()
        self.create_menus()
        self.create_status_bar()

        self.update_actions_state()

    def create_actions(self):
        self.open_action = QAction(
            "&打开...", self, shortcut=QKeySequence.Open, triggered=self.open_file
        )
        self.save_action = QAction(
            "&保存", self, shortcut=QKeySequence.Save, triggered=self.save_file
        )
        self.save_as_action = QAction(
            "另存为...", self, shortcut=QKeySequence.SaveAs, triggered=self.save_file_as
        )
        self.exit_action = QAction(
            "退出", self, shortcut=QKeySequence.Quit, triggered=self.close
        )

        self.undo_action = QAction("撤销", self, shortcut=QKeySequence.Undo)
        self.redo_action = QAction("重做", self, shortcut=QKeySequence.Redo)

        self.invert_action = QAction(
            "反相",
            self,
            triggered=lambda: self.apply_edit(image_processor.invert, "反相"),
        )
        self.add_brightness_action = QAction(
            "亮度...", self, triggered=self.add_brightness
        )
        self.he_action = QAction(
            "直方图均衡化",
            self,
            triggered=lambda: self.apply_edit(
                image_processor.histogram_equalization, "直方图均衡化"
            ),
        )
        self.gamma_action = QAction(
            "指数(Gamma)变换...", self, triggered=self.get_gamma_and_apply
        )

        self.h_flip_action = QAction(
            "水平翻转",
            self,
            triggered=lambda: self.apply_edit(
                image_processor.flip, "水平翻转", "horizontal"
            ),
        )
        self.v_flip_action = QAction(
            "垂直翻转",
            self,
            triggered=lambda: self.apply_edit(
                image_processor.flip, "垂直翻转", "vertical"
            ),
        )
        self.rotate90_cw_action = QAction(
            "顺时针旋转90°",
            self,
            triggered=lambda: self.apply_edit(image_processor.rotate, "旋转-90°", -90),
        )
        self.rotate90_ccw_action = QAction(
            "逆时针旋转90°",
            self,
            triggered=lambda: self.apply_edit(image_processor.rotate, "旋转90°", 90),
        )
        self.rotate180_action = QAction(
            "旋转180°",
            self,
            triggered=lambda: self.apply_edit(image_processor.rotate, "旋转180°", 180),
        )
        self.translate_action = QAction("平移...", self, triggered=self.translate_image)

        self.homomorphic_action = QAction(
            "同态滤波",
            self,
            triggered=lambda: self.apply_edit(
                image_processor.homomorphic_filter, "同态滤波"
            ),
        )
        self.laplacian_sharpen_action = QAction(
            "Laplace锐化",
            self,
            triggered=lambda: self.apply_edit(
                image_processor.laplacian_sharpen, "Laplace锐化"
            ),
        )

        self.roberts_action = QAction(
            "Roberts算子",
            self,
            triggered=lambda: self.apply_edit(
                image_processor.edge_detection, "Roberts边缘检测", operator="roberts"
            ),
        )
        self.prewitt_action = QAction(
            "Prewitt算子",
            self,
            triggered=lambda: self.apply_edit(
                image_processor.edge_detection, "Prewitt边缘检测", operator="prewitt"
            ),
        )
        self.sobel_action = QAction(
            "Sobel算子",
            self,
            triggered=lambda: self.apply_edit(
                image_processor.edge_detection, "Sobel边缘检测", operator="sobel"
            ),
        )
        self.laplacian_edge_action = QAction(
            "Laplacian算子",
            self,
            triggered=lambda: self.apply_edit(
                image_processor.edge_detection,
                "Laplacian边缘检测",
                operator="laplacian",
            ),
        )

        self.dilate_action = QAction(
            "膨胀",
            self,
            triggered=lambda: self.apply_edit(image_processor.dilate, "膨胀"),
        )
        self.erode_action = QAction(
            "腐蚀",
            self,
            triggered=lambda: self.apply_edit(image_processor.erode, "腐蚀"),
        )

        self.show_fft_action = QAction(
            "显示频谱 (FFT)", self, triggered=self.show_fft_spectrum
        )
        self.ifft_action = QAction(
            "从频谱恢复 (IFFT)", self, triggered=self.apply_ifft_from_memory
        )
        self.ifft_action.setEnabled(False)
        self.fft_conj_rot_action = QAction(
            "共轭旋转 (a-e步骤)",
            self,
            triggered=lambda: self.apply_edit(
                image_processor.fft_conjugate_rotation, "FFT共轭旋转"
            ),
        )
        self.fourier_descriptors_action = QAction(
            "傅里叶描述子重构...", self, triggered=self.reconstruct_from_descriptors
        )

        self.add_image_action = QAction(
            "图像相加...", self, triggered=self.handle_image_addition
        )
        self.sift_match_action = QAction(
            "SIFT特征匹配...", self, triggered=self.handle_sift_match
        )

    def create_menus(self):
        file_menu = self.menuBar().addMenu("文件(&F)")
        file_menu.addAction(self.open_action)
        file_menu.addAction(self.save_action)
        file_menu.addAction(self.save_as_action)
        file_menu.addSeparator()
        file_menu.addAction(self.exit_action)

        edit_menu = self.menuBar().addMenu("编辑(&E)")
        edit_menu.addAction(self.undo_action)
        edit_menu.addAction(self.redo_action)

        image_menu = self.menuBar().addMenu("图像(&I)")

        adjustment_menu = image_menu.addMenu("调整")
        # adjustment_menu.addAction(self.add_brightness_action)
        adjustment_menu.addAction(self.invert_action)
        adjustment_menu.addAction(self.he_action)
        adjustment_menu.addAction(self.gamma_action)

        transform_menu = image_menu.addMenu("变换")
        transform_menu.addAction(self.h_flip_action)
        transform_menu.addAction(self.v_flip_action)
        transform_menu.addSeparator()
        transform_menu.addAction(self.rotate90_cw_action)
        transform_menu.addAction(self.rotate90_ccw_action)
        transform_menu.addAction(self.rotate180_action)
        transform_menu.addSeparator()
        transform_menu.addAction(self.translate_action)

        image_menu.addSeparator()

        filter_menu = image_menu.addMenu("滤波")
        filter_menu.addAction(self.laplacian_sharpen_action)
        filter_menu.addAction(self.homomorphic_action)

        edge_menu = image_menu.addMenu("边缘检测")
        edge_menu.addAction(self.roberts_action)
        edge_menu.addAction(self.prewitt_action)
        edge_menu.addAction(self.sobel_action)
        edge_menu.addAction(self.laplacian_edge_action)

        morphology_menu = image_menu.addMenu("形态学")
        morphology_menu.addAction(self.dilate_action)
        morphology_menu.addAction(self.erode_action)

        fft_menu = self.menuBar().addMenu("傅里叶(&T)")
        fft_menu.addAction(self.show_fft_action)
        fft_menu.addAction(self.ifft_action)
        fft_menu.addAction(self.fft_conj_rot_action)
        fft_menu.addAction(self.fourier_descriptors_action)

        tools_menu = self.menuBar().addMenu("二元操作(&B)")
        tools_menu.addAction(self.add_image_action)
        tools_menu.addAction(self.sift_match_action)

    def create_status_bar(self):
        self.statusBar().showMessage("准备就绪")

    def get_current_widget(self) -> ImageWidget | None:
        return self.tab_widget.currentWidget()

    def update_actions_state(self):
        widget = self.get_current_widget()
        has_widget = widget is not None

        self.save_action.setEnabled(has_widget)
        self.save_as_action.setEnabled(has_widget)
        self.show_fft_action.setEnabled(has_widget)
        self.ifft_action.setEnabled(self.last_fft_result is not None)
        self.fft_conj_rot_action.setEnabled(has_widget)
        self.fourier_descriptors_action.setEnabled(has_widget)

        try:
            self.undo_action.triggered.disconnect()
            self.redo_action.triggered.disconnect()
        except (RuntimeError, TypeError):
            pass

        if has_widget:
            self.undo_action.setEnabled(widget.undo_stack.canUndo())
            self.redo_action.setEnabled(widget.undo_stack.canRedo())
            widget.undo_stack.canUndoChanged.connect(self.undo_action.setEnabled)
            widget.undo_stack.canRedoChanged.connect(self.redo_action.setEnabled)
            self.undo_action.triggered.connect(widget.undo_stack.undo)
            self.redo_action.triggered.connect(widget.undo_stack.redo)
        else:
            self.undo_action.setEnabled(False)
            self.redo_action.setEnabled(False)

        edit_actions = [
            # self.add_brightness_action,
            self.invert_action,
            self.he_action,
            self.gamma_action,
            self.h_flip_action,
            self.v_flip_action,
            self.rotate90_cw_action,
            self.rotate90_ccw_action,
            self.rotate180_action,
            self.translate_action,
            self.laplacian_sharpen_action,
            self.homomorphic_action,
            self.roberts_action,
            self.prewitt_action,
            self.sobel_action,
            self.laplacian_edge_action,
            self.dilate_action,
            self.erode_action,
            self.show_fft_action,
            self.ifft_action,
            self.fft_conj_rot_action,
            self.fourier_descriptors_action,
            self.add_image_action,
            self.sift_match_action,
        ]
        for action in edit_actions:
            action.setEnabled(has_widget)

    def open_file(self):
        file_filter = "图像文件 (*.bmp *.jpg *.jpeg *.png *.gif *.tiff);;所有文件 (*)"
        file_path, _ = QFileDialog.getOpenFileName(self, "打开图像", "", file_filter)
        if file_path:
            self.create_new_tab(file_path)

    def create_new_tab(self, file_path: str):
        new_widget = ImageWidget(file_path)
        new_widget.modification_changed.connect(self.update_tab_title)

        index = self.tab_widget.addTab(new_widget, new_widget.file_name)
        self.tab_widget.setCurrentIndex(index)
        self.statusBar().showMessage(f"已打开 {file_path}", 5000)

    def update_tab_title(self, is_modified: bool):
        widget = self.sender()
        if not isinstance(widget, ImageWidget):
            return

        index = self.tab_widget.indexOf(widget)
        if index != -1:
            title = widget.file_name
            if is_modified:
                title += " *"
            self.tab_widget.setTabText(index, title)

    def save_file(self):
        widget = self.get_current_widget()
        if not widget:
            return

        if widget.file_path and not widget.file_path.startswith("未命名"):
            if widget.save_image(widget.file_path):
                self.statusBar().showMessage(f"文件已保存到 {widget.file_path}", 5000)
            else:
                QMessageBox.warning(self, "保存失败", "无法保存文件。")
        else:
            self.save_file_as()

    def save_file_as(self):
        widget = self.get_current_widget()
        if not widget:
            return

        file_filter = (
            "PNG (*.png);;JPEG (*.jpg *.jpeg);;BMP (*.bmp);;TIFF (*.tiff);;所有文件 (*)"
        )
        save_path, _ = QFileDialog.getSaveFileName(
            self, "另存为", widget.file_path, file_filter
        )

        if save_path:
            if widget.save_image(save_path):
                self.statusBar().showMessage(f"文件已另存为 {save_path}", 5000)
                self.tab_widget.setTabText(
                    self.tab_widget.currentIndex(), widget.file_name
                )
            else:
                QMessageBox.warning(self, "保存失败", f"无法将文件保存到 {save_path}。")

    def close_tab(self, index):
        widget = self.tab_widget.widget(index)
        if widget.is_modified():
            reply = QMessageBox.question(
                self,
                "保存修改",
                f"文件 '{widget.file_name}' 已被修改。\n您想保存更改吗?",
                QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel,
                QMessageBox.Save,
            )
            if reply == QMessageBox.Save:
                self.save_file()
                if widget.is_modified():
                    return
            elif reply == QMessageBox.Cancel:
                return

        self.tab_widget.removeTab(index)

    def closeEvent(self, event):
        while self.tab_widget.count() > 0:
            current_index = self.tab_widget.currentIndex()
            count_before = self.tab_widget.count()
            self.close_tab(current_index)
            if self.tab_widget.count() == count_before:
                event.ignore()
                return
        event.accept()

    def apply_edit(self, operation_func, op_name, *args, **kwargs):
        widget = self.get_current_widget()
        if not widget:
            return

        command = EditCommand(widget, operation_func, op_name, *args, **kwargs)
        widget.undo_stack.push(command)

    def add_brightness(self):
        value, ok = QInputDialog.getInt(
            self, "调整亮度", "输入亮度值 (-100 to 100):", 0, -100, 100, 5
        )
        if ok:
            self.apply_edit(image_processor.add_brightness, "调整亮度", value)

    def translate_image(self):
        widget = self.get_current_widget()
        if not widget:
            return

        dx, ok1 = QInputDialog.getInt(
            self, "平移", "水平偏移 (dx):", 0, -widget.width(), widget.width(), 10
        )
        if not ok1:
            return
        dy, ok2 = QInputDialog.getInt(
            self, "平移", "垂直偏移 (dy):", 0, -widget.height(), widget.height(), 10
        )
        if ok2:
            self.apply_edit(image_processor.translate, "平移", dx, dy)

    def _get_second_image_path(self) -> str | None:
        file_filter = "图像文件 (*.bmp *.jpg *.jpeg *.png *.gif *.tiff);;所有文件 (*)"
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择第二张图像", "", file_filter
        )
        return file_path if file_path else None

    def handle_image_addition(self):
        widget = self.get_current_widget()
        if not widget:
            return

        second_image_path = self._get_second_image_path()
        if not second_image_path:
            return

        try:
            image2 = Image.open(second_image_path)
            self.apply_edit(image_processor.add_images, "图像相加", image2)
        except Exception as e:
            QMessageBox.critical(self, "错误", f"无法加载或处理第二张图片: {e}")

    def handle_sift_match(self):
        widget = self.get_current_widget()
        if not widget:
            return

        pil_image1 = qpixmap_to_pillow(widget.original_pixmap)

        second_image_path = self._get_second_image_path()
        if not second_image_path:
            return

        try:
            pil_image2 = Image.open(second_image_path)
            self.statusBar().showMessage("正在进行SIFT特征匹配，请稍候...")
            QApplication.setOverrideCursor(Qt.WaitCursor)

            result_pil_image = image_processor.sift_match(pil_image1, pil_image2)

            QApplication.restoreOverrideCursor()
            self.statusBar().showMessage("匹配完成！", 5000)

            result_qimage = pillow_to_qimage(result_pil_image)
            result_pixmap = QPixmap.fromImage(result_qimage)
            dialog = ResultDialog(result_pixmap, self)
            dialog.exec()

        except Exception as e:
            QApplication.restoreOverrideCursor()
            QMessageBox.critical(self, "SIFT匹配错误", f"处理过程中发生错误: {e}")

    def get_gamma_and_apply(self):
        gamma, ok = QInputDialog.getDouble(
            self, "指数(Gamma)变换", "输入Gamma值 (推荐 0.2 ~ 5.0):", 1.0, 0.1, 10.0, 2
        )
        if ok:
            self.apply_edit(image_processor.gamma_correction, "Gamma变换", gamma=gamma)

    def reconstruct_from_descriptors(self):
        widget = self.get_current_widget()
        if not widget:
            return

        num, ok = QInputDialog.getInt(
            self,
            "傅里叶描述子重构",
            "输入用于重构的描述子项数 (例如: 4, 8, 16):",
            8,
            2,
            512,
            1,
        )
        if ok:
            try:
                self.apply_edit(
                    image_processor.reconstruct_with_fourier_descriptors,
                    f"傅里叶描述子重构 ({num}项)",
                    num_descriptors=num,
                )
            except Exception as e:
                QMessageBox.critical(self, "重构失败", f"处理过程中发生错误: {e}")

    def show_fft_spectrum(self):
        widget = self.get_current_widget()
        if not widget:
            return

        try:
            pil_image = qpixmap_to_pillow(widget.original_pixmap)

            QApplication.setOverrideCursor(Qt.WaitCursor)
            visual_spectrum, complex_data = (
                image_processor.fft_transform_and_get_complex(pil_image)
            )
            QApplication.restoreOverrideCursor()

            self.last_fft_result = complex_data

            self.update_actions_state()

            spectrum_qimage = pillow_to_qimage(visual_spectrum)
            spectrum_pixmap = QPixmap.fromImage(spectrum_qimage)

            dialog = ResultDialog(spectrum_pixmap, self)
            dialog.setWindowTitle("FFT 幅度谱")
            dialog.exec()

        except Exception as e:
            QApplication.restoreOverrideCursor()
            self.last_fft_result = None
            self.update_actions_state()
            QMessageBox.critical(self, "FFT频谱显示错误", f"处理过程中发生错误: {e}")

    def apply_ifft_from_memory(self):
        if self.last_fft_result is None:
            QMessageBox.warning(
                self,
                "操作无效",
                "没有可用的FFT数据。请先对一张图片执行“显示频谱 (FFT)”。",
            )
            return

        widget = self.get_current_widget()
        if not widget:
            return

        operation_func = lambda img, data: image_processor.ifft_from_complex(data)

        command = EditCommand(
            widget, operation_func, "从频谱恢复", self.last_fft_result
        )
        widget.undo_stack.push(command)
