# -*- coding: utf-8 -*-
"""
革兰染色涂片智能分析软件 - 图形界面（PySide6版）
运行：python gram_gui.py
依赖：pip install PySide6 opencv-python numpy
"""
import sys
import os
import cv2
import numpy as np

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QHBoxLayout, QVBoxLayout, QGridLayout,
    QPushButton, QLabel, QComboBox, QSpinBox, QDoubleSpinBox, QSlider,
    QGroupBox, QTableWidget, QTableWidgetItem, QTextEdit, QFileDialog,
    QMessageBox, QListWidget, QGraphicsView, QGraphicsScene,
    QGraphicsPixmapItem, QCheckBox, QSplitter, QRadioButton, QButtonGroup
)
from PySide6.QtGui import QImage, QPixmap, QPainter, QPen, QColor, QFont
from PySide6.QtCore import Qt, QRectF, QPointF, Signal

from gram_analyzer import GramAnalyzer, AnalyzerConfig, make_synthetic_gram_image


def cv2_to_qpixmap(img_bgr):
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    h, w, ch = rgb.shape
    qimg = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
    return QPixmap.fromImage(qimg.copy())


class ImageCanvas(QGraphicsView):
    """图像画布：滚轮缩放、拖拽平移；点击/框选事件上报。"""
    clicked = Signal(int, int)
    rect_selected = Signal(int, int, int, int)

    def __init__(self):
        super().__init__()
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self._pix_item = None
        self.mode = "select"          # select / add / cal_pos / cal_neg
        self.setRenderHint(QPainter.Antialiasing)
        self.setDragMode(QGraphicsView.NoDrag)
        self._rubber_origin = None
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)

    def set_image(self, qpix):
        self._scene.clear()
        self._pix_item = QGraphicsPixmapItem(qpix)
        self._scene.addItem(self._pix_item)
        self._scene.setSceneRect(QRectF(qpix.rect()))
        self.fitInView(self._pix_item, Qt.KeepAspectRatio)

    def refresh_pixmap(self, qpix):
        if self._pix_item is not None:
            self._pix_item.setPixmap(qpix)

    def wheelEvent(self, e):
        factor = 1.25 if e.angleDelta().y() > 0 else 0.8
        self.scale(factor, factor)

    def mousePressEvent(self, e):
        if self._pix_item is None:
            return
        pos = self.mapToScene(e.pos())
        x, y = int(pos.x()), int(pos.y())
        if e.button() == Qt.LeftButton:
            if self.mode == "add":
                self._rubber_origin = (x, y)
            else:
                self.setDragMode(QGraphicsView.ScrollHandDrag)
                super().mousePressEvent(e)
                return
        elif e.button() == Qt.RightButton:
            self.clicked.emit(x, y)   # 右键 = 点选对象/校准取点
            return
        super().mousePressEvent(e)

    def mouseReleaseEvent(self, e):
        if self._rubber_origin is not None and e.button() == Qt.LeftButton:
            pos = self.mapToScene(e.pos())
            x2, y2 = int(pos.x()), int(pos.y())
            x1, y1 = self._rubber_origin
            self._rubber_origin = None
            x, y = min(x1, x2), min(y1, y2)
            w, h = abs(x2 - x1), abs(y2 - y1)
            if w >= 4 and h >= 4:
                self.rect_selected.emit(x, y, w, h)
        self.setDragMode(QGraphicsView.NoDrag)
        super().mouseReleaseEvent(e)

    def mouseMoveEvent(self, e):
        if self._rubber_origin is not None:
            pos = self.mapToScene(e.pos())
            x1, y1 = self._rubber_origin
            rect = QRectF(QPointF(x1, y1), pos).normalized()
            for item in self._scene.items():
                if getattr(item, "_preview", False):
                    self._scene.removeItem(item)
            pen = QPen(QColor(255, 60, 60), 2, Qt.DashLine)
            r = self._scene.addRect(rect, pen)
            r._preview = True
        super().mouseMoveEvent(e)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("革兰染色涂片智能分析系统（WS/T 805—2023）")
        self.resize(1500, 950)

        self.cfg = AnalyzerConfig()
        # 在软件所在目录搜索 gram_model.pt / gram_model.onnx，有则启用AI后端
        app_dir = os.path.dirname(os.path.abspath(__file__))
        self.analyzer = GramAnalyzer(self.cfg, model_dir=app_dir)
        self.image = None
        self.selected_ids = set()
        self.field_results = {}           # 文件名 -> summarize结果（多视野）
        self.image_paths = []

        self._build_ui()
        self._connect()

    # ------------------------------------------------------------------
    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)

        splitter = QSplitter(Qt.Horizontal)
        root.addWidget(splitter)

        # ============ 左：画布 ============
        left = QWidget()
        lv = QVBoxLayout(left)
        self.canvas = ImageCanvas()
        lv.addWidget(self.canvas, 1)

        mode_bar = QHBoxLayout()
        mode_bar.addWidget(QLabel("操作模式："))
        self.mode_group = QButtonGroup(self)
        self.rb_select = QRadioButton("浏览/平移")
        self.rb_add = QRadioButton("框选添加漏检")
        self.rb_cal_pos = QRadioButton("校准-阳性(紫)")
        self.rb_cal_neg = QRadioButton("校准-阴性(红)")
        self.rb_select.setChecked(True)
        for rb in (self.rb_select, self.rb_add, self.rb_cal_pos, self.rb_cal_neg):
            self.mode_group.addButton(rb)
            mode_bar.addWidget(rb)
        mode_bar.addStretch(1)
        self.cb_labels = QCheckBox("显示标注")
        self.cb_labels.setChecked(True)
        self.cb_labels.stateChanged.connect(lambda _: self.refresh_view())
        mode_bar.addWidget(self.cb_labels)
        lv.addLayout(mode_bar)

        hint = QLabel("滚轮缩放，左键平移；右键点选对象（可删除/重分类）；染色偏差大时用校准模式点击已知阴/阳性菌后重新识别。")
        hint.setStyleSheet("color:#666;")
        lv.addWidget(hint)
        splitter.addWidget(left)

        # ============ 右：控制面板 ============
        right = QWidget()
        rv = QVBoxLayout(right)

        g_file = QGroupBox("图像与视野")
        fl = QGridLayout(g_file)
        self.btn_open = QPushButton("打开图像(可多选)")
        self.btn_demo = QPushButton("生成演示图")
        self.btn_analyze = QPushButton("识别当前图")
        self.btn_analyze_all = QPushButton("识别全部视野")
        self.btn_analyze.setStyleSheet("background:#2d7;color:#fff;font-weight:bold;")
        fl.addWidget(self.btn_open, 0, 0)
        fl.addWidget(self.btn_demo, 0, 1)
        fl.addWidget(self.btn_analyze, 1, 0)
        fl.addWidget(self.btn_analyze_all, 1, 1)
        self.list_images = QListWidget()
        self.list_images.setMaximumHeight(90)
        fl.addWidget(self.list_images, 2, 0, 1, 2)
        rv.addWidget(g_file)

        g_param = QGroupBox("识别参数")
        pl = QGridLayout(g_param)
        pl.addWidget(QLabel("灵敏度"), 0, 0)
        self.sl_sens = QSlider(Qt.Horizontal)
        self.sl_sens.setRange(0, 100)
        self.sl_sens.setValue(self.cfg.sensitivity)
        pl.addWidget(self.sl_sens, 0, 1)
        self.lb_sens = QLabel(str(self.cfg.sensitivity))
        pl.addWidget(self.lb_sens, 0, 2)
        pl.addWidget(QLabel("最小面积px²"), 1, 0)
        self.sp_min_area = QSpinBox(); self.sp_min_area.setRange(1, 10000)
        self.sp_min_area.setValue(self.cfg.min_area)
        pl.addWidget(self.sp_min_area, 1, 1)
        pl.addWidget(QLabel("细胞面积px²"), 1, 2)
        self.sp_cell_area = QSpinBox(); self.sp_cell_area.setRange(500, 200000)
        self.sp_cell_area.setValue(self.cfg.cell_area)
        pl.addWidget(self.sp_cell_area, 1, 3)
        pl.addWidget(QLabel("阳性色相"), 2, 0)
        self.sp_hpos = QDoubleSpinBox(); self.sp_hpos.setRange(0, 180)
        self.sp_hpos.setValue(self.cfg.ref_hue_pos)
        pl.addWidget(self.sp_hpos, 2, 1)
        pl.addWidget(QLabel("阴性色相"), 2, 2)
        self.sp_hneg = QDoubleSpinBox(); self.sp_hneg.setRange(0, 180)
        self.sp_hneg.setValue(self.cfg.ref_hue_neg)
        pl.addWidget(self.sp_hneg, 2, 3)
        rv.addWidget(g_param)

        g_edit = QGroupBox("人工修正")
        el = QGridLayout(g_edit)
        self.btn_del = QPushButton("删除选中对象")
        self.cb_reclass = QComboBox()
        self.cb_reclass.addItems([
            "革兰阳性球菌", "革兰阳性球杆菌", "革兰阳性杆菌",
            "革兰阴性球菌", "革兰阴性球杆菌", "革兰阴性杆菌",
            "真菌孢子(提示)", "真菌菌丝(提示)", "白细胞(提示)", "鳞状上皮/其他细胞(提示)"])
        self.btn_reclass = QPushButton("重分类选中")
        self.btn_undo_sel = QPushButton("清空选择")
        el.addWidget(self.btn_del, 0, 0)
        el.addWidget(self.cb_reclass, 0, 1)
        el.addWidget(self.btn_reclass, 0, 2)
        el.addWidget(self.btn_undo_sel, 0, 3)
        rv.addWidget(g_edit)

        g_spec = QGroupBox("标本信息（报告用）")
        sl = QGridLayout(g_spec)
        sl.addWidget(QLabel("标本类型"), 0, 0)
        self.cb_specimen = QComboBox()
        self.cb_specimen.addItems(["痰", "气管抽吸物", "尿液", "生殖道分泌物",
                                   "脑脊液", "血液培养", "脓液/分泌物", "其他"])
        self.cb_specimen.currentTextChanged.connect(lambda _: self.refresh_report())
        sl.addWidget(self.cb_specimen, 0, 1)
        sl.addWidget(QLabel("视野类型"), 0, 2)
        self.cb_field = QComboBox()
        self.cb_field.addItems(["油镜视野(OIF)", "低倍视野(LPF)"])
        sl.addWidget(self.cb_field, 0, 3)
        sl.addWidget(QLabel("白细胞/LPF"), 1, 0)
        self.sp_wbc = QDoubleSpinBox(); self.sp_wbc.setRange(-1, 999)
        self.sp_wbc.setValue(-1); self.sp_wbc.setSpecialValueText("未计数")
        self.sp_wbc.valueChanged.connect(lambda _: self.refresh_report())
        sl.addWidget(self.sp_wbc, 1, 1)
        sl.addWidget(QLabel("鳞状上皮/LPF"), 1, 2)
        self.sp_sec = QDoubleSpinBox(); self.sp_sec.setRange(-1, 999)
        self.sp_sec.setValue(-1); self.sp_sec.setSpecialValueText("未计数")
        self.sp_sec.valueChanged.connect(lambda _: self.refresh_report())
        sl.addWidget(self.sp_sec, 1, 3)
        rv.addWidget(g_spec)

        g_res = QGroupBox("分类统计（数量=估算菌数）")
        rl = QVBoxLayout(g_res)
        self.table = QTableWidget(0, 3)
        self.table.setHorizontalHeaderLabels(["类别", "估算数量", "半定量(/视野)"])
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setMaximumHeight(170)
        rl.addWidget(self.table)
        rv.addWidget(g_res)

        g_rep = QGroupBox("报告预览")
        rp = QVBoxLayout(g_rep)
        self.txt_report = QTextEdit()
        self.txt_report.setFont(QFont("Microsoft YaHei", 10))
        rp.addWidget(self.txt_report)
        rv.addWidget(g_rep, 1)

        g_exp = QGroupBox("导出")
        xl = QHBoxLayout(g_exp)
        self.btn_csv = QPushButton("导出CSV")
        self.btn_img = QPushButton("导出标注图")
        self.btn_rep = QPushButton("导出报告TXT")
        self.btn_cfg = QPushButton("保存参数")
        xl.addWidget(self.btn_csv); xl.addWidget(self.btn_img)
        xl.addWidget(self.btn_rep); xl.addWidget(self.btn_cfg)
        rv.addWidget(g_exp)

        splitter.addWidget(right)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)

        self.statusBar().showMessage("就绪。分类标准：WS/T 805—2023《临床微生物检验基本技术标准》")

    # ------------------------------------------------------------------
    def _connect(self):
        self.btn_open.clicked.connect(self.open_images)
        self.btn_demo.clicked.connect(self.load_demo)
        self.btn_analyze.clicked.connect(self.analyze_current)
        self.btn_analyze_all.clicked.connect(self.analyze_all)
        self.btn_del.clicked.connect(self.delete_selected)
        self.btn_reclass.clicked.connect(self.reclassify_selected)
        self.btn_undo_sel.clicked.connect(self.clear_selection)
        self.btn_csv.clicked.connect(self.export_csv)
        self.btn_img.clicked.connect(self.export_image)
        self.btn_rep.clicked.connect(self.export_report)
        self.btn_cfg.clicked.connect(self.save_config)
        self.list_images.currentRowChanged.connect(self.switch_image)
        self.canvas.clicked.connect(self.on_canvas_click)
        self.canvas.rect_selected.connect(self.on_rect_selected)
        self.sl_sens.valueChanged.connect(lambda v: self.lb_sens.setText(str(v)))
        self.mode_group.buttonClicked.connect(self._mode_changed)

    def _mode_changed(self, rb):
        if rb is self.rb_select:
            self.canvas.mode = "select"
        elif rb is self.rb_add:
            self.canvas.mode = "add"
        elif rb is self.rb_cal_pos:
            self.canvas.mode = "cal_pos"
        else:
            self.canvas.mode = "cal_neg"

    # ------------------------------------------------------------------
    def open_images(self):
        paths, _ = QFileDialog.getOpenFileNames(
            self, "选择涂片图像", "", "图像文件 (*.png *.jpg *.jpeg *.bmp *.tif *.tiff)")
        if not paths:
            return
        self.image_paths = paths
        self.field_results.clear()
        self.list_images.clear()
        self.list_images.addItems([os.path.basename(p) for p in paths])
        self.list_images.setCurrentRow(0)

    def load_demo(self):
        img = make_synthetic_gram_image()
        path = os.path.join(os.path.expanduser("~"), "gram_demo.png")
        cv2.imwrite(path, img)
        self.image_paths = [path]
        self.field_results.clear()
        self.list_images.clear()
        self.list_images.addItem("演示图(合成)")
        self.list_images.setCurrentRow(0)

    def switch_image(self, row):
        if 0 <= row < len(self.image_paths):
            self.image = cv2.imread(self.image_paths[row])
            self.selected_ids.clear()
            if self.image is not None:
                self.canvas.set_image(cv2_to_qpixmap(self.image))
                self.statusBar().showMessage(f"已加载：{self.image_paths[row]}，请点击“识别当前图”")

    # ------------------------------------------------------------------
    def sync_params(self):
        self.cfg.sensitivity = self.sl_sens.value()
        self.cfg.min_area = self.sp_min_area.value()
        self.cfg.cell_area = self.sp_cell_area.value()
        self.cfg.ref_hue_pos = self.sp_hpos.value()
        self.cfg.ref_hue_neg = self.sp_hneg.value()

    def analyze_current(self):
        if self.image is None:
            QMessageBox.warning(self, "提示", "请先打开图像")
            return
        self.sync_params()
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            self.analyzer.analyze(self.image)
        finally:
            QApplication.restoreOverrideCursor()
        self.selected_ids.clear()
        self._record_field_result()
        self.refresh_view()
        self.refresh_results()
        self.statusBar().showMessage(
            f"识别完成（后端：{self.analyzer.backend_name}），共 {len(self.analyzer.objects)} 个对象，请人工复核")

    def analyze_all(self):
        if not self.image_paths:
            return
        self.sync_params()
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            for p in self.image_paths:
                img = cv2.imread(p)
                if img is None:
                    continue
                an = GramAnalyzer(self.cfg, backend=self.analyzer.backend)
                an.analyze(img)
                self.field_results[os.path.basename(p)] = an.summarize()
            if self.image is not None:
                self.analyzer.analyze(self.image)
        finally:
            QApplication.restoreOverrideCursor()
        self.refresh_view()
        self.refresh_results()
        self.statusBar().showMessage(f"已完成 {len(self.field_results)} 个视野的识别")

    def _record_field_result(self):
        row = self.list_images.currentRow()
        if 0 <= row < len(self.image_paths):
            name = os.path.basename(self.image_paths[row])
            self.field_results[name] = self.analyzer.summarize()

    # ------------------------------------------------------------------
    def refresh_view(self):
        if self.image is None:
            return
        disp = self.image
        if self.cb_labels.isChecked() and self.analyzer.annotated is not None:
            disp = self.analyzer.annotated
        if self.selected_ids:
            disp = disp.copy()
            for o in self.analyzer.objects:
                if o.obj_id in self.selected_ids:
                    cnt = np.array(o.contour, dtype=np.int32).reshape(-1, 1, 2)
                    cv2.drawContours(disp, [cnt], -1, (0, 0, 255), 4)
        self.canvas.refresh_pixmap(cv2_to_qpixmap(disp))

    def refresh_results(self):
        agg = {}
        n_fields = max(len(self.field_results), 1)
        for s in self.field_results.values():
            for cat, cnt in s["by_category_est"].items():
                agg[cat] = agg.get(cat, 0) + cnt
        if not agg and self.analyzer.objects:
            for cat, cnt in self.analyzer.summarize()["by_category_est"].items():
                agg[cat] = cnt
        self.table.setRowCount(len(agg))
        for r, (cat, cnt) in enumerate(sorted(agg.items(), key=lambda x: -x[1])):
            avg = cnt / n_fields
            if "细胞" in cat or "孢子" in cat:
                grade = "人工复核"
            else:
                grade = GramAnalyzer.grade_bacteria_oif(avg)
            self.table.setItem(r, 0, QTableWidgetItem(cat))
            self.table.setItem(r, 1, QTableWidgetItem(str(cnt)))
            self.table.setItem(r, 2, QTableWidgetItem(f"{grade} ({avg:.1f})"))
        self.refresh_report()

    def refresh_report(self):
        wbc = self.sp_wbc.value() if self.sp_wbc.value() >= 0 else None
        sec = self.sp_sec.value() if self.sp_sec.value() >= 0 else None
        n_fields = max(len(self.field_results), 1)
        txt = self.analyzer.report_text(
            specimen=self.cb_specimen.currentText(),
            field_type=self.cb_field.currentText(),
            n_fields=n_fields, wbc=wbc, sec=sec)
        txt += "\n\n" + self.analyzer.clinical_summary(
            specimen=self.cb_specimen.currentText())
        self.txt_report.setPlainText(txt)

    # ------------------------------------------------------------------
    def on_canvas_click(self, x, y):
        mode = self.canvas.mode
        if mode in ("cal_pos", "cal_neg"):
            if self.image is None:
                return
            g = "pos" if mode == "cal_pos" else "neg"
            hue = self.analyzer.calibrate(self.image, (x, y), g)
            if hue is None:
                self.statusBar().showMessage("校准失败：点击区域颜色太浅，请点击菌体中心")
            else:
                if g == "pos":
                    self.sp_hpos.setValue(round(hue, 1))
                else:
                    self.sp_hneg.setValue(round(hue, 1))
                self.statusBar().showMessage(
                    f"{'阳性(紫)' if g=='pos' else '阴性(红)'}参考色相已校准为 {hue:.1f}，请重新识别")
            return
        best, best_d = None, 30.0
        for o in self.analyzer.objects:
            d = ((o.centroid[0]-x)**2 + (o.centroid[1]-y)**2) ** 0.5
            if d < best_d:
                best, best_d = o, d
        if best is not None:
            if best.obj_id in self.selected_ids:
                self.selected_ids.discard(best.obj_id)
            else:
                self.selected_ids.add(best.obj_id)
            self.refresh_view()
            self.statusBar().showMessage(
                f"选中 #{best.obj_id}：{best.category}，面积{best.area:.0f}px²，"
                f"轴比{best.aspect_ratio:.2f}（已选 {len(self.selected_ids)} 个）")

    def on_rect_selected(self, x, y, w, h):
        o = self.analyzer.add_manual_object((x, y, w, h))
        if o is None:
            self.statusBar().showMessage("框选区域未找到明显菌体")
        else:
            self._record_field_result()
            self.refresh_view()
            self.refresh_results()
            self.statusBar().showMessage(f"已手动添加：{o.category}")

    def delete_selected(self):
        if not self.selected_ids:
            self.statusBar().showMessage("未选择任何对象（右键点击图像中的对象进行选择）")
            return
        self.analyzer.delete_objects(list(self.selected_ids))
        self.selected_ids.clear()
        self._record_field_result()
        self.refresh_view()
        self.refresh_results()

    def reclassify_selected(self):
        if not self.selected_ids:
            return
        cat = self.cb_reclass.currentText()
        for oid in list(self.selected_ids):
            self.analyzer.reclassify(oid, cat)
        self.selected_ids.clear()
        self._record_field_result()
        self.refresh_view()
        self.refresh_results()

    def clear_selection(self):
        self.selected_ids.clear()
        self.refresh_view()

    # ------------------------------------------------------------------
    def export_csv(self):
        if not self.analyzer.objects:
            QMessageBox.warning(self, "提示", "无识别结果")
            return
        path, _ = QFileDialog.getSaveFileName(self, "导出CSV", "革兰染色结果.csv", "CSV (*.csv)")
        if path:
            self.analyzer.export_csv(path)
            self.statusBar().showMessage(f"已导出：{path}")

    def export_image(self):
        if self.analyzer.annotated is None:
            return
        path, _ = QFileDialog.getSaveFileName(self, "导出标注图", "革兰染色标注图.png", "PNG (*.png)")
        if path:
            cv2.imwrite(path, self.analyzer.annotated)
            self.statusBar().showMessage(f"已导出：{path}")

    def export_report(self):
        path, _ = QFileDialog.getSaveFileName(self, "导出报告", "革兰染色报告.txt", "TXT (*.txt)")
        if path:
            with open(path, "w", encoding="utf-8") as f:
                f.write(self.txt_report.toPlainText())
            self.statusBar().showMessage(f"已导出：{path}")

    def save_config(self):
        path, _ = QFileDialog.getSaveFileName(self, "保存参数", "gram_config.json", "JSON (*.json)")
        if path:
            self.sync_params()
            self.cfg.to_json(path)
            self.statusBar().showMessage(f"参数已保存：{path}")


def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
