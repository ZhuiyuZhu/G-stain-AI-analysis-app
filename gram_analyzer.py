# -*- coding: utf-8 -*-
"""
革兰染色涂片智能分析引擎
分类与报告标准依据：
  1. WS/T 805—2023《临床微生物检验基本技术标准》（国家卫健委）
  2. 《全国临床检验操作规程》细菌涂片半定量报告方式

核心功能：
  - 颜色分割：革兰阳性（紫/蓝紫）vs 革兰阴性（粉红/红），参考色相距离分类
  - 形态分类：球菌 / 球杆菌 / 杆菌 / 长丝状 / 真菌孢子(提示) / 宿主细胞(提示)
  - 排列分析：成双、短链、长链、葡萄串状、散在
  - 半定量分级：油镜视野(OIF) 1+~4+；低倍视野(LPF)细胞 +~++++
  - 致密团块按覆盖面积估算菌数（临床惯例：葡萄状堆叠不可逐一分辨）
  - 颜色校准：点击已知对照菌（金葡=阳性紫、大肠=阴性红）自动校正参考色相
"""

import cv2
import numpy as np
from dataclasses import dataclass, field, asdict
from typing import List, Tuple, Optional, Dict
import json


@dataclass
class AnalyzerConfig:
    # 参考色相（OpenCV H: 0-180），按与参考的色相距离分类
    ref_hue_pos: float = 140.0   # 结晶紫紫蓝（革兰阳性）参考
    ref_hue_neg: float = 175.0   # 番红粉红（革兰阴性）参考
    hue_window: float = 32.0     # 允许的最大色相距离，超过判"不确定"
    sat_min: int = 40
    # 尺寸过滤（油镜1000x典型成像，需按实际像素标尺调整）
    min_area: int = 25
    yeast_area: int = 1200
    cell_area: int = 8000
    # 形态阈值
    coccus_ar: float = 1.35
    coccobacillus_ar: float = 1.9
    bacillus_ar: float = 5.0
    circularity_min: float = 0.55
    # 排列分析
    group_dist_factor: float = 1.6
    chain_elongation: float = 2.2
    # 分割
    sensitivity: int = 50
    # AI模型后端
    model_conf: float = 0.25   # YOLO置信度阈值

    def to_json(self, path):
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(asdict(self), f, ensure_ascii=False, indent=2)

    @staticmethod
    def from_json(path):
        with open(path, 'r', encoding='utf-8') as f:
            return AnalyzerConfig(**json.load(f))


@dataclass
class DetectedObject:
    obj_id: int
    centroid: Tuple[float, float]
    area: float
    perimeter: float
    aspect_ratio: float
    circularity: float
    mean_hue: float
    mean_rgb: Tuple[int, int, int]
    gram: str = "不确定"
    shape: str = "未知"
    category: str = "未分类"
    contour: list = field(default_factory=list)
    group_id: int = -1
    arrangement: str = ""
    manual: bool = False
    est_count: int = 1   # 致密不可分区域按面积估算的菌数


# ======================================================================
# 检测后端：传统算法 / YOLO深度学习模型（放入模型文件自动启用）
# ======================================================================

# 模型类别ID -> 软件类别（训练时 data.yaml 的类别顺序必须与此一致）
MODEL_CLASS_MAP = {
    0: ("革兰阳性球菌", "革兰阳性", "球菌"),
    1: ("革兰阴性球菌", "革兰阴性", "球菌"),
    2: ("革兰阳性杆菌", "革兰阳性", "杆菌"),
    3: ("革兰阴性杆菌", "革兰阴性", "杆菌"),
    4: ("真菌孢子(提示)", "革兰阳性", "孢子样"),
    5: ("真菌菌丝(提示)", "革兰阳性", "长丝状"),
    6: ("白细胞(提示)", "革兰阳性", "细胞样"),
    7: ("鳞状上皮/其他细胞(提示)", "革兰阴性", "细胞样"),
}

# 模型文件默认搜索名（放在 gram_gui.py 同目录即可）
MODEL_FILENAMES = ("gram_model.pt", "gram_model.onnx")


class YoloBackend:
    """YOLO检测后端：加载 gram_model.pt / gram_model.onnx，输出统一为 DetectedObject。
    依赖 ultralytics（pip install ultralytics），未安装或模型缺失时返回 None 回退传统算法。
    """
    name = "AI模型(YOLO)"

    def __init__(self, model_path: str, conf_thres: float = 0.25):
        from ultralytics import YOLO  # 延迟导入，未装不影响传统模式
        self.model = YOLO(model_path)
        self.conf = conf_thres

    @staticmethod
    def try_load(search_dir: str = ".", cfg: Optional["AnalyzerConfig"] = None):
        import os
        for fn in MODEL_FILENAMES:
            p = os.path.join(search_dir, fn)
            if os.path.exists(p):
                try:
                    conf = getattr(cfg, "model_conf", 0.25) if cfg else 0.25
                    return YoloBackend(p, conf)
                except Exception as e:
                    print(f"[YoloBackend] 模型加载失败，回退传统算法: {e}")
                    return None
        return None

    def detect(self, image_bgr, cfg: "AnalyzerConfig") -> List[DetectedObject]:
        h_img, w_img = image_bgr.shape[:2]
        results = self.model.predict(image_bgr, conf=self.conf, verbose=False)
        objs: List[DetectedObject] = []
        hsv_full = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
        oid = 0
        for r in results:
            boxes = r.boxes
            if boxes is None:
                continue
            for i in range(len(boxes)):
                cls_id = int(boxes.cls[i].item())
                if cls_id not in MODEL_CLASS_MAP:
                    continue
                cat, gram, shape = MODEL_CLASS_MAP[cls_id]
                x1, y1, x2, y2 = [int(v) for v in boxes.xyxy[i].tolist()]
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w_img - 1, x2), min(h_img - 1, y2)
                w, h = max(1, x2 - x1), max(1, y2 - y1)
                area = float(w * h) * 0.85          # 椭圆近似
                ar = max(w, h) / max(min(w, h), 1)
                circ = float(min(1.0, np.pi / 4 * (min(w, h) / max(w, h))))
                roi = hsv_full[y1:y2, y1:x2]
                mean_hue = 0.0
                if roi.size > 0:
                    hp = roi[:, :, 0][roi[:, :, 1] > 20].astype(float)
                    if len(hp):
                        mean_hue = float(GramAnalyzer._circular_mean_hue(hp))
                contour = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
                objs.append(DetectedObject(
                    obj_id=oid, centroid=((x1 + x2) / 2.0, (y1 + y2) / 2.0),
                    area=area, perimeter=float(2 * (w + h)), aspect_ratio=float(ar),
                    circularity=circ, mean_hue=mean_hue, mean_rgb=(0, 0, 0),
                    gram=gram, shape=shape, category=cat, contour=contour))
                oid += 1
        return objs


class GramAnalyzer:
    def __init__(self, config: Optional[AnalyzerConfig] = None,
                 backend: Optional[object] = None, model_dir: str = "."):
        self.cfg = config or AnalyzerConfig()
        self.objects: List[DetectedObject] = []
        self.image = None
        self.annotated = None
        # 后端：优先AI模型；未提供时自动在 model_dir 搜索模型文件；找不到则传统算法
        self.backend = backend if backend is not None else YoloBackend.try_load(model_dir, self.cfg)
        self.backend_name = self.backend.name if self.backend else "传统图像算法"

    # ------------------------------------------------------------------
    @staticmethod
    def _hue_dist(a: float, b: float) -> float:
        d = abs(a - b) % 180
        return min(d, 180 - d)

    def calibrate(self, image_bgr, point: Tuple[int, int], gram_type: str, radius: int = 8):
        """点击已知对照菌校正参考色相（金葡ATCC25923=pos，大肠ATCC25922=neg）。"""
        x, y = point
        hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
        patch = hsv[max(0, y-radius):y+radius, max(0, x-radius):x+radius]
        sat = patch[:, :, 1].astype(float)
        if sat.max() < self.cfg.sat_min:
            return None
        h = patch[:, :, 0].astype(float) * 2
        w = np.clip(sat - self.cfg.sat_min, 0, None)
        ang = np.deg2rad(h)
        mean_ang = np.arctan2((w*np.sin(ang)).sum(), (w*np.cos(ang)).sum())
        mean_hue = (np.rad2deg(mean_ang) % 360) / 2
        if gram_type == 'pos':
            self.cfg.ref_hue_pos = mean_hue
        else:
            self.cfg.ref_hue_neg = mean_hue
        return mean_hue

    # ------------------------------------------------------------------
    def analyze(self, image_bgr) -> List[DetectedObject]:
        self.image = image_bgr.copy()
        self.objects = []
        if self.backend is not None:
            # AI模型后端：检测 -> 排列分析 -> 标注绘制（复用统一后处理）
            objects = self.backend.detect(image_bgr, self.cfg)
            self._analyze_arrangement(objects)
            self.objects = objects
            self.annotated = self._draw(image_bgr, objects)
            return objects
        hsv = self._prep_hsv(image_bgr)
        mask = self._clean_mask(self._object_mask(hsv))
        markers, est_map = self._split_touching(mask)
        objects = self._extract_objects(markers, hsv, image_bgr, est_map)
        self._classify_all(objects)
        self._analyze_arrangement(objects)
        self.objects = objects
        self.annotated = self._draw(image_bgr, objects)
        return objects

    def _prep_hsv(self, img):
        """H、S取原图；V做光照除法校正（只校正亮度，避免洗掉大目标的颜色）。"""
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
        v = hsv[:, :, 2] + 1.0
        bg = cv2.GaussianBlur(v, (0, 0), 61)
        hsv[:, :, 2] = np.clip(v / bg * float(bg.mean()), 0, 255)
        return hsv.astype(np.uint8)

    def _object_mask(self, hsv):
        c = self.cfg
        H = hsv[:, :, 0].astype(np.float32)
        S = hsv[:, :, 1]
        V = hsv[:, :, 2]
        d_pos = np.minimum(np.abs(H - c.ref_hue_pos), 180 - np.abs(H - c.ref_hue_pos))
        d_neg = np.minimum(np.abs(H - c.ref_hue_neg), 180 - np.abs(H - c.ref_hue_neg))
        in_hue = (d_pos <= c.hue_window * 1.4) | (d_neg <= c.hue_window * 1.4)
        sat = S >= c.sat_min
        v_max = 245 - int((c.sensitivity - 50) * 0.6)
        dark = V <= v_max
        return (in_hue & sat & dark).astype(np.uint8) * 255

    def _clean_mask(self, mask):
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
        return mask

    # ------------------------------------------------------------------
    def _split_touching(self, mask):
        """逐连通域判断并分割粘连。返回 (markers, est_map)。"""
        markers_out = np.zeros(mask.shape, np.int32)
        est_map: Dict[int, int] = {}
        cur = 1
        n_comp, comp = cv2.connectedComponents(mask)
        dist_full = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
        for lab in range(1, n_comp):
            cmask = (comp == lab).astype(np.uint8) * 255
            area = int(cmask.sum() // 255)
            if area < self.cfg.min_area:
                continue
            d = dist_full.copy()
            d[cmask == 0] = 0
            cnts, _ = cv2.findContours(cmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cnt = max(cnts, key=cv2.contourArea)
            if len(cnt) >= 5:
                (_, _), (ma, MA), _ = cv2.fitEllipse(cnt)
                ar = max(ma, MA) / max(min(ma, MA), 1e-3)
            else:
                x, y, w, h = cv2.boundingRect(cnt)
                ar = max(w, h) / max(min(w, h), 1)

            # 宿主细胞整体保留，不分割
            if area >= self.cfg.cell_area:
                markers_out[cmask > 0] = cur
                cur += 1
                continue

            # 小核局部峰 -> 30分位数估计真实单细胞半径
            k0 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
            lm0 = cv2.dilate(d, k0)
            pk0 = (d >= lm0 - 0.3) & (d >= 2.5)
            vals = d[pk0]
            need_split = False
            elongated = ar >= 1.8
            r_est, single_est = 0.0, 0.0
            if len(vals) >= 2:
                r_est = float(np.percentile(vals, 30))
                single_est = np.pi * r_est * r_est
                if area >= 1.6 * single_est:
                    need_split = True
            if not need_split:
                markers_out[cmask > 0] = cur
                cur += 1
                continue

            # 截顶距离图 + 高斯平滑去像素阶梯 -> 实心长杆单峰、致密团多峰
            d2 = np.minimum(d, 1.3 * r_est).astype(np.float32)
            d2 = cv2.GaussianBlur(d2, (0, 0), 1.0)
            ksz = max(9, int(1.4 * r_est) | 1)
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksz, ksz))
            local_max = cv2.dilate(d2, k)
            min_peak = max(2.5, 0.60 * r_est - (self.cfg.sensitivity - 50) / 80.0)
            peaks = ((d2 >= local_max - 0.8) & (d2 >= min_peak)).astype(np.uint8)
            npk, pk = cv2.connectedComponents(peaks)
            if npk <= 2:
                markers_out[cmask > 0] = cur
                if not elongated:
                    est_map[cur] = max(1, int(round(area / single_est)))
                cur += 1
                continue
            m = pk.astype(np.int32)
            ws = cv2.cvtColor(cmask, cv2.COLOR_GRAY2BGR)
            cv2.watershed(ws, m)
            got = 0
            for sub in range(1, npk):
                region = (m == sub) & (cmask > 0)
                rarea = int(region.sum())
                if rarea >= self.cfg.min_area:
                    markers_out[region] = cur
                    est_map[cur] = 1 if elongated else max(1, int(round(rarea / single_est)))
                    cur += 1
                    got += 1
            if got == 0:
                markers_out[cmask > 0] = cur
                if not elongated:
                    est_map[cur] = max(1, int(round(area / single_est)))
                cur += 1
        return markers_out, est_map

    # ------------------------------------------------------------------
    def _extract_objects(self, markers, hsv, img, est_map):
        objs = []
        labels = [l for l in np.unique(markers) if l > 0]
        for i, lab in enumerate(labels):
            region = (markers == lab).astype(np.uint8) * 255
            contours, _ = cv2.findContours(region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                continue
            cnt = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(cnt)
            if area < self.cfg.min_area:
                continue
            peri = cv2.arcLength(cnt, True)
            if len(cnt) >= 5:
                (cx, cy), (ma, MA), _ = cv2.fitEllipse(cnt)
                long_ax, short_ax = max(ma, MA), max(min(ma, MA), 1e-3)
            else:
                x, y, w, h = cv2.boundingRect(cnt)
                cx, cy = x + w/2, y + h/2
                long_ax, short_ax = max(w, h), max(min(w, h), 1)
            ar = long_ax / short_ax
            circ = 4 * np.pi * area / (peri * peri + 1e-6)
            pix = img[region > 0]
            hp = hsv[:, :, 0][region > 0].astype(float)
            mean_rgb = tuple(int(v) for v in pix[:, ::-1].mean(axis=0))
            objs.append(DetectedObject(
                obj_id=i, centroid=(float(cx), float(cy)), area=float(area),
                perimeter=float(peri), aspect_ratio=float(ar),
                circularity=float(min(circ, 1.0)),
                mean_hue=float(self._circular_mean_hue(hp)),
                mean_rgb=mean_rgb,
                contour=cnt.reshape(-1, 2).tolist(),
                est_count=int(est_map.get(lab, 1))))
        return objs

    @staticmethod
    def _circular_mean_hue(hues):
        ang = np.deg2rad(hues * 2)
        m = np.arctan2(np.sin(ang).mean(), np.cos(ang).mean())
        return (np.rad2deg(m) % 360) / 2

    # ------------------------------------------------------------------
    def _classify_all(self, objs):
        c = self.cfg
        for o in objs:
            h = o.mean_hue
            d_pos = self._hue_dist(h, c.ref_hue_pos)
            d_neg = self._hue_dist(h, c.ref_hue_neg)
            if d_pos <= d_neg and d_pos <= c.hue_window:
                o.gram = "革兰阳性"
            elif d_neg < d_pos and d_neg <= c.hue_window:
                o.gram = "革兰阴性"
            else:
                o.gram = "不确定"

            if o.area >= c.cell_area:
                o.shape = "细胞样"
                o.category = "白细胞(提示)" if o.gram == "革兰阳性" else "鳞状上皮/其他细胞(提示)"
                continue
            if o.area >= c.yeast_area and o.aspect_ratio < c.coccus_ar and o.gram == "革兰阳性":
                o.shape = "孢子样"
                o.category = "真菌孢子(提示)"
                continue

            if o.aspect_ratio < c.coccus_ar and o.circularity >= c.circularity_min * 0.8:
                o.shape = "球菌"
            elif o.aspect_ratio < c.coccobacillus_ar:
                o.shape = "球杆菌"
            elif o.aspect_ratio < c.bacillus_ar:
                o.shape = "杆菌"
            else:
                o.shape = "长丝状"

            g = "阳性" if o.gram == "革兰阳性" else ("阴性" if o.gram == "革兰阴性" else "")
            s = {"球菌": "球菌", "球杆菌": "球杆菌", "杆菌": "杆菌", "长丝状": "杆菌(长丝)"}[o.shape]
            o.category = f"革兰{g}{s}" if g else f"染色不定{s}"

    # ------------------------------------------------------------------
    def _analyze_arrangement(self, objs):
        cocci = [o for o in objs if o.shape == "球菌"]
        if not cocci:
            return
        diams = [2 * np.sqrt(o.area / np.pi) for o in cocci]
        pts = np.array([o.centroid for o in cocci])
        n = len(cocci)
        parent = list(range(n))

        def find(a):
            while parent[a] != a:
                parent[a] = parent[parent[a]]
                a = parent[a]
            return a

        def union(a, b):
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[ra] = rb

        for i in range(n):
            for j in range(i+1, n):
                d = np.linalg.norm(pts[i] - pts[j])
                if d < self.cfg.group_dist_factor * max(diams[i], diams[j]):
                    union(i, j)

        groups: Dict[int, List[int]] = {}
        for i in range(n):
            groups.setdefault(find(i), []).append(i)

        for gid, members in groups.items():
            est_sum = sum(cocci[m].est_count for m in members)
            if len(members) == 1 and est_sum <= 2:
                desc = "散在"
            elif len(members) <= 2 and est_sum <= 2:
                desc = "成双排列"
            else:
                coords = pts[members] - pts[members].mean(axis=0)
                if len(members) >= 3:
                    eig = np.linalg.eigvalsh(np.cov(coords.T))[::-1]
                    elong = np.sqrt(eig[0] / max(eig[1], 1e-6))
                else:
                    elong = 1.0
                if elong >= self.cfg.chain_elongation:
                    desc = "长链排列" if est_sum >= 6 else "短链排列"
                else:
                    desc = "葡萄状排列" if est_sum >= 5 else "聚集排列"
            for m in members:
                cocci[m].group_id = gid
                cocci[m].arrangement = desc

    # ------------------------------------------------------------------
    def summarize(self) -> dict:
        cats, cats_est = {}, {}
        for o in self.objects:
            cats[o.category] = cats.get(o.category, 0) + 1
            cats_est[o.category] = cats_est.get(o.category, 0) + o.est_count
        return {
            "total": len(self.objects),
            "by_category": cats,
            "by_category_est": cats_est,
            "by_gram": {
                "革兰阳性": sum(o.est_count for o in self.objects if o.gram == "革兰阳性"),
                "革兰阴性": sum(o.est_count for o in self.objects if o.gram == "革兰阴性"),
                "不确定": sum(o.est_count for o in self.objects if o.gram == "不确定"),
            },
        }

    @staticmethod
    def grade_bacteria_oif(avg_per_field: float) -> str:
        """油镜视野细菌半定量（全国临床检验操作规程）"""
        if avg_per_field < 1:
            return "1+"
        if avg_per_field <= 5:
            return "2+"
        if avg_per_field <= 30:
            return "3+"
        return "4+"

    @staticmethod
    def grade_cells_lpf(avg_per_field: float) -> str:
        """低倍视野细胞半定量"""
        if avg_per_field < 1:
            return "+"
        if avg_per_field <= 9:
            return "++"
        if avg_per_field <= 25:
            return "+++"
        return "++++"

    @staticmethod
    def sputum_quality(wbc_per_lpf: float, sec_per_lpf: float) -> str:
        """痰标本质量评估（WS/T 805 / 操作规程）"""
        if wbc_per_lpf > 25 and sec_per_lpf < 10:
            return "合格（白细胞>25/LPF，鳞状上皮细胞<10/LPF）"
        if sec_per_lpf >= 25:
            return "不合格（鳞状上皮细胞≥25/LPF，提示口咽污染，建议重新留取）"
        if wbc_per_lpf < 10 and sec_per_lpf < 10:
            return "介于中间（如为免疫抑制/吸入性肺炎患者可按合格处理）"
        return "可接受（需结合临床，建议与培养结果核对）"

    # ------------------------------------------------------------------
    # 菌群比例简报：面向四个临床问题的简化输出
    # （菌群失调判断 / 血培养报警 / 高危形态提示 / 真菌检出）
    # ------------------------------------------------------------------
    def clinical_summary(self, specimen: str = "痰") -> str:
        est = self.summarize()["by_category_est"]
        bac = {k: v for k, v in est.items()
               if "细胞" not in k and "孢子" not in k and "菌丝" not in k}
        total = sum(bac.values())
        lines = ["【菌群比例简报（供临床参考，不能替代鉴定与药敏）】"]
        spores = sum(v for k, v in est.items() if "孢子" in k)
        hyphae = sum(v for k, v in est.items() if "菌丝" in k)

        if total == 0 and spores == 0 and hyphae == 0:
            lines.append("本视野未检出明确菌体及真菌结构。")
            return "\n".join(lines)

        # 1) 各类菌比例
        for k, v in sorted(bac.items(), key=lambda x: -x[1]):
            lines.append(f"  {k}：约 {v/total*100:.0f}%（{v}个）")
        gp = sum(v for k, v in bac.items() if k.startswith("革兰阳性"))
        gn = sum(v for k, v in bac.items() if k.startswith("革兰阴性"))
        if total > 0:
            lines.append(f"  革兰阳性 : 革兰阴性 ≈ {gp} : {gn}")
            top = max(bac, key=bac.get)
            lines.append(f"  优势菌群：{top}（约 {bac[top]/total*100:.0f}%）")

        # 2) 菌群失调提示（呼吸道标本）
        if specimen in ("痰", "气管抽吸物") and total > 0 and gn / total >= 0.5:
            lines.append("  >> 革兰阴性杆菌占比≥50%：阴性杆菌优势，警惕菌群失调/"
                         "条件致病菌定植或感染，建议结合培养结果。")

        # 3) 血培养阳性报警
        if specimen == "血液培养" and total > 0:
            top = max(bac, key=bac.get)
            lines.append(f"  >> 血培养涂片查见{top}：建议按危急值流程立即报告临床，"
                         "并注明形态染色特征供经验用药参考。")

        # 4) 真菌
        if hyphae > 0 and spores > 0:
            lines.append("  >> 查见孢子及菌丝/假菌丝：提示念珠菌属可能，"
                         "建议转种真菌显色培养基并人工复核。")
        elif hyphae > 0:
            lines.append("  >> 查见菌丝样结构：建议真菌培养并人工复核。")
        elif spores > 0:
            lines.append("  >> 查见孢子样结构：需与酵母样真菌鉴别，建议人工复核。")

        # 5) 高危形态学提示（基于 类别+排列 模式）
        arr_pairs = {(o.category, o.arrangement) for o in self.objects}
        if ("革兰阳性球菌", "葡萄状排列") in arr_pairs:
            lines.append("  >> 革兰阳性球菌呈葡萄状排列：警惕金黄色葡萄球菌，建议凝固酶试验/培养确认。")
        if ("革兰阳性球菌", "成双排列") in arr_pairs and specimen in ("痰", "气管抽吸物", "脑脊液"):
            lines.append("  >> 革兰阳性双球菌：呼吸道/脑脊液标本需警惕肺炎链球菌，建议奥普托欣/荚膜肿胀试验。")
        if ("革兰阴性球菌", "成双排列") in arr_pairs and specimen in ("生殖道分泌物", "脑脊液", "脓液/分泌物"):
            lines.append("  >> 革兰阴性双球菌：警惕奈瑟菌属（淋球菌/脑膜炎奈瑟菌），"
                         "注意是否位于中性粒细胞内，建议立即报告。")
        if any(k.startswith("革兰阴性杆菌") for k in bac) and specimen == "生殖道分泌物":
            lines.append("  >> 生殖道标本革兰阴性杆菌占优势：提示菌群失调可能（正常应以乳杆菌为主）。")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    def report_text(self, specimen="痰", field_type="油镜视野(OIF)",
                    n_fields=1, wbc=None, sec=None) -> str:
        lines = ["【革兰染色涂片镜检报告】", f"标本类型：{specimen}",
                 f"观察视野：{field_type} × {n_fields}"]
        if not self.objects:
            lines.append("镜检结果：革兰染色，未见明确菌体。")
        else:
            lines.append("镜检结果：")
            total_bacteria = sum(o.est_count for o in self.objects
                                 if "细胞" not in o.category and "孢子" not in o.category)
            agg = {}
            for o in self.objects:
                key = (o.category, o.arrangement)
                agg[key] = agg.get(key, 0) + o.est_count
            for (cat, arr), cnt in sorted(agg.items(), key=lambda x: -x[1]):
                avg = cnt / max(n_fields, 1)
                if "细胞" in cat or "孢子" in cat:
                    lines.append(f"  · 可见{cat}（{cnt}个，建议人工复核）")
                    continue
                grade = self.grade_bacteria_oif(avg)
                ratio = cnt / max(total_bacteria, 1) * 100
                arr_txt = f"，{arr}" if arr else ""
                lines.append(f"  · 查见{cat}{arr_txt}，半定量 {grade}"
                             f"（约{avg:.1f}个/视野，占菌体总数{ratio:.0f}%）")
        if wbc is not None or sec is not None:
            lines.append("细胞学：")
            if wbc is not None:
                lines.append(f"  · 白细胞：{self.grade_cells_lpf(wbc)}（{wbc:.0f}个/低倍视野）")
            if sec is not None:
                lines.append(f"  · 鳞状上皮细胞：{self.grade_cells_lpf(sec)}（{sec:.0f}个/低倍视野）")
            if wbc is not None and sec is not None and specimen in ("痰", "气管抽吸物"):
                lines.append(f"标本质量评估：{self.sputum_quality(wbc, sec)}")
        lines.append("（注：本结果由图像算法辅助生成，须由检验人员镜下复核后发出。）")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    COLOR_MAP = {
        "革兰阳性": (200, 0, 200),
        "革兰阴性": (0, 160, 0),
        "不确定": (0, 200, 255),
    }

    def _draw(self, img, objs):
        out = img.copy()
        font = cv2.FONT_HERSHEY_SIMPLEX
        for o in objs:
            color = self.COLOR_MAP.get(o.gram, (128, 128, 128))
            if "细胞" in o.category or "孢子" in o.category:
                color = (255, 128, 0)
            cnt = np.array(o.contour, dtype=np.int32).reshape(-1, 1, 2)
            cv2.drawContours(out, [cnt], -1, color, 2)
            x, y = int(o.centroid[0]), int(o.centroid[1])
            label = self._short_label(o)
            if o.est_count > 1:
                label += f"x{o.est_count}"
            cv2.putText(out, label, (x + 4, y - 4), font, 0.45, color, 1, cv2.LINE_AA)
        return out

    @staticmethod
    def _short_label(o):
        if "白细胞" in o.category:
            return "WBC?"
        if "鳞状上皮" in o.category or "其他细胞" in o.category:
            return "CELL?"
        if "孢子" in o.category:
            return "YST?"
        if "菌丝" in o.category:
            return "HYPH?"
        m = {"革兰阳性球菌": "G+C", "革兰阳性球杆菌": "G+CB", "革兰阳性杆菌": "G+B",
             "革兰阳性杆菌(长丝)": "G+B-", "革兰阴性球菌": "G-C", "革兰阴性球杆菌": "G-CB",
             "革兰阴性杆菌": "G-B", "革兰阴性杆菌(长丝)": "G-B-"}
        return m.get(o.category, "??")

    # ------------------------------------------------------------------
    def delete_objects(self, ids: List[int]):
        self.objects = [o for o in self.objects if o.obj_id not in ids]
        self._analyze_arrangement(self.objects)
        self.annotated = self._draw(self.image, self.objects)

    def add_manual_object(self, rect: Tuple[int, int, int, int]):
        x, y, w, h = rect
        roi = self.image[y:y+h, x:x+w]
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        S, V = hsv[:, :, 1], hsv[:, :, 2]
        sel = (S >= 20) & (V < 250)
        if sel.sum() < 5:
            return None
        cnt_mask = sel.astype(np.uint8) * 255
        contours, _ = cv2.findContours(cnt_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnt = max(contours, key=cv2.contourArea) + np.array([[[x, y]]])
        area = cv2.contourArea(cnt)
        peri = cv2.arcLength(cnt, True)
        if len(cnt) >= 5:
            (cx, cy), (ma, MA), _ = cv2.fitEllipse(cnt)
            ar = max(ma, MA) / max(min(ma, MA), 1e-3)
        else:
            cx, cy = x + w/2, y + h/2
            ar = max(w, h) / max(min(w, h), 1)
        circ = 4*np.pi*area/(peri*peri+1e-6)
        full_hsv = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)
        mask_full = np.zeros(self.image.shape[:2], np.uint8)
        cv2.drawContours(mask_full, [cnt], -1, 255, -1)
        hp = full_hsv[:, :, 0][mask_full > 0].astype(float)
        pix = self.image[mask_full > 0]
        new_id = max([o.obj_id for o in self.objects], default=-1) + 1
        o = DetectedObject(
            obj_id=new_id, centroid=(float(cx), float(cy)), area=float(area),
            perimeter=float(peri), aspect_ratio=float(ar),
            circularity=float(min(circ, 1.0)),
            mean_hue=float(self._circular_mean_hue(hp)),
            mean_rgb=tuple(int(v) for v in pix[:, ::-1].mean(axis=0)),
            contour=cnt.reshape(-1, 2).tolist(), manual=True)
        self._classify_all([o])
        self.objects.append(o)
        self._analyze_arrangement(self.objects)
        self.annotated = self._draw(self.image, self.objects)
        return o

    def reclassify(self, obj_id: int, new_category: str):
        for o in self.objects:
            if o.obj_id == obj_id:
                o.category = new_category
                o.manual = True
                if new_category.startswith("革兰阳性"):
                    o.gram = "革兰阳性"
                elif new_category.startswith("革兰阴性"):
                    o.gram = "革兰阴性"
                for s in ("球菌", "球杆菌", "杆菌", "孢子样", "长丝状"):
                    if s in new_category:
                        o.shape = s
                if "菌丝" in new_category:
                    o.shape = "长丝状"
                if "孢子" in new_category:
                    o.shape = "孢子样"
        self.annotated = self._draw(self.image, self.objects)

    def export_csv(self, path):
        import csv
        with open(path, 'w', newline='', encoding='utf-8-sig') as f:
            w = csv.writer(f)
            w.writerow(["编号", "中心X", "中心Y", "面积px²", "长短轴比", "圆度",
                        "平均色相", "革兰染色", "形态", "报告类别", "排列",
                        "估算菌数", "人工修正"])
            for o in self.objects:
                w.writerow([o.obj_id, f"{o.centroid[0]:.1f}", f"{o.centroid[1]:.1f}",
                            f"{o.area:.0f}", f"{o.aspect_ratio:.2f}", f"{o.circularity:.2f}",
                            f"{o.mean_hue:.1f}", o.gram, o.shape, o.category,
                            o.arrangement, o.est_count, "是" if o.manual else "否"])


def make_synthetic_gram_image(width=1200, height=900, seed=42,
                              n_gpc_clusters=5, n_gnr=25, add_cells=True):
    """生成模拟油镜视野：紫色球菌团 + 粉红杆菌 + 可选白细胞/上皮细胞（供试用）。"""
    rng = np.random.default_rng(seed)
    img = np.full((height, width, 3), (245, 240, 235), np.uint8)
    noise = rng.normal(0, 6, img.shape).astype(np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    PURPLE = (150, 40, 130)
    PINK = (120, 90, 210)

    for _ in range(n_gpc_clusters):
        cx, cy = int(rng.integers(80, width-80)), int(rng.integers(80, height-80))
        mode = rng.choice(['cluster', 'chain'])
        if mode == 'cluster':
            for _ in range(int(rng.integers(6, 14))):
                ox, oy = int(rng.normal(0, 14)), int(rng.normal(0, 14))
                r = int(rng.integers(6, 9))
                cv2.circle(img, (cx+ox, cy+oy), r, PURPLE, -1, cv2.LINE_AA)
        else:
            ang = rng.uniform(0, np.pi)
            for k in range(int(rng.integers(4, 8))):
                ox, oy = int(np.cos(ang)*k*13), int(np.sin(ang)*k*13)
                cv2.circle(img, (cx+ox, cy+oy), int(rng.integers(6, 8)), PURPLE, -1, cv2.LINE_AA)
    for _ in range(n_gnr):
        cx, cy = int(rng.integers(60, width-60)), int(rng.integers(60, height-60))
        L, Wd = int(rng.integers(22, 34)), int(rng.integers(8, 11))
        ang = float(rng.uniform(0, 180))
        cv2.ellipse(img, (cx, cy), (L//2, Wd//2), ang, 0, 360, PINK, -1, cv2.LINE_AA)
    if add_cells:
        for _ in range(2):
            cx, cy = int(rng.integers(150, width-150)), int(rng.integers(150, height-150))
            pts = []
            for a in np.linspace(0, 2*np.pi, 14, endpoint=False):
                rr = rng.integers(58, 78)
                pts.append([int(cx+rr*np.cos(a)), int(cy+rr*np.sin(a))])
            cv2.fillPoly(img, [np.array(pts, np.int32)], (120, 30, 110), cv2.LINE_AA)
        cx, cy = int(rng.integers(200, width-200)), int(rng.integers(200, height-200))
        pts = []
        for a in np.linspace(0, 2*np.pi, 9, endpoint=False):
            rr = rng.integers(90, 130)
            pts.append([int(cx+rr*np.cos(a)), int(cy+rr*np.sin(a))])
        cv2.fillPoly(img, [np.array(pts, np.int32)], (170, 170, 225), cv2.LINE_AA)

    return cv2.GaussianBlur(img, (3, 3), 0)


if __name__ == "__main__":
    test = make_synthetic_gram_image()
    an = GramAnalyzer()
    an.analyze(test)
    print(an.summarize())
    print(an.report_text(specimen="痰", n_fields=1, wbc=18, sec=3))
    cv2.imwrite("test_input.png", test)
    cv2.imwrite("test_annotated.png", an.annotated)
    an.export_csv("test_result.csv")
    print("self-test OK")
