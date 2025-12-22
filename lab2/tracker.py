import cv2
import logging
import numpy as np
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Tuple


# ===== НАСТРОЙКИ ПОЛЬЗОВАТЕЛЯ =====
INPUT_VIDEO_PATH = "input.avi"       # сюда положи своё видео
OUTPUT_VIDEO_PATH = "output.mov"     # путь к выходному видео
OBJECT_LABEL = "Object"              # подпись над объектом
DEBUG_MODE = True                    # включить детальное логирование
LOG_TO_FILE = False                   # сохранять логи в файл
# Показывать ли окна OpenCV (для пакетной обработки удобно отключить)
SHOW_WINDOWS = True

# сделаем трекинг более устойчивым
MIN_MATCHES = 10            # минимум good matches (было 20)
RATIO_TEST = 0.8            # порог теста Лоу (было 0.7 — даём больше матчей пройти)
MIN_INLIERS = 8             # минимум инлиеров RANSAC (было 15)
MAX_REPROJ_ERROR = 10.0     # макс. ошибка репроекции (было 10.0)
MIN_AREA_SCALE = 0.15       # рамка не должна быть меньше 0.15 от исходной площади (было 0.25)
MAX_AREA_SCALE = 20.0       # и не больше 6 раз (было 4.0)
OBJECT_LOST_TOLERANCE = 30  # через сколько кадров без хорошей гомографии считаем объект потерянным (было 15)

# параметры оптического потока
FLOW_MAX_POINTS = 300       # сколько точек отслеживаем максимум
FLOW_MIN_POINTS = 20        # минимум живых точек, чтобы считать гомографию

# сглаживание гомографии, чтобы рамка меньше «дёргалась»
# 0.0 — используем только старую матрицу, 1.0 — полностью новую (без сглаживания)
H_SMOOTHING_ALPHA = 0.3

# максимально допустимое отношение длины самой длинной стороны рамки к самой короткой
MAX_EDGE_RATIO = 5.0
# ==================================


LOGGER = logging.getLogger("tracker")


@dataclass
class TrackingConfig:
    """Конфигурация трекера."""

    input_path: str
    output_path: str
    object_label: str = OBJECT_LABEL
    debug: bool = DEBUG_MODE
    show_windows: bool = SHOW_WINDOWS
    log_to_file: bool = LOG_TO_FILE

    min_matches: int = MIN_MATCHES
    ratio_test: float = RATIO_TEST
    min_inliers: int = MIN_INLIERS
    max_reproj_error: float = MAX_REPROJ_ERROR
    min_area_scale: float = MIN_AREA_SCALE
    max_area_scale: float = MAX_AREA_SCALE
    object_lost_tolerance: int = OBJECT_LOST_TOLERANCE

    flow_max_points: int = FLOW_MAX_POINTS
    flow_min_points: int = FLOW_MIN_POINTS
    h_smoothing_alpha: float = H_SMOOTHING_ALPHA
    max_edge_ratio: float = MAX_EDGE_RATIO


def setup_logging(debug: bool, log_to_file: bool) -> logging.Logger:
    """
    Настройка логгера модуля.
    """
    logger = LOGGER
    logger.setLevel(logging.DEBUG if debug else logging.INFO)
    logger.propagate = False

    # Чистим старые хендлеры (актуально для пакетной обработки)
    if logger.handlers:
        logger.handlers.clear()

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG if debug else logging.INFO)
    console_formatter = logging.Formatter("[%(levelname)s] %(message)s")
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    if log_to_file:
        os.makedirs("logs", exist_ok=True)
        log_filename = os.path.join(
            "logs", f"tracker_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        )
        file_handler = logging.FileHandler(log_filename, encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter(
            "%(asctime)s [%(levelname)s] %(message)s"
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
        logger.info("Логи сохраняются в %s", log_filename)

    return logger


def smooth_homography(
    prev_H: Optional[np.ndarray],
    new_H: np.ndarray,
    alpha: float = H_SMOOTHING_ALPHA,
) -> np.ndarray:
    """
    Сглаживает матрицу гомографии между предыдущим и новым значением.
    """
    if prev_H is None:
        H_smooth = new_H.astype(np.float32)
    else:
        H_new = new_H.astype(np.float64)
        H_prev = prev_H.astype(np.float64)
        H_smooth = (1.0 - alpha) * H_prev + alpha * H_new

    if abs(H_smooth[2, 2]) > 1e-6:
        H_smooth /= H_smooth[2, 2]

    return H_smooth.astype(np.float32)


def init_video_writer(
    first_frame_shape: Tuple[int, int, int],
    fps: float,
    output_path: str,
) -> cv2.VideoWriter:
    """Создаёт объект VideoWriter с параметрами под первый кадр."""
    frame_h, frame_w = first_frame_shape[:2]
    if fps <= 1e-3:
        fps = 25.0  # запасной вариант, если FPS не прочитался

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    return cv2.VideoWriter(output_path, fourcc, fps, (frame_w, frame_h))


def polygon_area(pts):
    """
    Площадь многоугольника по координатам вершин (N x 2).
    """
    x = pts[:, 0]
    y = pts[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def compute_reprojection_error(H, src_pts, dst_pts):
    """
    Считаем среднеквадратичную ошибку репроекции:
    transform(src_pts, H) vs dst_pts.
    src_pts, dst_pts: (N, 1, 2)
    """
    src = src_pts.reshape(-1, 1, 2)
    dst = dst_pts.reshape(-1, 1, 2)

    proj = cv2.perspectiveTransform(src, H)  # (N,1,2)
    diff = dst - proj
    d2 = (diff[:, 0, 0] ** 2 + diff[:, 0, 1] ** 2)
    rmse = np.sqrt(np.mean(d2))
    return rmse


def is_homography_reasonable(H, src_pts, dst_pts, obj_shape, frame_shape, mask, debug=False):
    """
    Комплексная проверка гомографии:
    - достаточно инлиеров
    - маленькая ошибка репроекции
    - рамка в пределах кадра
    - площадь рамки не слишком отличается от исходной
    """
    # 3) рамка & площадь
    h_obj, w_obj = obj_shape
    h_frame, w_frame = frame_shape

    # Проверяем, является ли объект всем кадром
    is_full_frame = (w_obj == w_frame and h_obj == h_frame)

    # 1) инлиеры
    inliers = int(mask.sum())
    # Для полного кадра ослабляем требование к инлиерам
    min_inliers_required = MIN_INLIERS // 2 if is_full_frame else MIN_INLIERS
    if inliers < min_inliers_required:
        if debug:
            print(f"  [DEBUG] Недостаточно инлиеров: {inliers} < {min_inliers_required}")
        return False

    # 2) ошибка репроекции
    rmse = compute_reprojection_error(H, src_pts, dst_pts)
    # Для полного кадра сильнее ослабляем проверку ошибки репроекции
    # (так как при большом масштабе сцены и перспективных искажениях
    #  средняя ошибка в пикселях естественно растёт)
    if is_full_frame:
        max_error_allowed = MAX_REPROJ_ERROR * 10.0
    else:
        max_error_allowed = MAX_REPROJ_ERROR
    if rmse > max_error_allowed:
        if debug:
            print(f"  [DEBUG] Слишком большая ошибка репроекции: {rmse:.2f} > {max_error_allowed}")
        return False

    obj_corners = np.float32([
        [0, 0],
        [w_obj, 0],
        [w_obj, h_obj],
        [0, h_obj]
    ]).reshape(-1, 1, 2)

    dst_corners = cv2.perspectiveTransform(obj_corners, H)  # (4,1,2)
    pts = dst_corners.reshape(-1, 2)

    # Для полного кадра ослабляем проверку границ (или вообще пропускаем)
    if not is_full_frame:
        border_margin = 10
        if not (
            (pts[:, 0] >= -border_margin).all() and (pts[:, 0] <= w_frame + border_margin).all() and
            (pts[:, 1] >= -border_margin).all() and (pts[:, 1] <= h_frame + border_margin).all()
        ):
            if debug:
                print(f"  [DEBUG] Углы выходят за границы кадра")
                print(f"    Углы: {pts}")
                print(f"    Границы: [0, 0] - [{w_frame}, {h_frame}]")
            return False

    # 3б) проверка формы четырёхугольника — отсекаем сильно вытянутые рамки («стрелы»)
    edges = np.array([
        pts[1] - pts[0],
        pts[2] - pts[1],
        pts[3] - pts[2],
        pts[0] - pts[3],
    ])
    edge_lengths = np.linalg.norm(edges, axis=1)
    min_len = edge_lengths.min()
    max_len = edge_lengths.max()

    # защита от вырожденных сторон
    if min_len < 1e-3:
        if debug:
            print("  [DEBUG] Слишком маленькая сторона рамки")
        return False

    length_ratio = max_len / min_len
    if length_ratio > MAX_EDGE_RATIO:
        if debug:
            print(
                f"  [DEBUG] Слишком вытянутая рамка: max/min сторон = "
                f"{length_ratio:.2f} > {MAX_EDGE_RATIO}"
            )
        return False

    # 4) площадь
    area_obj = w_obj * h_obj
    area_dst = polygon_area(pts)
    if area_dst <= 1e-3:
        if debug:
            print(f"  [DEBUG] Площадь слишком мала: {area_dst}")
        return False

    scale = area_dst / float(area_obj)
    
    # Для полного кадра ослабляем проверку масштаба (допускаем намного больший диапазон)
    if is_full_frame:
        # допускаем, что проекция кадра может сжаться до 1% или вырасти до 5x
        min_scale = 0.01
        max_scale = 5.0
    else:
        min_scale = MIN_AREA_SCALE
        max_scale = MAX_AREA_SCALE
    
    if scale < min_scale or scale > max_scale:
        if debug:
            print(f"  [DEBUG] Масштаб вне допустимого диапазона: {scale:.3f} (допустимо [{min_scale}, {max_scale}])")
        return False

    if debug:
        print(f"  [DEBUG] Гомография прошла все проверки: inliers={inliers}, rmse={rmse:.2f}, scale={scale:.3f}")
    
    return True



def init_video_capture(path: str):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"Не удалось открыть видео: {path}")
    return cap


def select_object_roi(frame):
    """
    Автоматически выбирает весь кадр как объект для трекинга.
    Возвращает:
        obj_img  - само изображение объекта (весь кадр)
        rect     - (x, y, w, h) – координаты ROI в первом кадре (весь кадр)
    """
    h_frame, w_frame = frame.shape[:2]
    
    # Используем весь кадр
    x, y = 0, 0
    w_roi, h_roi = w_frame, h_frame

    LOGGER.info(
        "Автоматически выбран весь кадр как объект: (%d, %d, %d, %d)",
        x,
        y,
        w_roi,
        h_roi,
    )

    obj_img = frame.copy()
    return obj_img, (x, y, w_roi, h_roi)


def init_feature_extractor():
    """
    ORB — быстрый и бесплатный (в отличие от SIFT/SURF).
    """
    orb = cv2.ORB_create(
        nfeatures=1000,
        scaleFactor=1.2,
        nlevels=8,
        edgeThreshold=31,
        firstLevel=0,
        WTA_K=2,
        scoreType=cv2.ORB_HARRIS_SCORE,
        patchSize=31,
        fastThreshold=20
    )
    return orb


def compute_keypoints_and_descriptors(detector, img_gray):
    keypoints, descriptors = detector.detectAndCompute(img_gray, None)
    return keypoints, descriptors


def init_matcher():
    """
    BFMatcher для ORB-дескрипторов (бинарные, поэтому NORM_HAMMING).
    """
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    return bf


def filter_matches_by_ratio(matches, ratio=0.75):
    """
    Тест Лоу (Lowe’s ratio test): оставляем только хорошие совпадения.
    matches — список списков длины k (knnMatch с k=2)
    """
    good = []
    for m, n in matches:
        if m.distance < ratio * n.distance:
            good.append(m)
    return good


def draw_tracked_object(frame, homography, obj_shape, label: str):
    """
    Рисует рамку по гомографии + подпись.
    obj_shape: (h, w) объекта (ROI).
    """
    h_obj, w_obj = obj_shape

    # четыре угла объекта в его собственной системе координат
    obj_corners = np.float32([
        [0, 0],
        [w_obj, 0],
        [w_obj, h_obj],
        [0, h_obj]
    ]).reshape(-1, 1, 2)

    # проецируем углы на текущий кадр
    dst_corners = cv2.perspectiveTransform(obj_corners, homography)

    # приводим к удобному виду (4, 2)
    pts = dst_corners.reshape(-1, 2)

    # защита от NaN/Inf и слишком «улетевших» координат
    if not np.isfinite(pts).all():
        return frame

    h_frame, w_frame = frame.shape[:2]

    # если все точки сильно вне кадра, просто не рисуем рамку
    if (
        (pts[:, 0] < -w_frame).all() or
        (pts[:, 0] > 2 * w_frame).all() or
        (pts[:, 1] < -h_frame).all() or
        (pts[:, 1] > 2 * h_frame).all()
    ):
        return frame

    # подрезаем координаты, чтобы они не выходили далеко за границы
    pts[:, 0] = np.clip(pts[:, 0], 0, w_frame - 1)
    pts[:, 1] = np.clip(pts[:, 1], 0, h_frame - 1)

    dst_corners = pts.reshape(-1, 1, 2)

    # рисуем многоугольник
    frame = cv2.polylines(
        frame,
        [np.int32(dst_corners)],
        isClosed=True,
        color=(0, 255, 0),
        thickness=3
    )

    # подпись — берём среднюю точку по X и минимальную по Y
    xs = dst_corners[:, 0, 0]
    ys = dst_corners[:, 0, 1]
    cX = int(xs.mean())
    cY = int(ys.min()) - 10  # чуть выше объекта

    cv2.putText(
        frame,
        label,
        (cX, max(cY, 20)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2,
        cv2.LINE_AA
    )

    return frame

class Tracker:
    """
    Инкапсулирует состояние и шаги трекинга для одного видео.
    """

    def __init__(self, config: TrackingConfig, logger: Optional[logging.Logger] = None):
        self.config = config
        self.logger = logger or LOGGER

        # Фичи и матчер
        self.detector = init_feature_extractor()
        self.matcher = init_matcher()

        # Состояние объекта
        self.obj_img: Optional[np.ndarray] = None
        self.obj_rect: Optional[Tuple[int, int, int, int]] = None
        self.kp_obj = None
        self.des_obj = None

        # Для оптического потока
        self.flow_src_pts: Optional[np.ndarray] = None
        self.flow_prev_pts: Optional[np.ndarray] = None
        self.prev_gray: Optional[np.ndarray] = None

        # Состояние гомографии
        self.last_good_H: Optional[np.ndarray] = None
        self.frames_since_good: int = self.config.object_lost_tolerance + 1
        self.frame_count: int = 0

    def initialize(self, first_frame: np.ndarray) -> np.ndarray:
        """
        Инициализация трекера по первому кадру: ROI, фичи, точки потока, начальная гомография.
        Возвращает первый кадр с нарисованной исходной рамкой.
        """
        obj_img, obj_rect = select_object_roi(first_frame)
        self.obj_img = obj_img
        self.obj_rect = obj_rect

        obj_gray = cv2.cvtColor(obj_img, cv2.COLOR_BGR2GRAY)
        first_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)

        self.kp_obj, self.des_obj = compute_keypoints_and_descriptors(
            self.detector, obj_gray
        )

        if self.des_obj is None or len(self.kp_obj) == 0:
            raise RuntimeError(
                "Не удалось найти ключевые точки на объекте. "
                "Выбери более текстурный объект."
            )

        self.logger.info("Ключевых точек на объекте: %d", len(self.kp_obj))

        # === инициализация точек для KLT-оптического потока ===
        x, y, w, h = obj_rect
        kp_for_flow = self.kp_obj[: self.config.flow_max_points]
        self.flow_src_pts = np.float32(
            [kp.pt for kp in kp_for_flow]
        ).reshape(-1, 1, 2)  # координаты в системе объекта (ROI)
        self.flow_prev_pts = np.float32(
            [[kp.pt[0] + x, kp.pt[1] + y] for kp in kp_for_flow]
        ).reshape(-1, 1, 2)  # координаты в первом кадре (глобальные)
        self.prev_gray = first_gray.copy()
        # ======================================================

        # состояние трекинга
        h_obj, w_obj = obj_img.shape[:2]
        h_frame, w_frame = first_frame.shape[:2]
        is_full_frame = (w_obj == w_frame and h_obj == h_frame)

        if is_full_frame:
            # Для полного кадра используем единичную матрицу как начальную гомографию
            self.last_good_H = np.eye(3, dtype=np.float32)
            self.frames_since_good = 0
            self.logger.info("Объект - весь кадр, инициализирована единичная гомография")
        else:
            self.last_good_H = None
            # считаем, что объект ещё не найден (поэтому рамку не рисуем,
            # пока не появится первая хорошая гомография)
            self.frames_since_good = self.config.object_lost_tolerance + 1

        # обработка первого кадра (рисуем рамку по исходному ROI)
        frame_to_write = first_frame.copy()
        cv2.rectangle(
            frame_to_write,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            3,
        )
        cv2.putText(
            frame_to_write,
            self.config.object_label,
            (x, max(y - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
        return frame_to_write

    def _update_by_flow(
        self, frame_gray: np.ndarray, frame_bgr: np.ndarray
    ) -> bool:
        """
        Пытается обновить гомографию по оптическому потоку.
        Возвращает True, если гомография была успешно обновлена.
        """
        if self.flow_prev_pts is None or len(self.flow_prev_pts) == 0:
            return False

        p1, st, err = cv2.calcOpticalFlowPyrLK(
            self.prev_gray,
            frame_gray,
            self.flow_prev_pts,
            None,
            winSize=(21, 21),
            maxLevel=3,
            criteria=(
                cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                30,
                0.01,
            ),
        )

        st = st.reshape(-1)
        good_mask = st == 1
        flow_curr = p1[good_mask]
        flow_src_good = self.flow_src_pts[good_mask]

        updated_by_flow = False

        if len(flow_curr) >= self.config.flow_min_points:
            src_flow = flow_src_good.reshape(-1, 1, 2)
            dst_flow = flow_curr.reshape(-1, 1, 2)

            H_flow, mask_flow = cv2.findHomography(
                src_flow, dst_flow, cv2.RANSAC, 5.0
            )
            if H_flow is not None and mask_flow is not None:
                if self.config.debug and self.frame_count % 10 == 0:
                    self.logger.debug(
                        "[Frame %d] Проверка гомографии от оптического потока...",
                        self.frame_count,
                    )
                is_reasonable = is_homography_reasonable(
                    H_flow,
                    src_flow,
                    dst_flow,
                    self.obj_img.shape[:2],
                    frame_bgr.shape[:2],
                    mask_flow,
                    debug=self.config.debug and self.frame_count % 10 == 0,
                )
                if is_reasonable:
                    # сглаживаем гомографию, чтобы рамка вела себя плавнее
                    self.last_good_H = smooth_homography(
                        self.last_good_H,
                        H_flow,
                        alpha=self.config.h_smoothing_alpha,
                    )
                    self.frames_since_good = 0
                    updated_by_flow = True
                    if self.config.debug:
                        self.logger.debug(
                            "[Frame %d] ✓ Гомография от потока принята",
                            self.frame_count,
                        )

                    # визуализируем точки потока (синие)
                    for p in flow_curr:
                        xf, yf = p.ravel()
                        cv2.circle(
                            frame_bgr,
                            (int(xf), int(yf)),
                            2,
                            (255, 0, 0),
                            -1,
                        )
                elif self.config.debug and self.frame_count % 10 == 0:
                    self.logger.debug(
                        "[Frame %d] ✗ Гомография от потока отклонена",
                        self.frame_count,
                    )

        # обновляем точки и prev_gray для следующего шага потока
        if len(flow_curr) > 0:
            self.flow_prev_pts = flow_curr.reshape(-1, 1, 2)
            self.flow_src_pts = flow_src_good.reshape(-1, 1, 2)
        else:
            self.flow_prev_pts = None
            self.flow_src_pts = None

        self.prev_gray = frame_gray.copy()
        return updated_by_flow

    def _update_by_orb(
        self, frame_gray: np.ndarray, frame_bgr: np.ndarray
    ) -> int:
        """
        Пытается обновить гомографию по ORB-матчингам.
        Возвращает количество good matches.
        """
        kp_frame, des_frame = compute_keypoints_and_descriptors(
            self.detector, frame_gray
        )

        if des_frame is None or len(kp_frame) == 0:
            return 0

        # knnMatch (k=2) для теста Лоу
        matches_knn = self.matcher.knnMatch(self.des_obj, des_frame, k=2)

        # фильтрация совпадений
        good_matches = filter_matches_by_ratio(
            matches_knn, ratio=RATIO_TEST
        )

        if len(good_matches) < MIN_MATCHES:
            self.frames_since_good += 1
            if self.config.debug and self.frame_count % 10 == 0:
                self.logger.debug(
                    "[Frame %d] ✗ Недостаточно матчей: %d < %d, frames_since_good=%d",
                    self.frame_count,
                    len(good_matches),
                    MIN_MATCHES,
                    self.frames_since_good,
                )
            return len(good_matches)

        src_pts = np.float32(
            [self.kp_obj[m.queryIdx].pt for m in good_matches]
        ).reshape(-1, 1, 2)
        dst_pts = np.float32(
            [kp_frame[m.trainIdx].pt for m in good_matches]
        ).reshape(-1, 1, 2)

        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

        if H is not None and mask is not None:
            if self.config.debug and self.frame_count % 10 == 0:
                self.logger.debug(
                    "[Frame %d] Проверка гомографии от ORB (matches: %d)...",
                    self.frame_count,
                    len(good_matches),
                )
            is_reasonable = is_homography_reasonable(
                H,
                src_pts,
                dst_pts,
                self.obj_img.shape[:2],
                frame_bgr.shape[:2],
                mask,
                debug=self.config.debug and self.frame_count % 10 == 0,
            )
            if is_reasonable:
                # обновляем "последнюю хорошую" гомографию (со сглаживанием)
                self.last_good_H = smooth_homography(
                    self.last_good_H,
                    H,
                    alpha=self.config.h_smoothing_alpha,
                )

                self.frames_since_good = 0
                if self.config.debug:
                    self.logger.debug(
                        "[Frame %d] ✓ Гомография от ORB принята",
                        self.frame_count,
                    )

                # визуализация инлиеров (красные точки)
                inlier_mask = mask.ravel().astype(bool)
                inlier_pts = dst_pts[inlier_mask]  # (N,1,2)
                for p in inlier_pts:
                    xg, yg = p[0]
                    cv2.circle(
                        frame_bgr,
                        (int(xg), int(yg)),
                        3,
                        (0, 0, 255),
                        -1,
                    )
            else:
                self.frames_since_good += 1
                if self.config.debug and self.frame_count % 10 == 0:
                    self.logger.debug(
                        "[Frame %d] ✗ Гомография от ORB отклонена, frames_since_good=%d",
                        self.frame_count,
                        self.frames_since_good,
                    )
        else:
            self.frames_since_good += 1
            if self.config.debug and self.frame_count % 10 == 0:
                self.logger.debug(
                    "[Frame %d] ✗ Не удалось найти гомографию от ORB, frames_since_good=%d",
                    self.frame_count,
                    self.frames_since_good,
                )

        return len(good_matches)

    def _draw_overlay(self, frame_bgr: np.ndarray, matches_count: int) -> None:
        """
        Подпись со статистикой (сколько совпадений и сколько кадров без хорошей гомографии)
        и отрисовка рамки, если есть последняя хорошая гомография.
        """
        cv2.putText(
            frame_bgr,
            f"matches: {matches_count}, frames_since_good: {self.frames_since_good}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )

        # Рисуем рамку, только если недавно была хорошая гомография
        if (
            self.last_good_H is not None
            and self.frames_since_good <= self.config.object_lost_tolerance
        ):
            try:
                frame_bgr[:] = draw_tracked_object(
                    frame_bgr,
                    self.last_good_H,
                    self.obj_img.shape[:2],
                    self.config.object_label,
                )
            except cv2.error:
                if self.config.debug and self.frame_count % 10 == 0:
                    self.logger.debug(
                        "[Frame %d] Ошибка при рисовании рамки", self.frame_count
                    )
        else:
            if (
                self.config.debug
                and self.frame_count % 10 == 0
                and self.last_good_H is not None
            ):
                self.logger.debug(
                    "[Frame %d] Рамка не рисуется: frames_since_good=%d > %d",
                    self.frame_count,
                    self.frames_since_good,
                    self.config.object_lost_tolerance,
                )

    def process_frame(self, frame_bgr: np.ndarray) -> np.ndarray:
        """
        Обрабатывает очередной кадр: обновляет гомографию (поток / ORB),
        рисует статистику и рамку.
        """
        self.frame_count += 1
        frame_gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

        updated_by_flow = self._update_by_flow(frame_gray, frame_bgr)
        matches_count = 0
        if not updated_by_flow:
            matches_count = self._update_by_orb(frame_gray, frame_bgr)

        self._draw_overlay(frame_bgr, matches_count)
        return frame_bgr


def run_tracker(config: TrackingConfig) -> None:
    """
    Высокоуровневая функция запуска трекинга для одного видео.
    """
    logger = setup_logging(config.debug, config.log_to_file)

    cap = init_video_capture(config.input_path)

    # читаем первый кадр
    ret, first_frame = cap.read()
    if not ret:
        raise RuntimeError("Не удалось прочитать первый кадр видео.")

    tracker = Tracker(config, logger=logger)
    first_output = tracker.initialize(first_frame)

    # подготовка видеозаписи результата
    fps = cap.get(cv2.CAP_PROP_FPS)
    out = init_video_writer(first_frame.shape, fps, config.output_path)

    out.write(first_output)
    if config.show_windows:
        cv2.imshow("Tracking", first_output)

    logger.info("Запуск трекинга. Для выхода нажми 'q'.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break  # видео закончилось

        processed = tracker.process_frame(frame)

        # пишем кадр в выходное видео
        out.write(processed)

        # показываем в реальном времени (можно убрать, если мешает)
        if config.show_windows:
            cv2.imshow("Tracking", processed)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    out.release()
    if config.show_windows:
        cv2.destroyAllWindows()
    logger.info("Готово. Результат сохранён в %s", config.output_path)


def main():
    """
    Совместимый с существующим кодом entry-point.
    Использует глобальные настройки модуля.
    """
    config = TrackingConfig(
        input_path=INPUT_VIDEO_PATH,
        output_path=OUTPUT_VIDEO_PATH,
        object_label=OBJECT_LABEL,
        debug=DEBUG_MODE,
        show_windows=SHOW_WINDOWS,
        log_to_file=LOG_TO_FILE,
    )
    run_tracker(config)


if __name__ == "__main__":
    main()
