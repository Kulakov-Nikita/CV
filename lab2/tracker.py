import cv2
import numpy as np
import sys
from datetime import datetime


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
    
    print(f"[INFO] Автоматически выбран весь кадр как объект: ({x}, {y}, {w_roi}, {h_roi})")
    
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


def main():
    # Настройка логирования в файл
    log_file = None
    import builtins
    original_print = builtins.print
    
    if LOG_TO_FILE:
        log_filename = f"logs/tracker_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        log_file = open(log_filename, 'w', encoding='utf-8')
        original_print(f"[INFO] Логи сохраняются в {log_filename}")
    
    def log_print(*args, **kwargs):
        """Печатает в консоль и в файл"""
        original_print(*args, **kwargs)
        if log_file:
            original_print(*args, **kwargs, file=log_file)
            log_file.flush()
    
    # Переопределяем print для логирования
    if LOG_TO_FILE:
        builtins.print = log_print
    
    cap = init_video_capture(INPUT_VIDEO_PATH)

    # читаем первый кадр
    ret, first_frame = cap.read()
    if not ret:
        raise RuntimeError("Не удалось прочитать первый кадр видео.")

    # даём пользователю выбрать объект
    obj_img, obj_rect = select_object_roi(first_frame)
    obj_gray = cv2.cvtColor(obj_img, cv2.COLOR_BGR2GRAY)
    first_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
    x, y, w, h = obj_rect

    # инициализируем ORB и описатели для объекта
    detector = init_feature_extractor()
    kp_obj, des_obj = compute_keypoints_and_descriptors(detector, obj_gray)

    if des_obj is None or len(kp_obj) == 0:
        raise RuntimeError("Не удалось найти ключевые точки на объекте. Выбери более текстурный объект.")

    print(f"[INFO] Ключевых точек на объекте: {len(kp_obj)}")

    matcher = init_matcher()

    # === инициализация точек для KLT-оптического потока ===
    kp_for_flow = kp_obj[:FLOW_MAX_POINTS]  # ограничим число точек
    flow_src_pts = np.float32(
        [kp.pt for kp in kp_for_flow]
    ).reshape(-1, 1, 2)  # координаты в системе объекта (ROI)
    flow_prev_pts = np.float32(
        [[kp.pt[0] + x, kp.pt[1] + y] for kp in kp_for_flow]
    ).reshape(-1, 1, 2)  # координаты в первом кадре (глобальные)
    prev_gray = first_gray.copy()
    # ======================================================

    # подготовка видеозаписи результата
    frame_h, frame_w = first_frame.shape[:2]
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 1e-3:
        fps = 25.0  # запасной вариант, если FPS не прочитался

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, fps, (frame_w, frame_h))

    # обработка первого кадра (можно уже нарисовать рамку по исходному ROI)
    frame_to_write = first_frame.copy()
    x, y, w, h = obj_rect
    cv2.rectangle(frame_to_write, (x, y), (x + w, y + h), (0, 255, 0), 3)
    cv2.putText(
        frame_to_write,
        OBJECT_LABEL,
        (x, max(y - 10, 20)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2,
        cv2.LINE_AA
    )
    out.write(frame_to_write)
    if SHOW_WINDOWS:
        cv2.imshow("Tracking", frame_to_write)

    print("[INFO] Запуск трекинга. Для выхода нажми 'q'.")

    # состояние трекинга
    # Если объект - весь кадр, инициализируем единичной матрицей
    h_obj, w_obj = obj_img.shape[:2]
    h_frame, w_frame = first_frame.shape[:2]
    is_full_frame = (w_obj == w_frame and h_obj == h_frame)
    
    if is_full_frame:
        # Для полного кадра используем единичную матрицу как начальную гомографию
        last_good_H = np.eye(3, dtype=np.float32)
        frames_since_good = 0
        print("[INFO] Объект - весь кадр, инициализирована единичная гомография")
    else:
        last_good_H = None
        # считаем, что объект ещё не найден (поэтому рамку не рисуем, пока не появится первая хорошая гомография)
        frames_since_good = OBJECT_LOST_TOLERANCE + 1
    
    frame_count = 0

    while True:
        frame_count += 1
        ret, frame = cap.read()
        if not ret:
            break  # видео закончилось

        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # флаг: удалось ли в этом кадре обновить гомографию по оптическому потоку
        updated_by_flow = False
        good_matches = []

        # ======== шаг 1: попытка обновить гомографию по оптическому потоку ========
        if flow_prev_pts is not None and len(flow_prev_pts) > 0:
            p1, st, err = cv2.calcOpticalFlowPyrLK(
                prev_gray,
                frame_gray,
                flow_prev_pts,
                None,
                winSize=(21, 21),
                maxLevel=3,
                criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
            )

            st = st.reshape(-1)
            good_mask = st == 1
            flow_curr = p1[good_mask]
            flow_src_good = flow_src_pts[good_mask]

            if len(flow_curr) >= FLOW_MIN_POINTS:
                src_flow = flow_src_good.reshape(-1, 1, 2)
                dst_flow = flow_curr.reshape(-1, 1, 2)

                H_flow, mask_flow = cv2.findHomography(src_flow, dst_flow, cv2.RANSAC, 5.0)
                if H_flow is not None and mask_flow is not None:
                    if DEBUG_MODE and frame_count % 10 == 0:
                        print(f"[Frame {frame_count}] Проверка гомографии от оптического потока...")
                    is_reasonable = is_homography_reasonable(
                        H_flow,
                        src_flow,
                        dst_flow,
                        obj_img.shape[:2],
                        frame.shape[:2],
                        mask_flow,
                        debug=DEBUG_MODE and frame_count % 10 == 0
                    )
                    if is_reasonable:
                        # сглаживаем гомографию, чтобы рамка вела себя плавнее
                        if last_good_H is not None:
                            H_new = H_flow.astype(np.float64)
                            H_prev = last_good_H.astype(np.float64)
                            H_smooth = (1.0 - H_SMOOTHING_ALPHA) * H_prev + H_SMOOTHING_ALPHA * H_new
                            # нормализуем, чтобы H[2,2] ≈ 1
                            if abs(H_smooth[2, 2]) > 1e-6:
                                H_smooth /= H_smooth[2, 2]
                            last_good_H = H_smooth.astype(np.float32)
                        else:
                            last_good_H = H_flow

                        frames_since_good = 0
                        updated_by_flow = True
                        if DEBUG_MODE:
                            print(f"[Frame {frame_count}] ✓ Гомография от потока принята")

                        # визуализируем точки потока (синие)
                        for p in flow_curr:
                            xf, yf = p.ravel()
                            cv2.circle(frame, (int(xf), int(yf)), 2, (255, 0, 0), -1)
                    elif DEBUG_MODE and frame_count % 10 == 0:
                        print(f"[Frame {frame_count}] ✗ Гомография от потока отклонена")

            # обновляем точки и prev_gray для следующего шага потока
            if len(flow_curr) > 0:
                flow_prev_pts = flow_curr.reshape(-1, 1, 2)
                flow_src_pts = flow_src_good.reshape(-1, 1, 2)
            else:
                flow_prev_pts = None
                flow_src_pts = None
            prev_gray = frame_gray.copy()
        # ==========================================================================#

        # ======== шаг 2: ORB + матчинги (используем, только если поток не помог) ==
        if not updated_by_flow:
            # ключевые точки на текущем кадре
            kp_frame, des_frame = compute_keypoints_and_descriptors(detector, frame_gray)

            if des_frame is None or len(kp_frame) == 0:
                # нечего матчить — просто пишем кадр как есть
                out.write(frame)
                if SHOW_WINDOWS:
                    cv2.imshow("Tracking", frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                continue

            # knnMatch (k=2) для теста Лоу
            matches_knn = matcher.knnMatch(des_obj, des_frame, k=2)

            # фильтрация совпадений
            good_matches = filter_matches_by_ratio(matches_knn, ratio=RATIO_TEST)

            # если совпадений мало — считаем, что объект не найден в этом кадре
            if len(good_matches) < MIN_MATCHES:
                frames_since_good += 1
                if DEBUG_MODE and frame_count % 10 == 0:
                    print(f"[Frame {frame_count}] ✗ Недостаточно матчей: {len(good_matches)} < {MIN_MATCHES}, frames_since_good={frames_since_good}")
            elif len(good_matches) >= MIN_MATCHES:
                src_pts = np.float32(
                    [kp_obj[m.queryIdx].pt for m in good_matches]
                ).reshape(-1, 1, 2)
                dst_pts = np.float32(
                    [kp_frame[m.trainIdx].pt for m in good_matches]
                ).reshape(-1, 1, 2)

                H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

                if H is not None and mask is not None:
                    if DEBUG_MODE and frame_count % 10 == 0:
                        print(f"[Frame {frame_count}] Проверка гомографии от ORB (matches: {len(good_matches)})...")
                    is_reasonable = is_homography_reasonable(
                        H,
                        src_pts,
                        dst_pts,
                        obj_img.shape[:2],
                        frame.shape[:2],
                        mask,
                        debug=DEBUG_MODE and frame_count % 10 == 0
                    )
                    if is_reasonable:
                        # обновляем "последнюю хорошую" гомографию (со сглаживанием)
                        if last_good_H is not None:
                            H_new = H.astype(np.float64)
                            H_prev = last_good_H.astype(np.float64)
                            H_smooth = (1.0 - H_SMOOTHING_ALPHA) * H_prev + H_SMOOTHING_ALPHA * H_new
                            if abs(H_smooth[2, 2]) > 1e-6:
                                H_smooth /= H_smooth[2, 2]
                            last_good_H = H_smooth.astype(np.float32)
                        else:
                            last_good_H = H

                        frames_since_good = 0
                        if DEBUG_MODE:
                            print(f"[Frame {frame_count}] ✓ Гомография от ORB принята")

                        # визуализация инлиеров (красные точки)
                        inlier_mask = mask.ravel().astype(bool)
                        inlier_pts = dst_pts[inlier_mask]  # (N,1,2)
                        for p in inlier_pts:
                            xg, yg = p[0]
                            cv2.circle(frame, (int(xg), int(yg)), 3, (0, 0, 255), -1)
                    else:
                        frames_since_good += 1
                        if DEBUG_MODE and frame_count % 10 == 0:
                            print(f"[Frame {frame_count}] ✗ Гомография от ORB отклонена, frames_since_good={frames_since_good}")
                else:
                    frames_since_good += 1
                    if DEBUG_MODE and frame_count % 10 == 0:
                        print(f"[Frame {frame_count}] ✗ Не удалось найти гомографию от ORB, frames_since_good={frames_since_good}")
            else:
                frames_since_good += 1
        # ==========================================================================#

        # подпись со статистикой (сколько совпадений и сколько кадров без хорошей гомографии)
        cv2.putText(
            frame,
            f"matches: {len(good_matches)}, frames_since_good: {frames_since_good}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )

        # Рисуем рамку, только если недавно была хорошая гомография
        if last_good_H is not None and frames_since_good <= OBJECT_LOST_TOLERANCE:
            try:
                frame = draw_tracked_object(frame, last_good_H, obj_img.shape[:2], OBJECT_LABEL)
            except cv2.error:
                if DEBUG_MODE and frame_count % 10 == 0:
                    print(f"[Frame {frame_count}] Ошибка при рисовании рамки")
        else:
            if DEBUG_MODE and frame_count % 10 == 0 and last_good_H is not None:
                print(f"[Frame {frame_count}] Рамка не рисуется: frames_since_good={frames_since_good} > {OBJECT_LOST_TOLERANCE}")
        # иначе ничего не рисуем — считаем, что объект временно потерян

        # пишем кадр в выходное видео
        out.write(frame)

        # показываем в реальном времени (можно убрать, если мешает)
        if SHOW_WINDOWS:
            cv2.imshow("Tracking", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    out.release()
    if SHOW_WINDOWS:
        cv2.destroyAllWindows()
    print(f"[INFO] Готово. Результат сохранён в {OUTPUT_VIDEO_PATH}")
    
    if log_file:
        log_file.close()
        import builtins
        builtins.print = original_print


if __name__ == "__main__":
    main()
