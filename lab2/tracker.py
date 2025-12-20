import cv2
import numpy as np


# ===== НАСТРОЙКИ ПОЛЬЗОВАТЕЛЯ =====
INPUT_VIDEO_PATH = "input.avi"      # сюда положи своё видео
OUTPUT_VIDEO_PATH = "output.avi"    # путь к выходному видео
OBJECT_LABEL = "Object"             # подпись над объектом

# сделаем трекинг более устойчивым
MIN_MATCHES = 10          # минимум good matches (было 20)
RATIO_TEST = 0.8          # порог теста Лоу (было 0.7 — даём больше матчей пройти)
MIN_INLIERS = 8           # минимум инлиеров RANSAC (было 15)
MAX_REPROJ_ERROR = 15.0   # макс. ошибка репроекции (было 10.0)
MIN_AREA_SCALE = 0.15     # рамка не должна быть меньше 0.15 от исходной площади (было 0.25)
MAX_AREA_SCALE = 6.0      # и не больше 6 раз (было 4.0)
OBJECT_LOST_TOLERANCE = 30  # через сколько кадров без хорошей гомографии считаем объект потерянным (было 15)

# параметры оптического потока
FLOW_MAX_POINTS = 300       # сколько точек отслеживаем максимум
FLOW_MIN_POINTS = 20        # минимум живых точек, чтобы считать гомографию
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


def is_homography_reasonable(H, src_pts, dst_pts, obj_shape, frame_shape, mask):
    """
    Комплексная проверка гомографии:
    - достаточно инлиеров
    - маленькая ошибка репроекции
    - рамка в пределах кадра
    - площадь рамки не слишком отличается от исходной
    """
    # 1) инлиеры
    inliers = int(mask.sum())
    if inliers < MIN_INLIERS:
        return False

    # 2) ошибка репроекции
    rmse = compute_reprojection_error(H, src_pts, dst_pts)
    if rmse > MAX_REPROJ_ERROR:
        return False

    # 3) рамка & площадь
    h_obj, w_obj = obj_shape
    h_frame, w_frame = frame_shape

    obj_corners = np.float32([
        [0, 0],
        [w_obj, 0],
        [w_obj, h_obj],
        [0, h_obj]
    ]).reshape(-1, 1, 2)

    dst_corners = cv2.perspectiveTransform(obj_corners, H)  # (4,1,2)
    pts = dst_corners.reshape(-1, 2)

    # все точки должны быть в пределах кадра с небольшим запасом
    if not (
        (pts[:, 0] >= -10).all() and (pts[:, 0] <= w_frame + 10).all() and
        (pts[:, 1] >= -10).all() and (pts[:, 1] <= h_frame + 10).all()
    ):
        return False

    # 4) площадь
    area_obj = w_obj * h_obj
    area_dst = polygon_area(pts)
    if area_dst <= 1e-3:
        return False

    scale = area_dst / float(area_obj)
    if scale < MIN_AREA_SCALE or scale > MAX_AREA_SCALE:
        return False

    return True



def init_video_capture(path: str):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"Не удалось открыть видео: {path}")
    return cap


def select_object_roi(frame):
    """
    Даёт пользователю выделить объект на первом кадре.
    Возвращает:
        obj_img  - само изображение объекта (cropped ROI)
        rect     - (x, y, w, h) – координаты ROI в первом кадре
    """
    print("[INFO] Выдели объект мышкой и нажми ENTER/SPACE. Отмена - ESC.")
    roi = cv2.selectROI("Select Object", frame, fromCenter=False, showCrosshair=True)
    cv2.destroyWindow("Select Object")

    x, y, w, h = roi
    if w == 0 or h == 0:
        raise RuntimeError("Объект не был выделен (ROI пустой).")

    obj_img = frame[y:y + h, x:x + w]
    return obj_img, (x, y, w, h)


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
    cv2.imshow("Tracking", frame_to_write)

    print("[INFO] Запуск трекинга. Для выхода нажми 'q'.")

    # состояние трекинга
    last_good_H = None
    # считаем, что объект ещё не найден (поэтому рамку не рисуем, пока не появится первая хорошая гомография)
    frames_since_good = OBJECT_LOST_TOLERANCE + 1

    while True:
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
                if (
                    H_flow is not None
                    and mask_flow is not None
                    and is_homography_reasonable(
                        H_flow,
                        src_flow,
                        dst_flow,
                        obj_img.shape[:2],
                        frame.shape[:2],
                        mask_flow,
                    )
                ):
                    last_good_H = H_flow
                    frames_since_good = 0
                    updated_by_flow = True

                    # визуализируем точки потока (синие)
                    for p in flow_curr:
                        xf, yf = p.ravel()
                        cv2.circle(frame, (int(xf), int(yf)), 2, (255, 0, 0), -1)

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
                cv2.imshow("Tracking", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                continue

            # knnMatch (k=2) для теста Лоу
            matches_knn = matcher.knnMatch(des_obj, des_frame, k=2)

            # фильтрация совпадений
            good_matches = filter_matches_by_ratio(matches_knn, ratio=RATIO_TEST)

            # если совпадений мало — считаем, что объект не найден в этом кадре
            if len(good_matches) >= MIN_MATCHES:
                src_pts = np.float32(
                    [kp_obj[m.queryIdx].pt for m in good_matches]
                ).reshape(-1, 1, 2)
                dst_pts = np.float32(
                    [kp_frame[m.trainIdx].pt for m in good_matches]
                ).reshape(-1, 1, 2)

                H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

                if H is not None and mask is not None:
                    if is_homography_reasonable(
                        H,
                        src_pts,
                        dst_pts,
                        obj_img.shape[:2],
                        frame.shape[:2],
                        mask
                    ):
                        # обновляем "последнюю хорошую" гомографию
                        last_good_H = H
                        frames_since_good = 0

                        # визуализация инлиеров (красные точки)
                        inlier_mask = mask.ravel().astype(bool)
                        inlier_pts = dst_pts[inlier_mask]  # (N,1,2)
                        for p in inlier_pts:
                            xg, yg = p[0]
                            cv2.circle(frame, (int(xg), int(yg)), 3, (0, 0, 255), -1)
                    else:
                        frames_since_good += 1
                else:
                    frames_since_good += 1
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
                pass
        # иначе ничего не рисуем — считаем, что объект временно потерян

        # пишем кадр в выходное видео
        out.write(frame)

        # показываем в реальном времени (можно убрать, если мешает)
        cv2.imshow("Tracking", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"[INFO] Готово. Результат сохранён в {OUTPUT_VIDEO_PATH}")


if __name__ == "__main__":
    main()
