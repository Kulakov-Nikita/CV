import cv2
import numpy as np


# ===== НАСТРОЙКИ ПОЛЬЗОВАТЕЛЯ =====
INPUT_VIDEO_PATH = "input.avi"      # сюда положи своё видео
OUTPUT_VIDEO_PATH = "output.avi"    # путь к выходному видео
OBJECT_LABEL = "Object"             # подпись над объектом

MIN_MATCHES = 20          # минимум good matches
RATIO_TEST = 0.7          # порог теста Лоу
MIN_INLIERS = 15          # минимум инлиеров RANSAC
MAX_REPROJ_ERROR = 10.0   # макс. среднеквадратичная ошибка репроекции (пиксели)
MIN_AREA_SCALE = 0.25     # рамка не должна быть меньше 0.25 от исходной площади
MAX_AREA_SCALE = 4.0      # и не больше 4 раз
OBJECT_LOST_TOLERANCE = 15  # через сколько кадров без хорошей гомографии считаем объект потерянным
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

    # инициализируем ORB и описатели для объекта
    detector = init_feature_extractor()
    kp_obj, des_obj = compute_keypoints_and_descriptors(detector, obj_gray)

    if des_obj is None or len(kp_obj) == 0:
        raise RuntimeError("Не удалось найти ключевые точки на объекте. Выбери более текстурный объект.")

    print(f"[INFO] Ключевых точек на объекте: {len(kp_obj)}")

    matcher = init_matcher()

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

    while True:
        ret, frame = cap.read()
        if not ret:
            break  # видео закончилось

        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

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

        # если совпадений мало — считаем, что объект не найден
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
                    # обновляем "последнюю хорошую"
                    last_good_H = H
                    frames_since_good = 0
                else:
                    frames_since_good += 1
            else:
                frames_since_good += 1
        else:
            frames_since_good += 1

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
