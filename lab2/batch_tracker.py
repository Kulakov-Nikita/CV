import os
import tracker


def main():
    """
    Пройтись по всем видео в папке videos/input и сохранить результаты в videos/output.
    Для каждого файла <name>.<ext> создаётся результат <name>_tracked.avi.
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    input_dir = os.path.join(base_dir, "videos", "input")
    output_dir = os.path.join(base_dir, "videos", "output")

    os.makedirs(output_dir, exist_ok=True)

    # Для пакетной обработки отключаем окна OpenCV
    tracker.SHOW_WINDOWS = False

    videos = [
        f
        for f in os.listdir(input_dir)
        if os.path.isfile(os.path.join(input_dir, f))
    ]

    if not videos:
        print(f"[BATCH] В папке {input_dir} нет видеофайлов")
        return

    print(f"[BATCH] Найдено {len(videos)} видео(файлов) для обработки")

    for filename in videos:
        name, _ext = os.path.splitext(filename)
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, f"{name}_tracked.mov")

        print(f"[BATCH] Обработка: {input_path} -> {output_path}")

        tracker.INPUT_VIDEO_PATH = input_path
        tracker.OUTPUT_VIDEO_PATH = output_path

        try:
            tracker.main()
        except Exception as e:
            # Логируем ошибку, но продолжаем обрабатывать остальные файлы
            print(f"[BATCH] Ошибка при обработке {filename}: {e}")


if __name__ == "__main__":
    main()


