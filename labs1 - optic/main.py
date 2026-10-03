# -*- coding: utf-8 -*-
"""
Практическая работа №1. Оптика.
Обнаружение QR-кода камерой и оценка расстояния (OpenCV).

Алгоритм:
1. Загрузка калибровки камеры (results/calibration.npz)
2. Для каждого фото из data/qr_test/:
   - устранение дисторсии
   - детектирование QR
   - оценка расстояния двумя методами (простая формула + solvePnP)
   - расчёт погрешности относительно эталона
   - сохранение визуализации
3. Сохранение таблицы измерений (measurements.csv)
4. Построение графиков (plots.png) и сводной таблицы (pivot_table.csv)
"""

import re
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ---------- НАСТРОЙКИ ----------

RESULTS_DIR = Path('results')
TEST_DIR = Path('data/qr_test')
CALIB_FILE = RESULTS_DIR / 'calibration.npz'

QR_REAL_SIZE_MM = 100.0     # реальный размер QR-кода в мм (10 см)
ERROR_THRESHOLD = 5.0       # порог гипотезы, %

# Создаём папку результатов, если её нет
RESULTS_DIR.mkdir(exist_ok=True)


# ---------- ЗАГРУЗКА КАЛИБРОВКИ ----------

def load_calibration(path: Path):
    """Загружает матрицу камеры и коэффициенты дисторсии."""
    if not path.exists():
        raise FileNotFoundError(
            f'Файл калибровки не найден: {path}\n'
            f'Сначала запустите calibrate.py'
        )
    calib = np.load(path)
    mtx, dist = calib['mtx'], calib['dist']
    focal_px = mtx[0, 0]  # фокусное расстояние в пикселях по оси X
    print(f'[i] Калибровка загружена: fx = {focal_px:.2f} px, '
          f'cx = {mtx[0, 2]:.2f}, cy = {mtx[1, 2]:.2f}')
    return mtx, dist, focal_px


# ---------- ОЦЕНКА РАССТОЯНИЯ ----------

def distance_simple(qr_width_px: float, focal_px: float,
                    real_size_mm: float = QR_REAL_SIZE_MM) -> float:
    """
    Метод 1: подобие треугольников (pinhole-модель).
    d = f * W_real / W_pixels
    Возвращает расстояние в мм.
    """
    return (focal_px * real_size_mm) / qr_width_px


def distance_pnp(corners_2d: np.ndarray, mtx: np.ndarray, dist: np.ndarray,
                 real_size_mm: float = QR_REAL_SIZE_MM):
    """
    Метод 2: solvePnP для плоского квадратного маркера.
    Возвращает (distance_mm, rvec, tvec) или (None, None, None).
    """
    half = real_size_mm / 2.0
    # 3D-координаты углов QR в системе маркера (плоскость Z=0)
    obj_points = np.array([
        [-half, -half, 0],
        [half, -half, 0],
        [half,  half, 0],
        [-half,  half, 0],
    ], dtype=np.float32)

    img_points = np.asarray(corners_2d, dtype=np.float32).reshape(-1, 2)

    success, rvec, tvec = cv2.solvePnP(
        obj_points, img_points, mtx, dist,
        flags=cv2.SOLVEPNP_IPPE_SQUARE
    )
    if not success:
        return None, None, None
    return float(np.linalg.norm(tvec)), rvec, tvec


# ---------- ПАРСИНГ ИМЕНИ ФАЙЛА ----------


def parse_filename(stem: str):
    m = re.match(r'd(\d{2,3})_a(\d{1,2})_(\d+)', stem)
    if not m:
        return None, None, None
    return int(m.group(1)), int(m.group(2)), int(m.group(3))


# ---------- ОБРАБОТКА ОДНОГО ИЗОБРАЖЕНИЯ ----------

qr_detector = cv2.QRCodeDetector()


def process_image(img_path: Path, mtx, dist, focal_px,
                  real_distance_cm=None, angle_deg=None):
    """Обрабатывает одно изображение. Возвращает словарь с результатами."""
    img = cv2.imread(str(img_path))
    if img is None:
        print(f'[!] Не удалось прочитать: {img_path.name}')
        return None

    # Устраняем дисторсию объектива
    img_undist = cv2.undistort(img, mtx, dist)

    # Детектируем QR-код
    data, points, _ = qr_detector.detectAndDecode(img_undist)
    if points is None or len(points) == 0:
        print(f'[!] QR не найден: {img_path.name}')
        return None

    points = points.reshape(-1, 2)

    # Ширина QR в пикселях — среднее двух противоположных сторон
    top = np.linalg.norm(points[0] - points[1])
    bottom = np.linalg.norm(points[3] - points[2])
    qr_width_px = float((top + bottom) / 2.0)

    # Метод 1 — простая формула
    d_simple_mm = distance_simple(qr_width_px, focal_px)
    d_simple_cm = d_simple_mm / 10.0

    # Метод 2 — solvePnP
    d_pnp_mm, rvec, tvec = distance_pnp(points, mtx, dist)
    d_pnp_cm = d_pnp_mm / 10.0 if d_pnp_mm is not None else None

    # Погрешности (если есть эталон)
    err_simple = None
    err_pnp = None
    if real_distance_cm is not None:
        err_simple = abs(d_simple_cm - real_distance_cm) / \
            real_distance_cm * 100.0
        if d_pnp_cm is not None:
            err_pnp = abs(d_pnp_cm - real_distance_cm) / \
                real_distance_cm * 100.0

# ---------- ОБРАБОТКА ОДНОГО ИЗОБРАЖЕНИЯ ----------

    vis = img_undist.copy()
    cv2.polylines(vis, [points.astype(int)], True, (0, 255, 0), 2)
    for i, p in enumerate(points):
        pt = tuple(p.astype(int))
        cv2.circle(vis, pt, 5, (0, 0, 255), -1)
        cv2.putText(vis, str(i), (pt[0] + 5, pt[1] - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    cv2.putText(vis, f'Simple: {d_simple_cm:.1f} cm', (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    if d_pnp_cm is not None:
        cv2.putText(vis, f'PnP: {d_pnp_cm:.1f} cm', (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    if real_distance_cm is not None:
        cv2.putText(vis, f'Real: {real_distance_cm} cm', (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        if err_simple is not None:
            color = (0, 0, 255) if err_simple > ERROR_THRESHOLD else (0, 255, 0)
            cv2.putText(vis, f'Err Simple: {err_simple:.1f}%', (10, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    out_path = RESULTS_DIR / f'vis_{img_path.stem}.jpg'
    cv2.imwrite(str(out_path), vis)

    return {
    'Имя_файла': img_path.name,
    'Дата': data,
    'Дистанция': real_distance_cm,
    'Угол_град': angle_deg,
    'Ширина_QR_пикс': round(qr_width_px, 1),
    'Дистанция_простая_см': round(d_simple_cm, 2),
    'Дистанция_PnP_см': round(d_pnp_cm, 2) if d_pnp_cm is not None else None,
    'Ошибка_простая_%': round(err_simple, 2) if err_simple is not None else None,
    'Ошибка_PnP_%': round(err_pnp, 2) if err_pnp is not None else None,
}


# ---------- ПОСТРОЕНИЕ ГРАФИКОВ ----------

def build_plots(df: pd.DataFrame):
    """Строит графики погрешностей от расстояния и угла."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # --- График 1: ошибка от расстояния при угле 0° ---
    sub = (df[df['Угол_град'] == 0]
           .groupby('Дистанция')['Ошибка_простая_%']
           .agg(['mean', 'std']))
    if not sub.empty:
        axes[0].errorbar(sub.index, sub['mean'], yerr=sub['std'].fillna(0),
                         marker='o', capsize=5, label='Simple')
    sub_pnp = (df[df['Угол_град'] == 0]
               .groupby('Дистанция')['Ошибка_PnP_%']
               .agg(['mean', 'std']))
    if not sub_pnp.empty:
        axes[0].errorbar(sub_pnp.index, sub_pnp['mean'],
                         yerr=sub_pnp['std'].fillna(0),
                         marker='^', capsize=5, label='PnP', color='green')

    axes[0].axhline(ERROR_THRESHOLD, color='r', linestyle='--',
                    label=f'Порог {ERROR_THRESHOLD:.0f}%')
    axes[0].set_xlabel('Реальное расстояние, см')
    axes[0].set_ylabel('Относительная погрешность, %')
    axes[0].set_title('Погрешность от расстояния (угол 0°)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # --- График 2: ошибка от угла при расстоянии 60 см ---
    sub2 = (df[df['Дистанция'] == 60]
            .groupby('Угол_град')['Ошибка_простая_%']
            .agg(['mean', 'std']))
    if not sub2.empty:
        axes[1].errorbar(sub2.index, sub2['mean'], yerr=sub2['std'].fillna(0),
                         marker='s', capsize=5, color='orange', label='Simple')
    sub2_pnp = (df[df['Дистанция'] == 60]
                .groupby('Угол_град')['Ошибка_PnP_%']
                .agg(['mean', 'std']))
    if not sub2_pnp.empty:
        axes[1].errorbar(sub2_pnp.index, sub2_pnp['mean'],
                         yerr=sub2_pnp['std'].fillna(0),
                         marker='D', capsize=5, color='green', label='PnP')

    axes[1].axhline(ERROR_THRESHOLD, color='r', linestyle='--',
                    label=f'Порог {ERROR_THRESHOLD:.0f}%')
    axes[1].set_xlabel('Угол наклона QR, °')
    axes[1].set_ylabel('Относительная погрешность, %')
    axes[1].set_title('Погрешность от угла (расстояние 60 см)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    out = RESULTS_DIR / 'plots.png'
    plt.savefig(out, dpi=150)
    print(f'[i] Графики сохранены: {out}')
    plt.close()

# ---------- MAIN ----------

def main():
    print('=' * 60)
    print('Обработка изображений QR-кода')
    print('=' * 60)

    mtx, dist, focal_px = load_calibration(CALIB_FILE)

    images = sorted(TEST_DIR.glob('*.jpg')) + sorted(TEST_DIR.glob('*.png'))
    if not images:
        print(f'[!] В папке {TEST_DIR} нет изображений.')
        return

    print(f'[i] Найдено изображений: {len(images)}')

    results = []
    for img_path in images:
        d_real, angle, repeat = parse_filename(img_path.stem)
        row = process_image(img_path, mtx, dist, focal_px,
                            real_distance_cm=d_real, angle_deg=angle)
        if row:
            results.append(row)

    if not results:
        print('[!] Ни один QR не распознан. Проверьте качество фото.')
        return

    # --- Таблица измерений ---
    df = pd.DataFrame(results)
    csv_path = RESULTS_DIR / 'measurements.csv'
    df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f'\n[i] Таблица измерений: {csv_path}')
    print('\n=== Результаты измерений ===')
    print(df.to_string(index=False))

    # --- Сводная таблица погрешностей ---
    if df['Ошибка_простая_%'].notna().any():
        pivot = df.pivot_table(
            values='Ошибка_простая_%',
            index='Дистанция',
            columns='Угол_град',
            aggfunc='mean',
        ) 
        pivot_path = RESULTS_DIR / 'pivot_table.csv'
        pivot.round(2).to_csv(pivot_path, encoding='utf-8-sig')
        print(f'\n[i] Сводная таблица: {pivot_path}')
        print('\n=== Средняя погрешность (%) — метод Simple ===')
        print(pivot.round(2).to_string())

    # --- Графики ---
    build_plots(df)

    # --- Краткая статистика по гипотезе ---
    print('\n' + '=' * 60)
    print('ПРОВЕРКА ГИПОТЕЗЫ')
    print('=' * 60)
    sub = df[(df['Дистанция'].between(30, 150)) & (df['Угол_град'] <= 30)]
    if not sub.empty and sub['Ошибка_простая_%'].notna().any():
        mean_err = sub['Ошибка_простая_%'].mean()
        max_err = sub['Ошибка_простая_%'].max()
        print(f'Диапазон 30–150 см, угол ≤ 30°:')
        print(f'  Средняя погрешность Simple: {mean_err:.2f}%')
        print(f'  Максимальная погрешность:    {max_err:.2f}%')
        print(f'  Порог гипотезы:              {ERROR_THRESHOLD:.0f}%')
        if mean_err <= ERROR_THRESHOLD and max_err <= ERROR_THRESHOLD * 2:
            print('  → Гипотеза ПОДТВЕРЖДЕНА')
        else:
            print('  → Гипотеза ОПРОВЕРГНУТА')

    print('\nГотово.')


if __name__ == '__main__':
    main()