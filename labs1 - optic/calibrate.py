import cv2
import numpy as np
import glob
from pathlib import Path

# ---------- НАСТРОЙКИ ----------
CHESSBOARD = (9, 6)      # количество ВНУТРЕННИХ углов (ширина, высота)
SQUARE_SIZE = 25         # размер клетки в мм

CHESS_DIR = 'data/chessboard'
RESULTS_DIR = Path('results')
RESULTS_DIR.mkdir(exist_ok=True)

# ---------- ПРОВЕРКА ВХОДНЫХ ДАННЫХ ----------
images = glob.glob(f'{CHESS_DIR}/*.jpg') + glob.glob(f'{CHESS_DIR}/*.png')
print(f'Найдено изображений: {len(images)}')

if len(images) == 0:
    raise SystemExit(
        f'[!] В папке {CHESS_DIR} нет изображений.\n'
        f'    Положите 15-20 фото шахматной доски (jpg/png) и запустите снова.'
    )

# ---------- ЭТАЛОННЫЕ 3D-ТОЧКИ ----------
objp = np.zeros((CHESSBOARD[0] * CHESSBOARD[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHESSBOARD[0], 0:CHESSBOARD[1]].T.reshape(-1, 2)
objp *= SQUARE_SIZE

objpoints = []   # 3D-точки в реальном мире
imgpoints = []   # 2D-точки на изображениях
img_size = None  # размер изображения (ширина, высота)

# ---------- ПОИСК УГЛОВ НА КАЖДОМ ФОТО ----------
found = 0
for fname in images:
    img = cv2.imread(fname)
    if img is None:
        print(f'[!] Не удалось прочитать: {fname}')
        continue

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Запоминаем размер изображения (нужен для calibrateCamera)
    if img_size is None:
        img_size = gray.shape[::-1]  # (width, height)

    ret, corners = cv2.findChessboardCorners(gray, CHESSBOARD, None)
    if not ret:
        print(f'[!] Доска не найдена: {fname}')
        continue

    # Уточняем углы с субпиксельной точностью
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

    objpoints.append(objp)
    imgpoints.append(corners2)
    found += 1

    # Визуализация найденных углов
    vis = img.copy()
    cv2.drawChessboardCorners(vis, CHESSBOARD, corners2, ret)
    out_name = RESULTS_DIR / f'chessboard_detected_{Path(fname).stem}.jpg'
    cv2.imwrite(str(out_name), vis)

print(f'[i] Успешно обработано: {found} из {len(images)}')

if found < 5:
    raise SystemExit(
        f'[!] Слишком мало кадров с найденной доской ({found}).\n'
        f'    Нужно минимум 5, лучше 15-20. Переснимите доску.'
    )

# ---------- КАЛИБРОВКА ----------
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
    objpoints, imgpoints, img_size, None, None
)

print('\n=== РЕЗУЛЬТАТЫ КАЛИБРОВКИ ===')
print('Матрица камеры (mtx):')
print(mtx)
print('\nКоэффициенты дисторсии (dist):')
print(dist.ravel())
print(f'\nСредняя ошибка репроекции: {ret:.4f} пикселей')
if ret < 1.0:
    print('  → Отличная калибровка')
elif ret < 2.0:
    print('  → Приемлемая калибровка')
else:
    print('  → Плохая калибровка, переснимите доску')

# ---------- СОХРАНЕНИЕ ----------
out_path = RESULTS_DIR / 'calibration.npz'
np.savez(str(out_path), mtx=mtx, dist=dist)
print(f'\n[i] Калибровка сохранена: {out_path}')