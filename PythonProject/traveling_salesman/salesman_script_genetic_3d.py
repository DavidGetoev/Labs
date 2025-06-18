import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
from matplotlib.colors import LightSource, ListedColormap
from scipy.interpolate import griddata
import random
from math import sqrt

# Параметры задачи
n = 5  # Количество объектов
area_size = 100  # Размер области
max_height = 10  # Максимальная высота рельефа
salesman_speed = 2.5  # Скорость коммивояжера
object_speed = 0.25  # Скорость объектов
num_steps = 150  # Количество шагов анимации

# Генерация рельефа (более сложный ландшафт)
np.random.seed(12)
grid_size = 60
x = np.linspace(0, area_size, grid_size)
y = np.linspace(0, area_size, grid_size)
x_grid, y_grid = np.meshgrid(x, y)

# Создаем холмы и впадины
height = np.zeros((grid_size, grid_size))
for _ in range(5):
    x0, y0 = random.uniform(0, area_size), random.uniform(0, area_size)
    peak_height = random.uniform(10, max_height)
    dist = np.sqrt((x_grid - x0) ** 2 + (y_grid - y0) ** 2)
    height += peak_height * np.exp(-dist ** 2 / (0.2 * area_size ** 2))

# Добавляем общий наклон
height += 5 * (x_grid + y_grid) / area_size


# Функция получения высоты
def get_height(x, y):
    return griddata((x_grid.flatten(), y_grid.flatten()),
                    height.flatten(),
                    (x, y), method='cubic')


# Генерация случайных траекторий объектов
trajectories = np.zeros((num_steps, n, 3))
for i in range(n):
    # Начальная позиция (равномерно распределенная)
    start_x = random.uniform(0.1 * area_size, 0.9 * area_size)
    start_y = random.uniform(0.1 * area_size, 0.9 * area_size)

    # Случайное движение (броуновское)
    dx = np.cumsum(np.random.randn(num_steps) * object_speed)
    dy = np.cumsum(np.random.randn(num_steps) * object_speed)

    for t in range(num_steps):
        x = np.clip(start_x + dx[t], 0.05 * area_size, 0.95 * area_size)
        y = np.clip(start_y + dy[t], 0.05 * area_size, 0.95 * area_size)
        z = get_height(x, y)
        trajectories[t, i] = [x, y, z]

# Создание 3D визуализации с улучшенным рельефом
fig = plt.figure(figsize=(16, 12))
ax = fig.add_subplot(111, projection='3d')

# Освещение для лучшего отображения рельефа
ls = LightSource(azdeg=315, altdeg=45)
rgb = ls.shade(height, cmap=plt.cm.terrain, vert_exag=0.5, blend_mode='soft')

# Отрисовка рельефа с текстурами
surf = ax.plot_surface(x_grid, y_grid, height, rstride=1, cstride=1,
                       facecolors=rgb, linewidth=0, antialiased=True, alpha=0.7)

# Элементы визуализации
colors = plt.cm.tab20(np.linspace(0, 1, n))
scatters = [ax.scatter([], [], [], c=colors[i], s=80, depthshade=True,
                       label=f'Объект {i + 1}') for i in range(n)]
salesman_marker, = ax.plot([], [], [], 'ko', markersize=12,
                           markeredgewidth=2, markerfacecolor='yellow')
current_path, = ax.plot([], [], [], 'r-', lw=2, alpha=0.8)
trajectory_line, = ax.plot([], [], [], 'm-', lw=1.5, alpha=0.6)
visited_scatter = ax.scatter([], [], [], c='red', s=100, marker='x', linewidth=2)

# Настройки графика
ax.set_xlim(0, area_size)
ax.set_ylim(0, area_size)
ax.set_zlim(0, max_height * 1.1)
ax.set_xlabel('X координата', fontsize=10)
ax.set_ylabel('Y координата', fontsize=10)
ax.set_zlabel('Высота', fontsize=10)
ax.set_title('3D задача коммивояжера с реалистичным рельефом', fontsize=12)
ax.grid(True, linestyle=':', alpha=0.5)
ax.view_init(elev=35, azim=-45)

# Состояние системы
salesman_pos = np.array([area_size / 2, area_size / 2, get_height(area_size / 2, area_size / 2)])
current_target = None
visited = set()
salesman_trajectory = [salesman_pos.copy()]
completed = False


# Упрощенный алгоритм маршрутизации
def get_next_target(current_pos, salesman_pos, visited):
    unvisited = [i for i in range(n) if i not in visited]
    if not unvisited:
        return None

    # Выбираем ближайшую цель с учетом высоты
    distances = [distance(salesman_pos, current_pos[i]) for i in unvisited]
    return unvisited[np.argmin(distances)]


def distance(p1, p2):
    # Евклидово расстояние с коэффициентом для высоты
    dx = p1[0] - p2[0]
    dy = p1[1] - p2[1]
    dz = (p1[2] - p2[2]) * 0.5  # Меньший вес для высоты
    return sqrt(dx ** 2 + dy ** 2 + dz ** 2)


def update(frame):
    global salesman_pos, current_target, visited, completed

    current_points = trajectories[frame]

    # Выбор новой цели если нужно
    if current_target is None and not completed:
        current_target = get_next_target(current_points, salesman_pos, visited)
        if current_target is None:
            completed = True

    # Движение коммивояжера
    if current_target is not None:
        target_pos = current_points[current_target]
        dist = distance(salesman_pos, target_pos)

        if dist < 1.5:  # Достигли цели
            visited.add(current_target)
            current_target = None
        else:
            # Плавное движение к цели
            step_size = min(salesman_speed, dist)
            direction = (target_pos - salesman_pos) / dist
            salesman_pos += direction * step_size
            salesman_pos[2] = get_height(salesman_pos[0], salesman_pos[1])
            salesman_trajectory.append(salesman_pos.copy())

    # Обновление визуализации
    for i in range(n):
        scatters[i]._offsets3d = ([current_points[i, 0]],
                                  [current_points[i, 1]],
                                  [current_points[i, 2]])

    # Посещенные объекты
    if visited:
        visited_points = np.array([current_points[i] for i in visited])
        visited_scatter._offsets3d = (visited_points[:, 0],
                                      visited_points[:, 1],
                                      visited_points[:, 2])
    else:
        visited_scatter._offsets3d = ([], [], [])

    # Позиция коммивояжера
    salesman_marker.set_data([salesman_pos[0]], [salesman_pos[1]])
    salesman_marker.set_3d_properties([salesman_pos[2]])

    # Текущий маршрут
    if current_target is not None:
        path_points = np.vstack((salesman_pos, current_points[current_target]))
        current_path.set_data(path_points[:, 0], path_points[:, 1])
        current_path.set_3d_properties(path_points[:, 2])
    else:
        current_path.set_data([], [])
        current_path.set_3d_properties([])

    # Траектория коммивояжера
    if len(salesman_trajectory) > 1:
        trajectory = np.array(salesman_trajectory)
        trajectory_line.set_data(trajectory[:, 0], trajectory[:, 1])
        trajectory_line.set_3d_properties(trajectory[:, 2])

    # Статус
    ax.set_title(f'Шаг: {frame + 1}/{num_steps}. Посещено: {len(visited)}/{n}',
                 fontsize=12)

    return scatters + [salesman_marker, current_path, trajectory_line, visited_scatter]


# Запуск анимации
ani = FuncAnimation(fig, update, frames=num_steps, interval=150, blit=False)
plt.tight_layout()
plt.show()