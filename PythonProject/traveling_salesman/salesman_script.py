import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.interpolate import interp1d
from matplotlib.colors import ListedColormap
from matplotlib.patches import FancyArrow

# Параметры задачи
n = 10  # количество мобильных объектов
num_steps = 100  # увеличенное количество шагов
area_size = 100  # размер области движения
prediction_steps = 3  # глубина прогнозирования
salesman_speed = 1.2  # скорость коммивояжера (единиц/шаг)
visit_time = 0  # время посещения точки (в шагах)
object_speed = 0.7  # скорость движения объектов
time_per_step = 0.5  # время одного шага в секундах

# Генерация траекторий
np.random.seed(60)
trajectories = np.cumsum(np.random.randn(num_steps, n, 2) * object_speed, axis=0) + np.random.randint(0, area_size,
                                                                                                      (1, n, 2))


# Функция прогнозирования
def predict_movement(past_positions, steps=1):
    predicted = []
    for i in range(n):
        x = np.arange(len(past_positions))
        x_new = np.linspace(x[-1], x[-1] + steps, steps + 1)

        f_x = interp1d(x, past_positions[:, i, 0], kind='quadratic', fill_value="extrapolate")
        f_y = interp1d(x, past_positions[:, i, 1], kind='quadratic', fill_value="extrapolate")

        predicted.append(np.column_stack((f_x(x_new), f_y(x_new))))

    return np.array(predicted).transpose(1, 0, 2)


def build_route(points, salesman_pos, visited_indices):
    unvisited = [i for i in range(len(points)) if i not in visited_indices]
    if not unvisited:
        return []

    distances = [np.linalg.norm(salesman_pos - points[i]) for i in unvisited]
    path = [unvisited[np.argmin(distances)]]
    unvisited.remove(path[0])

    while unvisited:
        last = path[-1]
        next_node = min(unvisited, key=lambda x: np.linalg.norm(points[last] - points[x]))
        path.append(next_node)
        unvisited.remove(next_node)

    return path


# Создание анимации
fig, ax = plt.subplots(figsize=(14, 12))
ax.set_xlim(0, area_size)
ax.set_ylim(0, area_size)
ax.set_title('Задача коммивояжера с траекторией перемещения')

# Цветовая схема
colors = ['blue', 'red']
cmap = ListedColormap(colors)

# Элементы визуализации
scatter = ax.scatter([], [], c=[], cmap=cmap, vmin=0, vmax=1, s=100,
                     label='Объекты (синие - не посещены, красные - посещены)')
prediction_scatter = ax.scatter([], [], c='green', alpha=0.3, s=80, label='Прогноз позиций')
salesman_marker = ax.plot([], [], 'ko', markersize=14, markeredgewidth=2,
                          markerfacecolor='yellow', label='Коммивояжер')[0]
current_path = ax.plot([], [], 'r-', lw=3, alpha=0.8, label='Текущий маршрут')[0]
predicted_path = ax.plot([], [], 'g--', lw=2, alpha=0.6, label='Прогнозируемый маршрут')[0]
trajectory_line = ax.plot([], [], 'm-', lw=2, alpha=0.5, label='Траектория коммивояжера')[0]
time_text = ax.text(0.02, 0.96, '', transform=ax.transAxes, fontsize=12,
                    bbox=dict(facecolor='white', alpha=0.8))
status_text = ax.text(0.02, 0.90, '', transform=ax.transAxes, fontsize=10,
                      bbox=dict(facecolor='white', alpha=0.8))
final_info = ax.text(0.5, 0.5, '', transform=ax.transAxes, fontsize=14,
                     bbox=dict(facecolor='white', alpha=0.9), ha='center', va='center')
final_info.set_visible(False)
ax.legend(loc='upper right')

# Состояние системы
salesman_pos = np.array([area_size / 2, area_size / 2])
current_target = None
time_at_target = 0
visited = set()
total_distance = 0
travel_history = []
completed = False
start_time = 0
salesman_trajectory = [salesman_pos.copy()]  # История позиций коммивояжера


def update(frame):
    global salesman_pos, current_target, time_at_target, visited, total_distance
    global travel_history, completed, start_time, salesman_trajectory

    current_points = trajectories[frame]

    if frame == 0:
        start_time = frame

    # Прогнозирование движения
    if frame >= 2 and len(visited) < n:
        predicted = predict_movement(trajectories[:frame + 1], prediction_steps)
        predicted_points = predicted[-1]
    else:
        predicted_points = current_points

    # Построение маршрутов
    current_route = build_route(current_points, salesman_pos, visited)

    # Прогнозируемый маршрут только для непосещенных объектов
    if frame >= 2 and len(visited) < n:
        unvisited_predicted = [i for i in range(n) if i not in visited]
        if unvisited_predicted:
            predicted_route = build_route(predicted_points[unvisited_predicted], salesman_pos, set())
            predicted_route = [unvisited_predicted[i] for i in predicted_route]
        else:
            predicted_route = []
    else:
        predicted_route = []

    # Движение коммивояжера
    if not completed:
        if current_target is None and current_route:
            current_target = current_route[0]

        if current_target is not None and current_target < len(current_points):
            target_pos = current_points[current_target]
            direction = target_pos - salesman_pos
            distance = np.linalg.norm(direction)

            if distance < 0.8:  # Достигли цели
                if time_at_target >= visit_time:
                    visited.add(current_target)
                    travel_history.append((current_target, frame))
                    current_route = build_route(current_points, salesman_pos, visited)
                    current_target = current_route[0] if current_route else None
                    time_at_target = 0

                    if len(visited) == n and not completed:
                        completed = True
                        end_time = frame
                        total_time_seconds = (end_time - start_time) * time_per_step
                        final_info.set_text(f'Все объекты посещены!\n'
                                            f'Общее время: {total_time_seconds:.1f} секунд\n'
                                            f'Общее расстояние: {total_distance:.1f} единиц\n'
                                            f'Скорость: {salesman_speed:.1f} ед/шаг\n'
                                            f'Время посещения: {visit_time} шагов')
                        final_info.set_visible(True)
                else:
                    time_at_target += 1
            else:
                step = direction / distance * min(salesman_speed, distance)
                salesman_pos += step
                total_distance += np.linalg.norm(step)
                salesman_trajectory.append(salesman_pos.copy())

    # Визуализация
    point_colors = [1 if i in visited else 0 for i in range(n)]
    scatter.set_offsets(current_points)
    scatter.set_array(np.array(point_colors))

    salesman_marker.set_data([salesman_pos[0]], [salesman_pos[1]])

    # Прогнозируемые позиции
    if frame >= 2 and len(visited) < n:
        unvisited_indices = [i for i in range(n) if i not in visited]
        prediction_scatter.set_offsets(predicted_points[unvisited_indices])
    else:
        prediction_scatter.set_offsets(np.empty((0, 2)))

    # Текущий маршрут
    if current_route:
        route_points = np.vstack(([salesman_pos], current_points[current_route]))
        current_path.set_data(route_points[:, 0], route_points[:, 1])
    else:
        current_path.set_data([], [])

    # Прогнозируемый маршрут
    if predicted_route and not completed:
        pred_route_points = np.vstack(([salesman_pos], predicted_points[predicted_route]))
        predicted_path.set_data(pred_route_points[:, 0], pred_route_points[:, 1])
    else:
        predicted_path.set_data([], [])

    # Траектория коммивояжера
    if len(salesman_trajectory) > 1:
        trajectory = np.array(salesman_trajectory)
        trajectory_line.set_data(trajectory[:, 0], trajectory[:, 1])

    # Итоговый маршрут после завершения
    if completed:
        history_x = [salesman_trajectory[0][0]] + [trajectories[step][target][0] for target, step in travel_history]
        history_y = [salesman_trajectory[0][1]] + [trajectories[step][target][1] for target, step in travel_history]
        ax.plot(history_x, history_y, 'c-', lw=2, alpha=0.7, label='Итоговый маршрут')
        ax.legend(loc='upper right')

    # Обновление информации
    current_time_seconds = frame * time_per_step
    time_text.set_text(f'Время: {current_time_seconds:.1f} сек\n'
                       f'Шаг: {frame}/{num_steps}\n'
                       f'Скорость: {salesman_speed:.1f} ед/шаг')
    status_text.set_text(f'Цель: {current_target}\n'
                         f'Посещено: {len(visited)}/{n}\n'
                         f'Время у цели: {time_at_target}/{visit_time}')

    return (scatter, prediction_scatter, salesman_marker, current_path,
            predicted_path, trajectory_line, time_text, status_text, final_info)


# Запуск анимации
ani = animation.FuncAnimation(fig, update, frames=num_steps, interval=300, blit=True)
plt.tight_layout()
plt.show()