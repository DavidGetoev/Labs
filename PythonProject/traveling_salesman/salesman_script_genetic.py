import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from sklearn.linear_model import LinearRegression

# Параметры задачи
n = 10  # количество мобильных объектов
num_steps = 300  # количество временных шагов
area_size = 100  # размер области движения
salesman_speed = 1.8  # скорость коммивояжера
visit_time = 0  # время посещения точки
object_speed = 0.4  # скорость движения объектов
time_per_step = 0.3  # время одного шага в секундах
prediction_horizon = 5  # горизонт прогнозирования

# Генерация траекторий
np.random.seed(42)
trajectories = np.cumsum(np.random.randn(num_steps, n, 2) * object_speed, axis=0) + np.random.randint(0, area_size,
                                                                                                      (1, n, 2))


# алгоритм прогнозирования с линейной регрессией
def predict_trajectories(past_positions, steps=1):
    predicted = []
    history_length = min(5, len(past_positions))  # Используем последние 5 позиций

    for i in range(n):
        # Подготовка данных для регрессии
        x_data = np.arange(history_length).reshape(-1, 1)
        yx_data = past_positions[-history_length:, i, 0]
        yy_data = past_positions[-history_length:, i, 1]

        # Обучение моделей
        model_x = LinearRegression().fit(x_data, yx_data)
        model_y = LinearRegression().fit(x_data, yy_data)

        # Прогнозирование
        x_pred = np.arange(history_length, history_length + steps).reshape(-1, 1)
        yx_pred = model_x.predict(x_pred)
        yy_pred = model_y.predict(x_pred)

        predicted.append(np.column_stack((yx_pred, yy_pred)))

    return np.array(predicted).transpose(1, 0, 2)


# Модифицированный генетический алгоритм с учетом прогноза
class AdvancedTSP:
    def __init__(self, current_pos, predicted_pos, salesman_pos, speed):
        self.current_pos = current_pos
        self.predicted_pos = predicted_pos  # shape: (prediction_horizon, n, 2)
        self.salesman_pos = salesman_pos
        self.speed = speed
        self.n = len(current_pos)

    def _distance(self, a, b):
        return np.linalg.norm(a - b)

    def _route_time(self, route):
        if not route:
            return 0

        total_time = self._distance(self.salesman_pos, self.current_pos[route[0]]) / self.speed
        for i in range(len(route) - 1):
            total_time += self._distance(self.current_pos[route[i]], self.current_pos[route[i + 1]]) / self.speed
        return total_time

    def _predictive_route_time(self, route):
        if not route:
            return 0

        # Рассчитываем время с учетом прогнозируемых позиций
        time_to_first = self._distance(self.salesman_pos, self.current_pos[route[0]]) / self.speed
        arrival_time = time_to_first

        total_time = time_to_first
        current_pos = self.current_pos[route[0]]

        for i in range(1, len(route)):
            # Прогнозируем позицию следующей цели к моменту прибытия
            pred_idx = min(int(arrival_time), len(self.predicted_pos) - 1)
            next_pos = self.predicted_pos[pred_idx][route[i]]

            segment_time = self._distance(current_pos, next_pos) / self.speed
            total_time += segment_time
            arrival_time += segment_time
            current_pos = next_pos

        return total_time

    def optimize(self):
        # Начинаем с ближайшей точки
        unvisited = list(range(self.n))
        if not unvisited:
            return []

        # Жадное начальное решение с учетом прогноза
        route = []
        remaining = unvisited.copy()
        current_pos = self.salesman_pos

        while remaining:
            # Выбираем следующую точку с минимальным прогнозируемым временем прибытия
            next_node = min(remaining, key=lambda x: (
                self._distance(current_pos, self.current_pos[x]) / self.speed +
                self._distance(self.current_pos[x], self.predicted_pos[-1][x]) / self.speed
                if len(self.predicted_pos) > 0 else 0  # Защита от пустого массива
            ))
            route.append(next_node)
            remaining.remove(next_node)
            current_pos = self.predicted_pos[-1][next_node] if len(self.predicted_pos) > 0 else self.current_pos[
                next_node]

        # 2.5-opt оптимизация
        improved = True
        best_route = route
        best_time = self._predictive_route_time(best_route)

        while improved:
            improved = False
            for i in range(1, len(best_route) - 1):
                for j in range(i + 1, len(best_route)):
                    if j - i == 1: continue
                    # переставить сегмент
                    new_route = best_route[:i] + best_route[i:j][::-1] + best_route[j:]
                    new_time = self._predictive_route_time(new_route)
                    if new_time < best_time:
                        best_route = new_route
                        best_time = new_time
                        improved = True

        return best_route


# Создание анимации
fig, ax = plt.subplots(figsize=(14, 12))
ax.set_xlim(0, area_size)
ax.set_ylim(0, area_size)
ax.set_title('Алгоритм коммивояжера с прогнозированием')

# Цветовая схема
colors = ['blue', 'red']
cmap = ListedColormap(colors)

# Элементы визуализации
scatter = ax.scatter([], [], c=[], cmap=cmap, vmin=0, vmax=1, s=100,
                     label='Объекты (синие - не посещены, красные - посещены)')
prediction_scatter = ax.scatter([], [], c='green', alpha=0.3, s=80, label='Прогнозируемые позиции')
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
salesman_trajectory = [salesman_pos.copy()]
current_route = []


def update(frame):
    global salesman_pos, current_target, time_at_target, visited, total_distance
    global travel_history, completed, start_time, salesman_trajectory, current_route

    current_points = trajectories[frame]

    if frame == 0:
        start_time = frame

    # Прогнозирование движения объектов
    if frame >= 2:
        predicted_trajectories = predict_trajectories(trajectories[:frame + 1], prediction_horizon)
        predicted_points = predicted_trajectories[-1] if len(predicted_trajectories) > 0 else current_points
    else:
        predicted_points = current_points
        predicted_trajectories = np.array([current_points])

    # Построение маршрута с учетом прогноза
    if frame % 3 == 0 or len(current_route) == 0:  # Пересчитываем маршрут периодически
        tsp = AdvancedTSP(current_points, predicted_trajectories, salesman_pos, salesman_speed)
        current_route = tsp.optimize()

    # Движение коммивояжера
    if not completed:
        if current_target is None and current_route:
            current_target = current_route[0] if current_route[0] not in visited else None

        if current_target is not None and current_target < len(current_points):
            target_pos = current_points[current_target]
            direction = target_pos - salesman_pos
            distance = np.linalg.norm(direction)

            if distance < 0.8:  # Достигли цели
                if time_at_target >= visit_time:
                    visited.add(current_target)
                    travel_history.append((current_target, frame))

                    # Обновляем маршрут, исключая посещенные точки
                    unvisited = [i for i in current_route if i not in visited]
                    if unvisited:
                        tsp = AdvancedTSP(current_points[unvisited],
                                          predicted_trajectories[:, unvisited] if len(predicted_trajectories) > 0 else
                                          current_points[unvisited],
                                          salesman_pos, salesman_speed)
                        new_route_indices = tsp.optimize()
                        current_route = [i for i in current_route if i in visited] + [unvisited[i] for i in
                                                                                      new_route_indices]
                        current_target = current_route[len(visited)] if len(visited) < len(current_route) else None
                    else:
                        current_target = None
                        completed = True
                        end_time = frame
                        total_time_seconds = (end_time - start_time) * time_per_step
                        final_info.set_text(f'Все объекты посещены!\n'
                                            f'Общее время: {total_time_seconds:.1f} сек\n'
                                            f'Общее расстояние: {total_distance:.1f}')
                        final_info.set_visible(True)
                    time_at_target = 0
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
    if frame >= 2:
        unvisited_indices = [i for i in range(n) if i not in visited]
        if unvisited_indices and len(predicted_trajectories) > 0:
            prediction_scatter.set_offsets(predicted_trajectories[-1][unvisited_indices])
        else:
            prediction_scatter.set_offsets(np.empty((0, 2)))
    else:
        prediction_scatter.set_offsets(np.empty((0, 2)))

    # Текущий маршрут
    if current_route:
        unvisited_in_route = [i for i in current_route if i not in visited]
        if unvisited_in_route:
            route_points = np.vstack(([salesman_pos], current_points[unvisited_in_route]))
            current_path.set_data(route_points[:, 0], route_points[:, 1])
        else:
            current_path.set_data([], [])
    else:
        current_path.set_data([], [])

    # Прогнозируемый маршрут
    if frame >= 2 and not completed and len(visited) < n and len(predicted_trajectories) > 0:
        unvisited = [i for i in range(n) if i not in visited]
        if unvisited:
            tsp = AdvancedTSP(current_points[unvisited], predicted_trajectories[:, unvisited], salesman_pos,
                              salesman_speed)
            pred_route = tsp.optimize()
            pred_route_full = [unvisited[i] for i in pred_route]
            pred_route_points = np.vstack(([salesman_pos], current_points[pred_route_full]))
            predicted_path.set_data(pred_route_points[:, 0], pred_route_points[:, 1])
        else:
            predicted_path.set_data([], [])
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
                       f'Скорость: {salesman_speed:.1f} ед/сек')
    status_text.set_text(f'Цель: {current_target}\n'
                         f'Посещено: {len(visited)}/{n}\n'
                         f'Время у цели: {time_at_target}/{visit_time}')

    return (scatter, prediction_scatter, salesman_marker, current_path,
            predicted_path, trajectory_line, time_text, status_text, final_info)


# Запуск анимации
ani = animation.FuncAnimation(fig, update, frames=num_steps, interval=300, blit=True)
plt.tight_layout()
plt.show()
