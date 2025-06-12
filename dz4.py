import folium
import numpy as np
import openrouteservice

API_KEY = "5b3ce3597851110001cf62482a5273e316dc408b85bc069d53fa9310"
TIME_LIMIT_HOURS = 30
TRANSPORT_MODES = ["driving-car", "cycling-regular", "foot-walking"]

# === Точки Чувашии с приоритетами ===
points = [
    {"name": "Чебоксары", "lat": 56.1367, "lon": 47.2511, "priority": 10},
    {"name": "Новочебоксарск", "lat": 56.0939, "lon": 47.4977, "priority": 8},
    {"name": "Шумерля", "lat": 55.5283, "lon": 47.4277, "priority": 6},
    {"name": "Канаш", "lat": 55.4918, "lon": 47.4683, "priority": 7},
    {"name": "Мариинский Посад", "lat": 56.1756, "lon": 47.9143, "priority": 5},
    {"name": "Алатырь", "lat": 54.8451, "lon": 46.5671, "priority": 4},
    {"name": "Цивильск", "lat": 56.0916, "lon": 47.0886, "priority": 3},
]

coords = [(p["lon"], p["lat"]) for p in points]
priorities = [p["priority"] for p in points]
names = [p["name"] for p in points]

client = openrouteservice.Client(key=API_KEY)

# === Получаем матрицу времени между точками ===
def get_duration_matrix(client, coordinates, profile):
    matrix = client.distance_matrix(
        locations=coordinates,
        profile=profile,
        metrics=["duration"],
        resolve_locations=True,
        units="km",
    )
    durations = np.array(matrix["durations"])
    return durations / 3600.0  # в часах

# === Муравьиный алгоритм ===
class AntColony:
    def __init__(self, durations, priorities, time_limit, n_ants=10, n_iterations=100, alpha=1, beta=2, evaporation=0.5, Q=1):
        self.durations = durations
        self.priorities = priorities
        self.time_limit = time_limit
        self.n = len(durations)
        self.n_ants = n_ants
        self.n_iterations = n_iterations
        self.alpha = alpha
        self.beta = beta
        self.evaporation = evaporation
        self.Q = Q
        self.pheromone = np.ones((self.n, self.n))
        self.best_path = None
        self.best_score = -1

    def run(self):
        for iteration in range(self.n_iterations):
            all_paths = []
            all_scores = []
            for _ in range(self.n_ants):
                path, score = self.construct_solution()
                all_paths.append(path)
                all_scores.append(score)
                if score > self.best_score:
                    self.best_score = score
                    self.best_path = path
            self.update_pheromones(all_paths, all_scores)
        return self.best_path, self.best_score

    def construct_solution(self):
        path = [0]
        visited = set(path)
        total_time = 0
        score = self.priorities[0]

        while True:
            current = path[-1]
            candidates = [i for i in range(self.n) if i not in visited]

            if not candidates:
                break

            probs = []
            for next_city in candidates:
                tau = self.pheromone[current][next_city] ** self.alpha
                eta = (self.priorities[next_city] / (self.durations[current][next_city] + 1e-6)) ** self.beta
                probs.append(tau * eta)

            if sum(probs) == 0:
                break

            probs = probs / np.sum(probs)
            next_city = np.random.choice(candidates, p=probs)

            travel_time = self.durations[current][next_city]
            if total_time + travel_time > self.time_limit:
                break

            path.append(next_city)
            visited.add(next_city)
            total_time += travel_time
            score += self.priorities[next_city]

        return path, score

    def update_pheromones(self, paths, scores):
        self.pheromone *= (1 - self.evaporation)
        for path, score in zip(paths, scores):
            for i in range(len(path) - 1):
                self.pheromone[path[i]][path[i+1]] += self.Q * score

# === Визуализация маршрута ===
def plot_route(client, coords, path, profile):
    if not path or len(path) < 2:
        print(f"Нет маршрута для профиля '{profile}' с заданным временем.")
        return

    m = folium.Map(location=coords[path[0]][::-1], zoom_start=8)
    folium.Marker(coords[path[0]][::-1], popup=f"{names[path[0]]} (старт)", icon=folium.Icon(color='red')).add_to(m)

    total_duration = 0
    total_distance = 0

    for i in range(len(path) - 1):
        start = coords[path[i]]
        end = coords[path[i + 1]]
        try:
            route = client.directions([start, end], profile=profile, format="geojson")
            props = route["features"][0]["properties"]["summary"]
            geometry = route["features"][0]["geometry"]

            duration_hr = props["duration"] / 3600
            distance_km = props["distance"] / 1000
            speed = distance_km / duration_hr if duration_hr > 0 else 0

            total_duration += duration_hr
            total_distance += distance_km

            popup_text = (
                f"{names[path[i]]} → {names[path[i+1]]}<br>"
                f"{duration_hr:.1f} ч<br>"
                f"{distance_km:.1f} км<br>"
                f"{speed:.1f} км/ч"
            )

            folium.PolyLine(
                locations=[(lat, lon) for lon, lat in geometry["coordinates"]],
                color="blue", weight=5, opacity=0.7,
                popup=folium.Popup(popup_text, max_width=300)
            ).add_to(m)

            folium.Marker(end[::-1], popup=names[path[i+1]]).add_to(m)

        except Exception as e:
            print(f"Ошибка при построении маршрута {i} → {i+1}: {e}")

    print(f"Общее время: {total_duration:.2f} ч")
    print(f"Общее расстояние: {total_distance:.2f} км")
    if total_duration > 0:
        print(f"Средняя скорость: {total_distance / total_duration:.2f} км/ч")

    m.save(f"route_{profile}.html")
    print(f"Карта сохранена в route_{profile}.html")

for mode in TRANSPORT_MODES:
    print(f"\n== Транспорт: {mode} ==")
    durations = get_duration_matrix(client, coords, mode)
    ant_colony = AntColony(durations, priorities, TIME_LIMIT_HOURS, n_ants=20, n_iterations=200)
    path, score = ant_colony.run()
    print("Лучший маршрут:", " → ".join(names[i] for i in path))
    print("Сумма приоритетов:", score)
    plot_route(client, coords, path, mode)
