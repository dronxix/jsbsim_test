"""
Aircraft Visualization System using Matplotlib
3D визуализация воздушных боев с анимацией и интерактивными элементами
"""

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import json
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import colorsys
from matplotlib.patches import Circle
from matplotlib.collections import Line3DCollection
import matplotlib.patches as mpatches

# Цвета для команд
TEAM_COLORS = {
    "red": "#FF4444",
    "blue": "#4444FF",
    "missile": "#FFD700",
    "explosion": "#FF6600"
}

# Символы для типов самолетов
AIRCRAFT_SYMBOLS = {
    "f16": "^",
    "f15": "s", 
    "f18": "D",
    "f22": "o"
}

@dataclass
class VisualizationConfig:
    """Конфигурация для визуализации"""
    figure_size: Tuple[int, int] = (15, 10)
    dpi: int = 100
    interval: int = 100  # мс между кадрами
    trail_length: int = 50  # длина следа самолета
    missile_trail_length: int = 20
    show_radar_range: bool = True
    show_weapon_range: bool = True
    show_airspace_bounds: bool = True
    show_altitude_grid: bool = True
    camera_follow: Optional[str] = None  # ID самолета для слежения камерой
    
class AircraftVisualizer:
    """Класс для визуализации воздушных боев"""
    
    def __init__(self, config: VisualizationConfig = None):
        self.config = config or VisualizationConfig()
        
        # Создаем фигуру и 3D оси
        self.fig = plt.figure(figsize=self.config.figure_size, dpi=self.config.dpi)
        self.ax = self.fig.add_subplot(111, projection='3d')
        
        # Настройка осей
        self._setup_axes()
        
        # Хранилища для данных
        self.aircraft_data = {}
        self.missile_data = {}
        self.aircraft_trails = {}
        self.missile_trails = {}
        self.explosion_effects = []
        
        # Графические элементы
        self.aircraft_plots = {}
        self.missile_plots = {}
        self.trail_plots = {}
        self.range_circles = {}
        self.info_texts = {}
        
        # Анимация
        self.animation = None
        self.frame_data = []
        self.current_frame = 0
        
        print("🎨 Aircraft Visualizer initialized")
        print(f"   Figure size: {self.config.figure_size}")
        print(f"   Trail length: {self.config.trail_length}")
    
    def _setup_axes(self):
        """Настраивает 3D оси"""
        # Пределы по умолчанию (будут обновлены данными)
        self.ax.set_xlim([-100_000, 100_000])
        self.ax.set_ylim([-100_000, 100_000])
        self.ax.set_zlim([0, 20_000])
        
        # Подписи
        self.ax.set_xlabel('X (м)', fontsize=12)
        self.ax.set_ylabel('Y (м)', fontsize=12)
        self.ax.set_zlabel('Высота (м)', fontsize=12)
        
        # Цвет фона
        self.ax.xaxis.pane.fill = False
        self.ax.yaxis.pane.fill = False
        self.ax.zaxis.pane.fill = False
        self.ax.xaxis.pane.set_edgecolor('w')
        self.ax.yaxis.pane.set_edgecolor('w')
        self.ax.zaxis.pane.set_edgecolor('w')
        self.ax.grid(True, alpha=0.3)
        
        # Заголовок
        self.ax.set_title('Air Combat Simulation', fontsize=16, fontweight='bold')
        
        # Добавляем легенду
        self._create_legend()
    
    def _create_legend(self):
        """Создает легенду"""
        legend_elements = [
            mpatches.Patch(color=TEAM_COLORS["red"], label='Red Team'),
            mpatches.Patch(color=TEAM_COLORS["blue"], label='Blue Team'),
            mpatches.Patch(color=TEAM_COLORS["missile"], label='Missiles'),
            plt.Line2D([0], [0], marker='^', color='w', markerfacecolor='black', 
                      markersize=8, label='F-16'),
            plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='black', 
                      markersize=8, label='F-15'),
        ]
        
        self.ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.1, 1))
    
    def _setup_airspace_bounds(self, bounds: Dict[str, float]):
        """Отображает границы воздушного пространства"""
        if not self.config.show_airspace_bounds:
            return
        
        x_min, x_max = bounds['x_min'], bounds['x_max']
        y_min, y_max = bounds['y_min'], bounds['y_max']
        z_min, z_max = bounds['z_min'], bounds['z_max']
        
        # Обновляем пределы осей
        self.ax.set_xlim([x_min, x_max])
        self.ax.set_ylim([y_min, y_max])
        self.ax.set_zlim([z_min, z_max])
        
        # Рисуем границы как прозрачные плоскости
        corners = [
            [x_min, x_max, x_max, x_min, x_min],
            [y_min, y_min, y_max, y_max, y_min]
        ]
        
        # Нижняя граница
        self.ax.plot(corners[0], corners[1], [z_min]*5, 'k--', alpha=0.3, linewidth=1)
        # Верхняя граница  
        self.ax.plot(corners[0], corners[1], [z_max]*5, 'k--', alpha=0.3, linewidth=1)
        
        # Вертикальные линии
        for i in range(4):
            self.ax.plot([corners[0][i], corners[0][i]], 
                        [corners[1][i], corners[1][i]], 
                        [z_min, z_max], 'k--', alpha=0.3, linewidth=1)
    
    def _get_aircraft_color(self, aircraft: Dict[str, Any]) -> str:
        """Возвращает цвет самолета"""
        base_color = TEAM_COLORS[aircraft["team"]]
        
        # Затемняем если самолет подбит
        if aircraft["hp"] < 50:
            # Конвертируем в HSV для затемнения
            rgb = plt.colors.to_rgb(base_color)
            hsv = colorsys.rgb_to_hsv(*rgb)
            darkened_hsv = (hsv[0], hsv[1], hsv[2] * 0.6)
            return colorsys.hsv_to_rgb(*darkened_hsv)
        
        return base_color
    
    def _draw_aircraft(self, aircraft: Dict[str, Any], frame_idx: int):
        """Отрисовывает самолет"""
        aircraft_id = aircraft["id"]
        pos = aircraft["position"]
        
        if not aircraft["alive"]:
            # Убираем мертвые самолеты
            if aircraft_id in self.aircraft_plots:
                self.aircraft_plots[aircraft_id].remove()
                del self.aircraft_plots[aircraft_id]
            return
        
        # Цвет и размер
        color = self._get_aircraft_color(aircraft)
        size = 100 + (aircraft["hp"] / 100.0) * 50  # размер зависит от HP
        
        # Символ в зависимости от типа (если есть)
        symbol = AIRCRAFT_SYMBOLS.get(aircraft.get("aircraft_type", "f16"), "^")
        
        # Отрисовка
        if aircraft_id in self.aircraft_plots:
            self.aircraft_plots[aircraft_id].remove()
        
        self.aircraft_plots[aircraft_id] = self.ax.scatter(
            pos[0], pos[1], pos[2],
            c=[color], s=size, marker=symbol, 
            edgecolors='black', linewidths=1,
            alpha=0.9, depthshade=True
        )
        
        # Обновляем след
        self._update_aircraft_trail(aircraft_id, pos, color)
        
        # Показываем дальность радара если нужно
        if self.config.show_radar_range:
            self._draw_radar_range(aircraft_id, pos, aircraft.get("radar_range", 80000))
    
    def _update_aircraft_trail(self, aircraft_id: str, position: List[float], color: str):
        """Обновляет след самолета"""
        if aircraft_id not in self.aircraft_trails:
            self.aircraft_trails[aircraft_id] = []
        
        # Добавляем текущую позицию
        self.aircraft_trails[aircraft_id].append(position.copy())
        
        # Ограничиваем длину следа
        if len(self.aircraft_trails[aircraft_id]) > self.config.trail_length:
            self.aircraft_trails[aircraft_id].pop(0)
        
        # Рисуем след
        trail = self.aircraft_trails[aircraft_id]
        if len(trail) > 1:
            if aircraft_id in self.trail_plots:
                self.trail_plots[aircraft_id].remove()
            
            # Создаем градиент прозрачности для следа
            trail_array = np.array(trail)
            segments = []
            colors = []
            
            for i in range(len(trail) - 1):
                segments.append([trail[i], trail[i+1]])
                alpha = (i + 1) / len(trail) * 0.7  # нарастающая прозрачность
                colors.append((*plt.colors.to_rgb(color), alpha))
            
            if segments:
                line_collection = Line3DCollection(segments, colors=colors, linewidths=2)
                self.trail_plots[aircraft_id] = self.ax.add_collection3d(line_collection)
    
    def _draw_radar_range(self, aircraft_id: str, position: List[float], radar_range: float):
        """Отрисовывает радиус действия радара"""
        # Рисуем только для выбранных самолетов чтобы не загромождать
        if self.config.camera_follow != aircraft_id:
            return
            
        # Создаем окружность на текущей высоте
        theta = np.linspace(0, 2*np.pi, 50)
        x_circle = position[0] + radar_range * np.cos(theta)
        y_circle = position[1] + radar_range * np.sin(theta)
        z_circle = np.full_like(x_circle, position[2])
        
        if aircraft_id in self.range_circles:
            self.range_circles[aircraft_id].remove()
        
        self.range_circles[aircraft_id] = self.ax.plot(
            x_circle, y_circle, z_circle,
            '--', color='gray', alpha=0.3, linewidth=1
        )[0]
    
    def _draw_missile(self, missile: Dict[str, Any], frame_idx: int):
        """Отрисовывает ракету"""
        missile_id = missile["id"]
        pos = missile["position"]
        
        if not missile["active"]:
            # Убираем неактивные ракеты
            if missile_id in self.missile_plots:
                self.missile_plots[missile_id].remove()
                del self.missile_plots[missile_id]
            return
        
        # Отрисовка ракеты
        if missile_id in self.missile_plots:
            self.missile_plots[missile_id].remove()
        
        # Размер зависит от типа ракеты
        missile_type = missile.get("type", "")
        if "LONG_RANGE" in missile_type:
            size = 50
            marker = "d"
        elif "MEDIUM_RANGE" in missile_type:
            size = 40
            marker = "v"
        else:
            size = 30
            marker = "o"
        
        self.missile_plots[missile_id] = self.ax.scatter(
            pos[0], pos[1], pos[2],
            c=[TEAM_COLORS["missile"]], s=size, marker=marker,
            edgecolors='red', linewidths=1,
            alpha=0.8
        )
        
        # Обновляем след ракеты
        self._update_missile_trail(missile_id, pos)
    
    def _update_missile_trail(self, missile_id: str, position: List[float]):
        """Обновляет след ракеты"""
        if missile_id not in self.missile_trails:
            self.missile_trails[missile_id] = []
        
        self.missile_trails[missile_id].append(position.copy())
        
        # Ограничиваем длину следа
        if len(self.missile_trails[missile_id]) > self.config.missile_trail_length:
            self.missile_trails[missile_id].pop(0)
        
        # Рисуем след
        trail = self.missile_trails[missile_id]
        if len(trail) > 1:
            trail_array = np.array(trail)
            self.ax.plot(trail_array[:, 0], trail_array[:, 1], trail_array[:, 2],
                        '-', color=TEAM_COLORS["missile"], alpha=0.6, linewidth=1)
    
    def _draw_info_panel(self, frame_data: Dict[str, Any]):
        """Отрисовывает информационную панель"""
        info_text = []
        
        # Основная информация о бое
        info_text.append(f"Time: {frame_data['timestamp']:.1f}s")
        info_text.append(f"Red Team: {len([a for a in frame_data['aircraft'] if a['team'] == 'red' and a['alive']])}")
        info_text.append(f"Blue Team: {len([a for a in frame_data['aircraft'] if a['team'] == 'blue' and a['alive']])}")
        info_text.append(f"Active Missiles: {len([m for m in frame_data['missiles'] if m['active']])}")
        info_text.append(f"Total Kills: Red {frame_data['kills']['red']}, Blue {frame_data['kills']['blue']}")
        
        # Отображаем в левом верхнем углу
        info_str = '\n'.join(info_text)
        if hasattr(self, 'info_text_obj'):
            self.info_text_obj.remove()
        
        self.info_text_obj = self.fig.text(0.02, 0.98, info_str, 
                                          fontsize=10, verticalalignment='top',
                                          bbox=dict(boxstyle="round,pad=0.3", 
                                                   facecolor="white", alpha=0.8))
    
    def _draw_altitude_indicators(self, frame_data: Dict[str, Any]):
        """Отрисовывает индикаторы высоты"""
        if not self.config.show_altitude_grid:
            return
        
        # Находим диапазон высот
        altitudes = [a["position"][2] for a in frame_data["aircraft"] if a["alive"]]
        if not altitudes:
            return
        
        min_alt, max_alt = min(altitudes), max(altitudes)
        
        # Рисуем горизонтальные сетки на ключевых высотах
        key_altitudes = [5000, 10000, 15000]  # 5, 10, 15 км
        
        bounds = frame_data.get("airspace_bounds", {
            "x_min": -100000, "x_max": 100000,
            "y_min": -100000, "y_max": 100000
        })
        
        for alt in key_altitudes:
            if bounds["z_min"] <= alt <= bounds["z_max"]:
                x_grid = [bounds["x_min"], bounds["x_max"], bounds["x_max"], bounds["x_min"], bounds["x_min"]]
                y_grid = [bounds["y_min"], bounds["y_min"], bounds["y_max"], bounds["y_max"], bounds["y_min"]]
                z_grid = [alt] * 5
                
                self.ax.plot(x_grid, y_grid, z_grid, 
                           color='lightgray', alpha=0.2, linewidth=0.5)
    
    def _update_camera(self, frame_data: Dict[str, Any]):
        """Обновляет позицию камеры"""
        if not self.config.camera_follow:
            return
        
        # Ищем самолет для слежения
        followed_aircraft = None
        for aircraft in frame_data["aircraft"]:
            if aircraft["id"] == self.config.camera_follow and aircraft["alive"]:
                followed_aircraft = aircraft
                break
        
        if not followed_aircraft:
            return
        
        pos = followed_aircraft["position"]
        
        # Устанавливаем центр на самолет с некоторым смещением
        range_size = 50000  # радиус обзора 50км
        
        self.ax.set_xlim([pos[0] - range_size, pos[0] + range_size])
        self.ax.set_ylim([pos[1] - range_size, pos[1] + range_size])
        self.ax.set_zlim([max(0, pos[2] - 10000), pos[2] + 10000])
    
    def load_battle_data(self, data_source):
        """Загружает данные боя из файла или списка"""
        if isinstance(data_source, str):
            # Загружаем из JSON файла
            with open(data_source, 'r') as f:
                data = json.load(f)
            
            if "frames" in data:
                self.frame_data = data["frames"]
            else:
                self.frame_data = [data]  # Один кадр
        elif isinstance(data_source, list):
            # Прямо список кадров
            self.frame_data = data_source
        else:
            # Один кадр
            self.frame_data = [data_source]
        
        print(f"📊 Loaded {len(self.frame_data)} frames for visualization")
        
        # Настраиваем границы если есть данные
        if self.frame_data:
            first_frame = self.frame_data[0]
            if "airspace_bounds" in first_frame:
                self._setup_airspace_bounds(first_frame["airspace_bounds"])
    
    def animate_frame(self, frame_idx: int):
        """Анимирует один кадр"""
        if frame_idx >= len(self.frame_data):
            return
        
        frame_data = self.frame_data[frame_idx]
        self.current_frame = frame_idx
        
        # Очищаем предыдущие элементы (кроме следов)
        for plot_dict in [self.aircraft_plots, self.missile_plots, self.range_circles]:
            for plot_obj in list(plot_dict.values()):
                if hasattr(plot_obj, 'remove'):
                    plot_obj.remove()
            plot_dict.clear()
        
        # Отрисовываем самолеты
        for aircraft in frame_data.get("aircraft", []):
            self._draw_aircraft(aircraft, frame_idx)
        
        # Отрисовываем ракеты
        for missile in frame_data.get("missiles", []):
            self._draw_missile(missile, frame_idx)
        
        # Отрисовываем индикаторы высоты
        self._draw_altitude_indicators(frame_data)
        
        # Обновляем информационную панель
        self._draw_info_panel(frame_data)
        
        # Обновляем камеру
        self._update_camera(frame_data)
        
        # Обновляем заголовок с текущим временем
        timestamp = frame_data.get("timestamp", frame_idx)
        self.ax.set_title(f'Air Combat Simulation - T+{timestamp:.1f}s (Frame {frame_idx+1}/{len(self.frame_data)})', 
                         fontsize=16, fontweight='bold')
    
    def create_animation(self, interval: int = None) -> animation.FuncAnimation:
        """Создает анимацию"""
        if not self.frame_data:
            raise ValueError("No frame data loaded. Call load_battle_data first.")
        
        interval = interval or self.config.interval
        
        def animate(frame):
            self.animate_frame(frame)
            return []
        
        self.animation = animation.FuncAnimation(
            self.fig, animate, frames=len(self.frame_data),
            interval=interval, blit=False, repeat=True
        )
        
        return self.animation
    
    def save_animation(self, filename: str, fps: int = 10, dpi: int = None):
        """Сохраняет анимацию в файл"""
        if not self.animation:
            self.create_animation()
        
        dpi = dpi or self.config.dpi
        
        print(f"💾 Saving animation to {filename} (fps={fps}, dpi={dpi})")
        
        # Определяем writer по расширению
        if filename.endswith('.gif'):
            writer = animation.PillowWriter(fps=fps)
        elif filename.endswith('.mp4'):
            writer = animation.FFMpegWriter(fps=fps, bitrate=1800)
        else:
            writer = animation.FFMpegWriter(fps=fps, bitrate=1800)
        
        self.animation.save(filename, writer=writer, dpi=dpi)
        print(f"✅ Animation saved: {filename}")
    
    def show_static_frame(self, frame_idx: int = 0):
        """Показывает статический кадр"""
        self.animate_frame(frame_idx)
        plt.show()
    
    def show_interactive(self):
        """Показывает интерактивную визуализацию"""
        if not self.frame_data:
            raise ValueError("No frame data loaded")
        
        # Создаем интерактивные элементы управления
        from matplotlib.widgets import Slider, Button
        
        # Делаем место для слайдеров
        plt.subplots_adjust(bottom=0.2)
        
        # Слайдер для кадров
        ax_frame = plt.axes([0.2, 0.1, 0.5, 0.03])
        frame_slider = Slider(ax_frame, 'Frame', 0, len(self.frame_data)-1, 
                             valinit=0, valfmt='%d')
        
        # Кнопки управления
        ax_play = plt.axes([0.1, 0.05, 0.1, 0.04])
        play_button = Button(ax_play, 'Play')
        
        ax_pause = plt.axes([0.25, 0.05, 0.1, 0.04])
        pause_button = Button(ax_pause, 'Pause')
        
        def update_frame(val):
            frame_idx = int(frame_slider.val)
            self.animate_frame(frame_idx)
            self.fig.canvas.draw()
        
        def play_animation(event):
            if not self.animation:
                self.create_animation()
            # matplotlib animation control is complex, just create new animation
            
        def pause_animation(event):
            if self.animation:
                self.animation.event_source.stop()
        
        frame_slider.on_changed(update_frame)
        play_button.on_clicked(play_animation)
        pause_button.on_clicked(pause_animation)
        
        # Показываем первый кадр
        self.animate_frame(0)
        
        plt.show()

class BattleRecorderVisualizer:
    """Интеграция с системой записи боев"""
    
    def __init__(self, visualizer: AircraftVisualizer):
        self.visualizer = visualizer
    
    def export_from_env_data(self, env_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Конвертирует данные окружения в формат для визуализации"""
        frames = []
        
        for frame_data in env_data:
            viz_frame = {
                "timestamp": frame_data.get("timestamp", 0),
                "aircraft": [],
                "missiles": [],
                "kills": frame_data.get("kills", {"red": 0, "blue": 0}),
                "missiles_fired": frame_data.get("missiles_fired", {"red": 0, "blue": 0}),
                "airspace_bounds": frame_data.get("airspace_bounds", {
                    "x_min": -100000, "x_max": 100000,
                    "y_min": -100000, "y_max": 100000,
                    "z_min": 1000, "z_max": 20000
                }),
                "engagement_range": frame_data.get("engagement_range", 50000)
            }
            
            # Конвертируем данные самолетов
            for aircraft_data in frame_data.get("aircraft", []):
                viz_aircraft = {
                    "id": aircraft_data["id"],
                    "team": aircraft_data["team"],
                    "position": aircraft_data["position"],
                    "velocity": aircraft_data.get("velocity", [0, 0, 0]),
                    "alive": aircraft_data["alive"],
                    "hp": aircraft_data["hp"],
                    "fuel": aircraft_data.get("fuel", 1.0),
                    "heading": aircraft_data.get("heading", 0),
                    "altitude": aircraft_data["position"][2],
                    "speed": np.linalg.norm(aircraft_data.get("velocity", [0, 0, 0])),
                    "aircraft_type": aircraft_data.get("aircraft_type", "f16"),
                    "weapons": aircraft_data.get("weapons", {}),
                    "radar_range": aircraft_data.get("radar_range", 80000)
                }
                viz_frame["aircraft"].append(viz_aircraft)
            
            # Конвертируем данные ракет
            for missile_data in frame_data.get("missiles", []):
                viz_missile = {
                    "id": missile_data["id"],
                    "shooter_id": missile_data["shooter_id"],
                    "target_id": missile_data["target_id"],
                    "position": missile_data["position"],
                    "type": missile_data.get("type", "MEDIUM_RANGE_MISSILE"),
                    "active": missile_data["active"],
                    "fuel": missile_data.get("fuel", 1.0)
                }
                viz_frame["missiles"].append(viz_missile)
            
            frames.append(viz_frame)
        
        return frames

def create_battle_visualization(battle_data_path: str, 
                               config: VisualizationConfig = None,
                               save_path: str = None,
                               show_interactive: bool = True) -> AircraftVisualizer:
    """Удобная функция для создания визуализации боя"""
    
    visualizer = AircraftVisualizer(config)
    visualizer.load_battle_data(battle_data_path)
    
    if save_path:
        animation = visualizer.create_animation()
        visualizer.save_animation(save_path)
    
    if show_interactive:
        visualizer.show_interactive()
    
    return visualizer

def demo_visualization():
    """Демонстрация системы визуализации"""
    print("🎬 Creating demo air combat visualization...")
    
    # Создаем тестовые данные
    demo_frames = []
    
    for t in range(100):  # 10 секунд анимации
        frame = {
            "timestamp": t * 0.1,
            "aircraft": [],
            "missiles": [],
            "kills": {"red": 0, "blue": 0},
            "missiles_fired": {"red": 0, "blue": 0},
            "airspace_bounds": {
                "x_min": -50000, "x_max": 50000,
                "y_min": -50000, "y_max": 50000,
                "z_min": 5000, "z_max": 15000
            }
        }
        
        # Добавляем самолеты
        for i in range(2):  # 2 красных
            aircraft = {
                "id": f"red_{i}",
                "team": "red",
                "position": [
                    -20000 + i * 5000 + t * 100,  # движутся на восток
                    i * 10000 - 5000,
                    8000 + 1000 * np.sin(t * 0.1 + i)  # небольшие маневры
                ],
                "velocity": [100, 0, 10 * np.cos(t * 0.1 + i)],
                "alive": True,
                "hp": max(0, 100 - t * 0.5),  # постепенно теряют HP
                "fuel": max(0.1, 1.0 - t * 0.01),
                "heading": 90,
                "aircraft_type": "f16",
                "radar_range": 80000
            }
            frame["aircraft"].append(aircraft)
        
        for i in range(2):  # 2 синих
            aircraft = {
                "id": f"blue_{i}",
                "team": "blue",
                "position": [
                    20000 - i * 5000 - t * 100,  # движутся на запад
                    i * 10000 - 5000,
                    8000 + 1000 * np.cos(t * 0.1 + i + np.pi)
                ],
                "velocity": [-100, 0, -10 * np.sin(t * 0.1 + i)],
                "alive": True,
                "hp": max(0, 100 - t * 0.3),
                "fuel": max(0.1, 1.0 - t * 0.01),
                "heading": 270,
                "aircraft_type": "f15",
                "radar_range": 80000
            }
            frame["aircraft"].append(aircraft)
        
        # Добавляем несколько ракет в середине боя
        if 30 <= t <= 70 and t % 10 == 0:
            missile = {
                "id": f"missile_{t}",
                "shooter_id": "red_0",
                "target_id": "blue_0", 
                "position": [
                    -10000 + (t-30) * 200,
                    0,
                    8000
                ],
                "type": "MEDIUM_RANGE_MISSILE",
                "active": True,
                "fuel": 1.0 - (t-30) * 0.02
            }
            frame["missiles"].append(missile)
        
        demo_frames.append(frame)
    
    # Создаем визуализацию
    config = VisualizationConfig(
        figure_size=(16, 12),
        interval=100,
        trail_length=30,
        show_radar_range=False,  # отключаем для демо
        camera_follow="red_0"    # следим за первым красным
    )
    
    visualizer = AircraftVisualizer(config)
    visualizer.load_battle_data(demo_frames)
    
    # Показываем статический кадр
    print("📊 Showing demo frame...")
    visualizer.show_static_frame(50)  # середина боя
    
    return visualizer

# Утилиты для анализа
def analyze_battle_statistics(frame_data: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Анализирует статистику воздушного боя"""
    
    if not frame_data:
        return {}
    
    stats = {
        "duration": len(frame_data) * 0.1,  # секунды
        "participants": {},
        "casualties": {"red": 0, "blue": 0},
        "missiles_fired": {"red": 0, "blue": 0},
        "max_altitude": 0,
        "min_altitude": float('inf'),
        "max_speed": 0,
        "engagement_range": 0
    }
    
    # Собираем участников
    for frame in frame_data:
        for aircraft in frame.get("aircraft", []):
            if aircraft["id"] not in stats["participants"]:
                stats["participants"][aircraft["id"]] = {
                    "team": aircraft["team"],
                    "aircraft_type": aircraft.get("aircraft_type", "unknown"),
                    "max_altitude": aircraft["position"][2],
                    "min_altitude": aircraft["position"][2],
                    "max_speed": np.linalg.norm(aircraft.get("velocity", [0, 0, 0]))
                }
            
            # Обновляем статистику
            participant = stats["participants"][aircraft["id"]]
            participant["max_altitude"] = max(participant["max_altitude"], aircraft["position"][2])
            participant["min_altitude"] = min(participant["min_altitude"], aircraft["position"][2])
            speed = np.linalg.norm(aircraft.get("velocity", [0, 0, 0]))
            participant["max_speed"] = max(participant["max_speed"], speed)
            
            # Глобальные статистики
            stats["max_altitude"] = max(stats["max_altitude"], aircraft["position"][2])
            stats["min_altitude"] = min(stats["min_altitude"], aircraft["position"][2])
            stats["max_speed"] = max(stats["max_speed"], speed)
        
        # Подсчитываем потери
        final_kills = frame.get("kills", {"red": 0, "blue": 0})
        stats["casualties"] = final_kills
        
        # Ракеты
        final_missiles = frame.get("missiles_fired", {"red": 0, "blue": 0})
        stats["missiles_fired"] = final_missiles
    
    return stats

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "demo":
            demo_visualization()
        elif sys.argv[1] == "analyze" and len(sys.argv) > 2:
            # Анализ файла боя
            with open(sys.argv[2], 'r') as f:
                data = json.load(f)
            
            frames = data if isinstance(data, list) else data.get("frames", [data])
            stats = analyze_battle_statistics(frames)
            
            print("📊 Battle Analysis:")
            for key, value in stats.items():
                print(f"  {key}: {value}")
        else:
            print("Usage:")
            print("  python aircraft_visualization.py demo")
            print("  python aircraft_visualization.py analyze <battle_file.json>")
    else:
        print("🎨 Aircraft Visualization System")
        print("Features:")
        print("- 3D matplotlib visualization of air combat")
        print("- Real-time animation with aircraft trails")
        print("- Missile tracking and ballistics")  
        print("- Interactive controls and camera following")
        print("- Export to GIF/MP4")
        print("- Battle statistics analysis")