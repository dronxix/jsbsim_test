"""
JSBSim Aircraft Environment for Multi-Agent Dogfighting
Адаптировано из Arena 3D для воздушных боев с иерархическим управлением
"""

import numpy as np
import jsbsim
import math
from typing import Dict, Any, Optional, List, Tuple
from gymnasium import spaces
from ray.rllib.env import MultiAgentEnv

# Константы для авиации
MAX_AIRCRAFT = 8
AIRCRAFT_FEATURES = 15  # Расширенные авиационные параметры
ENEMY_FEATURES = 13     # Информация о противниках
SELF_FEATURES = 18      # Собственное состояние
GLOBAL_FEATURES = 64

# Параметры воздушного боя
ENGAGEMENT_RANGE = 50_000.0  # метры (50 км максимальная дальность ракет)
VISUAL_RANGE = 20_000.0      # метры (20 км визуальная дальность)
MISSILE_RANGE = 30_000.0     # метры (30 км дальность ракет средней дальности)
CANNON_RANGE = 2_000.0       # метры (2 км пушки)

# Границы воздушного пространства
AIRSPACE_BOUNDS = {
    'x_min': -100_000, 'x_max': 100_000,  # ±100 км
    'y_min': -100_000, 'y_max': 100_000,  # ±100 км  
    'z_min': 1_000,    'z_max': 20_000    # 1-20 км высота
}

# Типы вооружения
WEAPON_TYPES = {
    "CANNON": 0,
    "SHORT_RANGE_MISSILE": 1,  # AIM-9 типа
    "MEDIUM_RANGE_MISSILE": 2, # AIM-120 типа
    "LONG_RANGE_MISSILE": 3    # AIM-54 типа
}

# Маневры
MANEUVERS = {
    "STRAIGHT": 0,
    "CLIMB": 1,
    "DIVE": 2,
    "LEFT_TURN": 3,
    "RIGHT_TURN": 4,
    "BARREL_ROLL": 5,
    "SPLIT_S": 6,
    "IMMELMANN": 7,
    "DEFENSIVE_SPIRAL": 8
}

def _box(lo, hi, shape):
    return spaces.Box(low=lo, high=hi, shape=shape, dtype=np.float32)

class Missile:
    """Класс для моделирования ракеты"""
    def __init__(self, shooter_id: str, target_id: str, launch_pos: np.ndarray, 
                 target_pos: np.ndarray, missile_type: str, timestamp: float):
        self.id = f"missile_{timestamp:.3f}_{shooter_id}"
        self.shooter_id = shooter_id
        self.target_id = target_id
        self.launch_pos = launch_pos.copy()
        self.current_pos = launch_pos.copy()
        self.target_pos = target_pos.copy()
        self.missile_type = missile_type
        self.timestamp = timestamp
        self.active = True
        
        # Параметры ракеты в зависимости от типа
        if missile_type == "SHORT_RANGE_MISSILE":
            self.speed = 1200.0  # м/с (Mach 3.5)
            self.max_range = 18_000.0
            self.turn_rate = 50.0  # g
            self.pk = 0.85  # вероятность поражения
        elif missile_type == "MEDIUM_RANGE_MISSILE":
            self.speed = 1400.0  # м/с (Mach 4)
            self.max_range = 30_000.0
            self.turn_rate = 30.0  # g
            self.pk = 0.75
        elif missile_type == "LONG_RANGE_MISSILE":
            self.speed = 1600.0  # м/с (Mach 4.5)
            self.max_range = 60_000.0
            self.turn_rate = 20.0  # g
            self.pk = 0.65
        else:
            self.speed = 900.0
            self.max_range = 2_000.0
            self.turn_rate = 5.0
            self.pk = 0.95
        
        self.fuel = 1.0  # нормализованное топливо
        self.fuel_consumption = 0.01  # расход на секунду
    
    def update(self, dt: float, target_pos: np.ndarray) -> bool:
        """Обновляет позицию ракеты. Возвращает True если ракета активна"""
        if not self.active:
            return False
        
        # Расход топлива
        self.fuel -= self.fuel_consumption * dt
        if self.fuel <= 0:
            self.active = False
            return False
        
        # Наведение на цель
        to_target = target_pos - self.current_pos
        distance = np.linalg.norm(to_target)
        
        if distance < 100.0:  # Попадание (100м радиус поражения)
            self.active = False
            return False
        
        if distance > self.max_range:  # Вышла за дальность
            self.active = False
            return False
        
        # Движение к цели
        direction = to_target / (distance + 1e-8)
        self.current_pos += direction * self.speed * dt
        self.target_pos = target_pos.copy()
        
        return True
    
    def get_hit_probability(self, target_pos: np.ndarray, target_velocity: np.ndarray) -> float:
        """Рассчитывает вероятность попадания"""
        distance = np.linalg.norm(self.current_pos - target_pos)
        
        # Базовая вероятность уменьшается с дистанцией
        range_factor = max(0.1, 1.0 - (distance / self.max_range))
        
        # Учитываем скорость цели (маневрирование)
        target_speed = np.linalg.norm(target_velocity)
        speed_factor = max(0.3, 1.0 - (target_speed / 500.0))  # сложнее попасть в быстрые цели
        
        return self.pk * range_factor * speed_factor

class Aircraft:
    """Класс самолета с JSBSim"""
    def __init__(self, aircraft_id: str, team: str, initial_pos: np.ndarray, 
                 initial_heading: float, aircraft_type: str = "f16"):
        self.id = aircraft_id
        self.team = team
        self.aircraft_type = aircraft_type
        
        # JSBSim
        self.fdm = jsbsim.FGFDMExec()
        self.fdm.set_debug_level(0)
        
        # Загружаем модель самолета
        if aircraft_type == "f16":
            self.fdm.load_model('f16')
        elif aircraft_type == "f15":
            self.fdm.load_model('f15') 
        else:
            self.fdm.load_model('f16')  # по умолчанию F-16
        
        self.fdm.load_ic('reset00', False)
        self.fdm.run_ic()
        
        # Устанавливаем начальную позицию
        self.fdm.set_property_value('position/lat-gc-deg', initial_pos[0])
        self.fdm.set_property_value('position/long-gc-deg', initial_pos[1])
        self.fdm.set_property_value('position/h-sl-ft', initial_pos[2] * 3.28084)  # м в футы
        self.fdm.set_property_value('attitude/psi-deg', initial_heading)
        
        # Запускаем двигатель
        self.fdm.set_property_value('propulsion/engine/set-running', 1)
        self.fdm.set_property_value('fcs/throttle-cmd-norm', 0.8)
        
        # Состояние
        self.alive = True
        self.hp = 100.0
        self.fuel = 1.0
        
        # Вооружение
        self.weapons = {
            "CANNON": {"count": 500, "ready": True},
            "SHORT_RANGE_MISSILE": {"count": 4, "ready": True},
            "MEDIUM_RANGE_MISSILE": {"count": 6, "ready": True},
            "LONG_RANGE_MISSILE": {"count": 2, "ready": True},
        }
        
        # Сенсоры
        self.radar_range = 80_000.0  # 80 км радар
        self.rwr_active = True  # система предупреждения о радарном облучении
        
        # История для расчета производных
        self.position_history = [initial_pos.copy()]
        self.velocity_history = [np.zeros(3)]
        
    def get_state_vector(self) -> np.ndarray:
        """Получает вектор состояния самолета"""
        # Позиция и ориентация
        lat = self.fdm.get_property_value('position/lat-gc-deg')
        lon = self.fdm.get_property_value('position/long-gc-deg')
        alt = self.fdm.get_property_value('position/h-sl-ft') * 0.3048  # футы в метры
        
        # Углы Эйлера
        phi = self.fdm.get_property_value('attitude/phi-deg')
        theta = self.fdm.get_property_value('attitude/theta-deg')
        psi = self.fdm.get_property_value('attitude/psi-deg')
        
        # Скорости
        u = self.fdm.get_property_value('velocities/u-fps') * 0.3048  # футы/с в м/с
        v = self.fdm.get_property_value('velocities/v-fps') * 0.3048
        w = self.fdm.get_property_value('velocities/w-fps') * 0.3048
        
        # Угловые скорости
        p = self.fdm.get_property_value('velocities/p-rad_sec')
        q = self.fdm.get_property_value('velocities/q-rad_sec')  
        r = self.fdm.get_property_value('velocities/r-rad_sec')
        
        # Воздушная скорость и число Маха
        tas = self.fdm.get_property_value('velocities/vt-fps') * 0.3048
        mach = self.fdm.get_property_value('velocities/mach')
        
        # Перегрузки
        nx = self.fdm.get_property_value('accelerations/n-pilot-x-norm')
        ny = self.fdm.get_property_value('accelerations/n-pilot-y-norm')
        nz = self.fdm.get_property_value('accelerations/n-pilot-z-norm')
        
        # Топливо и системы
        fuel_norm = self.fdm.get_property_value('propulsion/total-fuel-lbs') / 20000.0  # нормализуем
        
        return np.array([
            lat, lon, alt / 20000.0,  # позиция (высота нормализована)
            phi / 180.0, theta / 180.0, psi / 180.0,  # углы (нормализованы)
            u / 500.0, v / 500.0, w / 500.0,  # скорости (нормализованы к 500 м/с)
            p, q, r,  # угловые скорости
            tas / 500.0, mach / 2.0,  # воздушная скорость и мах
            nx / 10.0, ny / 10.0, nz / 10.0,  # перегрузки (нормализованы к 10g)
            fuel_norm, self.hp / 100.0  # топливо и здоровье
        ], dtype=np.float32)
    
    def apply_controls(self, action: Dict[str, Any]) -> bool:
        """Применяет управляющие действия"""
        if not self.alive:
            return False
        
        # Управление полетом
        aileron = np.clip(action.get("aileron", 0.0), -1.0, 1.0)
        elevator = np.clip(action.get("elevator", 0.0), -1.0, 1.0)
        rudder = np.clip(action.get("rudder", 0.0), -1.0, 1.0)
        throttle = np.clip(action.get("throttle", 0.8), 0.0, 1.0)
        
        # Применяем к JSBSim
        self.fdm.set_property_value('fcs/aileron-cmd-norm', aileron)
        self.fdm.set_property_value('fcs/elevator-cmd-norm', elevator)
        self.fdm.set_property_value('fcs/rudder-cmd-norm', rudder)
        self.fdm.set_property_value('fcs/throttle-cmd-norm', throttle)
        
        return True
    
    def update(self, dt: float) -> bool:
        """Обновляет состояние самолета"""
        if not self.alive:
            return False
        
        # Обновляем JSBSim
        self.fdm.run()
        
        # Проверяем границы воздушного пространства
        pos = self.get_position()
        if not self._check_boundaries(pos):
            self.hp -= 50.0  # штраф за выход из зоны
            if self.hp <= 0:
                self.alive = False
                return False
        
        # Обновляем историю
        self.position_history.append(pos)
        if len(self.position_history) > 10:
            self.position_history.pop(0)
        
        # Рассчитываем скорость
        if len(self.position_history) >= 2:
            velocity = (self.position_history[-1] - self.position_history[-2]) / dt
            self.velocity_history.append(velocity)
            if len(self.velocity_history) > 10:
                self.velocity_history.pop(0)
        
        return True
    
    def get_position(self) -> np.ndarray:
        """Получает текущую позицию в метрах (x, y, z)"""
        lat = self.fdm.get_property_value('position/lat-gc-deg')
        lon = self.fdm.get_property_value('position/long-gc-deg')
        alt = self.fdm.get_property_value('position/h-sl-ft') * 0.3048
        
        # Преобразуем в прямоугольные координаты (упрощенно)
        x = lon * 111320.0  # градус долготы ≈ 111.32 км
        y = lat * 110540.0  # градус широты ≈ 110.54 км
        z = alt
        
        return np.array([x, y, z], dtype=np.float32)
    
    def get_velocity(self) -> np.ndarray:
        """Получает текущую скорость в м/с"""
        if len(self.velocity_history) > 0:
            return self.velocity_history[-1]
        return np.zeros(3, dtype=np.float32)
    
    def _check_boundaries(self, pos: np.ndarray) -> bool:
        """Проверяет границы воздушного пространства"""
        return (AIRSPACE_BOUNDS['x_min'] <= pos[0] <= AIRSPACE_BOUNDS['x_max'] and
                AIRSPACE_BOUNDS['y_min'] <= pos[1] <= AIRSPACE_BOUNDS['y_max'] and
                AIRSPACE_BOUNDS['z_min'] <= pos[2] <= AIRSPACE_BOUNDS['z_max'])
    
    def can_engage_target(self, target_pos: np.ndarray, weapon_type: str) -> bool:
        """Проверяет возможность атаки цели"""
        if not self.alive or weapon_type not in self.weapons:
            return False
        
        if self.weapons[weapon_type]["count"] <= 0:
            return False
        
        distance = np.linalg.norm(self.get_position() - target_pos)
        
        if weapon_type == "CANNON":
            return distance <= CANNON_RANGE
        elif weapon_type == "SHORT_RANGE_MISSILE":
            return distance <= 18_000.0
        elif weapon_type == "MEDIUM_RANGE_MISSILE":
            return distance <= MISSILE_RANGE
        elif weapon_type == "LONG_RANGE_MISSILE":
            return distance <= 60_000.0
        
        return False
    
    def fire_weapon(self, target_id: str, target_pos: np.ndarray, weapon_type: str) -> Optional[Missile]:
        """Выстрел по цели"""
        if not self.can_engage_target(target_pos, weapon_type):
            return None
        
        if weapon_type == "CANNON":
            # Пушка - мгновенное попадание/промах
            self.weapons[weapon_type]["count"] -= 10  # очередь
            distance = np.linalg.norm(self.get_position() - target_pos)
            hit_prob = max(0.1, 1.0 - (distance / CANNON_RANGE))
            return None  # пушка не создает объект снаряда
        else:
            # Ракета
            self.weapons[weapon_type]["count"] -= 1
            missile = Missile(
                shooter_id=self.id,
                target_id=target_id,
                launch_pos=self.get_position(),
                target_pos=target_pos,
                missile_type=weapon_type,
                timestamp=self.fdm.get_property_value('simulation/sim-time-sec')
            )
            return missile

class DogfightEnv(MultiAgentEnv):
    """Окружение для воздушных боев с JSBSim"""
    
    def __init__(self, env_config: Optional[Dict[str, Any]] = None):
        self.cfg = env_config or {}
        self.rng = np.random.default_rng(self.cfg.get("seed", 0))
        self.max_len = int(self.cfg.get("episode_len", 3000))  # 5 минут при 10 Hz
        
        # Размеры команд
        self.red_choices = list(self.cfg.get("red_choices", [2, 4]))
        self.blue_choices = list(self.cfg.get("blue_choices", [2, 4]))
        self.max_aircraft = int(self.cfg.get("max_aircraft", MAX_AIRCRAFT))
        
        # Временной шаг
        self.dt = float(self.cfg.get("dt", 0.1))  # 10 Hz
        
        # Observation space для авиации
        self.single_obs_space = spaces.Dict({
            "self": _box(-10, 10, (SELF_FEATURES,)),
            "allies": _box(-10, 10, (self.max_aircraft, AIRCRAFT_FEATURES)),
            "allies_mask": spaces.MultiBinary(self.max_aircraft),
            "enemies": _box(-10, 10, (self.max_aircraft, ENEMY_FEATURES)),
            "enemies_mask": spaces.MultiBinary(self.max_aircraft),
            "global_state": _box(-10, 10, (GLOBAL_FEATURES,)),
            "enemy_action_mask": spaces.MultiBinary(self.max_aircraft),
        })
        
        # Action space для пилота (иерархическое управление)
        self.single_act_space = spaces.Dict({
            # Высокоуровневые команды
            "target": spaces.Discrete(self.max_aircraft),
            "maneuver": spaces.Discrete(len(MANEUVERS)),
            "weapon": spaces.Discrete(len(WEAPON_TYPES)),
            "fire": spaces.Discrete(2),
            
            # Низкоуровневое управление полетом
            "aileron": spaces.Box(-1, 1, (1,)),
            "elevator": spaces.Box(-1, 1, (1,)),
            "rudder": spaces.Box(-1, 1, (1,)),
            "throttle": spaces.Box(0, 1, (1,)),
        })
        
        # Состояние симуляции
        self.aircraft: Dict[str, Aircraft] = {}
        self.missiles: List[Missile] = []
        self.red_team: List[str] = []
        self.blue_team: List[str] = []
        self.t = 0
        
        # Метрики
        self.kills = {"red": 0, "blue": 0}
        self.missiles_fired = {"red": 0, "blue": 0}
        self.out_of_bounds_penalties = 0
        
        print("🛩️ DogfightEnv initialized with JSBSim")
        print(f"   Max aircraft: {self.max_aircraft}")
        print(f"   Airspace: {AIRSPACE_BOUNDS}")
        print(f"   Engagement range: {ENGAGEMENT_RANGE/1000:.0f} km")

    @property
    def observation_space(self): return self.single_obs_space

    @property  
    def action_space(self): return self.single_act_space

    def _spawn_aircraft(self):
        """Создает самолеты для боя"""
        n_red = int(self.rng.choice(self.red_choices))
        n_blue = int(self.rng.choice(self.blue_choices))
        
        self.red_team = [f"red_{i}" for i in range(n_red)]
        self.blue_team = [f"blue_{i}" for i in range(n_blue)]
        
        self.aircraft.clear()
        self.missiles.clear()
        
        # Спавн красной команды (запад)
        for i, aircraft_id in enumerate(self.red_team):
            pos = np.array([
                -50_000 + self.rng.uniform(-10_000, 10_000),  # x: западная сторона
                (i - n_red/2) * 5_000 + self.rng.uniform(-2_000, 2_000),  # y: разнесены
                8_000 + self.rng.uniform(-2_000, 2_000)  # z: ~8км высота
            ])
            heading = 90 + self.rng.uniform(-30, 30)  # на восток ±30°
            
            aircraft = Aircraft(aircraft_id, "red", pos, heading, "f16")
            self.aircraft[aircraft_id] = aircraft
        
        # Спавн синей команды (восток) 
        for j, aircraft_id in enumerate(self.blue_team):
            pos = np.array([
                50_000 + self.rng.uniform(-10_000, 10_000),   # x: восточная сторона
                (j - n_blue/2) * 5_000 + self.rng.uniform(-2_000, 2_000),  # y: разнесены
                8_000 + self.rng.uniform(-2_000, 2_000)   # z: ~8км высота
            ])
            heading = 270 + self.rng.uniform(-30, 30)  # на запад ±30°
            
            aircraft = Aircraft(aircraft_id, "blue", pos, heading, "f16")
            self.aircraft[aircraft_id] = aircraft
        
        print(f"🛩️ Spawned {n_red} red vs {n_blue} blue aircraft")

    def _update_missiles(self):
        """Обновляет состояние всех ракет"""
        active_missiles = []
        
        for missile in self.missiles:
            if not missile.active:
                continue
            
            # Находим цель
            if missile.target_id not in self.aircraft or not self.aircraft[missile.target_id].alive:
                missile.active = False
                continue
            
            target_pos = self.aircraft[missile.target_id].get_position()
            
            # Обновляем ракету
            if missile.update(self.dt, target_pos):
                active_missiles.append(missile)
                
                # Проверяем попадание
                distance = np.linalg.norm(missile.current_pos - target_pos)
                if distance < 100.0:  # радиус поражения
                    hit_prob = missile.get_hit_probability(
                        target_pos, self.aircraft[missile.target_id].get_velocity()
                    )
                    
                    if self.rng.random() < hit_prob:
                        # Попадание!
                        self.aircraft[missile.target_id].hp -= 80.0
                        if self.aircraft[missile.target_id].hp <= 0:
                            self.aircraft[missile.target_id].alive = False
                            
                            # Засчитываем убийство
                            shooter_team = "red" if missile.shooter_id.startswith("red_") else "blue"
                            self.kills[shooter_team] += 1
                            
                        print(f"🎯 {missile.shooter_id} hit {missile.target_id} with {missile.missile_type}")
                    
                    missile.active = False
        
        self.missiles = active_missiles

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        
        self.t = 0
        self.kills = {"red": 0, "blue": 0}
        self.missiles_fired = {"red": 0, "blue": 0}
        self.out_of_bounds_penalties = 0
        
        self._spawn_aircraft()
        
        obs, infos = {}, {}
        for aircraft_id in self.red_team + self.blue_team:
            obs[aircraft_id] = self._build_obs(aircraft_id)
            infos[aircraft_id] = {
                "team_sizes": (len(self.red_team), len(self.blue_team)),
                "airspace": AIRSPACE_BOUNDS,
                "engagement_range": ENGAGEMENT_RANGE
            }
        
        return obs, infos

    def step(self, action_dict: Dict[str, Any]):
        # 1) Применяем действия
        for aircraft_id, action in action_dict.items():
            if aircraft_id not in self.aircraft or not self.aircraft[aircraft_id].alive:
                continue
            
            aircraft = self.aircraft[aircraft_id]
            
            # Низкоуровневое управление полетом
            flight_controls = {
                "aileron": float(action.get("aileron", [0.0])[0]),
                "elevator": float(action.get("elevator", [0.0])[0]), 
                "rudder": float(action.get("rudder", [0.0])[0]),
                "throttle": float(action.get("throttle", [0.8])[0])
            }
            aircraft.apply_controls(flight_controls)
            
            # Высокоуровневые команды (стрельба)
            if action.get("fire", 0) == 1:
                target_idx = int(action.get("target", 0))
                weapon_type = list(WEAPON_TYPES.keys())[int(action.get("weapon", 0)) % len(WEAPON_TYPES)]
                
                # Находим цель
                enemy_team = self.blue_team if aircraft_id.startswith("red_") else self.red_team
                if target_idx < len(enemy_team):
                    target_id = enemy_team[target_idx]
                    if target_id in self.aircraft and self.aircraft[target_id].alive:
                        target_pos = self.aircraft[target_id].get_position()
                        
                        missile = aircraft.fire_weapon(target_id, target_pos, weapon_type)
                        if missile:
                            self.missiles.append(missile)
                            team = "red" if aircraft_id.startswith("red_") else "blue"
                            self.missiles_fired[team] += 1
        
        # 2) Обновляем физику
        for aircraft in self.aircraft.values():
            aircraft.update(self.dt)
        
        # 3) Обновляем ракеты
        self._update_missiles()
        
        self.t += 1
        
        # 4) Вычисляем награды и завершение
        obs, rews, terms, truncs, infos = {}, {}, {}, {}, {}
        
        red_alive = sum(1 for aid in self.red_team if self.aircraft[aid].alive)
        blue_alive = sum(1 for aid in self.blue_team if self.aircraft[aid].alive)
        
        done = (red_alive == 0) or (blue_alive == 0) or (self.t >= self.max_len)
        
        # Награды за выживание и убийства
        for aircraft_id in self.red_team + self.blue_team:
            if not self.aircraft[aircraft_id].alive:
                continue
                
            obs[aircraft_id] = self._build_obs(aircraft_id)
            
            # Базовая награда за выживание
            reward = 0.01
            
            # Награда за убийства команды
            team = "red" if aircraft_id.startswith("red_") else "blue"
            reward += self.kills[team] * 10.0
            
            # Штраф за потери команды
            enemy_kills = self.kills["blue" if team == "red" else "red"]
            reward -= enemy_kills * 5.0
            
            # Штраф за выход из границ
            pos = self.aircraft[aircraft_id].get_position()
            if not self._check_airspace_bounds(pos):
                reward -= 1.0
                self.out_of_bounds_penalties += 1
            
            # Награда за высоту (тактическое преимущество)
            altitude_bonus = min(0.1, pos[2] / AIRSPACE_BOUNDS['z_max']) * 0.05
            reward += altitude_bonus
            
            rews[aircraft_id] = float(reward)
            terms[aircraft_id] = False
            truncs[aircraft_id] = False
            
            infos[aircraft_id] = {
                "kills": self.kills[team],
                "missiles_fired": self.missiles_fired[team],
                "altitude": pos[2],
                "fuel": self.aircraft[aircraft_id].fuel,
                "hp": self.aircraft[aircraft_id].hp,
                "active_missiles": len(self.missiles),
                "position": pos.tolist(),
                "velocity": self.aircraft[aircraft_id].get_velocity().tolist()
            }
        
        # Бонус за победу
        if done:
            winner = None
            if red_alive > 0 and blue_alive == 0:
                winner = "red"
                for aid in self.red_team:
                    if aid in rews:
                        rews[aid] += 50.0
                for aid in self.blue_team:
                    if aid in rews:
                        rews[aid] -= 25.0
            elif blue_alive > 0 and red_alive == 0:
                winner = "blue"
                for aid in self.blue_team:
                    if aid in rews:
                        rews[aid] += 50.0
                for aid in self.red_team:
                    if aid in rews:
                        rews[aid] -= 25.0
            
            print(f"🏁 Battle ended: {winner or 'draw'} wins")
            print(f"   Kills: Red {self.kills['red']}, Blue {self.kills['blue']}")
            print(f"   Survivors: Red {red_alive}, Blue {blue_alive}")
        
        terms["__all__"] = False
        truncs["__all__"] = done
        
        return obs, rews, terms, truncs, infos

    def _check_airspace_bounds(self, pos: np.ndarray) -> bool:
        """Проверяет границы воздушного пространства"""
        return (AIRSPACE_BOUNDS['x_min'] <= pos[0] <= AIRSPACE_BOUNDS['x_max'] and
                AIRSPACE_BOUNDS['y_min'] <= pos[1] <= AIRSPACE_BOUNDS['y_max'] and
                AIRSPACE_BOUNDS['z_min'] <= pos[2] <= AIRSPACE_BOUNDS['z_max'])

    def _build_obs(self, aircraft_id: str) -> Dict[str, np.ndarray]:
        """Строит наблюдения для самолета"""
        aircraft = self.aircraft[aircraft_id]
        team = "red" if aircraft_id.startswith("red_") else "blue"
        
        # Собственное состояние (расширенное для авиации)
        self_state = aircraft.get_state_vector()
        
        # Добавляем информацию о вооружении
        weapon_state = np.array([
            aircraft.weapons["CANNON"]["count"] / 500.0,
            aircraft.weapons["SHORT_RANGE_MISSILE"]["count"] / 4.0,
            aircraft.weapons["MEDIUM_RANGE_MISSILE"]["count"] / 6.0,
            aircraft.weapons["LONG_RANGE_MISSILE"]["count"] / 2.0,
        ], dtype=np.float32)
        
        self_vec = np.concatenate([self_state, weapon_state], axis=0)
        
        # Дополняем до нужного размера
        if len(self_vec) < SELF_FEATURES:
            padding = np.zeros(SELF_FEATURES - len(self_vec), dtype=np.float32)
            self_vec = np.concatenate([self_vec, padding])
        else:
            self_vec = self_vec[:SELF_FEATURES]
        
        # Союзники
        ally_team = self.red_team if team == "red" else self.blue_team
        allies_data = np.zeros((self.max_aircraft, AIRCRAFT_FEATURES), dtype=np.float32)
        allies_mask = np.zeros(self.max_aircraft, dtype=np.int32)
        
        my_pos = aircraft.get_position()
        ally_idx = 0
        
        for ally_id in ally_team:
            if ally_id == aircraft_id or ally_idx >= self.max_aircraft:
                continue
            
            if ally_id in self.aircraft and self.aircraft[ally_id].alive:
                ally = self.aircraft[ally_id]
                ally_pos = ally.get_position()
                relative_pos = ally_pos - my_pos
                
                # Нормализуем относительную позицию
                relative_pos = relative_pos / ENGAGEMENT_RANGE
                
                ally_state = ally.get_state_vector()[:12]  # первые 12 параметров
                
                # Относительные параметры
                distance = np.linalg.norm(ally_pos - my_pos) / ENGAGEMENT_RANGE
                bearing = np.arctan2(relative_pos[1], relative_pos[0]) / np.pi
                elevation = np.arctan2(relative_pos[2], np.linalg.norm(relative_pos[:2])) / np.pi
                
                allies_data[ally_idx, :3] = relative_pos
                allies_data[ally_idx, 3:15] = ally_state
                allies_data[ally_idx, 15] = distance
                
                allies_mask[ally_idx] = 1
                ally_idx += 1
        
        # Враги
        enemy_team = self.blue_team if team == "red" else self.red_team
        enemies_data = np.zeros((self.max_aircraft, ENEMY_FEATURES), dtype=np.float32)
        enemies_mask = np.zeros(self.max_aircraft, dtype=np.int32)
        enemy_action_mask = np.zeros(self.max_aircraft, dtype=np.int32)
        
        enemy_idx = 0
        
        for enemy_id in enemy_team:
            if enemy_idx >= self.max_aircraft:
                break
            
            if enemy_id in self.aircraft and self.aircraft[enemy_id].alive:
                enemy = self.aircraft[enemy_id]
                enemy_pos = enemy.get_position()
                relative_pos = enemy_pos - my_pos
                distance = np.linalg.norm(relative_pos)
                
                # Нормализуем относительную позицию
                relative_pos = relative_pos / ENGAGEMENT_RANGE
                
                # Информация о враге (ограниченная - только то, что видно радаром)
                enemy_state = enemy.get_state_vector()[:10]  # ограниченная информация
                
                bearing = np.arctan2(relative_pos[1], relative_pos[0]) / np.pi
                elevation = np.arctan2(relative_pos[2], np.linalg.norm(relative_pos[:2])) / np.pi
                norm_distance = distance / ENGAGEMENT_RANGE
                
                enemies_data[enemy_idx, :3] = relative_pos
                enemies_data[enemy_idx, 3:13] = enemy_state
                enemies_mask[enemy_idx] = 1
                
                # Маска возможности атаки (в зависимости от дистанции и вооружения)
                can_attack = False
                for weapon_type in WEAPON_TYPES.keys():
                    if aircraft.can_engage_target(enemy_pos, weapon_type):
                        can_attack = True
                        break
                
                enemy_action_mask[enemy_idx] = 1 if can_attack else 0
                enemy_idx += 1
        
        # Глобальное состояние
        global_state = np.zeros(GLOBAL_FEATURES, dtype=np.float32)
        
        red_alive = sum(1 for aid in self.red_team if self.aircraft[aid].alive)
        blue_alive = sum(1 for aid in self.blue_team if self.aircraft[aid].alive)
        
        global_state[0] = red_alive / len(self.red_team)
        global_state[1] = blue_alive / len(self.blue_team)
        global_state[2] = self.kills["red"]
        global_state[3] = self.kills["blue"]
        global_state[4] = len(self.missiles) / 20.0  # активные ракеты
        global_state[5] = self.t / self.max_len  # прогресс боя
        
        # Центры команд
        if red_alive > 0:
            red_positions = [self.aircraft[aid].get_position() for aid in self.red_team 
                           if self.aircraft[aid].alive]
            red_center = np.mean(red_positions, axis=0) / ENGAGEMENT_RANGE
            global_state[6:9] = red_center
        
        if blue_alive > 0:
            blue_positions = [self.aircraft[aid].get_position() for aid in self.blue_team 
                            if self.aircraft[aid].alive]
            blue_center = np.mean(blue_positions, axis=0) / ENGAGEMENT_RANGE
            global_state[9:12] = blue_center
        
        # Информация о границах и тактической обстановке
        global_state[12] = AIRSPACE_BOUNDS['z_max'] / 20_000.0  # нормализованная высота потолка
        global_state[13] = ENGAGEMENT_RANGE / 100_000.0  # нормализованная дальность боя
        global_state[14] = self.missiles_fired["red"]
        global_state[15] = self.missiles_fired["blue"]
        
        # Заполняем оставшиеся элементы шумом для разнообразия
        global_state[16:] = self.rng.normal(0, 0.1, size=GLOBAL_FEATURES-16).astype(np.float32)
        
        return {
            "self": np.clip(self_vec, -10.0, 10.0),
            "allies": np.clip(allies_data, -10.0, 10.0),
            "allies_mask": allies_mask,
            "enemies": np.clip(enemies_data, -10.0, 10.0), 
            "enemies_mask": enemies_mask,
            "global_state": np.clip(global_state, -10.0, 10.0),
            "enemy_action_mask": enemy_action_mask,
        }

    def get_battle_summary(self) -> Dict[str, Any]:
        """Возвращает сводку боя"""
        red_alive = sum(1 for aid in self.red_team if self.aircraft[aid].alive)
        blue_alive = sum(1 for aid in self.blue_team if self.aircraft[aid].alive)
        
        return {
            "step": self.t,
            "red_aircraft": len(self.red_team),
            "blue_aircraft": len(self.blue_team), 
            "red_alive": red_alive,
            "blue_alive": blue_alive,
            "red_kills": self.kills["red"],
            "blue_kills": self.kills["blue"],
            "missiles_fired_red": self.missiles_fired["red"],
            "missiles_fired_blue": self.missiles_fired["blue"],
            "active_missiles": len(self.missiles),
            "out_of_bounds_penalties": self.out_of_bounds_penalties,
            "airspace": AIRSPACE_BOUNDS,
            "engagement_range": ENGAGEMENT_RANGE
        }

    def export_for_visualization(self) -> Dict[str, Any]:
        """Экспортирует данные для визуализации"""
        aircraft_states = []
        
        for aircraft_id, aircraft in self.aircraft.items():
            if aircraft_id in self.aircraft:
                pos = aircraft.get_position()
                vel = aircraft.get_velocity()
                state_vec = aircraft.get_state_vector()
                
                aircraft_states.append({
                    "id": aircraft_id,
                    "team": aircraft.team,
                    "position": pos.tolist(),
                    "velocity": vel.tolist(),
                    "alive": aircraft.alive,
                    "hp": aircraft.hp,
                    "fuel": aircraft.fuel,
                    "heading": state_vec[5] * 180.0,  # psi в градусах
                    "altitude": pos[2],
                    "speed": np.linalg.norm(vel),
                    "weapons": aircraft.weapons
                })
        
        missile_states = []
        for missile in self.missiles:
            missile_states.append({
                "id": missile.id,
                "shooter_id": missile.shooter_id,
                "target_id": missile.target_id,
                "position": missile.current_pos.tolist(),
                "type": missile.missile_type,
                "active": missile.active,
                "fuel": missile.fuel
            })
        
        return {
            "timestamp": self.t,
            "aircraft": aircraft_states,
            "missiles": missile_states,
            "kills": self.kills.copy(),
            "missiles_fired": self.missiles_fired.copy(),
            "airspace_bounds": AIRSPACE_BOUNDS,
            "engagement_range": ENGAGEMENT_RANGE
        }


# Тестирование окружения
def test_jsbsim_environment():
    """Тестирует JSBSim окружение"""
    print("🧪 Testing JSBSim Aircraft Environment")
    
    try:
        env = DogfightEnv({
            "red_choices": [2],
            "blue_choices": [2],
            "episode_len": 100
        })
        
        obs, info = env.reset()
        print(f"✅ Environment reset successful")
        print(f"   Aircraft: {list(obs.keys())}")
        print(f"   Observation space: {env.observation_space}")
        print(f"   Action space: {env.action_space}")
        
        # Тестируем несколько шагов
        for step in range(10):
            actions = {}
            for aircraft_id in obs.keys():
                actions[aircraft_id] = {
                    "target": 0,
                    "maneuver": np.random.randint(0, len(MANEUVERS)),
                    "weapon": np.random.randint(0, len(WEAPON_TYPES)),
                    "fire": np.random.randint(0, 2),
                    "aileron": [np.random.uniform(-0.3, 0.3)],
                    "elevator": [np.random.uniform(-0.3, 0.3)],
                    "rudder": [np.random.uniform(-0.3, 0.3)],
                    "throttle": [np.random.uniform(0.7, 1.0)]
                }
            
            obs, rewards, terms, truncs, infos = env.step(actions)
            
            if step % 3 == 0:
                summary = env.get_battle_summary()
                print(f"Step {step}: Red {summary['red_alive']}/{summary['red_aircraft']}, "
                      f"Blue {summary['blue_alive']}/{summary['blue_aircraft']}, "
                      f"Missiles: {summary['active_missiles']}")
            
            if terms.get("__all__") or truncs.get("__all__"):
                break
        
        print("✅ JSBSim environment test completed successfully")
        return True
        
    except Exception as e:
        print(f"❌ JSBSim environment test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        test_jsbsim_environment()
    else:
        print("🛩️ JSBSim Aircraft Environment for Multi-Agent Dogfighting")
        print("Usage: python aircraft_env.py test")
        print("\nFeatures:")
        print("- JSBSim flight dynamics")
        print("- Multi-agent air-to-air combat")
        print("- Hierarchical control (high-level tactics + low-level flight controls)")
        print("- Missile simulation with realistic ballistics")
        print("- Radar and sensor modeling")
        print("- Airspace boundaries and constraints")