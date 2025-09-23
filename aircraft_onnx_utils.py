"""
ONNX утилиты для авиационных моделей
Инференс, валидация и тестирование экспортированных моделей самолетов
"""

import numpy as np
import onnxruntime as ort
import json
import os
import glob
from typing import Dict, List, Optional, Tuple, Any
import time

class AircraftONNXInference:
    """Класс для инференса авиационных ONNX моделей"""
    
    def __init__(self, onnx_path: str, meta_path: Optional[str] = None):
        self.onnx_path = onnx_path
        self.meta_path = meta_path or self._find_meta_file(onnx_path)
        
        # Загружаем метаданные
        self.meta = self._load_meta()
        
        # Создаем ONNX сессию
        self.session = ort.InferenceSession(
            onnx_path,
            providers=["CPUExecutionProvider"]
        )
        
        # Получаем информацию о входах и выходах
        self.input_info = {inp.name: inp for inp in self.session.get_inputs()}
        self.output_info = {out.name: out for out in self.session.get_outputs()}
        
        print(f"✈️ Loaded Aircraft ONNX model: {os.path.basename(onnx_path)}")
        print(f"   Policy: {self.meta.get('policy_id', 'unknown')}")
        print(f"   Max aircraft: {self.meta['max_aircraft']}")
        print(f"   Airspace: {self.meta.get('airspace_bounds', 'unknown')}")
        print(f"   Inputs: {list(self.input_info.keys())}")
        print(f"   Outputs: {list(self.output_info.keys())}")
    
    def _find_meta_file(self, onnx_path: str) -> Optional[str]:
        """Ищет мета-файл для ONNX модели"""
        onnx_dir = os.path.dirname(onnx_path)
        onnx_name = os.path.basename(onnx_path)
        
        possible_names = [
            onnx_name.replace('.onnx', '_meta.json'),
            onnx_name.replace('.onnx', '.json'),
            'meta.json',
            'aircraft_meta.json'
        ]
        
        for name in possible_names:
            meta_path = os.path.join(onnx_dir, name)
            if os.path.exists(meta_path):
                return meta_path
        
        print(f"Warning: No meta file found for {onnx_path}")
        return None
    
    def _load_meta(self) -> Dict[str, Any]:
        """Загружает метаданные модели"""
        if self.meta_path and os.path.exists(self.meta_path):
            try:
                with open(self.meta_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                print(f"Error loading meta file: {e}")
        
        # Значения по умолчанию для авиационных моделей
        return {
            "policy_id": "unknown",
            "max_aircraft": 8,
            "aircraft_model": True,
            "obs_space": {
                "self": [18],
                "allies": [8, 15],
                "enemies": [8, 13],
                "allies_mask": [8],
                "enemies_mask": [8],
                "enemy_action_mask": [8],
                "global_state": [64]
            },
            "airspace_bounds": {
                "x_min": -100000, "x_max": 100000,
                "y_min": -100000, "y_max": 100000,
                "z_min": 1000, "z_max": 20000
            },
            "engagement_range": 50000
        }
    
    def prepare_aircraft_observation(self, 
                                   self_state: np.ndarray,
                                   allies: List[np.ndarray],
                                   enemies: List[np.ndarray],
                                   global_state: Optional[np.ndarray] = None,
                                   batch_size: int = 1) -> Dict[str, np.ndarray]:
        """Подготавливает наблюдения для авиационной модели"""
        
        max_aircraft = self.meta["max_aircraft"]
        obs_shapes = self.meta["obs_space"]
        
        # Подготавливаем массивы
        obs = {}
        
        # Self state (состояние своего самолета)
        self_shape = obs_shapes["self"]
        obs["self"] = np.zeros((batch_size,) + tuple(self_shape), dtype=np.float32)
        if batch_size == 1:
            obs["self"][0] = self_state[:self_shape[0]]
        
        # Allies (союзники)
        allies_shape = obs_shapes["allies"]
        obs["allies"] = np.zeros((batch_size,) + tuple(allies_shape), dtype=np.float32)
        obs["allies_mask"] = np.zeros((batch_size, allies_shape[0]), dtype=np.int32)
        
        n_allies = min(len(allies), allies_shape[0])
        for i in range(n_allies):
            if batch_size == 1:
                obs["allies"][0, i, :allies_shape[1]] = allies[i][:allies_shape[1]]
                obs["allies_mask"][0, i] = 1
        
        # Enemies (враги)
        enemies_shape = obs_shapes["enemies"] 
        obs["enemies"] = np.zeros((batch_size,) + tuple(enemies_shape), dtype=np.float32)
        obs["enemies_mask"] = np.zeros((batch_size, enemies_shape[0]), dtype=np.int32)
        obs["enemy_action_mask"] = np.zeros((batch_size, enemies_shape[0]), dtype=np.int32)
        
        n_enemies = min(len(enemies), enemies_shape[0])
        for j in range(n_enemies):
            if batch_size == 1:
                obs["enemies"][0, j, :enemies_shape[1]] = enemies[j][:enemies_shape[1]]
                obs["enemies_mask"][0, j] = 1
                obs["enemy_action_mask"][0, j] = 1  # Все враги доступны для атаки по умолчанию
        
        # Global state
        global_shape = obs_shapes["global_state"]
        obs["global_state"] = np.zeros((batch_size,) + tuple(global_shape), dtype=np.float32)
        if global_state is not None and batch_size == 1:
            obs["global_state"][0, :len(global_state)] = global_state[:global_shape[0]]
        
        return obs
    
    def predict(self, observations: Dict[str, np.ndarray]) -> np.ndarray:
        """Выполняет предсказание для авиационной модели"""
        
        # Подготавливаем входы для ONNX
        onnx_inputs = {}
        for input_name in self.input_info.keys():
            if input_name in observations:
                onnx_inputs[input_name] = observations[input_name]
            else:
                raise ValueError(f"Required input '{input_name}' not found in observations")
        
        # Выполняем предсказание
        try:
            outputs = self.session.run(["action_logits"], onnx_inputs)
            return outputs[0]
        except Exception as e:
            print(f"Aircraft ONNX inference error: {e}")
            print(f"Input shapes: {[(k, v.shape) for k, v in onnx_inputs.items()]}")
            raise
    
    def decode_aircraft_action(self, action_logits: np.ndarray) -> Dict[str, Any]:
        """Декодирует выход модели в авиационные действия"""
        if action_logits.ndim == 1:
            action_logits = action_logits.reshape(1, -1)
        
        batch_size = action_logits.shape[0]
        max_aircraft = self.meta["max_aircraft"]
        
        # Парсим логиты согласно структуре авиационных действий
        idx = 0
        
        # Target selection
        target_logits = action_logits[:, idx:idx+max_aircraft]; idx += max_aircraft
        target = np.argmax(target_logits, axis=-1)
        
        # Maneuver type (9 типов маневров)
        maneuver_logits = action_logits[:, idx:idx+9]; idx += 9
        maneuver = np.argmax(maneuver_logits, axis=-1)
        
        # Weapon type (4 типа оружия)
        weapon_logits = action_logits[:, idx:idx+4]; idx += 4
        weapon = np.argmax(weapon_logits, axis=-1)
        
        # Fire decision
        fire_logits = action_logits[:, idx:idx+1]; idx += 1
        fire = (fire_logits > 0).astype(int).flatten()
        
        # Flight controls (continuous)
        aileron = np.tanh(action_logits[:, idx:idx+1]); idx += 1
        elevator = np.tanh(action_logits[:, idx:idx+1]); idx += 1
        rudder = np.tanh(action_logits[:, idx:idx+1]); idx += 1
        throttle = np.sigmoid(action_logits[:, idx:idx+1]); idx += 1
        
        decoded = {
            "target": target,
            "maneuver": maneuver,
            "weapon": weapon, 
            "fire": fire,
            "aileron": aileron.flatten(),
            "elevator": elevator.flatten(),
            "rudder": rudder.flatten(),
            "throttle": throttle.flatten(),
        }
        
        if batch_size == 1:
            # Убираем batch dimension для single sample
            decoded = {k: v[0] if isinstance(v[0], (int, np.integer)) else v[0] 
                      for k, v in decoded.items()}
        
        return decoded
    
    def get_action_explanation(self, decoded_action: Dict[str, Any]) -> str:
        """Возвращает человекопонятное описание действий"""
        
        maneuvers = ["STRAIGHT", "CLIMB", "DIVE", "LEFT_TURN", "RIGHT_TURN", 
                    "BARREL_ROLL", "SPLIT_S", "IMMELMANN", "DEFENSIVE_SPIRAL"]
        weapons = ["CANNON", "SHORT_RANGE_MISSILE", "MEDIUM_RANGE_MISSILE", "LONG_RANGE_MISSILE"]
        
        maneuver_name = maneuvers[int(decoded_action["maneuver"]) % len(maneuvers)]
        weapon_name = weapons[int(decoded_action["weapon"]) % len(weapons)]
        
        explanation = [
            f"Target: Enemy #{decoded_action['target']}",
            f"Maneuver: {maneuver_name}",
            f"Weapon: {weapon_name}",
            f"Fire: {'YES' if decoded_action['fire'] else 'NO'}",
            f"Flight Controls:",
            f"  Aileron: {decoded_action['aileron']:+.3f}",
            f"  Elevator: {decoded_action['elevator']:+.3f}",
            f"  Rudder: {decoded_action['rudder']:+.3f}",
            f"  Throttle: {decoded_action['throttle']:.3f}",
        ]
        
        return "\n".join(explanation)

def test_aircraft_onnx_inference(onnx_path: str, batch_size: int = 2, verbose: bool = True):
    """Тестирует инференс авиационной ONNX модели"""
    
    print(f"=== Aircraft ONNX Inference Test ===")
    print(f"Model: {onnx_path}")
    print(f"Batch size: {batch_size}")
    
    # Создаем inference engine
    engine = AircraftONNXInference(onnx_path)
    
    # Генерируем тестовые авиационные данные
    rng = np.random.default_rng(42)
    
    # Self state (состояние своего самолета)
    # [lat, lon, alt, phi, theta, psi, u, v, w, p, q, r, tas, mach, nx, ny, nz, fuel, hp]
    self_state = np.array([
        35.0,  # latitude (degrees)
        -118.0,  # longitude (degrees)  
        8000/20000,  # altitude normalized
        0.1, 0.05, 90/180,  # attitude (phi, theta, psi normalized)
        200/500, 0/500, 10/500,  # velocities (u, v, w normalized)
        0.1, 0.0, 0.2,  # angular rates
        250/500, 0.7/2.0,  # airspeed, mach
        1.2/10, 0.1/10, 1.0/10,  # accelerations (normalized to 10g)
        0.8, 0.9  # fuel, hp
    ], dtype=np.float32)
    
    # Союзники (случайные относительные данные)
    n_allies = rng.integers(1, 4)
    allies = []
    for i in range(n_allies):
        ally_data = rng.normal(0, 0.3, size=15).astype(np.float32)
        allies.append(ally_data)
    
    # Враги (случайные относительные данные)
    n_enemies = rng.integers(1, 4)
    enemies = []
    for i in range(n_enemies):
        enemy_data = rng.normal(0, 0.4, size=13).astype(np.float32)
        enemies.append(enemy_data)
    
    # Глобальное состояние
    global_state = rng.normal(0, 0.2, size=64).astype(np.float32)
    
    if verbose:
        print(f"\nTest scenario:")
        print(f"  Self state shape: {self_state.shape}")
        print(f"  Allies: {n_allies}")
        print(f"  Enemies: {n_enemies}")
        print(f"  Global state shape: {global_state.shape}")
    
    # Подготавливаем наблюдения
    observations = engine.prepare_aircraft_observation(
        self_state=self_state,
        allies=allies,
        enemies=enemies,
        global_state=global_state,
        batch_size=batch_size
    )
    
    if verbose:
        print(f"\nPrepared observations:")
        for name, arr in observations.items():
            print(f"  {name}: {arr.shape}")
    
    # Выполняем предсказание
    try:
        start_time = time.time()
        action_logits = engine.predict(observations)
        inference_time = time.time() - start_time
        
        if verbose:
            print(f"\nPrediction successful!")
            print(f"  Inference time: {inference_time*1000:.2f}ms")
            print(f"  Action logits shape: {action_logits.shape}")
        
        # Декодируем действия
        for b in range(batch_size):
            decoded = engine.decode_aircraft_action(action_logits[b:b+1])
            
            if verbose:
                print(f"\n=== Batch {b} Aircraft Actions ===")
                explanation = engine.get_action_explanation(decoded)
                print(explanation)
        
        return action_logits
        
    except Exception as e:
        print(f"✗ Aircraft inference failed: {e}")
        raise

def benchmark_aircraft_performance(onnx_path: str, num_iterations: int = 1000):
    """Тест производительности авиационного инференса"""
    
    print(f"\n=== Aircraft Performance Benchmark ===")
    print(f"Iterations: {num_iterations}")
    
    engine = AircraftONNXInference(onnx_path)
    
    # Подготавливаем статические данные
    rng = np.random.default_rng(42)
    
    self_state = rng.normal(0, 0.5, size=18).astype(np.float32)
    allies = [rng.normal(0, 0.3, size=15).astype(np.float32) for _ in range(2)]
    enemies = [rng.normal(0, 0.4, size=13).astype(np.float32) for _ in range(3)]
    global_state = rng.normal(0, 0.2, size=64).astype(np.float32)
    
    observations = engine.prepare_aircraft_observation(
        self_state=self_state,
        allies=allies,
        enemies=enemies,
        global_state=global_state,
        batch_size=1
    )
    
    # Разогрев
    print("Warming up...")
    for _ in range(10):
        engine.predict(observations)
    
    # Бенчмарк
    print("Running benchmark...")
    times = []
    
    for i in range(num_iterations):
        start_time = time.time()
        engine.predict(observations)
        end_time = time.time()
        times.append(end_time - start_time)
        
        if i % 100 == 0:
            print(f"  Progress: {i}/{num_iterations}")
    
    # Статистика
    times = np.array(times)
    
    print(f"\nPerformance results:")
    print(f"  Mean time per inference: {np.mean(times)*1000:.3f}ms")
    print(f"  Std deviation: {np.std(times)*1000:.3f}ms")
    print(f"  Min time: {np.min(times)*1000:.3f}ms")
    print(f"  Max time: {np.max(times)*1000:.3f}ms")
    print(f"  95th percentile: {np.percentile(times, 95)*1000:.3f}ms")
    print(f"  Inferences per second: {1/np.mean(times):.1f}")
    
    # Оценка для реального времени
    real_time_freq = 10  # 10 Hz для авиации
    real_time_budget = 1000 / real_time_freq  # 100ms бюджет
    
    print(f"\nReal-time feasibility (10 Hz):")
    print(f"  Time budget: {real_time_budget:.1f}ms")
    print(f"  Average usage: {np.mean(times)*1000/real_time_budget*100:.1f}%")
    feasible = np.mean(times)*1000 < real_time_budget * 0.8  # 80% запас
    print(f"  Real-time capable: {'✅ YES' if feasible else '❌ NO'}")

def find_latest_aircraft_models(export_dir: str = "./aircraft_onnx_exports") -> List[str]:
    """Находит последние экспортированные авиационные ONNX модели"""
    
    if not os.path.exists(export_dir):
        print(f"Export directory not found: {export_dir}")
        return []
    
    # Ищем latest символическую ссылку
    latest_dir = os.path.join(export_dir, "latest")
    if os.path.exists(latest_dir):
        onnx_files = glob.glob(os.path.join(latest_dir, "*aircraft*.onnx"))
        if not onnx_files:
            onnx_files = glob.glob(os.path.join(latest_dir, "*.onnx"))
        
        if onnx_files:
            print(f"Found {len(onnx_files)} aircraft ONNX files in latest export:")
            for f in onnx_files:
                print(f"  {os.path.basename(f)}")
            return onnx_files
    
    # Если latest нет, ищем самую новую итерацию
    iter_dirs = glob.glob(os.path.join(export_dir, "iter_*"))
    if iter_dirs:
        latest_iter = max(iter_dirs, key=os.path.getmtime)
        onnx_files = glob.glob(os.path.join(latest_iter, "*aircraft*.onnx"))
        if not onnx_files:
            onnx_files = glob.glob(os.path.join(latest_iter, "*.onnx"))
            
        if onnx_files:
            print(f"Found {len(onnx_files)} aircraft ONNX files in {os.path.basename(latest_iter)}:")
            for f in onnx_files:
                print(f"  {os.path.basename(f)}")
            return onnx_files
    
    print("No aircraft ONNX files found")
    return []

def validate_aircraft_model(onnx_path: str) -> bool:
    """Валидирует авиационную ONNX модель"""
    
    print(f"🔍 Validating aircraft ONNX model: {os.path.basename(onnx_path)}")
    
    try:
        engine = AircraftONNXInference(onnx_path)
        
        # Проверяем что это авиационная модель
        if not engine.meta.get("aircraft_model", False):
            print("⚠️ Warning: Model metadata doesn't indicate aircraft model")
        
        # Проверяем размеры
        expected_inputs = ["self", "allies", "allies_mask", "enemies", "enemies_mask", "enemy_action_mask", "global_state"]
        missing_inputs = [inp for inp in expected_inputs if inp not in engine.input_info]
        
        if missing_inputs:
            print(f"❌ Missing expected inputs: {missing_inputs}")
            return False
        
        # Проверяем мета-информацию
        meta = engine.meta
        required_meta = ["max_aircraft", "obs_space"]
        missing_meta = [key for key in required_meta if key not in meta]
        
        if missing_meta:
            print(f"⚠️ Missing metadata: {missing_meta}")
        
        # Тестовый инференс
        try:
            test_aircraft_onnx_inference(onnx_path, batch_size=1, verbose=False)
            print("✅ Inference test passed")
        except Exception as e:
            print(f"❌ Inference test failed: {e}")
            return False
        
        # Проверяем производительность
        try:
            times = []
            engine_test = AircraftONNXInference(onnx_path)
            
            # Быстрый тест производительности
            rng = np.random.default_rng(42)
            self_state = rng.normal(0, 0.5, size=18).astype(np.float32)
            allies = [rng.normal(0, 0.3, size=15).astype(np.float32)]
            enemies = [rng.normal(0, 0.4, size=13).astype(np.float32)]
            global_state = rng.normal(0, 0.2, size=64).astype(np.float32)
            
            observations = engine_test.prepare_aircraft_observation(
                self_state, allies, enemies, global_state, batch_size=1
            )
            
            # 10 тестовых запусков
            for _ in range(10):
                start = time.time()
                engine_test.predict(observations)
                times.append(time.time() - start)
            
            avg_time = np.mean(times) * 1000
            print(f"✅ Performance: {avg_time:.2f}ms average")
            
            if avg_time > 100:  # 100ms для авиации может быть критично
                print(f"⚠️ Warning: Inference time may be too slow for real-time ({avg_time:.1f}ms)")
            
        except Exception as e:
            print(f"⚠️ Performance test failed: {e}")
        
        print("✅ Aircraft model validation passed")
        return True
        
    except Exception as e:
        print(f"❌ Validation failed: {e}")
        return False

def create_aircraft_model_report(onnx_path: str, output_path: str = None):
    """Создает отчет об авиационной модели"""
    
    if output_path is None:
        output_path = onnx_path.replace('.onnx', '_report.json')
    
    print(f"📊 Creating aircraft model report...")
    
    try:
        engine = AircraftONNXInference(onnx_path)
        
        # Собираем информацию о модели
        report = {
            "model_path": onnx_path,
            "model_name": os.path.basename(onnx_path),
            "validation_timestamp": time.time(),
            "metadata": engine.meta,
            "input_shapes": {name: info.shape for name, info in engine.input_info.items()},
            "output_shapes": {name: info.shape for name, info in engine.output_info.items()},
        }
        
        # Тест производительности
        print("Running performance test...")
        rng = np.random.default_rng(42)
        
        times = []
        accuracies = []
        
        for i in range(50):  # 50 тестов
            # Генерируем случайные данные
            self_state = rng.normal(0, 0.5, size=18).astype(np.float32)
            allies = [rng.normal(0, 0.3, size=15).astype(np.float32) for _ in range(rng.integers(0, 3))]
            enemies = [rng.normal(0, 0.4, size=13).astype(np.float32) for _ in range(rng.integers(1, 4))]
            global_state = rng.normal(0, 0.2, size=64).astype(np.float32)
            
            observations = engine.prepare_aircraft_observation(
                self_state, allies, enemies, global_state, batch_size=1
            )
            
            # Измеряем время
            start = time.time()
            action_logits = engine.predict(observations)
            inference_time = time.time() - start
            times.append(inference_time)
            
            # Проверяем валидность выходов
            decoded = engine.decode_aircraft_action(action_logits)
            
            # Простые проверки валидности
            valid_target = 0 <= decoded["target"] < engine.meta["max_aircraft"]
            valid_controls = all(-1 <= decoded[ctrl] <= 1 for ctrl in ["aileron", "elevator", "rudder"])
            valid_throttle = 0 <= decoded["throttle"] <= 1
            
            accuracy = sum([valid_target, valid_controls, valid_throttle]) / 3
            accuracies.append(accuracy)
        
        # Статистика производительности
        times = np.array(times)
        accuracies = np.array(accuracies)
        
        report["performance"] = {
            "mean_inference_time_ms": float(np.mean(times) * 1000),
            "std_inference_time_ms": float(np.std(times) * 1000),
            "min_inference_time_ms": float(np.min(times) * 1000),
            "max_inference_time_ms": float(np.max(times) * 1000),
            "p95_inference_time_ms": float(np.percentile(times, 95) * 1000),
            "inferences_per_second": float(1 / np.mean(times)),
            "mean_accuracy": float(np.mean(accuracies)),
            "min_accuracy": float(np.min(accuracies)),
        }
        
        # Real-time capability
        real_time_budget = 100  # 100ms для 10 Hz
        report["real_time"] = {
            "target_frequency_hz": 10,
            "time_budget_ms": real_time_budget,
            "budget_utilization_percent": float(np.mean(times) * 1000 / real_time_budget * 100),
            "real_time_capable": bool(np.mean(times) * 1000 < real_time_budget * 0.8),
        }
        
        # Пример декодированных действий
        print("Generating sample actions...")
        sample_obs = engine.prepare_aircraft_observation(
            self_state=rng.normal(0, 0.5, size=18).astype(np.float32),
            allies=[rng.normal(0, 0.3, size=15).astype(np.float32)],
            enemies=[rng.normal(0, 0.4, size=13).astype(np.float32) for _ in range(2)],
            global_state=rng.normal(0, 0.2, size=64).astype(np.float32),
            batch_size=1
        )
        
        sample_logits = engine.predict(sample_obs)
        sample_actions = engine.decode_aircraft_action(sample_logits)
        
        report["sample_actions"] = {
            str(k): float(v) if np.isscalar(v) else v.tolist() 
            for k, v in sample_actions.items()
        }
        
        # Сохраняем отчет
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"✅ Aircraft model report saved: {output_path}")
        
        # Краткий вывод
        perf = report["performance"]
        rt = report["real_time"]
        
        print(f"\n📊 Model Summary:")
        print(f"   Policy: {report['metadata'].get('policy_id', 'unknown')}")
        print(f"   Max aircraft: {report['metadata']['max_aircraft']}")
        print(f"   Inference time: {perf['mean_inference_time_ms']:.2f}ms ±{perf['std_inference_time_ms']:.2f}ms")
        print(f"   Throughput: {perf['inferences_per_second']:.1f} inferences/sec")
        print(f"   Real-time capable (10 Hz): {'✅ YES' if rt['real_time_capable'] else '❌ NO'}")
        print(f"   Accuracy: {perf['mean_accuracy']*100:.1f}%")
        
        return report
        
    except Exception as e:
        print(f"❌ Report generation failed: {e}")
        return None

def compare_aircraft_models(model_paths: List[str]):
    """Сравнивает несколько авиационных моделей"""
    
    print(f"⚔️ Comparing {len(model_paths)} aircraft models...")
    
    results = []
    
    for model_path in model_paths:
        print(f"\nTesting {os.path.basename(model_path)}...")
        
        try:
            engine = AircraftONNXInference(model_path)
            
            # Быстрый тест производительности
            rng = np.random.default_rng(42)
            times = []
            
            for _ in range(20):
                self_state = rng.normal(0, 0.5, size=18).astype(np.float32)
                allies = [rng.normal(0, 0.3, size=15).astype(np.float32)]
                enemies = [rng.normal(0, 0.4, size=13).astype(np.float32) for _ in range(2)]
                global_state = rng.normal(0, 0.2, size=64).astype(np.float32)
                
                observations = engine.prepare_aircraft_observation(
                    self_state, allies, enemies, global_state, batch_size=1
                )
                
                start = time.time()
                engine.predict(observations)
                times.append(time.time() - start)
            
            avg_time = np.mean(times) * 1000
            
            results.append({
                "model": os.path.basename(model_path),
                "path": model_path,
                "policy_id": engine.meta.get("policy_id", "unknown"),
                "max_aircraft": engine.meta.get("max_aircraft", "unknown"),
                "inference_time_ms": avg_time,
                "throughput_hz": 1 / np.mean(times),
                "real_time_capable": avg_time < 80,  # 80ms для запаса
            })
            
        except Exception as e:
            print(f"❌ Failed to test {model_path}: {e}")
            results.append({
                "model": os.path.basename(model_path),
                "path": model_path,
                "error": str(e)
            })
    
    # Выводим сравнение
    print(f"\n📊 Aircraft Models Comparison:")
    print(f"{'Model':<30} {'Policy':<15} {'Time (ms)':<12} {'Real-time':<12}")
    print("-" * 80)
    
    for result in results:
        if "error" in result:
            print(f"{result['model']:<30} {'ERROR':<15} {'-':<12} {'-':<12}")
        else:
            rt_status = "✅ YES" if result["real_time_capable"] else "❌ NO"
            print(f"{result['model']:<30} {result['policy_id']:<15} {result['inference_time_ms']:<12.2f} {rt_status:<12}")
    
    # Находим лучшую модель
    valid_results = [r for r in results if "error" not in r]
    if valid_results:
        best_model = min(valid_results, key=lambda x: x["inference_time_ms"])
        print(f"\n🏆 Fastest model: {best_model['model']} ({best_model['inference_time_ms']:.2f}ms)")
    
    return results

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "test" and len(sys.argv) > 2:
            # Тест конкретной модели
            test_aircraft_onnx_inference(sys.argv[2], verbose=True)
            
        elif sys.argv[1] == "benchmark" and len(sys.argv) > 2:
            # Бенчмарк производительности
            benchmark_aircraft_performance(sys.argv[2], num_iterations=500)
            
        elif sys.argv[1] == "validate" and len(sys.argv) > 2:
            # Валидация модели
            validate_aircraft_model(sys.argv[2])
            
        elif sys.argv[1] == "report" and len(sys.argv) > 2:
            # Создание отчета
            output_path = sys.argv[3] if len(sys.argv) > 3 else None
            create_aircraft_model_report(sys.argv[2], output_path)
            
        elif sys.argv[1] == "compare":
            # Сравнение моделей
            if len(sys.argv) > 2:
                model_paths = sys.argv[2:]
            else:
                model_paths = find_latest_aircraft_models()
            
            if model_paths:
                compare_aircraft_models(model_paths)
            else:
                print("No models found for comparison")
                
        elif sys.argv[1] == "find":
            # Поиск моделей
            export_dir = sys.argv[2] if len(sys.argv) > 2 else "./aircraft_onnx_exports"
            models = find_latest_aircraft_models(export_dir)
            
        else:
            print("Usage:")
            print("  python aircraft_onnx_utils.py test <model.onnx>")
            print("  python aircraft_onnx_utils.py benchmark <model.onnx>")
            print("  python aircraft_onnx_utils.py validate <model.onnx>")
            print("  python aircraft_onnx_utils.py report <model.onnx> [output.json]")
            print("  python aircraft_onnx_utils.py compare [model1.onnx model2.onnx ...]")
            print("  python aircraft_onnx_utils.py find [export_dir]")
    else:
        print("🛩️ Aircraft ONNX Utilities")
        print("Features:")
        print("- ONNX model inference for aircraft AI")
        print("- Performance benchmarking")
        print("- Model validation and testing")
        print("- Detailed reporting")
        print("- Model comparison")
        print("- Real-time capability assessment")
        print("\nRun with arguments for specific operations.")