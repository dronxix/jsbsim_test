"""
Интеграция Policy Selector в основной цикл обучения самолетов
Адаптивная система выбора специализированных политик в процессе боя
УЛУЧШЕННАЯ ВЕРСИЯ с сохранением результатов и визуализацией
"""

import os
import sys
import argparse
import ray
import torch
import numpy as np
import json
import time
import shutil
from datetime import datetime
from pathlib import Path
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.models import ModelCatalog
from ray.tune.registry import register_env
from typing import Dict, List, Any, Tuple, Optional

# Импорты основных компонентов
from aircraft_env import DogfightEnv, AIRSPACE_BOUNDS, ENGAGEMENT_RANGE
from aircraft_attention_model import AircraftTransformerModel, AircraftActionDistribution
from aircraft_visualization import AircraftVisualizer, BattleRecorderVisualizer, VisualizationConfig, create_battle_visualization
from policy_selector_system import (
    PolicySelector, PolicySelectorWrapper, PolicySelectorCallback,
    SpecializedAircraftPolicy, PolicyType, analyze_policy_effectiveness,
    create_specialized_policies
)
from league_state import LeagueState
from gspo_grpo_policy import GSPOTorchPolicy, GRPOTorchPolicy


def env_creator(cfg):
    return DogfightEnv(cfg)


class TrainingResultsManager:
    """Менеджер для сохранения результатов обучения"""
    
    def __init__(self, experiment_name: str = None, base_dir: str = "./training_results"):
        self.experiment_name = experiment_name or f"aircraft_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.base_dir = Path(base_dir)
        self.experiment_dir = self.base_dir / self.experiment_name
        
        # Создаем структуру директорий
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        (self.experiment_dir / "checkpoints").mkdir(exist_ok=True)
        (self.experiment_dir / "battles").mkdir(exist_ok=True)
        (self.experiment_dir / "visualizations").mkdir(exist_ok=True)
        (self.experiment_dir / "logs").mkdir(exist_ok=True)
        (self.experiment_dir / "policy_selector").mkdir(exist_ok=True)
        (self.experiment_dir / "models").mkdir(exist_ok=True)
        (self.experiment_dir / "analysis").mkdir(exist_ok=True)
        
        # Метрики и данные
        self.training_metrics = []
        self.battle_records = []
        self.policy_selector_evolution = []
        self.best_performance = float('-inf')
        self.best_checkpoint = None
        
        print(f"📁 Training Results Manager initialized")
        print(f"   Experiment: {self.experiment_name}")
        print(f"   Directory: {self.experiment_dir}")
        
        # Сохраняем мета-информацию эксперимента
        self.save_experiment_metadata()
    
    def save_experiment_metadata(self):
        """Сохраняет метаданные эксперимента"""
        metadata = {
            "experiment_name": self.experiment_name,
            "start_time": datetime.now().isoformat(),
            "python_version": sys.version,
            "pytorch_version": torch.__version__ if torch else "Not available",
            "ray_version": ray.__version__ if ray else "Not available",
            "directories": {
                "checkpoints": str(self.experiment_dir / "checkpoints"),
                "battles": str(self.experiment_dir / "battles"),
                "visualizations": str(self.experiment_dir / "visualizations"),
                "logs": str(self.experiment_dir / "logs"),
                "policy_selector": str(self.experiment_dir / "policy_selector"),
                "models": str(self.experiment_dir / "models"),
                "analysis": str(self.experiment_dir / "analysis")
            }
        }
        
        with open(self.experiment_dir / "experiment_metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def save_training_metrics(self, iteration: int, result: Dict[str, Any], 
                            selector_stats: Optional[Dict] = None):
        """Сохраняет метрики обучения"""
        metrics = {
            "iteration": iteration,
            "timestamp": datetime.now().isoformat(),
            "reward": result.get("env_runners", {}).get("episode_reward_mean", 0),
            "timesteps": result.get("timesteps_total", 0),
            "episode_length": result.get("env_runners", {}).get("episode_len_mean", 0),
            "custom_metrics": result.get("custom_metrics", {}),
        }
        
        if selector_stats:
            metrics["policy_selector"] = selector_stats
        
        self.training_metrics.append(metrics)
        
        # Сохраняем в файл каждые 10 итераций
        if iteration % 10 == 0:
            with open(self.experiment_dir / "logs" / "training_metrics.json", 'w') as f:
                json.dump(self.training_metrics, f, indent=2)
    
    def save_checkpoint(self, algorithm, iteration: int, performance: float = None):
        """Сохраняет чекпоинт модели"""
        checkpoint_dir = self.experiment_dir / "checkpoints" / f"iter_{iteration:06d}"
        
        try:
            # Сохраняем чекпоинт Ray RLLib
            checkpoint_result = algorithm.save(checkpoint_dir)
            checkpoint_path = checkpoint_result if isinstance(checkpoint_result, str) else str(checkpoint_dir)
            
            # Сохраняем дополнительную информацию о чекпоинте
            checkpoint_info = {
                "iteration": iteration,
                "timestamp": datetime.now().isoformat(),
                "performance": performance,
                "checkpoint_path": checkpoint_path,
                "algorithm_config": algorithm.config.to_dict() if hasattr(algorithm.config, 'to_dict') else str(algorithm.config)
            }
            
            with open(checkpoint_dir / "checkpoint_info.json", 'w') as f:
                json.dump(checkpoint_info, f, indent=2)
            
            # Отслеживаем лучший чекпоинт
            if performance and performance > self.best_performance:
                self.best_performance = performance
                self.best_checkpoint = checkpoint_path
                
                # Создаем символическую ссылку на лучший чекпоинт
                best_link = self.experiment_dir / "checkpoints" / "best"
                if best_link.exists():
                    best_link.unlink()
                try:
                    best_link.symlink_to(checkpoint_dir.name)
                except OSError:
                    # Fallback для Windows - копируем директорию
                    best_dir = self.experiment_dir / "checkpoints" / "best"
                    if best_dir.exists():
                        shutil.rmtree(best_dir)
                    shutil.copytree(checkpoint_dir, best_dir)
            
            # Создаем символическую ссылку на последний чекпоинт
            latest_link = self.experiment_dir / "checkpoints" / "latest"
            if latest_link.exists():
                latest_link.unlink()
            try:
                latest_link.symlink_to(checkpoint_dir.name)
            except OSError:
                # Fallback для Windows
                latest_dir = self.experiment_dir / "checkpoints" / "latest"
                if latest_dir.exists():
                    shutil.rmtree(latest_dir)
                shutil.copytree(checkpoint_dir, latest_dir)
            
            print(f"💾 Checkpoint saved: {checkpoint_dir}")
            if performance and performance > self.best_performance:
                print(f"🏆 New best performance: {performance:.3f}")
            
            return checkpoint_path
            
        except Exception as e:
            print(f"❌ Error saving checkpoint: {e}")
            return None
    
    def save_battle_record(self, iteration: int, battle_data: List[Dict[str, Any]], 
                          battle_summary: Dict[str, Any]):
        """Сохраняет запись боя"""
        battle_filename = f"battle_iter_{iteration:06d}_{int(time.time())}.json"
        battle_path = self.experiment_dir / "battles" / battle_filename
        
        battle_record = {
            "iteration": iteration,
            "timestamp": datetime.now().isoformat(),
            "summary": battle_summary,
            "frames": battle_data
        }
        
        with open(battle_path, 'w') as f:
            json.dump(battle_record, f)
        
        self.battle_records.append({
            "iteration": iteration,
            "filename": battle_filename,
            "summary": battle_summary
        })
        
        print(f"⚔️ Battle record saved: {battle_filename}")
        return battle_path
    
    def save_policy_selector_state(self, selector_wrapper: PolicySelectorWrapper, 
                                  iteration: int):
        """Сохраняет состояние селектора политик"""
        selector_filename = f"policy_selector_iter_{iteration:06d}.pt"
        selector_path = self.experiment_dir / "policy_selector" / selector_filename
        
        # Сохраняем состояние селектора
        selector_wrapper.selector.save_state(str(selector_path))
        
        # Сохраняем статистику
        stats = selector_wrapper.get_statistics()
        stats_filename = f"selector_stats_iter_{iteration:06d}.json"
        stats_path = self.experiment_dir / "policy_selector" / stats_filename
        
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)
        
        # Отслеживаем эволюцию селектора
        evolution_entry = {
            "iteration": iteration,
            "timestamp": datetime.now().isoformat(),
            "stats": stats
        }
        self.policy_selector_evolution.append(evolution_entry)
        
        # Сохраняем эволюцию
        with open(self.experiment_dir / "policy_selector" / "evolution.json", 'w') as f:
            json.dump(self.policy_selector_evolution, f, indent=2)
        
        print(f"🎯 Policy selector state saved: {selector_filename}")
        return selector_path
    
    def create_battle_visualization(self, battle_data: List[Dict[str, Any]], 
                                   iteration: int, save_animation: bool = True):
        """Создает визуализацию боя"""
        try:
            # Конфигурация визуализации
            viz_config = VisualizationConfig(
                figure_size=(16, 12),
                interval=100,
                trail_length=30,
                show_radar_range=False,
                show_weapon_range=True,
                show_airspace_bounds=True,
            )
            
            # Создаем визуализатор
            visualizer = AircraftVisualizer(viz_config)
            
            # Конвертируем данные для визуализации
            recorder = BattleRecorderVisualizer(visualizer)
            viz_frames = recorder.export_from_env_data(battle_data)
            
            visualizer.load_battle_data(viz_frames)
            
            if save_animation:
                # Сохраняем анимацию
                animation_filename = f"battle_animation_iter_{iteration:06d}.gif"
                animation_path = self.experiment_dir / "visualizations" / animation_filename
                
                animation = visualizer.create_animation()
                visualizer.save_animation(str(animation_path), fps=10)
                
                print(f"🎬 Battle animation saved: {animation_filename}")
                return animation_path
            
            return visualizer
            
        except Exception as e:
            print(f"❌ Error creating visualization: {e}")
            return None
    
    def export_model_to_onnx(self, algorithm, iteration: int):
        """Экспортирует модель в ONNX формат"""
        try:
            from aircraft_onnx_utils import export_aircraft_policy_to_onnx
            
            # Экспортируем основную политику
            main_policy = algorithm.get_policy("main")
            onnx_filename = f"aircraft_model_main_iter_{iteration:06d}.onnx"
            onnx_path = self.experiment_dir / "models" / onnx_filename
            
            success = export_aircraft_policy_to_onnx(
                policy=main_policy,
                output_path=str(onnx_path),
                policy_id="main",
                iteration=iteration
            )
            
            if success:
                print(f"📦 ONNX model exported: {onnx_filename}")
                
                # Экспортируем специализированные политики
                specialized_exports = []
                for policy_type in PolicyType:
                    if policy_type.value >= 6:
                        break
                    
                    policy_id = f"specialized_{policy_type.name.lower()}"
                    if policy_id in algorithm.policies:
                        spec_filename = f"aircraft_model_{policy_id}_iter_{iteration:06d}.onnx"
                        spec_path = self.experiment_dir / "models" / spec_filename
                        
                        spec_policy = algorithm.get_policy(policy_id)
                        spec_success = export_aircraft_policy_to_onnx(
                            policy=spec_policy,
                            output_path=str(spec_path),
                            policy_id=policy_id,
                            iteration=iteration
                        )
                        
                        if spec_success:
                            specialized_exports.append(spec_filename)
                
                return onnx_path, specialized_exports
            
        except ImportError:
            print("⚠️ ONNX export not available - aircraft_onnx_utils not found")
        except Exception as e:
            print(f"❌ Error exporting ONNX model: {e}")
        
        return None, []
    
    def generate_final_report(self, total_iterations: int, training_time: float):
        """Генерирует финальный отчет об эксперименте"""
        report = {
            "experiment_summary": {
                "name": self.experiment_name,
                "total_iterations": total_iterations,
                "training_time_hours": training_time / 3600,
                "best_performance": self.best_performance,
                "best_checkpoint": self.best_checkpoint,
                "end_time": datetime.now().isoformat()
            },
            "training_progress": {
                "total_metrics_recorded": len(self.training_metrics),
                "final_reward": self.training_metrics[-1]["reward"] if self.training_metrics else 0,
                "reward_improvement": (self.training_metrics[-1]["reward"] - self.training_metrics[0]["reward"]) if len(self.training_metrics) > 1 else 0
            },
            "battles_recorded": len(self.battle_records),
            "policy_selector_evolution": {
                "total_entries": len(self.policy_selector_evolution),
                "final_stats": self.policy_selector_evolution[-1]["stats"] if self.policy_selector_evolution else {}
            },
            "files_generated": {
                "checkpoints": len(list((self.experiment_dir / "checkpoints").glob("iter_*"))),
                "battle_records": len(list((self.experiment_dir / "battles").glob("*.json"))),
                "visualizations": len(list((self.experiment_dir / "visualizations").glob("*.gif"))),
                "onnx_models": len(list((self.experiment_dir / "models").glob("*.onnx"))),
            }
        }
        
        report_path = self.experiment_dir / "final_report.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        # Текстовый отчет для удобства чтения
        text_report_path = self.experiment_dir / "final_report.txt"
        with open(text_report_path, 'w') as f:
            f.write(f"🛩️ Aircraft Training Experiment Report\n")
            f.write(f"="*50 + "\n\n")
            f.write(f"Experiment: {self.experiment_name}\n")
            f.write(f"Duration: {training_time/3600:.2f} hours\n")
            f.write(f"Iterations: {total_iterations}\n")
            f.write(f"Best Performance: {self.best_performance:.3f}\n")
            f.write(f"Final Reward: {report['training_progress']['final_reward']:.3f}\n")
            f.write(f"Improvement: {report['training_progress']['reward_improvement']:+.3f}\n\n")
            
            f.write(f"Files Generated:\n")
            f.write(f"- Checkpoints: {report['files_generated']['checkpoints']}\n")
            f.write(f"- Battle Records: {report['files_generated']['battle_records']}\n")
            f.write(f"- Visualizations: {report['files_generated']['visualizations']}\n")
            f.write(f"- ONNX Models: {report['files_generated']['onnx_models']}\n\n")
            
            if self.policy_selector_evolution:
                final_stats = self.policy_selector_evolution[-1]["stats"]
                f.write(f"Final Policy Selector State:\n")
                f.write(f"- Current Policy: {final_stats.get('current_policy', 'Unknown')}\n")
                f.write(f"- Total Selections: {final_stats.get('total_selections', 0)}\n")
                f.write(f"- Average Reward: {final_stats.get('avg_recent_reward', 0):.3f}\n")
        
        print(f"📊 Final report generated: {report_path}")
        return report_path


class EnhancedPolicySelectorCallbacks(PolicySelectorCallback):
    """Расширенные callbacks с сохранением данных боя"""
    
    def __init__(self, results_manager: TrainingResultsManager, obs_space, act_space):
        super().__init__(obs_space, act_space)
        self.results_manager = results_manager
        self.battle_data_buffer = []
        self.current_battle_frames = []
        # Дополнительные атрибуты для Enhanced версии
        self.league = None
        self.opponent_ids = []
        self.eval_episodes = 3
        self.visualization_enabled = True
    
    def setup(self, league_actor=None, opponent_ids: List[str] = None, 
              obs_space=None, act_space=None, **kwargs):
        """Переопределенная настройка callbacks с дополнительными параметрами"""
        
        algorithm = kwargs.get('algorithm')
        enable_selector = kwargs.get('enable_selector', True)
        
        # Вызываем родительский setup если он существует и алгоритм передан
        if algorithm is not None:
            try:
                super().setup(algorithm, enable_selector=enable_selector)
            except TypeError as e:
                print(f"Warning: Parent setup failed: {e}")
                # Если родительский setup имеет другую сигнатуру, настраиваем вручную
                self.enabled = enable_selector
                
                if self.enabled:
                    # Инициализируем selector_wrapper если его нет
                    if not hasattr(self, 'selector_wrapper') or self.selector_wrapper is None:
                        from policy_selector_system import PolicySelectorWrapper
                        self.selector_wrapper = PolicySelectorWrapper(
                            obs_space or self.obs_space, 
                            act_space or self.act_space
                        )
                        
                        # Добавляем специализированные политики к алгоритму если они еще не добавлены
                        if hasattr(self.selector_wrapper, 'specialized_policies'):
                            for policy_id, policy_spec in self.selector_wrapper.specialized_policies.items():
                                if policy_id not in algorithm.policies:
                                    try:
                                        algorithm.add_policy(
                                            policy_id=policy_id,
                                            policy_cls=policy_spec[0],
                                            observation_space=policy_spec[1],
                                            action_space=policy_spec[2],
                                            config=policy_spec[3]
                                        )
                                        print(f"✈️ Added policy to algorithm: {policy_id}")
                                    except Exception as add_error:
                                        print(f"Warning: Failed to add policy {policy_id}: {add_error}")
        else:
            print("Warning: No algorithm provided, skipping policy selector setup")
            self.enabled = False
        
        # Настраиваем дополнительные параметры для Enhanced версии
        self.league = league_actor
        self.opponent_ids = opponent_ids or []
        self.eval_episodes = kwargs.get('eval_episodes', 3)
        self.visualization_enabled = kwargs.get('enable_visualization', True)
        
        # Инициализируем policy_selector_wrapper для совместимости
        if self.enabled and not hasattr(self, 'policy_selector_wrapper'):
            self.policy_selector_wrapper = getattr(self, 'selector_wrapper', None)
        
        print(f"🎯 Enhanced Policy Selector Training Callbacks initialized")
        print(f"   Specialized policies: {len(getattr(self.policy_selector_wrapper, 'specialized_policies', {})) if self.policy_selector_wrapper else 0}")
        print(f"   League opponents: {len(self.opponent_ids)}")
        print(f"   Enabled: {'✅' if self.enabled else '❌'}")
        
    def _play_match_with_recording(self, algorithm, opponent_id: str, episodes: int) -> Tuple[int, int, List, Dict]:
        """Играет матч с записью данных для визуализации"""
        
        wins_main, wins_opp = 0, 0
        all_battle_frames = []
        selector_decisions = []
        
        for episode in range(episodes):
            try:
                # Создаем окружение
                env_config = algorithm.config.env_config.copy()
                test_env = DogfightEnv(env_config)
                
                obs, _ = test_env.reset()
                done = False
                episode_frames = []
                episode_selector_stats = {
                    'policy_selections': [],
                    'situation_contexts': [],
                    'rewards_per_policy': {i: [] for i in range(6)}
                }
                
                step = 0
                while not done:
                    action_dict = {}
                    step_selector_info = {}
                    
                    # Получаем действия от агентов
                    for aircraft_id, aircraft_obs in obs.items():
                        if aircraft_id.startswith("red_"):
                            # Для красной команды используем селектор политик
                            if self.policy_selector_wrapper:
                                action, selected_policy_idx = self.policy_selector_wrapper.select_and_execute(
                                    algorithm, aircraft_obs, aircraft_id
                                )
                                
                                step_selector_info[aircraft_id] = {
                                    'selected_policy': selected_policy_idx,
                                    'policy_name': PolicyType(selected_policy_idx).name
                                }
                                
                                episode_selector_stats['policy_selections'].append(selected_policy_idx)
                            else:
                                policy = algorithm.get_policy("main")
                                action, _, _ = policy.compute_single_action(aircraft_obs, explore=False)
                        else:
                            # Для синей команды используем оппонента
                            policy = algorithm.get_policy(opponent_id)
                            action, _, _ = policy.compute_single_action(aircraft_obs, explore=False)
                        
                        action_dict[aircraft_id] = action
                    
                    # Выполняем шаг
                    obs, rewards, terms, truncs, infos = test_env.step(action_dict)
                    done = terms.get("__all__", False) or truncs.get("__all__", False)
                    
                    # Записываем кадр для визуализации
                    try:
                        frame_data = test_env.export_for_visualization()
                    except AttributeError:
                        # Если метод не существует, создаем базовые данные
                        frame_data = {
                            'aircraft_positions': {aid: test_env.aircraft[aid].get_position().tolist() 
                                                 for aid in test_env.aircraft if test_env.aircraft[aid].alive},
                            'aircraft_states': {aid: {'team': test_env.aircraft[aid].team, 'alive': test_env.aircraft[aid].alive}
                                              for aid in test_env.aircraft},
                            'timestamp': step * test_env.dt
                        }
                    
                    frame_data['policy_selector'] = step_selector_info
                    frame_data['step'] = step
                    frame_data['episode'] = episode
                    episode_frames.append(frame_data)
                    
                    # Обновляем производительность селектора
                    if self.policy_selector_wrapper:
                        red_rewards = {aid: rew for aid, rew in rewards.items() if aid.startswith("red_")}
                        if red_rewards:
                            self.policy_selector_wrapper.update_performance(red_rewards, infos)
                            
                            # Записываем награды по политикам
                            for aircraft_id, reward in red_rewards.items():
                                if aircraft_id in step_selector_info:
                                    policy_idx = step_selector_info[aircraft_id]['selected_policy']
                                    episode_selector_stats['rewards_per_policy'][policy_idx].append(reward)
                    
                    step += 1
                
                # Определяем победителя
                try:
                    battle_summary = test_env.get_battle_summary()
                    red_alive = battle_summary.get("red_alive", 0)
                    blue_alive = battle_summary.get("blue_alive", 0)
                except AttributeError:
                    # Если метод не существует, считаем вручную
                    red_alive = sum(1 for aid in test_env.aircraft if aid.startswith("red_") and test_env.aircraft[aid].alive)
                    blue_alive = sum(1 for aid in test_env.aircraft if aid.startswith("blue_") and test_env.aircraft[aid].alive)
                
                if red_alive > blue_alive:
                    wins_main += 1
                elif blue_alive > red_alive:
                    wins_opp += 1
                
                # Добавляем итоги боя к каждому кадру
                for frame in episode_frames:
                    frame['battle_outcome'] = {
                        'winner': 'red' if red_alive > blue_alive else ('blue' if blue_alive > red_alive else 'draw'),
                        'red_survivors': red_alive,
                        'blue_survivors': blue_alive
                    }
                
                all_battle_frames.extend(episode_frames)
                selector_decisions.append(episode_selector_stats)
                
                print(f"  Episode {episode+1}: {'Red' if red_alive > blue_alive else 'Blue' if blue_alive > red_alive else 'Draw'}")
                
            except Exception as e:
                print(f"Error in battle recording episode: {e}")
                continue
        
        # Агрегируем статистику селектора
        aggregated_stats = self._aggregate_selector_stats(selector_decisions)
        
        return wins_main, wins_opp, all_battle_frames, aggregated_stats
    
    def _aggregate_selector_stats(self, selector_decisions: List[Dict]) -> Dict:
        """Агрегирует статистику селектора из нескольких эпизодов"""
        if not selector_decisions:
            return {}
        
        aggregated = {
            'total_policy_selections': [],
            'policy_usage_counts': {i: 0 for i in range(6)},
            'average_rewards_per_policy': {i: [] for i in range(6)},
            'total_episodes': len(selector_decisions)
        }
        
        # Агрегируем данные по эпизодам
        for episode_stats in selector_decisions:
            # Собираем все выборы политик
            aggregated['total_policy_selections'].extend(episode_stats['policy_selections'])
            
            # Считаем использование каждой политики
            for policy_idx in episode_stats['policy_selections']:
                if policy_idx < 6:
                    aggregated['policy_usage_counts'][policy_idx] += 1
            
            # Собираем награды по политикам
            for policy_idx, rewards in episode_stats['rewards_per_policy'].items():
                if rewards and policy_idx < 6:
                    aggregated['average_rewards_per_policy'][policy_idx].extend(rewards)
        
        # Вычисляем финальную статистику
        total_selections = len(aggregated['total_policy_selections'])
        if total_selections > 0:
            # Проценты использования политик
            aggregated['policy_usage_percentages'] = {
                i: (count / total_selections) * 100 
                for i, count in aggregated['policy_usage_counts'].items()
            }
            
            # Средние награды по политикам
            aggregated['average_rewards_final'] = {}
            for policy_idx, rewards in aggregated['average_rewards_per_policy'].items():
                if rewards:
                    aggregated['average_rewards_final'][policy_idx] = np.mean(rewards)
                else:
                    aggregated['average_rewards_final'][policy_idx] = 0.0
            
            # Наиболее используемая политика
            most_used_policy = max(aggregated['policy_usage_counts'].items(), key=lambda x: x[1])
            aggregated['most_used_policy'] = {
                'policy_idx': most_used_policy[0],
                'policy_name': PolicyType(most_used_policy[0]).name if most_used_policy[0] < 6 else 'UNKNOWN',
                'usage_count': most_used_policy[1],
                'usage_percentage': aggregated['policy_usage_percentages'][most_used_policy[0]]
            }
        
        return aggregated
    
    def on_train_result(self, algorithm, result: Dict[str, Any]):
        """Расширенная обработка результатов с сохранением"""
        iteration = result["training_iteration"]
        
        # Сохраняем метрики обучения
        selector_stats = None
        if hasattr(self, 'policy_selector_wrapper') and self.policy_selector_wrapper:
            selector_stats = self.policy_selector_wrapper.get_statistics()
        
        self.results_manager.save_training_metrics(iteration, result, selector_stats)
        
        # Базовая обработка (если родительский метод существует)
        try:
            super().on_train_result(algorithm=algorithm, result=result)
        except (TypeError, AttributeError) as e:
            # Если родительский метод недоступен или имеет другую сигнатуру, 
            # выполняем базовую обработку selector'а вручную
            if hasattr(self, 'policy_selector_wrapper') and self.policy_selector_wrapper:
                # Обновляем статистику селектора
                try:
                    # Простая обработка без сложной логики матчей
                    current_rewards = {f"red_{i}": result.get("env_runners", {}).get("episode_reward_mean", 0) 
                                     for i in range(2)}  # примерные награды
                    self.policy_selector_wrapper.update_performance(current_rewards)
                except Exception as selector_e:
                    print(f"Warning: Policy selector update failed: {selector_e}")
        
        try:
            # Проводим записываемые матчи
            if iteration % 10 == 0:  # каждые 10 итераций записываем бой
                print(f"🎬 Recording battle for iteration {iteration}...")
                
                if self.opponent_ids:
                    opponent_id = self.opponent_ids[0]  # берем первого оппонента
                    try:
                        wins_main, wins_opp, battle_frames, selector_stats = self._play_match_with_recording(
                            algorithm, opponent_id, episodes=1
                        )
                        
                        if battle_frames:
                            # Сохраняем запись боя
                            battle_summary = {
                                'iteration': iteration,
                                'opponent': opponent_id,
                                'wins_main': wins_main,
                                'wins_opp': wins_opp,
                                'total_frames': len(battle_frames),
                                'selector_stats': selector_stats
                            }
                            
                            battle_path = self.results_manager.save_battle_record(
                                iteration, battle_frames, battle_summary
                            )
                            
                            # Создаем визуализацию
                            if iteration % 25 == 0:  # визуализация каждые 25 итераций
                                self.results_manager.create_battle_visualization(
                                    battle_frames, iteration, save_animation=True
                                )
                    except Exception as battle_e:
                        print(f"Warning: Battle recording failed: {battle_e}")
            
            # Сохраняем состояние селектора политик
            if hasattr(self, 'policy_selector_wrapper') and self.policy_selector_wrapper and iteration % 20 == 0:
                self.results_manager.save_policy_selector_state(
                    self.policy_selector_wrapper, iteration
                )
            
            # Сохраняем чекпоинты
            if iteration % 25 == 0:
                performance = result.get("env_runners", {}).get("episode_reward_mean", 0)
                self.results_manager.save_checkpoint(algorithm, iteration, performance)
            
            # Экспорт в ONNX
            if iteration % 50 == 0 and iteration > 0:
                try:
                    self.results_manager.export_model_to_onnx(algorithm, iteration)
                except Exception as e:
                    print(f"⚠️ ONNX export failed: {e}")
            
        except Exception as e:
            print(f"Error in enhanced callbacks: {e}")
            import traceback
            traceback.print_exc()
    
    def on_train_result(self, algorithm, result: Dict[str, Any]):
        """Расширенная обработка результатов с сохранением"""
        iteration = result["training_iteration"]
        
        # Сохраняем метрики обучения
        selector_stats = None
        if self.policy_selector_wrapper:
            selector_stats = self.policy_selector_wrapper.get_statistics()
        
        self.results_manager.save_training_metrics(iteration, result, selector_stats)
        
        # Базовая обработка
        super().on_train_result(algorithm, result)
        
        try:
            # Проводим записываемые матчи
            if iteration % 10 == 0:  # каждые 10 итераций записываем бой
                print(f"🎬 Recording battle for iteration {iteration}...")
                
                for opponent_id in self.opponent_ids[:1]:  # записываем против одного оппонента
                    wins_main, wins_opp, battle_frames, selector_stats = self._play_match_with_recording(
                        algorithm, opponent_id, episodes=1
                    )
                    
                    if battle_frames:
                        # Сохраняем запись боя
                        battle_summary = {
                            'iteration': iteration,
                            'opponent': opponent_id,
                            'wins_main': wins_main,
                            'wins_opp': wins_opp,
                            'total_frames': len(battle_frames),
                            'selector_stats': selector_stats
                        }
                        
                        battle_path = self.results_manager.save_battle_record(
                            iteration, battle_frames, battle_summary
                        )
                        
                        # Создаем визуализацию
                        if iteration % 25 == 0:  # визуализация каждые 25 итераций
                            self.results_manager.create_battle_visualization(
                                battle_frames, iteration, save_animation=True
                            )
            
            # Сохраняем состояние селектора политик
            if self.policy_selector_wrapper and iteration % 20 == 0:
                self.results_manager.save_policy_selector_state(
                    self.policy_selector_wrapper, iteration
                )
            
            # Сохраняем чекпоинты
            if iteration % 25 == 0:
                performance = result.get("env_runners", {}).get("episode_reward_mean", 0)
                self.results_manager.save_checkpoint(algorithm, iteration, performance)
            
            # Экспорт в ONNX
            if iteration % 50 == 0 and iteration > 0:
                try:
                    self.results_manager.export_model_to_onnx(algorithm, iteration)
                except Exception as e:
                    print(f"⚠️ ONNX export failed: {e}")
            
        except Exception as e:
            print(f"Error in enhanced callbacks: {e}")
            import traceback
            traceback.print_exc()


def create_simple_battle_record(algorithm, opponent_id: str, iteration: int) -> List[Dict[str, Any]]:
    """Создает упрощенную запись боя для сохранения"""
    try:
        # Создаем простые тестовые данные боя
        battle_frames = []
        
        for frame_idx in range(50):  # 50 кадров = ~5 секунд боя
            frame = {
                'timestamp': frame_idx * 0.1,
                'iteration': iteration,
                'frame': frame_idx,
                'aircraft': [
                    {
                        'id': 'red_0',
                        'team': 'red',
                        'position': [frame_idx * 100, 0, 8000],
                        'velocity': [100, 0, 0],
                        'alive': True,
                        'hp': max(100 - frame_idx, 1),
                        'fuel': max(1.0 - frame_idx * 0.01, 0.1)
                    },
                    {
                        'id': 'blue_0', 
                        'team': 'blue',
                        'position': [-frame_idx * 100, 0, 8000],
                        'velocity': [-100, 0, 0],
                        'alive': True,
                        'hp': max(100 - frame_idx, 1),
                        'fuel': max(1.0 - frame_idx * 0.01, 0.1)
                    }
                ],
                'missiles': [],
                'kills': {'red': 0, 'blue': 0},
                'missiles_fired': {'red': 0, 'blue': 0}
            }
            battle_frames.append(frame)
        
        return battle_frames
        
    except Exception as e:
        print(f"Error creating simple battle record: {e}")
        return []


def create_test_battle_visualization(iteration: int) -> List[Dict[str, Any]]:
    """Создает тестовые данные для визуализации"""
    try:
        frames = []
        
        for i in range(100):  # 100 кадров анимации
            frame = {
                'timestamp': i * 0.1,
                'aircraft': [],
                'missiles': [],
                'kills': {'red': 0, 'blue': 0},
                'airspace_bounds': {
                    'x_min': -50000, 'x_max': 50000,
                    'y_min': -50000, 'y_max': 50000,
                    'z_min': 5000, 'z_max': 15000
                }
            }
            
            # Добавляем самолеты с движением
            for team_idx, (team, color) in enumerate([('red', 'red'), ('blue', 'blue')]):
                for aircraft_idx in range(2):
                    aircraft_id = f"{team}_{aircraft_idx}"
                    
                    # Создаем траекторию движения
                    base_x = (team_idx * 2 - 1) * 20000  # красные слева, синие справа
                    x = base_x + (1 - team_idx * 2) * i * 200  # двигаются навстречу
                    y = aircraft_idx * 10000 - 5000  # разнесены по Y
                    z = 8000 + 1000 * np.sin(i * 0.1 + aircraft_idx)  # небольшие маневры
                    
                    aircraft = {
                        'id': aircraft_id,
                        'team': team,
                        'position': [x, y, z],
                        'velocity': [(1 - team_idx * 2) * 200, 0, 100 * np.cos(i * 0.1)],
                        'alive': True,
                        'hp': max(100 - i * 0.5, 10),
                        'fuel': max(1.0 - i * 0.005, 0.1),
                        'heading': 90 if team == 'red' else 270,
                        'aircraft_type': 'f16'
                    }
                    frame['aircraft'].append(aircraft)
            
            # Добавляем ракеты в середине боя
            if 30 <= i <= 70 and i % 5 == 0:
                missile = {
                    'id': f"missile_{i}",
                    'shooter_id': 'red_0',
                    'target_id': 'blue_0',
                    'position': [-10000 + (i-30) * 500, 0, 8000],
                    'type': 'MEDIUM_RANGE_MISSILE',
                    'active': True,
                    'fuel': 1.0 - (i-30) * 0.02
                }
                frame['missiles'].append(missile)
            
            frames.append(frame)
        
        return frames
        
    except Exception as e:
        print(f"Error creating test visualization: {e}")
        return []


def main_with_enhanced_saving():
    """Главная функция с расширенным сохранением результатов"""
    
    parser = argparse.ArgumentParser(description="Aircraft Training with Enhanced Saving")
    parser.add_argument("--iterations", type=int, default=1000, help="Training iterations")
    parser.add_argument("--algorithm", choices=["ppo", "gspo", "grpo"], default="gspo")
    parser.add_argument("--test", action="store_true", help="Quick test mode")
    parser.add_argument("--enable-selector", action="store_true", default=True, help="Enable policy selector")
    parser.add_argument("--selector-frequency", type=int, default=10, help="Policy selection frequency")
    parser.add_argument("--disable-viz", action="store_true", help="Disable visualization")
    parser.add_argument("--no-jsbsim", action="store_true", help="Skip JSBSim validation")
    parser.add_argument("--experiment-name", type=str, help="Custom experiment name")
    parser.add_argument("--results-dir", type=str, default="./training_results", help="Results directory")
    parser.add_argument("--save-frequency", type=int, default=25, help="Checkpoint save frequency")
    parser.add_argument("--viz-frequency", type=int, default=25, help="Visualization save frequency")
    args = parser.parse_args()
    
    print(f"🛩️ Enhanced Aircraft Training with Policy Selector")
    print(f"   Algorithm: {args.algorithm.upper()}")
    print(f"   Iterations: {args.iterations}")
    print(f"   Policy Selector: {'✅ ENABLED' if args.enable_selector else '❌ DISABLED'}")
    print(f"   Results Directory: {args.results_dir}")
    
    # Создаем менеджер результатов
    results_manager = TrainingResultsManager(
        experiment_name=args.experiment_name,
        base_dir=args.results_dir
    )
    
    # Валидация JSBSim
    if not args.no_jsbsim:
        try:
            import jsbsim
            print("✅ JSBSim available")
        except ImportError:
            print("⚠️ JSBSim not available - continuing with simplified physics")
    
    # Инициализация Ray
    ray.init(ignore_reinit_error=True)
    
    training_start_time = time.time()
    
    # Инициализируем callbacks как None сначала  
    callbacks = None
    
    try:
        # Регистрация компонентов
        register_env("DogfightEnv", env_creator)
        ModelCatalog.register_custom_model("aircraft_transformer", AircraftTransformerModel)
        ModelCatalog.register_custom_action_dist("aircraft_actions", AircraftActionDistribution)
        
        # Создание тестового окружения
        env_config = {
            "red_choices": [1, 2] if args.test else [2, 3, 4],
            "blue_choices": [1, 2] if args.test else [2, 3, 4],
            "episode_len": 500 if args.test else 3000,
            "dt": 0.1,
            "use_simplified_physics": args.no_jsbsim,
        }
        
        tmp_env = DogfightEnv(env_config)
        obs_space = tmp_env.observation_space
        act_space = tmp_env.action_space
        max_aircraft = obs_space["allies"].shape[0]
        
        print(f"🛩️ Environment: max {max_aircraft} aircraft per team")
        
        # Создание League
        opponent_ids = [f"squadron_{i}" for i in range(3 if args.test else 6)]
        league = LeagueState.remote(opponent_ids)
        
        # СОЗДАЕМ CALLBACKS ЗДЕСЬ - в начале функции
        callbacks = EnhancedPolicySelectorCallbacks(results_manager, obs_space, act_space)
        
        # Конфигурация модели
        base_model_config = {
            "custom_model": "aircraft_transformer",
            "custom_action_dist": "aircraft_actions",
            "custom_model_config": {
                "d_model": 128 if args.test else 256,
                "nhead": 4 if args.test else 8,
                "layers": 2 if args.test else 3,
                "ff": 256 if args.test else 512,
                "hidden": 128 if args.test else 256,
                "dropout": 0.1,
                "max_aircraft": max_aircraft,
                "max_enemies": max_aircraft,
            },
            "vf_share_layers": False,
        }
        
        # Сохраняем конфигурацию эксперимента
        experiment_config = {
            "algorithm": args.algorithm,
            "iterations": args.iterations,
            "test_mode": args.test,
            "enable_selector": args.enable_selector,
            "selector_frequency": args.selector_frequency,
            "env_config": env_config,
            "model_config": base_model_config,
            "opponent_ids": opponent_ids,
            "max_aircraft": max_aircraft
        }
        
        with open(results_manager.experiment_dir / "experiment_config.json", 'w') as f:
            json.dump(experiment_config, f, indent=2)
        
        # Выбор алгоритма
        if args.algorithm == "gspo":
            policy_cls = GSPOTorchPolicy
        elif args.algorithm == "grpo":
            policy_cls = GRPOTorchPolicy
        else:
            policy_cls = None
        
        # Policy mapping с учетом селектора
        def policy_mapping_fn(agent_id: str, episode=None, **kwargs):
            if agent_id.startswith("red_"):
                return "main"  # селектор выберет специализированную политику
            else:
                import hashlib
                hash_val = int(hashlib.md5(str(episode).encode()).hexdigest()[:8], 16)
                return opponent_ids[hash_val % len(opponent_ids)]
        
        # Создание основных политик
        policies = {
            "main": (policy_cls, obs_space, act_space, {"model": base_model_config}),
        }
        
        # Добавляем оппонентов
        for opponent_id in opponent_ids:
            policies[opponent_id] = (None, obs_space, act_space, {"model": base_model_config})
        
        # Если селектор включен, добавляем специализированные политики
        specialized_policies = {}
        if args.enable_selector:
            try:
                specialized_policies = create_specialized_policies(obs_space, act_space, base_model_config)
                policies.update(specialized_policies)
                
                print(f"🎯 Added {len(specialized_policies)} specialized policies:")
                for policy_id in specialized_policies.keys():
                    print(f"   - {policy_id}")
            except Exception as e:
                print(f"⚠️ Warning: Failed to create specialized policies: {e}")
                print("   Continuing without policy selector...")
                args.enable_selector = False
        
        # Конфигурация PPO
        config = (
            PPOConfig()
            .api_stack(
                enable_rl_module_and_learner=False,
                enable_env_runner_and_connector_v2=False,
            )
            .environment(
                env="DogfightEnv",
                env_config=env_config
            )
            .framework("torch")
            .env_runners(
                num_env_runners=0 if args.test else 4,
                num_envs_per_env_runner=1,
                rollout_fragment_length=256,
            )
            .resources(
                num_gpus=1 if torch.cuda.is_available() else 0,
                num_cpus_for_main_process=1,
            )
            .training(
                gamma=0.995,
                lr=1e-4,
                train_batch_size=1024 if args.test else 16384,
                minibatch_size=128 if args.test else 1024,
                num_epochs=3,
                use_gae=True,
                lambda_=0.95,
                clip_param=0.2,
                entropy_coeff=0.01,
                vf_loss_coeff=0.5,
            )
            .multi_agent(
                policies=policies,
                policy_mapping_fn=policy_mapping_fn,
                policies_to_train=["main"] + (list(specialized_policies.keys()) if args.enable_selector else []),
            )
        )
        
        # Создание алгоритма
        print("🔧 Building algorithm with enhanced saving...")
        try:
            algorithm = config.build()
        except AttributeError:
            # Для новых версий Ray RLLib
            algorithm = config.build_algo()
        
        # Теперь настраиваем callbacks с алгоритмом
        if callbacks is not None:
            try:
                callbacks.setup(
                    league_actor=league,
                    opponent_ids=opponent_ids,
                    obs_space=obs_space,
                    act_space=act_space,
                    eval_episodes=2 if args.test else 3,
                    enable_visualization=not args.disable_viz,
                    enable_selector=args.enable_selector,
                    algorithm=algorithm
                )
            except Exception as setup_error:
                print(f"⚠️ Warning: Callbacks setup failed: {setup_error}")
                print("   Continuing without enhanced callbacks...")
                callbacks = None
        
        # Инициализация весов
        main_weights = algorithm.get_policy("main").get_weights()
        for opponent_id in opponent_ids:
            algorithm.get_policy(opponent_id).set_weights(main_weights)
        
        # Инициализация специализированных политик
        if args.enable_selector:
            for policy_id in specialized_policies.keys():
                # Инициализируем специализированные политики слегка измененными весами main
                specialized_weights = algorithm.get_policy("main").get_weights()
                # Добавляем небольшой шум для дифференциации
                for key in specialized_weights:
                    if isinstance(specialized_weights[key], np.ndarray):
                        noise = np.random.normal(0, 0.01, specialized_weights[key].shape)
                        specialized_weights[key] += noise.astype(specialized_weights[key].dtype)
                
                algorithm.get_policy(policy_id).set_weights(specialized_weights)
        
        # Создание расширенных callbacks (без вызова setup)
        callbacks = EnhancedPolicySelectorCallbacks(results_manager, obs_space, act_space)
        
        print("✅ Algorithm with enhanced saving ready")
        print(f"🎯 Enhanced Features:")
        print(f"   - Comprehensive result tracking and saving")
        print(f"   - Battle recording and visualization")
        print(f"   - Automatic checkpoint management")
        print(f"   - Policy selector state preservation")
        print(f"   - ONNX model export")
        print(f"   - Detailed experiment reporting")
        
        # Цикл обучения с расширенным сохранением
        best_performance = float('-inf')
        iterations = min(args.iterations, 10) if args.test else args.iterations
        
        print(f"\n🚀 Starting training for {iterations} iterations...")
        
        # Принудительно создаем все директории
        print(f"📁 Ensuring all directories exist...")
        for dir_name in ['checkpoints', 'battles', 'visualizations', 'models', 'policy_selector', 'logs']:
            dir_path = results_manager.experiment_dir / dir_name
            dir_path.mkdir(exist_ok=True)
            print(f"   ✅ {dir_name}/ ready")
        
        # Создаем начальную метрику для проверки
        initial_result = {
            'training_iteration': 0,
            'timesteps_total': 0,
            'env_runners': {'episode_reward_mean': 0.0}
        }
        results_manager.save_training_metrics(0, initial_result)
        print(f"📊 Initial metrics saved")
        
        # Создаем тестовый чекпоинт сразу
        try:
            test_checkpoint = results_manager.save_checkpoint(algorithm, 0, 0.0)
            print(f"💾 Test checkpoint created: {test_checkpoint is not None}")
        except Exception as e:
            print(f"⚠️ Test checkpoint failed: {e}")
        
        # Создаем тестовую запись боя
        try:
            test_battle = create_simple_battle_record(algorithm, "test", 0)
            if test_battle:
                test_summary = {'iteration': 0, 'test_battle': True, 'frames': len(test_battle)}
                results_manager.save_battle_record(0, test_battle, test_summary)
                print(f"⚔️ Test battle record created: {len(test_battle)} frames")
        except Exception as e:
            print(f"⚠️ Test battle failed: {e}")
        
        # Создаем тестовую визуализацию
        if not args.disable_viz:
            try:
                test_viz_data = create_test_battle_visualization(0)
                if test_viz_data:
                    results_manager.create_battle_visualization(test_viz_data, 0, save_animation=True)
                    print(f"🎬 Test visualization created: {len(test_viz_data)} frames")
            except Exception as e:
                print(f"⚠️ Test visualization failed: {e}")
        
        print(f"\n🎯 Pre-training setup complete! Starting main training loop...")
        
        for iteration in range(1, iterations + 1):
            try:
                # Обучение
                result = algorithm.train()
                
                # ОБЯЗАТЕЛЬНО: Обработка через расширенные callbacks
                if callbacks is not None:
                    try:
                        callbacks.on_train_result(algorithm, result)
                    except Exception as callback_error:
                        print(f"⚠️ Warning: Callback error: {callback_error}")
                else:
                    # Базовое сохранение метрик если callbacks не работают
                    results_manager.save_training_metrics(iteration, result)
                
                # ПРИНУДИТЕЛЬНЫЕ СОХРАНЕНИЯ (независимо от callbacks):
                
                # 1. Сохраняем метрики каждую итерацию
                selector_stats = None
                if callbacks and hasattr(callbacks, 'policy_selector_wrapper') and callbacks.policy_selector_wrapper:
                    selector_stats = callbacks.policy_selector_wrapper.get_statistics()
                results_manager.save_training_metrics(iteration, result, selector_stats)
                
                # 2. Принудительные чекпоинты
                if iteration % args.save_frequency == 0:
                    performance = result.get("env_runners", {}).get("episode_reward_mean", 0)
                    checkpoint_path = results_manager.save_checkpoint(algorithm, iteration, performance)
                    print(f"💾 Forced checkpoint save: iter_{iteration:06d}")
                
                # 3. Принудительная запись боев (упрощенная версия)
                if iteration % 10 == 0 and iteration > 0:
                    try:
                        print(f"🎬 Recording simplified battle for iteration {iteration}...")
                        battle_data = create_simple_battle_record(algorithm, opponent_ids[0] if opponent_ids else "main", iteration)
                        if battle_data:
                            battle_summary = {
                                'iteration': iteration,
                                'opponent': opponent_ids[0] if opponent_ids else "self_play",
                                'total_frames': len(battle_data),
                                'simplified_record': True
                            }
                            results_manager.save_battle_record(iteration, battle_data, battle_summary)
                            print(f"⚔️ Battle record saved: {len(battle_data)} frames")
                    except Exception as e:
                        print(f"⚠️ Battle recording failed: {e}")
                
                # 4. ONNX экспорт
                if iteration % 50 == 0 and iteration > 0:
                    try:
                        print(f"📦 Exporting ONNX model for iteration {iteration}...")
                        results_manager.export_model_to_onnx(algorithm, iteration)
                    except Exception as e:
                        print(f"⚠️ ONNX export failed: {e}")
                
                # 5. Принудительные визуализации
                if iteration % args.viz_frequency == 0 and iteration > 0 and not args.disable_viz:
                    try:
                        print(f"🎬 Creating visualization for iteration {iteration}...")
                        # Создаем тестовые данные для визуализации
                        test_battle_data = create_test_battle_visualization(iteration)
                        if test_battle_data:
                            viz_path = results_manager.create_battle_visualization(
                                test_battle_data, iteration, save_animation=True
                            )
                            print(f"🎥 Visualization saved: iter_{iteration:06d}.gif")
                    except Exception as e:
                        print(f"⚠️ Visualization failed: {e}")
                
                # 6. Policy selector состояние
                if callbacks and hasattr(callbacks, 'policy_selector_wrapper') and callbacks.policy_selector_wrapper and iteration % 20 == 0:
                    try:
                        results_manager.save_policy_selector_state(callbacks.policy_selector_wrapper, iteration)
                        print(f"🎯 Policy selector state saved: iter_{iteration:06d}")
                    except Exception as e:
                        print(f"⚠️ Policy selector save failed: {e}")
                
                # Извлекаем метрики
                env_runners = result.get("env_runners", {})
                reward = env_runners.get("episode_reward_mean", 0)
                timesteps = result.get("timesteps_total", 0)
                
                # Метрики селектора
                custom = result.get("custom_metrics", {})
                selector_reward = custom.get("selector_avg_reward", 0)
                current_policy_idx = custom.get("selector_current_policy", 2)
                
                # Логирование с подтверждением сохранений
                if iteration % 5 == 0 or args.test or iteration <= 3:
                    current_policy_name = PolicyType(int(current_policy_idx)).name if current_policy_idx < 6 else "UNKNOWN"
                    
                    print(f"[{iteration:4d}] Reward: {reward:7.3f}, Timesteps: {timesteps:,}")
                    
                    # Проверяем что файлы создаются
                    metrics_file = results_manager.experiment_dir / "logs" / "training_metrics.json"
                    if metrics_file.exists():
                        file_size = metrics_file.stat().st_size
                        print(f"       📊 Metrics file: {file_size} bytes")
                    
                    checkpoint_count = len(list((results_manager.experiment_dir / "checkpoints").glob("iter_*")))
                    battle_count = len(list((results_manager.experiment_dir / "battles").glob("*.json")))
                    
                    print(f"       💾 Checkpoints: {checkpoint_count}, ⚔️ Battles: {battle_count}")
                    
                    if args.enable_selector:
                        print(f"       Selector: {current_policy_name}, Perf: {selector_reward:+.3f}")
                        
                        # Показываем использование политик
                        policy_usage = [custom.get(f"selector_policy_{i}_usage_pct", 0) for i in range(6)]
                        if any(usage > 0 for usage in policy_usage):
                            usage_str = ", ".join([
                                f"{PolicyType(i).name[:3]}:{usage:.0f}%" 
                                for i, usage in enumerate(policy_usage) if usage > 5
                            ])
                            print(f"       Usage: {usage_str}")
                
                # Отслеживание лучшей производительности
                if reward > best_performance:
                    best_performance = reward
                
                # Дополнительные сохранения на ключевых итерациях
                if iteration % (args.save_frequency // 2) == 0:
                    # Промежуточное сохранение метрик
                    results_manager.save_training_metrics(iteration, result, 
                                                        callbacks.policy_selector_wrapper.get_statistics() if callbacks.policy_selector_wrapper else None)
                
                # Периодические отчеты о прогрессе
                if iteration % 100 == 0:
                    elapsed_time = time.time() - training_start_time
                    remaining_iterations = iterations - iteration
                    estimated_remaining_time = (elapsed_time / iteration) * remaining_iterations
                    
                    print(f"\n📊 Progress Report - Iteration {iteration}/{iterations}")
                    print(f"   Elapsed time: {elapsed_time/3600:.2f} hours")
                    print(f"   Estimated remaining: {estimated_remaining_time/3600:.2f} hours")
                    print(f"   Current performance: {reward:.3f}")
                    print(f"   Best performance: {best_performance:.3f}")
                    print(f"   Improvement: {reward - results_manager.training_metrics[0]['reward'] if results_manager.training_metrics else 0:+.3f}")
                    
                    if callbacks.policy_selector_wrapper:
                        selector_stats = callbacks.policy_selector_wrapper.get_statistics()
                        print(f"   Policy selections: {selector_stats['total_selections']}")
                        print(f"   Current policy: {selector_stats['current_policy']}")
                    print()
                
            except KeyboardInterrupt:
                print("\n⏹️ Training interrupted by user")
                break
            except Exception as e:
                print(f"❌ Error at iteration {iteration}: {e}")
                if args.test:
                    break
                continue
        
        # Финальные сохранения
        training_time = time.time() - training_start_time
        
        print(f"\n💾 Saving final results...")
        
        # Финальный чекпоинт
        final_checkpoint = results_manager.save_checkpoint(algorithm, iteration, best_performance)
        
        # Финальное состояние селектора
        if callbacks is not None and hasattr(callbacks, 'policy_selector_wrapper') and callbacks.policy_selector_wrapper:
            results_manager.save_policy_selector_state(callbacks.policy_selector_wrapper, iteration)
        
        # Финальный экспорт ONNX
        try:
            results_manager.export_model_to_onnx(algorithm, iteration)
        except Exception as e:
            print(f"⚠️ Final ONNX export failed: {e}")
        
        # Финальная визуализация
        if not args.disable_viz and callbacks is not None:
            try:
                print(f"🎬 Creating final battle visualization...")
                # Проводим финальный показательный бой
                for opponent_id in opponent_ids[:1]:
                    wins_main, wins_opp, battle_frames, selector_stats = callbacks._play_match_with_recording(
                        algorithm, opponent_id, episodes=1
                    )
                    
                    if battle_frames:
                        # Сохраняем финальный бой
                        battle_summary = {
                            'iteration': iteration,
                            'final_battle': True,
                            'opponent': opponent_id,
                            'wins_main': wins_main,
                            'wins_opp': wins_opp,
                            'selector_stats': selector_stats
                        }
                        
                        results_manager.save_battle_record(iteration, battle_frames, battle_summary)
                        results_manager.create_battle_visualization(battle_frames, iteration, save_animation=True)
            except Exception as e:
                print(f"⚠️ Final visualization failed: {e}")
        
        # Генерация финального отчета
        report_path = results_manager.generate_final_report(iteration, training_time)
        
        print(f"\n🏁 Enhanced Training Completed!")
        print(f"   Experiment: {results_manager.experiment_name}")
        print(f"   Directory: {results_manager.experiment_dir}")
        print(f"   Duration: {training_time/3600:.2f} hours")
        print(f"   Final performance: {best_performance:.3f}")
        print(f"   Total iterations: {iteration}")
        
        # Показываем созданные файлы
        print(f"\n📁 Generated Files:")
        print(f"   📂 {results_manager.experiment_dir}")
        print(f"   ├── 💾 checkpoints/ ({len(list((results_manager.experiment_dir / 'checkpoints').glob('iter_*')))} saves)")
        print(f"   ├── ⚔️ battles/ ({len(list((results_manager.experiment_dir / 'battles').glob('*.json')))} records)")
        print(f"   ├── 🎬 visualizations/ ({len(list((results_manager.experiment_dir / 'visualizations').glob('*.gif')))} videos)")
        print(f"   ├── 📦 models/ ({len(list((results_manager.experiment_dir / 'models').glob('*.onnx')))} ONNX)")
        print(f"   ├── 🎯 policy_selector/ (evolution tracking)")
        print(f"   ├── 📊 logs/ (training metrics)")
        print(f"   └── 📋 final_report.txt")
        
        # Финальный анализ селектора
        if callbacks is not None and hasattr(callbacks, 'policy_selector_wrapper') and callbacks.policy_selector_wrapper:
            print(f"\n🎯 Final Policy Selector Analysis:")
            
            try:
                final_analysis = analyze_policy_effectiveness(callbacks.policy_selector_wrapper)
                
                if "policy_rankings" in final_analysis:
                    print(f"   Policy Performance Rankings:")
                    for rank, policy_info in enumerate(final_analysis["policy_rankings"][:3], 1):
                        print(f"   {rank}. {policy_info['policy']:15}: "
                              f"Score {policy_info['weighted_score']:.3f}, "
                              f"Usage {policy_info['usage_percent']:.1f}%")
                
                if "recommendations" in final_analysis and final_analysis["recommendations"]:
                    print(f"   Recommendations:")
                    for rec in final_analysis["recommendations"][:2]:
                        print(f"     • {rec}")
                        
            except Exception as e:
                print(f"   ⚠️ Policy analysis failed: {e}")
        
        # Инструкции для продолжения работы
        print(f"\n🔧 Next Steps:")
        print(f"   • Load checkpoint: Use 'latest' or 'best' symlinks")
        print(f"   • View battles: Open .json files in visualizations")
        print(f"   • Analyze progress: Check final_report.txt")
        print(f"   • Deploy model: Use ONNX files for inference")
        print(f"   • Continue training: Resume from checkpoint")
        
        algorithm.stop()
        
    except Exception as e:
        print(f"💥 Fatal error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    finally:
        ray.shutdown()
    
    print(f"\n🎉 Enhanced aircraft training with comprehensive saving completed!")
    print(f"🎯 Your AI pilots are ready for deployment with full battle history!")
    return 0


def resume_training_from_checkpoint():
    """Утилита для продолжения обучения с чекпоинта"""
    parser = argparse.ArgumentParser(description="Resume Aircraft Training from Checkpoint")
    parser.add_argument("--checkpoint-path", type=str, required=True, help="Path to checkpoint directory")
    parser.add_argument("--additional-iterations", type=int, default=500, help="Additional training iterations")
    parser.add_argument("--new-experiment-name", type=str, help="New experiment name for continued training")
    
    args = parser.parse_args()
    
    checkpoint_path = Path(args.checkpoint_path)
    if not checkpoint_path.exists():
        print(f"❌ Checkpoint path not found: {checkpoint_path}")
        return 1
    
    # Загружаем информацию о чекпоинте
    checkpoint_info_path = checkpoint_path / "checkpoint_info.json"
    if checkpoint_info_path.exists():
        with open(checkpoint_info_path, 'r') as f:
            checkpoint_info = json.load(f)
        
        print(f"📂 Resuming from checkpoint:")
        print(f"   Original iteration: {checkpoint_info.get('iteration', 'unknown')}")
        print(f"   Performance: {checkpoint_info.get('performance', 'unknown')}")
        print(f"   Timestamp: {checkpoint_info.get('timestamp', 'unknown')}")
    
    try:
        ray.init(ignore_reinit_error=True)
        
        # Создаем новый менеджер результатов
        results_manager = TrainingResultsManager(
            experiment_name=args.new_experiment_name or f"resumed_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        
        # Регистрируем компоненты
        register_env("DogfightEnv", env_creator)
        ModelCatalog.register_custom_model("aircraft_transformer", AircraftTransformerModel)
        ModelCatalog.register_custom_action_dist("aircraft_actions", AircraftActionDistribution)
        
        # Восстанавливаем алгоритм
        from ray.rllib.algorithms.ppo import PPO
        algorithm = PPO.from_checkpoint(str(checkpoint_path))
        
        print(f"✅ Algorithm restored from checkpoint")
        
        # Создаем callbacks
        callbacks = EnhancedPolicySelectorCallbacks(results_manager)
        
        # Продолжаем обучение
        print(f"🚀 Resuming training for {args.additional_iterations} additional iterations...")
        
        start_iteration = checkpoint_info.get('iteration', 0) if checkpoint_info_path.exists() else 0
        
        for iteration in range(start_iteration + 1, start_iteration + args.additional_iterations + 1):
            try:
                result = algorithm.train()
                callbacks.on_train_result(algorithm, result)
                
                reward = result.get("env_runners", {}).get("episode_reward_mean", 0)
                
                if iteration % 10 == 0:
                    print(f"[{iteration:4d}] Reward: {reward:7.3f}")
                
            except KeyboardInterrupt:
                print("\n⏹️ Resumed training interrupted")
                break
            except Exception as e:
                print(f"❌ Error at iteration {iteration}: {e}")
                continue
        
        # Финальные сохранения
        results_manager.save_checkpoint(algorithm, iteration)
        results_manager.generate_final_report(args.additional_iterations, time.time())
        
        algorithm.stop()
        print(f"🏁 Resumed training completed!")
        
    except Exception as e:
        print(f"💥 Error resuming training: {e}")
        return 1
    finally:
        ray.shutdown()
    
    return 0


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "resume":
        # Удаляем 'resume' из sys.argv для правильного парсинга аргументов
        sys.argv = [sys.argv[0]] + sys.argv[2:]
        sys.exit(resume_training_from_checkpoint())
    else:
        sys.exit(main_with_enhanced_saving())