"""
Интеграция Policy Selector в основной цикл обучения самолетов
Адаптивная система выбора специализированных политик в процессе боя
"""

import os
import sys
import argparse
import ray
import torch
import numpy as np
import json
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.models import ModelCatalog
from ray.tune.registry import register_env
from typing import Dict, List, Any, Tuple

# Импорты основных компонентов
from aircraft_env import DogfightEnv, AIRSPACE_BOUNDS, ENGAGEMENT_RANGE
from aircraft_attention_model import AircraftTransformerModel, AircraftActionDistribution
from aircraft_visualization import AircraftVisualizer, BattleRecorderVisualizer, VisualizationConfig
from policy_selector_system import (
    PolicySelector, PolicySelectorWrapper, PolicySelectorCallback,
    SpecializedAircraftPolicy, PolicyType, analyze_policy_effectiveness
)
from league_state import LeagueState
from gspo_grpo_policy import GSPOTorchPolicy, GRPOTorchPolicy

def env_creator(cfg):
    return DogfightEnv(cfg)

class PolicySelectorTrainingCallbacks:
    """Расширенные callbacks для обучения с селектором политик"""
    
    def __init__(self):
        self.policy_selector_wrapper = None
        self.league = None
        self.opponent_ids = []
        self.eval_episodes = 3
        self.visualization_enabled = True
        
        # Статистика селектора
        self.policy_effectiveness_history = []
        self.selection_frequency_stats = {}
        self.battle_context_history = []
        
    def setup(self, league_actor, opponent_ids: List[str], 
              obs_space, act_space, **kwargs):
        """Настройка callbacks с селектором политик"""
        
        self.league = league_actor
        self.opponent_ids = opponent_ids
        self.eval_episodes = kwargs.get('eval_episodes', 3)
        self.visualization_enabled = kwargs.get('enable_visualization', True)
        
        # Инициализируем селектор политик
        self.policy_selector_wrapper = PolicySelectorWrapper(
            obs_space=obs_space,
            act_space=act_space,
            num_policies=6
        )
        
        print(f"🎯 Policy Selector Training Callbacks initialized")
        print(f"   Specialized policies: {len(self.policy_selector_wrapper.specialized_policies)}")
        print(f"   League opponents: {len(opponent_ids)}")
    
    def on_train_result(self, algorithm, result: Dict[str, Any]):
        """Обработка результатов тренировки с селектором политик"""
        
        iteration = result["training_iteration"]
        timesteps = result.get("timesteps_total", 0)
        
        try:
            # 1) Стандартные матчи против оппонентов
            for opponent_id in self.opponent_ids:
                wins_main, wins_opp, battle_data, selector_stats = self._play_match_with_selector(
                    algorithm, opponent_id, self.eval_episodes
                )
                
                # Обновляем league рейтинги
                if self.league:
                    ray.get(self.league.update_pair_result.remote(wins_main, wins_opp, opponent_id))
                
                # Сохраняем статистику селектора
                if selector_stats:
                    self.policy_effectiveness_history.append({
                        'iteration': iteration,
                        'opponent': opponent_id,
                        'stats': selector_stats,
                        'wins_main': wins_main,
                        'wins_opp': wins_opp
                    })
            
            # 2) Анализ эффективности селектора политик
            if iteration % 10 == 0 and self.policy_selector_wrapper:
                effectiveness_analysis = analyze_policy_effectiveness(
                    self.policy_selector_wrapper, num_episodes=50
                )
                
                # Добавляем анализ к результатам
                result.setdefault("custom_metrics", {}).update({
                    f"selector_policy_{i}_performance": perf for i, perf 
                    in enumerate(effectiveness_analysis["policy_performance"])
                })
                
                result["custom_metrics"].update({
                    f"selector_policy_{i}_usage_pct": usage for i, usage
                    in enumerate(np.array(effectiveness_analysis["policy_usage"]) / 
                               max(1, sum(effectiveness_analysis["policy_usage"])) * 100)
                })
            
            # 3) Обновление статистики селектора в результатах
            if self.policy_selector_wrapper:
                selector_stats = self.policy_selector_wrapper.get_statistics()
                result.setdefault("custom_metrics", {}).update({
                    "selector_total_selections": selector_stats["total_selections"],
                    "selector_current_policy": PolicyType(
                        self.policy_selector_wrapper.selector.current_policy
                    ).value,
                    "selector_avg_reward": selector_stats["avg_recent_reward"]
                })
            
            # 4) Клонирование слабых оппонентов
            if iteration % 20 == 0 and iteration > 0 and self.league:
                self._refresh_weakest_opponent_with_selector(algorithm)
            
            # 5) Логирование селектора политик
            if iteration % 5 == 0:
                self._log_selector_statistics(iteration)
                
        except Exception as e:
            print(f"Error in policy selector callbacks: {e}")
            import traceback
            traceback.print_exc()
    
    def _play_match_with_selector(self, algorithm, opponent_id: str, episodes: int) -> Tuple[int, int, List, Dict]:
        """Играет матч с использованием селектора политик"""
        
        wins_main, wins_opp = 0, 0
        battle_frames = []
        selector_decisions = []
        
        for episode in range(episodes):
            try:
                # Создаем окружение
                env_config = algorithm.config.env_config.copy()
                test_env = DogfightEnv(env_config)
                
                obs, _ = test_env.reset()
                done = False
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
                            action, selected_policy_idx = self.policy_selector_wrapper.select_and_execute(
                                algorithm, aircraft_obs, aircraft_id
                            )
                            
                            step_selector_info[aircraft_id] = {
                                'selected_policy': selected_policy_idx,
                                'policy_name': PolicyType(selected_policy_idx).name
                            }
                            
                            episode_selector_stats['policy_selections'].append(selected_policy_idx)
                            
                        else:
                            # Для синей команды используем оппонента
                            policy = algorithm.get_policy(opponent_id)
                            action, _, _ = policy.compute_single_action(aircraft_obs, explore=False)
                        
                        action_dict[aircraft_id] = action
                    
                    # Выполняем шаг
                    obs, rewards, terms, truncs, infos = test_env.step(action_dict)
                    done = terms.get("__all__", False) or truncs.get("__all__", False)
                    
                    # Обновляем производительность селектора
                    red_rewards = {aid: rew for aid, rew in rewards.items() if aid.startswith("red_")}
                    if red_rewards:
                        self.policy_selector_wrapper.update_performance(red_rewards, infos)
                        
                        # Записываем награды по политикам
                        for aircraft_id, reward in red_rewards.items():
                            if aircraft_id in step_selector_info:
                                policy_idx = step_selector_info[aircraft_id]['selected_policy']
                                episode_selector_stats['rewards_per_policy'][policy_idx].append(reward)
                    
                    # Сохраняем кадр для визуализации
                    if step % 5 == 0:  # каждый 5-й кадр
                        frame_data = test_env.export_for_visualization()
                        # Добавляем информацию о селекторе
                        frame_data['policy_selector'] = step_selector_info
                        battle_frames.append(frame_data)
                    
                    step += 1
                
                # Определяем победителя
                red_aircraft = [aid for aid in obs.keys() if aid.startswith("red_")]
                blue_aircraft = [aid for aid in obs.keys() if aid.startswith("blue_")]
                
                red_hp = sum(test_env.aircraft[aid].hp for aid in red_aircraft if test_env.aircraft[aid].alive)
                blue_hp = sum(test_env.aircraft[aid].hp for aid in blue_aircraft if test_env.aircraft[aid].alive)
                
                if red_hp > blue_hp:
                    wins_main += 1
                elif blue_hp > red_hp:
                    wins_opp += 1
                
                selector_decisions.append(episode_selector_stats)
                
            except Exception as e:
                print(f"Error in selector match episode: {e}")
                continue
        
        # Агрегируем статистику селектора
        aggregated_stats = self._aggregate_selector_stats(selector_decisions)
        
        return wins_main, wins_opp, battle_frames, aggregated_stats
    
    def _aggregate_selector_stats(self, selector_decisions: List[Dict]) -> Dict[str, Any]:
        """Агрегирует статистику селектора по эпизодам"""
        
        if not selector_decisions:
            return {}
        
        # Подсчитываем использование политик
        all_selections = []
        policy_rewards = {i: [] for i in range(6)}
        
        for episode_stats in selector_decisions:
            all_selections.extend(episode_stats['policy_selections'])
            
            for policy_idx, rewards in episode_stats['rewards_per_policy'].items():
                policy_rewards[policy_idx].extend(rewards)
        
        # Статистика использования
        policy_usage = np.zeros(6)
        for selection in all_selections:
            policy_usage[selection] += 1
        
        # Средние награды по политикам
        policy_avg_rewards = {}
        for policy_idx, rewards in policy_rewards.items():
            if rewards:
                policy_avg_rewards[policy_idx] = np.mean(rewards)
            else:
                policy_avg_rewards[policy_idx] = 0.0
        
        return {
            'total_selections': len(all_selections),
            'policy_usage': policy_usage.tolist(),
            'policy_usage_pct': (policy_usage / max(1, len(all_selections)) * 100).tolist(),
            'policy_avg_rewards': policy_avg_rewards,
            'most_used_policy': int(np.argmax(policy_usage)) if len(all_selections) > 0 else 0,
            'best_performing_policy': max(policy_avg_rewards.items(), key=lambda x: x[1])[0] if policy_avg_rewards else 0
        }
    
    def _refresh_weakest_opponent_with_selector(self, algorithm):
        """Обновляет слабого оппонента, учитывая селектор политик"""
        
        if not self.league:
            return
        
        try:
            scores = ray.get(self.league.get_all_scores.remote())
            opponent_scores = [(oid, scores[oid][0] - 3*scores[oid][1]) 
                             for oid in self.opponent_ids if oid in scores]
            
            if opponent_scores:
                weakest = min(opponent_scores, key=lambda x: x[1])[0]
                
                # Клонируем не только main политику, но и лучшую специализированную
                if self.policy_selector_wrapper:
                    selector_stats = self.policy_selector_wrapper.get_statistics()
                    
                    # Находим лучшую специализированную политику
                    best_policy_idx = np.argmax(self.policy_selector_wrapper.selector.policy_performance)
                    best_policy_type = PolicyType(best_policy_idx)
                    best_specialized_id = f"specialized_{best_policy_type.name.lower()}"
                    
                    if best_specialized_id in algorithm.policies:
                        # Клонируем лучшую специализированную политику
                        best_weights = algorithm.get_policy(best_specialized_id).get_weights()
                        algorithm.get_policy(weakest).set_weights(best_weights)
                        
                        print(f"🔄 Cloned {best_policy_type.name} policy into opponent {weakest}")
                    else:
                        # Fallback на main политику
                        main_weights = algorithm.get_policy("main").get_weights()
                        algorithm.get_policy(weakest).set_weights(main_weights)
                        
                        print(f"🔄 Cloned main policy into opponent {weakest}")
                
                ray.get(self.league.clone_main_into.remote(weakest))
                
        except Exception as e:
            print(f"Error refreshing opponent with selector: {e}")
    
    def _log_selector_statistics(self, iteration: int):
        """Логирует статистику селектора политик"""
        
        if not self.policy_selector_wrapper:
            return
        
        stats = self.policy_selector_wrapper.get_statistics()
        selector = self.policy_selector_wrapper.selector
        
        print(f"\n🎯 Policy Selector Stats (Iteration {iteration}):")
        print(f"   Current policy: {stats['current_policy']}")
        print(f"   Total selections: {stats['total_selections']}")
        print(f"   Recent performance: {stats['avg_recent_reward']:.3f}")
        
        # Детальная статистика по политикам
        if stats['total_selections'] > 0:
            usage = np.array(stats['policy_usage'])
            performance = np.array(stats['policy_performance'])
            
            print(f"   Policy Usage & Performance:")
            for i in range(len(usage)):
                if usage[i] > 0:
                    policy_name = PolicyType(i).name
                    usage_pct = (usage[i] / stats['total_selections']) * 100
                    perf = performance[i]
                    print(f"     {policy_name:15}: {usage_pct:5.1f}% usage, {perf:+.3f} performance")
        
        # Текущая ситуация
        current_explanation = selector.get_policy_explanation()
        print(f"   Current situation: {current_explanation.split('.')[0]}")
        print()

def main_with_policy_selector():
    """Главная функция с интеграцией селектора политик"""
    
    parser = argparse.ArgumentParser(description="Aircraft Training with Policy Selector")
    parser.add_argument("--iterations", type=int, default=1000, help="Training iterations")
    parser.add_argument("--algorithm", choices=["ppo", "gspo", "grpo"], default="gspo")
    parser.add_argument("--test", action="store_true", help="Quick test mode")
    parser.add_argument("--enable-selector", action="store_true", default=True, help="Enable policy selector")
    parser.add_argument("--selector-frequency", type=int, default=10, help="Policy selection frequency")
    parser.add_argument("--disable-viz", action="store_true", help="Disable visualization")
    parser.add_argument("--no-jsbsim", action="store_true", help="Skip JSBSim validation")
    args = parser.parse_args()
    
    print(f"🛩️ Aircraft Training with Policy Selector")
    print(f"   Algorithm: {args.algorithm.upper()}")
    print(f"   Iterations: {args.iterations}")
    print(f"   Policy Selector: {'✅ ENABLED' if args.enable_selector else '❌ DISABLED'}")
    print(f"   Selection Frequency: every {args.selector_frequency} steps")
    
    # Валидация JSBSim
    if not args.no_jsbsim:
        try:
            import jsbsim
            print("✅ JSBSim available")
        except ImportError:
            print("⚠️ JSBSim not available - continuing with simplified physics")
    
    # Инициализация Ray
    ray.init(ignore_reinit_error=True)
    
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
        }
        
        tmp_env = DogfightEnv(env_config)
        obs_space = tmp_env.observation_space
        act_space = tmp_env.action_space
        max_aircraft = obs_space["allies"].shape[0]
        
        print(f"🛩️ Environment: max {max_aircraft} aircraft per team")
        
        # Создание League
        opponent_ids = [f"squadron_{i}" for i in range(3 if args.test else 6)]
        league = LeagueState.remote(opponent_ids)
        
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
        if args.enable_selector:
            from policy_selector_system import create_specialized_policies
            specialized_policies = create_specialized_policies(obs_space, act_space, base_model_config)
            policies.update(specialized_policies)
            
            print(f"🎯 Added {len(specialized_policies)} specialized policies:")
            for policy_id in specialized_policies.keys():
                print(f"   - {policy_id}")
        
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
        print("🔧 Building algorithm with policy selector...")
        algorithm = config.build()
        
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
        
        # Создание callbacks с селектором
        callbacks = PolicySelectorTrainingCallbacks()
        callbacks.setup(
            league_actor=league,
            opponent_ids=opponent_ids,
            obs_space=obs_space,
            act_space=act_space,
            eval_episodes=2 if args.test else 3,
            enable_visualization=not args.disable_viz
        )
        
        print("✅ Algorithm with policy selector ready")
        print(f"🎯 Features:")
        print(f"   - Adaptive policy selection every {args.selector_frequency} steps")
        print(f"   - 6 specialized combat policies")
        print(f"   - Situation-aware decision making")
        print(f"   - Performance tracking and learning")
        print(f"   - League-based opponent system")
        
        # Цикл обучения
        best_performance = float('-inf')
        final_checkpoint = None
        selector_evolution = []
        
        iterations = min(args.iterations, 10) if args.test else args.iterations
        
        for iteration in range(1, iterations + 1):
            try:
                result = algorithm.train()
                
                # Обработка callbacks
                callbacks.on_train_result(algorithm, result)
                
                # Извлекаем метрики
                env_runners = result.get("env_runners", {})
                reward = env_runners.get("episode_reward_mean", 0)
                timesteps = result.get("timesteps_total", 0)
                
                # Метрики селектора
                custom = result.get("custom_metrics", {})
                selector_reward = custom.get("selector_avg_reward", 0)
                current_policy_idx = custom.get("selector_current_policy", 2)  # NORMAL_SHOOTER по умолчанию
                
                # Логирование
                if iteration % 3 == 0 or args.test:
                    current_policy_name = PolicyType(int(current_policy_idx)).name if current_policy_idx < 6 else "UNKNOWN"
                    
                    print(f"[{iteration:4d}] Reward: {reward:7.3f}, Timesteps: {timesteps:,}")
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
                
                # Отслеживаем эволюцию селектора
                if args.enable_selector:
                    selector_evolution.append({
                        'iteration': iteration,
                        'current_policy': current_policy_name,
                        'reward': selector_reward,
                        'main_reward': reward
                    })
                
                # Отслеживание лучшей производительности
                if reward > best_performance:
                    best_performance = reward
                
                # Сохранение чекпоинтов
                checkpoint_freq = 2 if args.test else 25
                if iteration % checkpoint_freq == 0:
                    checkpoint_result = algorithm.save()
                    try:
                        if hasattr(checkpoint_result, 'to_directory'):
                            checkpoint_path = checkpoint_result.to_directory()
                        else:
                            checkpoint_path = str(checkpoint_result)
                    except:
                        checkpoint_path = f"selector_checkpoint_{iteration}"
                    
                    print(f"💾 Checkpoint saved with policy selector state")
                    
                    # Сохраняем состояние селектора отдельно
                    if args.enable_selector and callbacks.policy_selector_wrapper:
                        selector_path = f"policy_selector_state_iter_{iteration:06d}.pt"
                        callbacks.policy_selector_wrapper.selector.save_state(selector_path)
                    
                    final_checkpoint = checkpoint_path
                
            except KeyboardInterrupt:
                print("\n⏹️ Training interrupted by user")
                break
            except Exception as e:
                print(f"❌ Error at iteration {iteration}: {e}")
                if args.test:
                    break
                continue
        
        # Финальное сохранение
        if final_checkpoint is None:
            final_checkpoint = algorithm.save()
        
        print(f"\n🏁 Training with Policy Selector Completed!")
        print(f"   Final checkpoint: {final_checkpoint}")
        print(f"   Best performance: {best_performance:.3f}")
        print(f"   Total iterations: {iteration}")
        
        # Финальный анализ селектора
        if args.enable_selector and callbacks.policy_selector_wrapper:
            print(f"\n🎯 Final Policy Selector Analysis:")
            
            final_stats = callbacks.policy_selector_wrapper.get_statistics()
            total_selections = final_stats['total_selections']
            
            if total_selections > 0:
                print(f"   Total policy selections: {total_selections}")
                print(f"   Final policy: {final_stats['current_policy']}")
                print(f"   Final performance: {final_stats['avg_recent_reward']:.3f}")
                
                # Эволюция использования политик
                print(f"   Policy Evolution Summary:")
                usage_counts = np.array(final_stats['policy_usage'])
                usage_pcts = (usage_counts / max(1, usage_counts.sum())) * 100
                
                for i, (count, pct) in enumerate(zip(usage_counts, usage_pcts)):
                    if count > 0:
                        policy_name = PolicyType(i).name
                        performance = final_stats['policy_performance'][i]
                        print(f"     {policy_name:15}: {count:3d} uses ({pct:5.1f}%), Perf: {performance:+.3f}")
                
                # Рекомендации
                final_analysis = analyze_policy_effectiveness(callbacks.policy_selector_wrapper)
                if final_analysis.get("recommendations"):
                    print(f"   Optimization Recommendations:")
                    for rec in final_analysis["recommendations"][:3]:  # топ-3
                        print(f"     • {rec}")
            
            # Сохраняем финальное состояние селектора
            final_selector_path = "final_policy_selector_state.pt"
            callbacks.policy_selector_wrapper.selector.save_state(final_selector_path)
            print(f"   Policy selector state saved: {final_selector_path}")
        
        # Показываем созданные файлы
        print(f"\n📁 Generated Files:")
        print(f"   Checkpoints: saved successfully")
        if args.enable_selector:
            print(f"   Policy selector states: *.pt files")
        
        # Сохраняем эволюцию селектора в JSON
        if args.enable_selector and selector_evolution:
            evolution_path = "policy_selector_evolution.json"
            with open(evolution_path, 'w') as f:
                json.dump(selector_evolution, f, indent=2)
            print(f"   Selector evolution data: {evolution_path}")
        
        algorithm.stop()
        
    except Exception as e:
        print(f"💥 Fatal error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    finally:
        ray.shutdown()
    
    print(f"\n🎉 Advanced aircraft training with policy selector completed!")
    if args.enable_selector:
        print(f"🎯 Your AI pilots now have adaptive tactical intelligence!")
        print(f"🧠 Each pilot can dynamically choose from 6 specialized combat roles")
        print(f"⚡ Real-time adaptation to changing battle conditions")
    return 0


def analyze_selector_evolution(evolution_file: str = "policy_selector_evolution.json"):
    """Анализирует эволюцию селектора политик"""
    
    try:
        with open(evolution_file, 'r') as f:
            evolution_data = json.load(f)
        
        print(f"📊 Policy Selector Evolution Analysis")
        print(f"   Data points: {len(evolution_data)}")
        
        if not evolution_data:
            return
        
        # Анализ изменения политик по времени
        policy_changes = []
        current_policy = evolution_data[0]['current_policy']
        
        for entry in evolution_data[1:]:
            if entry['current_policy'] != current_policy:
                policy_changes.append({
                    'iteration': entry['iteration'],
                    'from': current_policy,
                    'to': entry['current_policy']
                })
                current_policy = entry['current_policy']
        
        print(f"   Policy changes: {len(policy_changes)}")
        print(f"   Final policy: {evolution_data[-1]['current_policy']}")
        
        # Наиболее частые переходы
        if policy_changes:
            transitions = {}
            for change in policy_changes:
                transition = f"{change['from']} → {change['to']}"
                transitions[transition] = transitions.get(transition, 0) + 1
            
            print(f"   Most frequent transitions:")
            for transition, count in sorted(transitions.items(), key=lambda x: x[1], reverse=True)[:3]:
                print(f"     {transition}: {count} times")
        
        # Прогресс производительности
        rewards = [entry['selector_reward'] for entry in evolution_data]
        if rewards:
            print(f"   Performance progression:")
            print(f"     Initial: {rewards[0]:+.3f}")
            print(f"     Final: {rewards[-1]:+.3f}")
            print(f"     Best: {max(rewards):+.3f}")
            print(f"     Improvement: {rewards[-1] - rewards[0]:+.3f}")
        
    except FileNotFoundError:
        print(f"❌ Evolution file not found: {evolution_file}")
    except Exception as e:
        print(f"❌ Analysis failed: {e}")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "analyze":
        evolution_file = sys.argv[2] if len(sys.argv) > 2 else "policy_selector_evolution.json"
        analyze_selector_evolution(evolution_file)
    else:
        sys.exit(main_with_policy_selector())