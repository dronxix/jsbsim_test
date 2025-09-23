"""
Policy Selector System для Air-to-Air Combat
Адаптивная система выбора специализированных политик на основе текущей ситуации
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models import ModelCatalog
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.algorithms.ppo.ppo_torch_policy import PPOTorchPolicy
from enum import IntEnum
import json
from dataclasses import dataclass
from collections import deque

# Типы специализированных политик
class PolicyType(IntEnum):
    OBSERVER = 0          # Наблюдатель - пассивная разведка
    AGGRESSIVE_SHOOTER = 1 # Агрессивный стрелок - активная атака
    NORMAL_SHOOTER = 2    # Обычный стрелок - сбалансированный подход
    DEFENSIVE = 3         # Оборонительная политика
    INTERCEPTOR = 4       # Перехватчик - преследование целей
    SUPPORT = 5          # Поддержка союзников

@dataclass
class SituationContext:
    """Контекст текущей ситуации для выбора политики"""
    # Тактическая ситуация
    enemy_count: int
    ally_count: int
    enemy_distance_avg: float
    enemy_distance_min: float
    
    # Состояние самолета
    own_hp: float
    own_fuel: float
    own_altitude_relative: float
    own_speed: float
    
    # Вооружение
    missiles_remaining: int
    cannon_ammo: int
    
    # Угрозы
    incoming_missiles: int
    radar_locked: bool
    enemy_advantage: bool
    
    # История
    recent_hits_taken: int
    recent_hits_given: int
    time_since_last_engagement: float

class SituationAnalyzer(nn.Module):
    """Анализатор тактической ситуации"""
    
    def __init__(self, obs_dim: int, context_dim: int = 32):
        super().__init__()
        self.obs_dim = obs_dim
        self.context_dim = context_dim
        
        # Энкодер наблюдений в контекст ситуации
        self.context_encoder = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, context_dim),
            nn.Tanh()
        )
        
        # Классификатор типа ситуации
        self.situation_classifier = nn.Sequential(
            nn.Linear(context_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 6)  # 6 типов ситуаций
        )
        
    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Возвращает контекст ситуации и классификацию
        """
        context = self.context_encoder(obs)
        situation_logits = self.situation_classifier(context)
        return context, situation_logits
    
    def extract_context(self, obs_dict: Dict[str, torch.Tensor]) -> SituationContext:
        """Извлекает структурированный контекст из наблюдений"""
        # Извлекаем информацию из словаря наблюдений
        self_obs = obs_dict["self"]
        enemies = obs_dict["enemies"]
        allies = obs_dict["allies"]
        enemies_mask = obs_dict["enemies_mask"]
        allies_mask = obs_dict["allies_mask"]
        
        # Подсчитываем врагов и союзников
        enemy_count = int(enemies_mask.sum().item())
        ally_count = int(allies_mask.sum().item())
        
        # Анализируем расстояния до врагов
        enemy_positions = enemies[:, :3]  # первые 3 - относительные координаты
        enemy_distances = torch.norm(enemy_positions, dim=-1)
        valid_enemy_distances = enemy_distances[enemies_mask > 0]
        
        if len(valid_enemy_distances) > 0:
            enemy_distance_avg = float(valid_enemy_distances.mean().item())
            enemy_distance_min = float(valid_enemy_distances.min().item())
        else:
            enemy_distance_avg = 1.0
            enemy_distance_min = 1.0
        
        # Извлекаем состояние своего самолета
        # Предполагаем структуру: [pos(3), attitude(3), velocity(3), angular_vel(3), tas, mach, accel(3), fuel, hp]
        own_hp = float(self_obs[-1].item())  # последний элемент - HP
        own_fuel = float(self_obs[-2].item())  # предпоследний - топливо
        own_altitude_relative = float(self_obs[2].item())  # z-координата
        own_speed = float(self_obs[12].item() if len(self_obs) > 12 else 0.5)  # воздушная скорость
        
        # Анализ вооружения (примерные значения, в реальности из info)
        missiles_remaining = 4  # будет обновляться из info
        cannon_ammo = 500
        
        # Анализ угроз (упрощенно)
        radar_locked = enemy_distance_min < 0.3  # близкий враг = вероятная угроза
        enemy_advantage = enemy_count > ally_count
        incoming_missiles = 0  # будет обновляться из info
        
        return SituationContext(
            enemy_count=enemy_count,
            ally_count=ally_count,
            enemy_distance_avg=enemy_distance_avg,
            enemy_distance_min=enemy_distance_min,
            own_hp=own_hp,
            own_fuel=own_fuel,
            own_altitude_relative=own_altitude_relative,
            own_speed=own_speed,
            missiles_remaining=missiles_remaining,
            cannon_ammo=cannon_ammo,
            incoming_missiles=incoming_missiles,
            radar_locked=radar_locked,
            enemy_advantage=enemy_advantage,
            recent_hits_taken=0,  # будет отслеживаться
            recent_hits_given=0,  # будет отслеживаться
            time_since_last_engagement=0.0
        )

class PolicySelectorNetwork(nn.Module):
    """Нейронная сеть для выбора политик"""
    
    def __init__(self, context_dim: int = 32, num_policies: int = 6, hidden_dim: int = 64):
        super().__init__()
        self.context_dim = context_dim
        self.num_policies = num_policies
        
        # Энкодер контекста ситуации
        self.situation_encoder = nn.Sequential(
            nn.Linear(context_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Энкодер истории выбора политик
        self.history_encoder = nn.Sequential(
            nn.Linear(num_policies * 10, hidden_dim//2),  # последние 10 выборов
            nn.ReLU(),
            nn.Linear(hidden_dim//2, hidden_dim//2)
        )
        
        # Энкодер результатов политик
        self.performance_encoder = nn.Sequential(
            nn.Linear(num_policies, hidden_dim//4),  # производительность каждой политики
            nn.ReLU(),
            nn.Linear(hidden_dim//4, hidden_dim//4)
        )
        
        # Финальный селектор политик
        self.policy_selector = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim//2 + hidden_dim//4, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.ReLU(),
            nn.Linear(hidden_dim//2, num_policies)
        )
        
        # Value функция для оценки качества выбора
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim//2 + hidden_dim//4, hidden_dim//2),
            nn.ReLU(),
            nn.Linear(hidden_dim//2, 1)
        )
    
    def forward(self, situation_context: torch.Tensor, 
                policy_history: torch.Tensor,
                policy_performance: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Выбирает политику на основе контекста, истории и производительности
        """
        # Кодируем входы
        situation_encoded = self.situation_encoder(situation_context)
        history_encoded = self.history_encoder(policy_history)
        performance_encoded = self.performance_encoder(policy_performance)
        
        # Объединяем все признаки
        combined_features = torch.cat([
            situation_encoded, 
            history_encoded, 
            performance_encoded
        ], dim=-1)
        
        # Выбираем политику
        policy_logits = self.policy_selector(combined_features)
        
        # Оцениваем качество выбора
        value = self.value_head(combined_features)
        
        return policy_logits, value

class PolicySelector:
    """Основной класс селектора политик"""
    
    def __init__(self, obs_space, num_policies: int = 6, selection_frequency: int = 10):
        self.num_policies = num_policies
        self.selection_frequency = selection_frequency
        self.obs_space = obs_space
        
        # Анализатор ситуации
        obs_dim = self._calculate_obs_dim()
        self.situation_analyzer = SituationAnalyzer(obs_dim)
        
        # Селектор политик
        self.selector_network = PolicySelectorNetwork(num_policies=num_policies)
        
        # История и статистика
        self.policy_history = deque(maxlen=100)  # последние 100 выборов
        self.policy_performance = np.zeros(num_policies, dtype=np.float32)
        self.policy_usage_count = np.zeros(num_policies, dtype=np.int32)
        
        # Текущее состояние
        self.current_policy = PolicyType.NORMAL_SHOOTER
        self.steps_since_selection = 0
        self.last_situation_context = None
        
        # Обучение селектора
        self.selector_optimizer = torch.optim.Adam(
            list(self.situation_analyzer.parameters()) + 
            list(self.selector_network.parameters()), 
            lr=3e-4
        )
        
        print(f"🎯 Policy Selector initialized:")
        print(f"   Policies: {num_policies}")
        print(f"   Selection frequency: every {selection_frequency} steps")
        print(f"   Available policies: {[p.name for p in PolicyType][:num_policies]}")
    
    def _calculate_obs_dim(self) -> int:
        """Вычисляет размерность наблюдений для анализатора"""
        # Приблизительная оценка на основе пространства наблюдений
        total_dim = 0
        if hasattr(self.obs_space, 'spaces'):
            for key, space in self.obs_space.spaces.items():
                if key != 'global_state':  # исключаем глобальное состояние
                    if len(space.shape) == 1:
                        total_dim += space.shape[0]
                    elif len(space.shape) == 2:
                        total_dim += space.shape[0] * space.shape[1]
        return max(total_dim, 64)  # минимум 64
    
    def select_policy(self, obs_dict: Dict[str, torch.Tensor], 
                     infos: Optional[Dict] = None) -> int:
        """
        Выбирает политику на основе текущих наблюдений
        """
        self.steps_since_selection += 1
        
        # Выбираем политику только каждые N шагов
        if self.steps_since_selection < self.selection_frequency:
            return int(self.current_policy)
        
        self.steps_since_selection = 0
        
        with torch.no_grad():
            # Анализируем ситуацию
            obs_flat = self._flatten_observations(obs_dict)
            situation_context, situation_logits = self.situation_analyzer(obs_flat)
            
            # Подготавливаем историю выбора политик
            policy_history_tensor = self._prepare_policy_history()
            
            # Подготавливаем производительность политик
            policy_performance_tensor = torch.from_numpy(self.policy_performance)
            
            # Выбираем политику
            policy_logits, value = self.selector_network(
                situation_context,
                policy_history_tensor,
                policy_performance_tensor.unsqueeze(0)
            )
            
            # Принимаем решение (с небольшой случайностью для исследования)
            if np.random.random() < 0.1:  # 10% исследования
                selected_policy = np.random.randint(0, self.num_policies)
            else:
                selected_policy = int(torch.argmax(policy_logits, dim=-1).item())
            
            self.current_policy = PolicyType(selected_policy)
            
            # Обновляем статистику
            self.policy_usage_count[selected_policy] += 1
            self.policy_history.append({
                'policy': selected_policy,
                'situation_context': situation_context.cpu().numpy(),
                'value': float(value.item()),
                'step': len(self.policy_history)
            })
            
            # Извлекаем структурированный контекст
            self.last_situation_context = self.situation_analyzer.extract_context(obs_dict)
            
            # Логируем выбор политики
            self._log_policy_selection(selected_policy, infos)
        
        return int(self.current_policy)
    
    def _flatten_observations(self, obs_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Преобразует словарь наблюдений в плоский тензор"""
        obs_parts = []
        
        # Добавляем различные части наблюдений
        for key in ["self", "allies", "enemies"]:
            if key in obs_dict:
                tensor = obs_dict[key]
                if tensor.dim() > 2:  # для батчей
                    tensor = tensor.flatten(start_dim=1)
                elif tensor.dim() == 2:  # для одиночных наблюдений
                    tensor = tensor.flatten()
                obs_parts.append(tensor)
        
        # Добавляем маски
        for key in ["allies_mask", "enemies_mask", "enemy_action_mask"]:
            if key in obs_dict:
                tensor = obs_dict[key].float()
                if tensor.dim() == 1:
                    obs_parts.append(tensor)
                else:
                    obs_parts.append(tensor.flatten(start_dim=1))
        
        if not obs_parts:
            return torch.zeros(64)  # fallback
        
        return torch.cat(obs_parts, dim=-1 if obs_parts[0].dim() > 1 else 0)
    
    def _prepare_policy_history(self) -> torch.Tensor:
        """Подготавливает тензор истории выбора политик"""
        history_length = 10
        history_tensor = torch.zeros(1, self.num_policies * history_length)
        
        if len(self.policy_history) > 0:
            recent_history = list(self.policy_history)[-history_length:]
            for i, entry in enumerate(recent_history):
                policy_idx = entry['policy']
                history_tensor[0, i * self.num_policies + policy_idx] = 1.0
        
        return history_tensor
    
    def update_policy_performance(self, rewards: Dict[str, float], 
                                 infos: Optional[Dict] = None):
        """Обновляет производительность текущей политики"""
        if not rewards:
            return
        
        # Средняя награда по всем агентам
        avg_reward = np.mean(list(rewards.values()))
        
        # Обновляем производительность текущей политики с экспоненциальным сглаживанием
        alpha = 0.1  # скорость обучения
        policy_idx = int(self.current_policy)
        
        self.policy_performance[policy_idx] = (
            (1 - alpha) * self.policy_performance[policy_idx] + 
            alpha * avg_reward
        )
        
        # Дополнительные метрики из infos
        if infos:
            for agent_id, info in infos.items():
                if isinstance(info, dict):
                    # Учитываем специфичные для авиации метрики
                    if 'kills' in info:
                        self.policy_performance[policy_idx] += info['kills'] * 0.5
                    if 'hits_taken' in info:
                        self.policy_performance[policy_idx] -= info['hits_taken'] * 0.2
                    if 'missiles_fired' in info and 'hits_given' in info:
                        hit_rate = info.get('hits_given', 0) / max(1, info.get('missiles_fired', 1))
                        self.policy_performance[policy_idx] += hit_rate * 0.3
    
    def train_selector(self, batch_rewards: List[float], 
                      batch_contexts: List[torch.Tensor]):
        """Обучает селектор политик"""
        if len(batch_rewards) < 10:  # нужно достаточно данных
            return
        
        self.selector_optimizer.zero_grad()
        
        # Подготавливаем данные для обучения
        rewards_tensor = torch.tensor(batch_rewards, dtype=torch.float32)
        contexts_tensor = torch.stack(batch_contexts)
        
        # Прямой проход через сеть
        policy_history_batch = torch.zeros(len(batch_rewards), self.num_policies * 10)
        performance_batch = torch.tile(
            torch.from_numpy(self.policy_performance), 
            (len(batch_rewards), 1)
        )
        
        policy_logits, values = self.selector_network(
            contexts_tensor, policy_history_batch, performance_batch
        )
        
        # Вычисляем loss
        # Value loss
        value_loss = F.mse_loss(values.squeeze(), rewards_tensor)
        
        # Policy loss (REINFORCE)
        advantages = rewards_tensor - values.squeeze().detach()
        policy_probs = F.softmax(policy_logits, dim=-1)
        
        # Псевдо-действия на основе производительности
        pseudo_actions = torch.argmax(policy_logits, dim=-1)
        policy_loss = -torch.mean(
            torch.log(policy_probs.gather(1, pseudo_actions.unsqueeze(1))) * 
            advantages.unsqueeze(1)
        )
        
        # Энтропийный бонус для исследования
        entropy_bonus = -0.01 * torch.mean(
            torch.sum(policy_probs * torch.log(policy_probs + 1e-8), dim=-1)
        )
        
        total_loss = value_loss + policy_loss + entropy_bonus
        
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(self.situation_analyzer.parameters()) + 
            list(self.selector_network.parameters()), 
            max_norm=0.5
        )
        self.selector_optimizer.step()
    
    def _log_policy_selection(self, selected_policy: int, infos: Optional[Dict]):
        """Логирует выбор политики"""
        policy_name = PolicyType(selected_policy).name
        
        print(f"🎯 Policy selected: {policy_name} (step {len(self.policy_history)})")
        
        if self.last_situation_context:
            ctx = self.last_situation_context
            print(f"   Situation: {ctx.enemy_count} enemies, HP {ctx.own_hp:.2f}, "
                  f"Fuel {ctx.own_fuel:.2f}, Min enemy distance {ctx.enemy_distance_min:.2f}")
        
        # Показываем статистику использования политик
        if len(self.policy_history) % 50 == 0:  # каждые 50 выборов
            self._print_policy_statistics()
    
    def _print_policy_statistics(self):
        """Выводит статистику использования политик"""
        print(f"\n📊 Policy Usage Statistics:")
        total_usage = sum(self.policy_usage_count)
        
        for i in range(self.num_policies):
            policy_name = PolicyType(i).name
            usage_pct = (self.policy_usage_count[i] / max(1, total_usage)) * 100
            performance = self.policy_performance[i]
            
            print(f"   {policy_name:15}: {usage_pct:5.1f}% usage, "
                  f"Performance: {performance:+.3f}")
        
        print()
    
    def get_policy_explanation(self) -> str:
        """Возвращает объяснение текущего выбора политики"""
        policy_name = self.current_policy.name
        
        explanations = {
            PolicyType.OBSERVER: "Maintaining safe distance, gathering intelligence on enemy positions and movements",
            PolicyType.AGGRESSIVE_SHOOTER: "Closing distance rapidly, prioritizing offensive engagement with available weapons",
            PolicyType.NORMAL_SHOOTER: "Balanced approach, engaging targets while maintaining tactical awareness",
            PolicyType.DEFENSIVE: "Prioritizing survival, using evasive maneuvers and defensive positioning",
            PolicyType.INTERCEPTOR: "Pursuing and engaging high-priority targets with speed and precision",
            PolicyType.SUPPORT: "Providing cover and assistance to allied aircraft in combat"
        }
        
        explanation = explanations.get(self.current_policy, "Unknown policy behavior")
        
        if self.last_situation_context:
            ctx = self.last_situation_context
            situation_desc = (
                f"Current situation: {ctx.enemy_count} enemies detected, "
                f"own status {ctx.own_hp:.0f}% HP, {ctx.own_fuel:.0f}% fuel, "
                f"{'radar lock detected' if ctx.radar_locked else 'no immediate threats'}"
            )
            return f"{policy_name}: {explanation}\n{situation_desc}"
        
        return f"{policy_name}: {explanation}"
    
    def save_state(self, filepath: str):
        """Сохраняет состояние селектора"""
        state = {
            'selector_network_state': self.selector_network.state_dict(),
            'situation_analyzer_state': self.situation_analyzer.state_dict(),
            'policy_performance': self.policy_performance.tolist(),
            'policy_usage_count': self.policy_usage_count.tolist(),
            'current_policy': int(self.current_policy),
            'policy_history': list(self.policy_history)
        }
        
        torch.save(state, filepath)
        print(f"💾 Policy selector state saved: {filepath}")
    
    def load_state(self, filepath: str):
        """Загружает состояние селектора"""
        try:
            state = torch.load(filepath)
            
            self.selector_network.load_state_dict(state['selector_network_state'])
            self.situation_analyzer.load_state_dict(state['situation_analyzer_state'])
            self.policy_performance = np.array(state['policy_performance'])
            self.policy_usage_count = np.array(state['policy_usage_count'])
            self.current_policy = PolicyType(state['current_policy'])
            
            # Восстанавливаем историю
            self.policy_history.clear()
            for entry in state['policy_history']:
                self.policy_history.append(entry)
            
            print(f"📂 Policy selector state loaded: {filepath}")
            
        except Exception as e:
            print(f"❌ Failed to load policy selector state: {e}")

# Специализированные политики
class SpecializedAircraftPolicy(PPOTorchPolicy):
    """Базовый класс для специализированных авиационных политик"""
    
    def __init__(self, *args, **kwargs):
        self.policy_type = kwargs.pop('policy_type', PolicyType.NORMAL_SHOOTER)
        super().__init__(*args, **kwargs)
        
        print(f"🛩️ Initialized specialized policy: {self.policy_type.name}")
    
    def postprocess_trajectory(self, sample_batch: SampleBatch, 
                             other_agent_batches=None, episode=None):
        """Пост-обработка с учетом специализации политики"""
        
        # Базовая обработка
        sample_batch = super().postprocess_trajectory(
            sample_batch, other_agent_batches, episode)
        
        # Модификация наград в зависимости от типа политики
        if self.policy_type == PolicyType.OBSERVER:
            # Наблюдатель получает награду за выживание и разведку
            self._modify_rewards_for_observer(sample_batch)
        elif self.policy_type == PolicyType.AGGRESSIVE_SHOOTER:
            # Агрессивный стрелок - за атаки и урон
            self._modify_rewards_for_aggressive(sample_batch)
        elif self.policy_type == PolicyType.DEFENSIVE:
            # Оборонительная - за уклонение и выживание
            self._modify_rewards_for_defensive(sample_batch)
        elif self.policy_type == PolicyType.SUPPORT:
            # Поддержка - за помощь союзникам
            self._modify_rewards_for_support(sample_batch)
        
        return sample_batch
    
    def _modify_rewards_for_observer(self, sample_batch: SampleBatch):
        """Модифицирует награды для политики наблюдателя"""
        rewards = sample_batch["rewards"]
        infos = sample_batch.get("infos", [])
        
        for i, (reward, info) in enumerate(zip(rewards, infos)):
            if isinstance(info, dict):
                # Бонус за выживание
                if info.get("hp", 100) > 50:
                    rewards[i] += 0.1
                
                # Бонус за обнаружение врагов (без атаки)
                if info.get("enemies_detected", 0) > 0 and info.get("missiles_fired", 0) == 0:
                    rewards[i] += 0.2
                
                # Штраф за агрессивное поведение
                if info.get("missiles_fired", 0) > 0:
                    rewards[i] -= 0.1
    
    def _modify_rewards_for_aggressive(self, sample_batch: SampleBatch):
        """Модифицирует награды для агрессивной политики"""
        rewards = sample_batch["rewards"]
        infos = sample_batch.get("infos", [])
        
        for i, (reward, info) in enumerate(zip(rewards, infos)):
            if isinstance(info, dict):
                # Бонус за атаки и попадания
                missiles_fired = info.get("missiles_fired", 0)
                hits_given = info.get("hits_given", 0)
                
                rewards[i] += missiles_fired * 0.2  # бонус за активность
                rewards[i] += hits_given * 0.5      # большой бонус за попадания
                
                # Бонус за сближение с врагами
                min_enemy_distance = info.get("min_enemy_distance", 1.0)
                if min_enemy_distance < 0.3:  # близко к врагу
                    rewards[i] += 0.3
    
    def _modify_rewards_for_defensive(self, sample_batch: SampleBatch):
        """Модифицирует награды для оборонительной политики"""
        rewards = sample_batch["rewards"]
        infos = sample_batch.get("infos", [])
        
        for i, (reward, info) in enumerate(zip(rewards, infos)):
            if isinstance(info, dict):
                # Бонус за уклонение от атак
                hits_taken = info.get("hits_taken", 0)
                if hits_taken == 0:
                    rewards[i] += 0.2
                else:
                    rewards[i] -= hits_taken * 0.3
                
                # Бонус за поддержание высокого HP
                hp = info.get("hp", 100)
                rewards[i] += (hp / 100.0) * 0.1
    
    def _modify_rewards_for_support(self, sample_batch: SampleBatch):
        """Модифицирует награды для политики поддержки"""
        rewards = sample_batch["rewards"]
        infos = sample_batch.get("infos", [])
        
        for i, (reward, info) in enumerate(zip(rewards, infos)):
            if isinstance(info, dict):
                # Бонус за помощь союзникам
                allies_supported = info.get("allies_supported", 0)
                rewards[i] += allies_supported * 0.3
                
                # Бонус за координацию действий
                team_coordination = info.get("team_coordination", 0.0)
                rewards[i] += team_coordination * 0.2

def create_specialized_policies(obs_space, act_space, model_config: Dict) -> Dict:
    """Создает набор специализированных политик"""
    
    policies = {}
    
    # Базовая конфигурация для всех политик
    base_config = model_config.copy()
    
    for policy_type in PolicyType:
        if policy_type.value >= 6:  # только первые 6 политик
            break
            
        # Модифицируем конфигурацию под специализацию
        specialized_config = base_config.copy()
        
        # Настройки модели в зависимости от типа политики
        if policy_type == PolicyType.OBSERVER:
            # Наблюдатель: больше внимания к анализу ситуации
            specialized_config["custom_model_config"]["d_model"] = 192
            specialized_config["custom_model_config"]["nhead"] = 6
            specialized_config["custom_model_config"]["layers"] = 3
            
        elif policy_type == PolicyType.AGGRESSIVE_SHOOTER:
            # Агрессивный: быстрые решения, меньше размышлений
            specialized_config["custom_model_config"]["d_model"] = 128
            specialized_config["custom_model_config"]["nhead"] = 4
            specialized_config["custom_model_config"]["layers"] = 2
            
        elif policy_type == PolicyType.DEFENSIVE:
            # Оборонительный: сосредоточен на анализе угроз
            specialized_config["custom_model_config"]["d_model"] = 160
            specialized_config["custom_model_config"]["nhead"] = 8
            specialized_config["custom_model_config"]["layers"] = 3
        
        # Создаем политику
        policy_id = f"specialized_{policy_type.name.lower()}"
        policies[policy_id] = (
            SpecializedAircraftPolicy,
            obs_space,
            act_space,
            {
                "model": specialized_config,
                "policy_type": policy_type
            }
        )
        
        print(f"✈️ Created specialized policy: {policy_id}")
    
    return policies


class PolicySelectorWrapper:
    """Обертка для интеграции селектора политик с RLLib"""
    
    def __init__(self, obs_space, act_space, num_policies: int = 6):
        self.obs_space = obs_space
        self.act_space = act_space
        self.num_policies = num_policies
        
        # Создаем селектор
        self.selector = PolicySelector(obs_space, num_policies)
        
        # Создаем специализированные политики
        base_model_config = {
            "custom_model": "aircraft_transformer",
            "custom_action_dist": "aircraft_actions",
            "custom_model_config": {
                "d_model": 256,
                "nhead": 8,
                "layers": 3,
                "max_aircraft": obs_space["allies"].shape[0],
                "max_enemies": obs_space["enemies"].shape[0],
            },
            "vf_share_layers": False,
        }
        
        self.specialized_policies = create_specialized_policies(
            obs_space, act_space, base_model_config
        )
        
        # Статистика
        self.total_selections = 0
        self.selection_rewards = []
        self.current_policy_performance = {}
        
        print(f"🎯 Policy Selector Wrapper initialized")
        print(f"   Total specialized policies: {len(self.specialized_policies)}")
    
    def select_and_execute(self, algorithm, obs_dict: Dict, agent_id: str, 
                          infos: Optional[Dict] = None) -> Tuple[Any, int]:
        """Выбирает политику и выполняет действие"""
        
        # Конвертируем наблюдения в тензоры если нужно
        if isinstance(obs_dict, dict):
            obs_tensors = {}
            for key, value in obs_dict.items():
                if isinstance(value, np.ndarray):
                    obs_tensors[key] = torch.from_numpy(value).float()
                else:
                    obs_tensors[key] = value
        else:
            obs_tensors = obs_dict
        
        # Выбираем политику
        selected_policy_idx = self.selector.select_policy(obs_tensors, infos)
        policy_type = PolicyType(selected_policy_idx)
        
        # Получаем ID политики
        policy_id = f"specialized_{policy_type.name.lower()}"
        
        # Выполняем действие через выбранную политику
        if policy_id in algorithm.policies:
            policy = algorithm.get_policy(policy_id)
            action, state, action_info = policy.compute_single_action(
                obs_dict, explore=True
            )
        else:
            # Fallback на основную политику
            policy = algorithm.get_policy("main")
            action, state, action_info = policy.compute_single_action(
                obs_dict, explore=True
            )
        
        self.total_selections += 1
        return action, selected_policy_idx
    
    def update_performance(self, rewards: Dict[str, float], 
                          infos: Optional[Dict] = None):
        """Обновляет производительность политик"""
        self.selector.update_policy_performance(rewards, infos)
        
        # Сохраняем награды для обучения селектора
        if rewards:
            avg_reward = np.mean(list(rewards.values()))
            self.selection_rewards.append(avg_reward)
        
        # Обучаем селектор каждые 100 шагов
        if len(self.selection_rewards) >= 100:
            contexts = []
            for _ in range(min(50, len(self.selection_rewards))):
                # Генерируем псевдо-контексты для обучения
                # В реальности нужно сохранять реальные контексты
                pseudo_context = torch.randn(32)  # размер контекста
                contexts.append(pseudo_context)
            
            recent_rewards = self.selection_rewards[-50:]
            self.selector.train_selector(recent_rewards, contexts)
            
            # Очищаем часть истории
            self.selection_rewards = self.selection_rewards[-50:]
    
    def get_statistics(self) -> Dict[str, Any]:
        """Возвращает статистику селектора"""
        return {
            "total_selections": self.total_selections,
            "policy_usage": self.selector.policy_usage_count.tolist(),
            "policy_performance": self.selector.policy_performance.tolist(),
            "current_policy": self.selector.current_policy.name,
            "avg_recent_reward": np.mean(self.selection_rewards[-20:]) if self.selection_rewards else 0.0
        }


# Интеграция с основным циклом обучения
class PolicySelectorCallback:
    """Callback для интеграции селектора политик в обучение"""
    
    def __init__(self, obs_space, act_space):
        self.obs_space = obs_space
        self.act_space = act_space
        self.selector_wrapper = None
        self.enabled = False
        
    def setup(self, algorithm, enable_selector: bool = True):
        """Настройка callback'а"""
        self.enabled = enable_selector
        
        if self.enabled:
            self.selector_wrapper = PolicySelectorWrapper(
                self.obs_space, self.act_space
            )
            
            # Добавляем специализированные политики к алгоритму
            for policy_id, policy_spec in self.selector_wrapper.specialized_policies.items():
                if policy_id not in algorithm.policies:
                    algorithm.add_policy(
                        policy_id=policy_id,
                        policy_cls=policy_spec[0],
                        observation_space=policy_spec[1],
                        action_space=policy_spec[2],
                        config=policy_spec[3]
                    )
            
            print(f"🎯 Policy Selector enabled with {len(self.selector_wrapper.specialized_policies)} specialized policies")
    
    def on_episode_start(self, *, worker, base_env, policies, episode, **kwargs):
        """Сброс состояния в начале эпизода"""
        if self.enabled and self.selector_wrapper:
            # Сброс статистики эпизода
            pass
    
    def on_episode_step(self, *, worker, base_env, episode, **kwargs):
        """Обработка шага эпизода"""
        if not self.enabled or not self.selector_wrapper:
            return
        
        # Здесь можно добавить дополнительную логику
        pass
    
    def on_episode_end(self, *, worker, base_env, policies, episode, **kwargs):
        """Обработка окончания эпизода"""
        if not self.enabled or not self.selector_wrapper:
            return
        
        # Обновляем производительность на основе результатов эпизода
        if hasattr(episode, 'agent_rewards'):
            rewards = {aid: sum(episode.agent_rewards[aid]) for aid in episode.agent_rewards}
            self.selector_wrapper.update_performance(rewards)
    
    def on_train_result(self, *, algorithm, result, **kwargs):
        """Обработка результатов обучения"""
        if not self.enabled or not self.selector_wrapper:
            return
        
        # Добавляем статистику селектора к результатам
        selector_stats = self.selector_wrapper.get_statistics()
        
        custom_metrics = result.get("custom_metrics", {})
        custom_metrics.update({
            f"policy_selector_{k}": v for k, v in selector_stats.items()
            if isinstance(v, (int, float))
        })
        
        # Логируем использование политик
        if selector_stats["total_selections"] % 100 == 0:
            print(f"\n🎯 Policy Selector Statistics:")
            print(f"   Current policy: {selector_stats['current_policy']}")
            print(f"   Total selections: {selector_stats['total_selections']}")
            print(f"   Average recent reward: {selector_stats['avg_recent_reward']:.3f}")
            
            usage = np.array(selector_stats["policy_usage"])
            if usage.sum() > 0:
                usage_pct = (usage / usage.sum()) * 100
                for i, pct in enumerate(usage_pct):
                    if pct > 0:
                        policy_name = PolicyType(i).name
                        print(f"   {policy_name}: {pct:.1f}%")
        
        result["custom_metrics"] = custom_metrics


# Утилиты для анализа политик
def analyze_policy_effectiveness(selector_wrapper: PolicySelectorWrapper, 
                               num_episodes: int = 100) -> Dict[str, Any]:
    """Анализирует эффективность различных политик"""
    
    print(f"📊 Analyzing policy effectiveness over {num_episodes} episodes...")
    
    analysis = {
        "policy_performance": selector_wrapper.selector.policy_performance.tolist(),
        "policy_usage": selector_wrapper.selector.policy_usage_count.tolist(),
        "total_selections": selector_wrapper.total_selections,
    }
    
    # Рассчитываем метрики эффективности
    performance = selector_wrapper.selector.policy_performance
    usage = selector_wrapper.selector.policy_usage_count
    
    # Нормализуем использование
    total_usage = usage.sum()
    usage_normalized = usage / max(total_usage, 1)
    
    # Находим наиболее эффективные политики
    policy_scores = performance * usage_normalized  # взвешенная эффективность
    
    analysis["policy_rankings"] = []
    for i in np.argsort(policy_scores)[::-1]:  # сортировка по убыванию
        if usage[i] > 0:  # только использованные политики
            policy_name = PolicyType(i).name
            analysis["policy_rankings"].append({
                "policy": policy_name,
                "performance": float(performance[i]),
                "usage_percent": float(usage_normalized[i] * 100),
                "weighted_score": float(policy_scores[i])
            })
    
    # Рекомендации по оптимизации
    analysis["recommendations"] = []
    
    # Недоиспользованные эффективные политики
    for i, score in enumerate(policy_scores):
        if performance[i] > np.mean(performance) and usage_normalized[i] < 0.1:
            policy_name = PolicyType(i).name
            analysis["recommendations"].append(
                f"Consider using {policy_name} more often - high performance but low usage"
            )
    
    # Переиспользованные неэффективные политики
    for i, score in enumerate(policy_scores):
        if performance[i] < np.mean(performance) and usage_normalized[i] > 0.3:
            policy_name = PolicyType(i).name
            analysis["recommendations"].append(
                f"Reduce usage of {policy_name} - low performance but high usage"
            )
    
    print("✅ Policy effectiveness analysis completed")
    return analysis

def create_policy_selector_config(base_config: Dict, enable_selector: bool = True) -> Dict:
    """Создает конфигурацию с интегрированным селектором политик"""
    
    if not enable_selector:
        return base_config
    
    # Модифицируем базовую конфигурацию
    config = base_config.copy()
    
    # Добавляем callback для селектора политик
    config["callbacks"] = PolicySelectorCallback
    
    # Добавляем специализированные политики
    obs_space = config.get("observation_space")
    act_space = config.get("action_space") 
    
    if obs_space and act_space:
        # Создаем временный wrapper для получения политик
        temp_wrapper = PolicySelectorWrapper(obs_space, act_space)
        specialized_policies = temp_wrapper.specialized_policies
        
        # Добавляем к мульти-агент конфигурации
        if "policies" in config:
            config["policies"].update(specialized_policies)
        
        # Добавляем к списку обучаемых политик
        if "policies_to_train" in config:
            specialized_ids = list(specialized_policies.keys())
            config["policies_to_train"].extend(specialized_ids)
    
    print(f"🎯 Policy selector configuration created")
    print(f"   Selector enabled: {enable_selector}")
    
    return config


if __name__ == "__main__":
    print("🎯 Policy Selector System for Air-to-Air Combat")
    print("\nFeatures:")
    print("- Adaptive policy selection based on situation analysis")
    print("- 6 specialized policies: Observer, Aggressive Shooter, Normal Shooter, Defensive, Interceptor, Support")
    print("- Neural network-based situation analyzer")
    print("- Performance tracking and learning")
    print("- Selection every 10 steps with exploration")
    print("- Integration with RLLib training pipeline")
    print("\nSpecialized Policies:")
    
    for policy_type in PolicyType:
        if policy_type.value >= 6:
            break
        print(f"  {policy_type.value + 1}. {policy_type.name}: ", end="")
        
        descriptions = {
            PolicyType.OBSERVER: "Passive reconnaissance, intelligence gathering",
            PolicyType.AGGRESSIVE_SHOOTER: "Active engagement, close-range combat",
            PolicyType.NORMAL_SHOOTER: "Balanced approach, general combat",
            PolicyType.DEFENSIVE: "Survival focus, evasive maneuvers",
            PolicyType.INTERCEPTOR: "Target pursuit, high-speed engagement",
            PolicyType.SUPPORT: "Team coordination, ally assistance"
        }
        
        print(descriptions.get(policy_type, "Unknown specialization"))