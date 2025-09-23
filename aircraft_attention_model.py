"""
Aircraft Attention Model - Трансформерная модель для управления самолетами
Адаптирована под JSBSim и воздушные бои с иерархическим управлением
ИСПРАВЛЕНА ВЕРСИЯ - добавлены недостающие импорты и исправлены ошибки
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Dict, Any
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.torch.torch_action_dist import TorchDistributionWrapper  # ИСПРАВЛЕНИЕ: добавлен импорт
from ray.rllib.models import ModelCatalog
import math

class AircraftPositionalEncoding(nn.Module):
    """Позиционное кодирование для авиационных данных"""
    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)
        
    def forward(self, x):
        pe = self.pe.to(x.device)
        return x + pe[:, :x.size(1), :]

class AviationMLP(nn.Module):
    """MLP для авиационных данных с дропаутом и нормализацией"""
    def __init__(self, dims: List[int], dropout: float = 0.1, act=nn.GELU):
        super().__init__()
        layers = []
        for i in range(len(dims)-1):
            layers += [
                nn.Linear(dims[i], dims[i+1]),
                nn.LayerNorm(dims[i+1]),
                act(),
                nn.Dropout(dropout)
            ]
        # Убираем последний dropout и activation
        if len(layers) >= 2:
            layers = layers[:-2]
        
        self.net = nn.Sequential(*layers)
        
        # Xavier инициализация
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
                
    def forward(self, x): 
        return self.net(x)

class AircraftMultiHeadAttention(nn.Module):
    """ONNX-совместимая multi-head attention для авиационных данных"""
    
    def __init__(self, d_model: int, nhead: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % nhead == 0
        
        self.d_model = d_model
        self.nhead = nhead
        self.d_k = d_model // nhead
        
        # Отдельные проекции для Q, K, V
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.d_k)
        
    def forward(self, query, key, value, key_padding_mask=None):
        batch_size, seq_len, d_model = query.size()
        
        # Проекции Q, K, V
        Q = self.w_q(query)
        K = self.w_k(key)
        V = self.w_v(value)
        
        # Reshape для multi-head: [B, L, D] -> [B, H, L, D/H]
        Q = Q.view(batch_size, seq_len, self.nhead, self.d_k).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.nhead, self.d_k).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.nhead, self.d_k).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        
        # Применяем маску если есть
        if key_padding_mask is not None:
            mask = key_padding_mask.unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(mask, float('-inf'))
        
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Применяем внимание к values
        attn_output = torch.matmul(attn_weights, V)
        
        # Concat heads: [B, H, L, D/H] -> [B, L, D]
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, d_model)
        
        output = self.w_o(attn_output)
        
        return output, attn_weights.mean(dim=1)  # Усредняем веса по головам

class AircraftAttentionBlock(nn.Module):
    """Блок трансформера для авиационных данных"""
    def __init__(self, d_model: int, nhead: int, dim_ff: int, dropout: float = 0.1):
        super().__init__()
        self.mha = AircraftMultiHeadAttention(d_model, nhead, dropout)
        self.ln1 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, dim_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_ff, d_model),
            nn.Dropout(dropout)
        )
        self.ln2 = nn.LayerNorm(d_model)
        self.last_attn = None
        
    def forward(self, x, key_padding_mask):
        # Self-attention с residual connection
        attn_out, attn_w = self.mha(x, x, x, key_padding_mask=key_padding_mask)
        self.last_attn = attn_w
        x = self.ln1(x + attn_out)
        
        # Feed-forward с residual connection
        ff_out = self.ff(x)
        x = self.ln2(x + ff_out)
        
        return x

class AircraftTransformerModel(TorchModelV2, nn.Module):
    """Трансформерная модель для управления самолетами"""
    
    def __init__(self, obs_space, action_space, num_outputs, model_config, name, **kwargs):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)

        cfg = model_config.get("custom_model_config", {})
        d_model = int(cfg.get("d_model", 256))
        nhead   = int(cfg.get("nhead", 8))
        layers  = int(cfg.get("layers", 3))
        ff      = int(cfg.get("ff", 512))
        hidden  = int(cfg.get("hidden", 256))
        dropout = float(cfg.get("dropout", 0.1))

        # Извлекаем размеры для авиации
        if hasattr(obs_space, 'spaces'):
            self_feats = obs_space["self"].shape[0]
            aircraft_shape = obs_space["allies"].shape
            enemies_shape = obs_space["enemies"].shape
            self.max_aircraft = aircraft_shape[0]
            self.max_enemies = enemies_shape[0]
            aircraft_feats = aircraft_shape[1]
            enemy_feats = enemies_shape[1]
            global_feats = obs_space["global_state"].shape[0]
        else:
            self.max_aircraft = int(cfg.get("max_aircraft", 8))
            self.max_enemies = int(cfg.get("max_enemies", 8))
            self_feats = 18
            aircraft_feats = 15
            enemy_feats = 13
            global_feats = 64

        # Сохраняем конфигурацию для экспорта
        self.d_model = d_model
        self.nhead = nhead
        self.layers = layers
        self.dropout = dropout

        # Энкодеры для авиационных данных
        self.self_encoder = AviationMLP([self_feats, d_model], dropout)
        self.aircraft_encoder = AviationMLP([aircraft_feats, d_model], dropout)
        self.enemy_encoder = AviationMLP([enemy_feats, d_model], dropout)
        
        # Специализированные энкодеры для авиационной специфики
        self.flight_dynamics_encoder = AviationMLP([6, d_model // 4], dropout)  # attitude + velocity
        self.tactical_encoder = AviationMLP([4, d_model // 4], dropout)  # weapons + fuel + hp
        
        # Трансформер блоки
        self.posenc = AircraftPositionalEncoding(d_model, max_len=max(self.max_aircraft * 2 + 1, 64))
        self.blocks = nn.ModuleList([
            AircraftAttentionBlock(d_model, nhead, ff, dropout) for _ in range(layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.last_attn = None

        # Иерархические выходные головы
        
        # Высокоуровневые тактические решения
        self.tactical_head = AviationMLP([d_model, hidden, hidden//2], dropout)
        self.target_head = nn.Linear(hidden//2, self.max_enemies)  # выбор цели
        self.maneuver_head = nn.Linear(hidden//2, 9)  # тип маневра
        self.weapon_head = nn.Linear(hidden//2, 4)    # тип оружия
        self.fire_head = nn.Linear(hidden//2, 1)      # решение о стрельбе
        
        # Низкоуровневое управление полетом
        self.flight_control_head = AviationMLP([d_model, hidden, hidden//2], dropout)
        self.aileron_head = nn.Linear(hidden//2, 1)   # элероны
        self.elevator_head = nn.Linear(hidden//2, 1)  # руль высоты
        self.rudder_head = nn.Linear(hidden//2, 1)    # руль направления
        self.throttle_head = nn.Linear(hidden//2, 1)  # газ
        
        # Специализированная value function для авиации
        self.value_net = AviationMLP([global_feats + d_model, hidden, 1], dropout)
        self._value_out: Optional[torch.Tensor] = None

    def _ensure_tensor_device(self, tensor, target_device):
        """Безопасное перемещение тензора на нужное устройство"""
        if isinstance(tensor, torch.Tensor):
            return tensor.to(target_device)
        elif isinstance(tensor, np.ndarray):
            return torch.from_numpy(tensor).to(target_device)
        else:
            return torch.tensor(tensor).to(target_device)

    def _ensure_obs_device_consistency(self, obs):
        """Убеждаемся что все наблюдения на одном устройстве"""
        target_device = next(self.parameters()).device
        
        obs_fixed = {}
        for key, value in obs.items():
            if isinstance(value, (torch.Tensor, np.ndarray)):
                obs_fixed[key] = self._ensure_tensor_device(value, target_device)
                
                if key in ["allies_mask", "enemies_mask", "enemy_action_mask"]:
                    obs_fixed[key] = obs_fixed[key].long()
                else:
                    obs_fixed[key] = obs_fixed[key].float()
            else:
                obs_fixed[key] = value
                
        return obs_fixed, target_device

    def _extract_aviation_features(self, obs, target_device):
        """Извлекает специализированные авиационные признаки"""
        batch_size = obs["self"].shape[0]
        
        # Извлекаем данные о собственном полете
        self_state = obs["self"]
        
        # Предполагаем структуру: [lat, lon, alt, phi, theta, psi, u, v, w, p, q, r, tas, mach, nx, ny, nz, fuel, hp]
        if self_state.shape[-1] >= 18:
            # Динамика полета (углы и скорости)
            flight_dynamics = self_state[:, 3:9]  # phi, theta, psi, u, v, w
            
            # Тактические параметры (топливо, здоровье, вооружение)
            tactical_params = self_state[:, -4:]  # последние 4 параметра
        else:
            # Fallback если структура другая
            flight_dynamics = self_state[:, :6] if self_state.shape[-1] >= 6 else torch.zeros(batch_size, 6, device=target_device)
            tactical_params = self_state[:, -4:] if self_state.shape[-1] >= 4 else torch.zeros(batch_size, 4, device=target_device)
        
        # Кодируем специализированные признаки
        flight_encoding = self.flight_dynamics_encoder(flight_dynamics)  # [B, d_model//4]
        tactical_encoding = self.tactical_encoder(tactical_params)       # [B, d_model//4]
        
        # Вычисляем относительные метрики
        allies_data = obs["allies"]
        enemies_data = obs["enemies"]
        
        # Среднее расстояние до союзников и врагов
        allies_mask = obs["allies_mask"].float().unsqueeze(-1)
        enemies_mask = obs["enemies_mask"].float().unsqueeze(-1)
        
        # Извлекаем относительные позиции (первые 3 компонента)
        allies_positions = allies_data[:, :, :3]
        enemies_positions = enemies_data[:, :, :3]
        
        allies_distances = torch.norm(allies_positions, dim=-1, keepdim=True)
        enemies_distances = torch.norm(enemies_positions, dim=-1, keepdim=True)
        
        avg_ally_dist = (allies_distances * allies_mask).sum(dim=1) / (allies_mask.sum(dim=1) + 1e-8)
        avg_enemy_dist = (enemies_distances * enemies_mask).sum(dim=1) / (enemies_mask.sum(dim=1) + 1e-8)
        
        # Высота и скорость (тактические преимущества)
        altitude = self_state[:, 2:3]  # нормализованная высота
        speed = torch.norm(self_state[:, 6:9], dim=-1, keepdim=True)  # скорость
        
        # Дополнительные авиационные признаки
        aviation_features = torch.cat([
            avg_ally_dist,    # [B, 1]
            avg_enemy_dist,   # [B, 1]
            altitude,         # [B, 1]
            speed             # [B, 1]
        ], dim=-1)  # [B, 4]
        
        return flight_encoding, tactical_encoding, aviation_features

    def forward(self, input_dict, state, seq_lens):
        raw_obs = input_dict["obs"]
        
        if isinstance(raw_obs, dict):
            obs = raw_obs
        else:
            obs = dict(raw_obs)
        
        # Убеждаемся что все тензоры на правильном устройстве
        obs, target_device = self._ensure_obs_device_consistency(obs)
        
        try:
            # Извлекаем авиационные признаки
            flight_encoding, tactical_encoding, aviation_features = self._extract_aviation_features(obs, target_device)
            
            # Энкодинг основных токенов
            self_tok = self.self_encoder(obs["self"])
            allies_tok = self.aircraft_encoder(obs["allies"])
            enemies_tok = self.enemy_encoder(obs["enemies"])
            
            # Добавляем специализированные авиационные кодировки к self токену
            aviation_encoding = torch.cat([
                flight_encoding, 
                tactical_encoding, 
                torch.zeros(flight_encoding.shape[0], 
                           self_tok.shape[1] - flight_encoding.shape[1] - tactical_encoding.shape[1], 
                           device=target_device)
            ], dim=1)
            
            self_tok = self_tok + aviation_encoding
            
            # Объединяем все токены
            x = torch.cat([self_tok.unsqueeze(1), allies_tok, enemies_tok], dim=1)

            # Создаем padding mask
            B = x.size(0)
            pad_self = torch.zeros(B, 1, dtype=torch.bool, device=target_device)
            am = obs["allies_mask"] > 0
            em = obs["enemies_mask"] > 0
            pad_mask = torch.cat([pad_self, ~am, ~em], dim=1)

            # Positional encoding + трансформер блоки
            x = self.posenc(x)
            for blk in self.blocks:
                x = blk(x, key_padding_mask=pad_mask)
            self.last_attn = self.blocks[-1].last_attn if self.blocks else None
            x = self.norm(x)

            # Self-токен как агрегированное представление
            h = x[:, 0, :]

            # Иерархические выходы
            
            # Тактический уровень
            tactical_features = self.tactical_head(h)
            logits_target = self.target_head(tactical_features)
            logits_maneuver = self.maneuver_head(tactical_features)
            logits_weapon = self.weapon_head(tactical_features)
            logit_fire = self.fire_head(tactical_features)
            
            # Применяем маску для целей
            mask = obs["enemy_action_mask"].float()
            inf_mask = (1.0 - mask) * torch.finfo(logits_target.dtype).min
            masked_target_logits = logits_target + inf_mask
            
            # Управление полетом
            flight_features = self.flight_control_head(h)
            aileron = torch.tanh(self.aileron_head(flight_features))
            elevator = torch.tanh(self.elevator_head(flight_features))
            rudder = torch.tanh(self.rudder_head(flight_features))
            throttle = torch.sigmoid(self.throttle_head(flight_features))  # 0-1 для газа
            
            # Склейка всех выходов
            out = torch.cat([
                masked_target_logits,  # target selection
                logits_maneuver,       # maneuver type
                logits_weapon,         # weapon type
                logit_fire,            # fire decision
                aileron,               # aileron control
                elevator,              # elevator control  
                rudder,                # rudder control
                throttle               # throttle control
            ], dim=-1)

            # Value function с учетом авиационной специфики
            global_with_aviation = torch.cat([obs["global_state"], h], dim=-1)
            v = self.value_net(global_with_aviation).squeeze(-1)
            self._value_out = v
            
            return out, state
            
        except Exception as e:
            print(f"ERROR in Aircraft Transformer forward: {e}")
            print(f"Observation shapes: {[(k, v.shape if hasattr(v, 'shape') else type(v)) for k, v in obs.items()]}")
            raise

    def value_function(self):
        return self._value_out

    def get_attention_weights(self):
        """Возвращает веса внимания для анализа"""
        if self.last_attn is not None:
            return self.last_attn.detach().cpu().numpy()
        return None

    def get_aircraft_action_dim(self):
        """Возвращает размерность действий для самолета"""
        # target + maneuver + weapon + fire + aileron + elevator + rudder + throttle
        return self.max_enemies + 9 + 4 + 1 + 1 + 1 + 1 + 1

# Дистрибуция действий для авиации
class AircraftActionDistribution(TorchDistributionWrapper):
    """Дистрибуция действий для самолетов с иерархическим управлением"""
    
    def __init__(self, inputs, model):
        super().__init__(inputs, model)
        
        # Получаем размеры из модели
        self.max_enemies = getattr(model, 'max_enemies', 8)
        
        # Распаковываем входы
        idx = 0
        
        # Тактические решения
        target_logits = inputs[..., idx:idx+self.max_enemies]; idx += self.max_enemies
        maneuver_logits = inputs[..., idx:idx+9]; idx += 9
        weapon_logits = inputs[..., idx:idx+4]; idx += 4
        fire_logits = inputs[..., idx:idx+1]; idx += 1
        
        # Управление полетом (непрерывные действия)
        aileron = inputs[..., idx:idx+1]; idx += 1
        elevator = inputs[..., idx:idx+1]; idx += 1
        rudder = inputs[..., idx:idx+1]; idx += 1
        throttle = inputs[..., idx:idx+1]; idx += 1
        
        # Создаем дистрибуции
        from torch.distributions import Categorical, Bernoulli
        
        self.target_dist = Categorical(logits=target_logits)
        self.maneuver_dist = Categorical(logits=maneuver_logits)
        self.weapon_dist = Categorical(logits=weapon_logits)
        self.fire_dist = Bernoulli(logits=fire_logits)
        
        # Для управления полетом используем ограниченные распределения
        self.aileron_val = aileron
        self.elevator_val = elevator
        self.rudder_val = rudder
        self.throttle_val = throttle
        
        # Для совместимости с RLLib
        self.dist = self.target_dist
        self.last_sample = None

    @staticmethod
    def required_model_output_shape(action_space, model_config):
        """Размер выхода модели для авиационных действий"""
        cfg = model_config.get("custom_model_config", {})
        max_enemies = cfg.get("max_enemies", 8)
        
        # target + maneuver + weapon + fire + flight_controls(4)
        return max_enemies + 9 + 4 + 1 + 4

    def _convert_to_dict(self, tensor_action):
        """Преобразует тензор в словарь действий"""
        if tensor_action.is_cuda:
            tensor_action = tensor_action.cpu()
        
        if tensor_action.dim() == 1:
            tensor_action = tensor_action.unsqueeze(0)
            squeeze_batch = True
        else:
            squeeze_batch = False
        
        # Извлекаем компоненты
        target = tensor_action[..., 0].long()
        maneuver = tensor_action[..., 1].long()
        weapon = tensor_action[..., 2].long()
        fire = tensor_action[..., 3].long()
        aileron = tensor_action[..., 4:5]
        elevator = tensor_action[..., 5:6]
        rudder = tensor_action[..., 6:7]
        throttle = tensor_action[..., 7:8]
        
        result = {
            "target": target.numpy(),
            "maneuver": maneuver.numpy(),
            "weapon": weapon.numpy(),
            "fire": fire.numpy(),
            "aileron": aileron.numpy(),
            "elevator": elevator.numpy(),
            "rudder": rudder.numpy(),
            "throttle": throttle.numpy(),
        }
        
        if squeeze_batch:
            result = {k: v[0] for k, v in result.items()}
        
        return result

    def sample(self):
        """Сэмплирует действия"""
        target = self.target_dist.sample().float()
        maneuver = self.maneuver_dist.sample().float()
        weapon = self.weapon_dist.sample().float()
        fire = self.fire_dist.sample().float()
        
        # Управление полетом (уже ограничено моделью)
        aileron = self.aileron_val.squeeze(-1)
        elevator = self.elevator_val.squeeze(-1)
        rudder = self.rudder_val.squeeze(-1)
        throttle = self.throttle_val.squeeze(-1)
        
        # Объединяем
        flat_action = torch.cat([
            target.unsqueeze(-1), maneuver.unsqueeze(-1), 
            weapon.unsqueeze(-1), fire.unsqueeze(-1),
            aileron.unsqueeze(-1), elevator.unsqueeze(-1),
            rudder.unsqueeze(-1), throttle.unsqueeze(-1)
        ], dim=-1)
        
        self.last_sample = flat_action
        return self._convert_to_dict(flat_action)

    def deterministic_sample(self):
        """Детерминированный сэмпл"""
        target = torch.argmax(self.target_dist.logits, dim=-1).float()
        maneuver = torch.argmax(self.maneuver_dist.logits, dim=-1).float()
        weapon = torch.argmax(self.weapon_dist.logits, dim=-1).float()
        fire = (self.fire_dist.logits > 0).float()
        
        # Управление полетом
        aileron = self.aileron_val.squeeze(-1)
        elevator = self.elevator_val.squeeze(-1)
        rudder = self.rudder_val.squeeze(-1)
        throttle = self.throttle_val.squeeze(-1)
        
        flat_action = torch.cat([
            target.unsqueeze(-1), maneuver.unsqueeze(-1),
            weapon.unsqueeze(-1), fire.unsqueeze(-1),
            aileron.unsqueeze(-1), elevator.unsqueeze(-1),
            rudder.unsqueeze(-1), throttle.unsqueeze(-1)
        ], dim=-1)
        
        self.last_sample = flat_action
        return self._convert_to_dict(flat_action)

    def logp(self, x):
        """Вычисляет log probability"""
        if isinstance(x, dict):
            # Конвертируем словарь в тензор
            target = torch.tensor(x["target"]).long()
            maneuver = torch.tensor(x["maneuver"]).long()
            weapon = torch.tensor(x["weapon"]).long()
            fire = torch.tensor(x["fire"]).float()
            aileron = torch.tensor(x["aileron"]).float()
            elevator = torch.tensor(x["elevator"]).float()
            rudder = torch.tensor(x["rudder"]).float()
            throttle = torch.tensor(x["throttle"]).float()
            
            x = torch.cat([
                target.float().unsqueeze(-1), maneuver.float().unsqueeze(-1),
                weapon.float().unsqueeze(-1), fire.unsqueeze(-1),
                aileron.unsqueeze(-1), elevator.unsqueeze(-1),
                rudder.unsqueeze(-1), throttle.unsqueeze(-1)
            ], dim=-1)
        elif isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float()
        
        if x.device != self.target_dist.logits.device:
            x = x.to(self.target_dist.logits.device)
        
        # Извлекаем компоненты
        target_idx = x[..., 0].long()
        maneuver_idx = x[..., 1].long()
        weapon_idx = x[..., 2].long()
        fire_val = x[..., 3]
        aileron_val = x[..., 4]
        elevator_val = x[..., 5]
        rudder_val = x[..., 6]
        throttle_val = x[..., 7]
        
        # Вычисляем log probabilities
        lp_target = self.target_dist.log_prob(target_idx)
        lp_maneuver = self.maneuver_dist.log_prob(maneuver_idx)
        lp_weapon = self.weapon_dist.log_prob(weapon_idx)
        
        # Fire probability
        fire_p = torch.sigmoid(self.fire_dist.logits.squeeze(-1))
        lp_fire = torch.where(fire_val > 0.5, 
                             torch.log(fire_p + 1e-8), 
                             torch.log(1 - fire_p + 1e-8))
        
        # Для непрерывных управляющих действий используем простую гауссову оценку
        lp_aileron = -0.5 * (aileron_val - self.aileron_val.squeeze(-1)).pow(2)
        lp_elevator = -0.5 * (elevator_val - self.elevator_val.squeeze(-1)).pow(2)
        lp_rudder = -0.5 * (rudder_val - self.rudder_val.squeeze(-1)).pow(2)
        lp_throttle = -0.5 * (throttle_val - self.throttle_val.squeeze(-1)).pow(2)
        
        return (lp_target + lp_maneuver + lp_weapon + lp_fire + 
                lp_aileron + lp_elevator + lp_rudder + lp_throttle)

    def sampled_action_logp(self):
        if self.last_sample is None:
            self.sample()
        return self.logp(self.last_sample)

    def entropy(self):
        """Вычисляем энтропию"""
        target_H = self.target_dist.entropy()
        maneuver_H = self.maneuver_dist.entropy()
        weapon_H = self.weapon_dist.entropy()
        
        fire_p = torch.sigmoid(self.fire_dist.logits.squeeze(-1))
        fire_H = -(fire_p * torch.log(fire_p + 1e-8) + (1-fire_p) * torch.log(1-fire_p + 1e-8))
        
        # Для непрерывных действий добавляем фиксированную энтропию
        flight_H = torch.ones_like(target_H) * 2.0  # константная энтропия для управления полетом
        
        return target_H + maneuver_H + weapon_H + fire_H + flight_H

# Регистрируем модель и дистрибуцию
ModelCatalog.register_custom_model("aircraft_transformer", AircraftTransformerModel)
ModelCatalog.register_custom_action_dist("aircraft_actions", AircraftActionDistribution)