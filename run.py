import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical, Normal
import numpy as np
import random
from collections import deque
import math
import importlib
import os
import glob
import inspect
from typing import Dict, Any, List, Callable, Optional, Tuple
from dataclasses import dataclass
from abc import ABC, abstractmethod
import threading
import copy
import warnings

# ==============================
# 1. Core Interfaces & Base Classes
# ==============================

@dataclass
class ObjectiveConfig:
    """Configuration for a dynamic objective module."""
    name: str
    enabled: bool = True
    weight: float = 1.0
    mode: str = "fixed"  # "fixed" or "adaptive"
    threshold: float = 0.0
    window_size: int = 100
    decay_rate: float = 0.95
    use_scaling: bool = True
    use_normalization: bool = True
    priority: int = 0
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

class ObjectiveModule(ABC):
    """
    Abstract base class for all dynamic objective modules.
    Defines the contract for reward computation, configuration, and state management.
    """

    @abstractmethod
    def validate(self) -> bool:
        """Validate the objective's configuration and dependencies."""
        pass

    @abstractmethod
    def execute(self, state: torch.Tensor, action: torch.Tensor, next_state: torch.Tensor,
                log_prob: torch.Tensor, old_log_prob: torch.Tensor, value: torch.Tensor,
                old_value: torch.Tensor, episode_data: Dict[str, Any]) -> torch.Tensor:
        """
        Compute the objective-specific reward.
        Returns: scalar tensor of reward value.
        """
        pass

    @abstractmethod
    def reset(self):
        """Reset internal state of the objective."""
        pass

    @abstractmethod
    def get_state(self) -> Dict[str, Any]:
        """Return current internal state for serialization."""
        pass

    @abstractmethod
    def set_state(self, state: Dict[str, Any]):
        """Restore internal state from serialized form."""
        pass

    @abstractmethod
    def get_config(self) -> ObjectiveConfig:
        """Return current configuration."""
        pass


# ==============================
# 2. Objective Registry & Plugin System
# ==============================

class ObjectiveRegistry:
    """
    Centralized registry for managing all objective modules.
    Supports dynamic discovery, loading, and composition.
    """

    def __init__(self):
        self._registry: Dict[str, type] = {}
        self._instances: Dict[str, ObjectiveModule] = {}
        self._lock = threading.RLock()
        self._discovery_paths: List[str] = ['plugins/', 'objectives/']
        self._plugin_cache: Dict[str, Any] = {}

    def register(self, name: str, module_class: type):
        """Register a new objective module class."""
        if not issubclass(module_class, ObjectiveModule):
            raise TypeError(f"Class {module_class.__name__} must inherit from ObjectiveModule")
        if name in self._registry:
            warnings.warn(f"Objective '{name}' already registered. Overwriting.")
        self._registry[name] = module_class

    def get(self, name: str) -> Optional[ObjectiveModule]:
        """Retrieve an instance of a registered objective."""
        with self._lock:
            if name not in self._instances:
                if name not in self._registry:
                    return None
                module_class = self._registry[name]
                instance = module_class()
                self._instances[name] = instance
            return self._instances[name]

    def list_all(self) -> List[str]:
        """List all registered objective names."""
        return list(self._registry.keys())

    def discover_and_load(self, paths: Optional[List[str]] = None):
        """Discover and load objective modules from external directories."""
        paths = paths or self._discovery_paths
        loaded = 0
        for path in paths:
            if not os.path.exists(path):
                continue
            for file_path in glob.glob(os.path.join(path, "*.py")):
                module_name = os.path.basename(file_path)[:-3]
                if module_name.startswith("__"):
                    continue
                try:
                    spec = importlib.util.spec_from_file_location(module_name, file_path)
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)
                    # Look for classes that inherit from ObjectiveModule
                    for name, obj in inspect.getmembers(module, inspect.isclass):
                        if issubclass(obj, ObjectiveModule) and obj is not ObjectiveModule:
                            self.register(name, obj)
                            loaded += 1
                except Exception as e:
                    warnings.warn(f"Failed to load plugin from {file_path}: {e}")
        print(f"Discovered and loaded {loaded} objective modules.")

    def clear(self):
        """Clear all registered and instantiated objectives."""
        with self._lock:
            self._registry.clear()
            self._instances.clear()

    def get_active(self, config: Dict[str, ObjectiveConfig]) -> List[ObjectiveModule]:
        """Return list of active objective instances based on config."""
        active = []
        for name, cfg in config.items():
            if cfg.enabled:
                obj = self.get(name)
                if obj is not None:
                    active.append(obj)
                else:
                    warnings.warn(f"Objective '{name}' not found in registry.")
        return active

    def get_by_priority(self, config: Dict[str, ObjectiveConfig]) -> List[ObjectiveModule]:
        """Return active objectives sorted by priority."""
        active = self.get_active(config)
        return sorted(active, key=lambda x: x.get_config().priority, reverse=True)


# ==============================
# 3. Baseline Manager (Enhanced)
# ==============================

class BaselineManager:
    """
    Dual-mode baseline manager supporting fixed and adaptive baselines.
    Integrates with objective modules for reward scaling.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self._baseline_data: deque = deque(maxlen=config.get("baseline_size", 1000))
        self._centroid: Optional[torch.Tensor] = None
        self._covariance: Optional[torch.Tensor] = None
        self._is_initialized = False
        self._lock = threading.Lock()
        self._last_update = 0
        self._update_freq = config.get("baseline_update_freq", 100)
        self._use_fixed_baseline = config.get("use_fixed_baseline", False)
        self._fixed_baseline_value = config.get("fixed_baseline_value", 0.0)
        self._decay_rate = config.get("decay_rate", 0.95)
        self._running_mean = 0.0
        self._running_var = 1.0
        self._n = 0

    def add(self, data: torch.Tensor):
        """Add new state or reward data to baseline."""
        with self._lock:
            self._baseline_data.append(data.detach().cpu().numpy())
            self._last_update += 1
            if self._last_update % self._update_freq == 0:
                self._update_manifold()

    def _update_manifold(self):
        """Compute centroid and covariance from baseline data."""
        if len(self._baseline_data) < 2:
            return
        data_array = np.array(self._baseline_data)
        self._centroid = torch.tensor(data_array.mean(axis=0), dtype=torch.float32)
        self._covariance = torch.tensor(np.cov(data_array, rowvar=False), dtype=torch.float32)
        self._is_initialized = True

    def get_centroid(self) -> Optional[torch.Tensor]:
        """Return baseline centroid."""
        return self._centroid

    def get_covariance(self) -> Optional[torch.Tensor]:
        """Return baseline covariance matrix."""
        return self._covariance

    def get_manifold(self) -> np.ndarray:
        """Return stored baseline data."""
        return np.array(self._baseline_data)

    def get_similarity_score(self, x: torch.Tensor) -> torch.Tensor:
        """Compute Mahalanobis distance-based similarity to baseline."""
        if not self._is_initialized or self._centroid is None or self._covariance is None:
            return torch.tensor(0.0, device=x.device)
        diff = x - self._centroid
        inv_cov = torch.inverse(self._covariance + 1e-6 * torch.eye(self._covariance.size(0)))
        return torch.exp(-0.5 * diff @ inv_cov @ diff.t()).squeeze()

    def get_proximity_score(self, x: torch.Tensor, baseline_pair: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        """Compute kernel-based proximity between x and two baseline points."""
        if len(baseline_pair) != 2:
            return torch.tensor(0.0, device=x.device)
        a, b = baseline_pair
        dist_a = torch.norm(x - a, dim=-1)
        dist_b = torch.norm(x - b, dim=-1)
        return torch.exp(-self.config.get("kernel_bandwidth", 1.0) * (dist_a + dist_b))

    def get_adaptive_baseline(self, reward: float) -> float:
        """Compute adaptive baseline using exponential smoothing."""
        self._n += 1
        delta = reward - self._running_mean
        self._running_mean += delta / self._n
        self._running_var += delta * (reward - self._running_mean)
        return self._running_mean

    def get_fixed_baseline(self) -> float:
        """Return fixed baseline value."""
        return self._fixed_baseline_value

    def get_baseline_value(self, reward: float, mode: str = "adaptive") -> float:
        """Get current baseline value based on mode."""
        if self._use_fixed_baseline:
            return self.get_fixed_baseline()
        return self.get_adaptive_baseline(reward)

    def get_scaled_reward(self, raw_reward: torch.Tensor, baseline_mode: str = "adaptive") -> torch.Tensor:
        """Scale raw reward using baseline and standard deviation."""
        if not self._is_initialized:
            return raw_reward

        # Use adaptive baseline if enabled
        baseline = self.get_baseline_value(raw_reward.item(), mode=baseline_mode)
        std = torch.sqrt(torch.tensor(self._running_var + 1e-6))
        scaled = (raw_reward - torch.tensor(baseline)) / (std + 1e-6)
        return scaled


# ==============================
# 4. Individual Objective Modules
# ==============================

class ExplorationObjective(ObjectiveModule):
    """Reward based on action entropy (exploration)."""

    def __init__(self, config: ObjectiveConfig):
        self.config = config
        self._entropy_buffer = deque(maxlen=self.config.window_size)

    def validate(self) -> bool:
        return isinstance(self.config, ObjectiveConfig) and self.config.enabled

    def execute(self, state: torch.Tensor, action: torch.Tensor, next_state: torch.Tensor,
                log_prob: torch.Tensor, old_log_prob: torch.Tensor, value: torch.Tensor,
                old_value: torch.Tensor, episode_data: Dict[str, Any]) -> torch.Tensor:
        if not self.config.enabled:
            return torch.tensor(0.0, device=state.device)
        # Compute entropy of action distribution
        if isinstance(action, torch.Tensor):
            dist = Normal(action, torch.ones_like(action) * 0.1)
            entropy = dist.entropy().mean()
        else:
            dist = Categorical(logits=action)
            entropy = dist.entropy().mean()
        reward = entropy * self.config.weight
        if self.config.use_scaling:
            reward = self.get_scaled_reward(reward)
        return reward

    def reset(self):
        self._entropy_buffer.clear()

    def get_state(self) -> Dict[str, Any]:
        return {"entropy_buffer": list(self._entropy_buffer)}

    def set_state(self, state: Dict[str, Any]):
        self._entropy_buffer = deque(state.get("entropy_buffer", []), maxlen=self.config.window_size)

    def get_config(self) -> ObjectiveConfig:
        return self.config

    def get_scaled_reward(self, raw_reward: torch.Tensor) -> torch.Tensor:
        return raw_reward


class ExploitationObjective(ObjectiveModule):
    """Reward based on high-value state exploitation."""

    def __init__(self, config: ObjectiveConfig):
        self.config = config
        self._value_buffer = deque(maxlen=self.config.window_size)

    def validate(self) -> bool:
        return isinstance(self.config, ObjectiveConfig) and self.config.enabled

    def execute(self, state: torch.Tensor, action: torch.Tensor, next_state: torch.Tensor,
                log_prob: torch.Tensor, old_log_prob: torch.Tensor, value: torch.Tensor,
                old_value: torch.Tensor, episode_data: Dict[str, Any]) -> torch.Tensor:
        if not self.config.enabled:
            return torch.tensor(0.0, device=state.device)
        # Reward high value estimates
        reward = value.mean() * self.config.weight
        if self.config.use_scaling:
            reward = self.get_scaled_reward(reward)
        return reward

    def reset(self):
        self._value_buffer.clear()

    def get_state(self) -> Dict[str, Any]:
        return {"value_buffer": list(self._value_buffer)}

    def set_state(self, state: Dict[str, Any]):
        self._value_buffer = deque(state.get("value_buffer", []), maxlen=self.config.window_size)

    def get_config(self) -> ObjectiveConfig:
        return self.config

    def get_scaled_reward(self, raw_reward: torch.Tensor) -> torch.Tensor:
        return raw_reward


class ExtrapolationObjective(ObjectiveModule):
    """Reward for extrapolating from baseline trajectories."""

    def __init__(self, config: ObjectiveConfig, baseline_manager: BaselineManager):
        self.config = config
        self.baseline_manager = baseline_manager
        self._extrapolation_buffer = deque(maxlen=self.config.window_size)

    def validate(self) -> bool:
        return isinstance(self.config, ObjectiveConfig) and self.config.enabled and self.baseline_manager is not None

    def execute(self, state: torch.Tensor, action: torch.Tensor, next_state: torch.Tensor,
                log_prob: torch.Tensor, old_log_prob: torch.Tensor, value: torch.Tensor,
                old_value: torch.Tensor, episode_data: Dict[str, Any]) -> torch.Tensor:
        if not self.config.enabled:
            return torch.tensor(0.0, device=state.device)
        if self.baseline_manager.get_centroid() is None:
            return torch.tensor(0.0, device=state.device)
        # Compute distance from centroid
        centroid = self.baseline_manager.get_centroid()
        dist = torch.norm(state - centroid, dim=-1).mean()
        reward = torch.exp(-dist) * self.config.weight
        if self.config.use_scaling:
            reward = self.get_scaled_reward(reward)
        return reward

    def reset(self):
        self._extrapolation_buffer.clear()

    def get_state(self) -> Dict[str, Any]:
        return {"extrapolation_buffer": list(self._extrapolation_buffer)}

    def set_state(self, state: Dict[str, Any]):
        self._extrapolation_buffer = deque(state.get("extrapolation_buffer", []), maxlen=self.config.window_size)

    def get_config(self) -> ObjectiveConfig:
        return self.config

    def get_scaled_reward(self, raw_reward: torch.Tensor) -> torch.Tensor:
        return raw_reward


class SpreadingObjective(ObjectiveModule):
    """Reward for spreading data away from baseline centroid."""

    def __init__(self, config: ObjectiveConfig, baseline_manager: BaselineManager):
        self.config = config
        self.baseline_manager = baseline_manager

    def validate(self) -> bool:
        return isinstance(self.config, ObjectiveConfig) and self.config.enabled and self.baseline_manager is not None

    def execute(self, state: torch.Tensor, action: torch.Tensor, next_state: torch.Tensor,
                log_prob: torch.Tensor, old_log_prob: torch.Tensor, value: torch.Tensor,
                old_value: torch.Tensor, episode_data: Dict[str, Any]) -> torch.Tensor:
        if not self.config.enabled:
            return torch.tensor(0.0, device=state.device)
        if self.baseline_manager.get_centroid() is None:
            return torch.tensor(0.0, device=state.device)
        centroid = self.baseline_manager.get_centroid()
        dist = torch.norm(state - centroid, dim=-1).mean()
        reward = dist * self.config.weight
        if self.config.use_scaling:
            reward = self.get_scaled_reward(reward)
        return reward

    def reset(self):
        pass

    def get_state(self) -> Dict[str, Any]:
        return {}

    def set_state(self, state: Dict[str, Any]):
        pass

    def get_config(self) -> ObjectiveConfig:
        return self.config

    def get_scaled_reward(self, raw_reward: torch.Tensor) -> torch.Tensor:
        return raw_reward


class GatheringObjective(ObjectiveModule):
    """Reward for gathering data near baseline centroid."""

    def __init__(self, config: ObjectiveConfig, baseline_manager: BaselineManager):
        self.config = config
        self.baseline_manager = baseline_manager

    def validate(self) -> bool:
        return isinstance(self.config, ObjectiveConfig) and self.config.enabled and self.baseline_manager is not None

    def execute(self, state: torch.Tensor, action: torch.Tensor, next_state: torch.Tensor,
                log_prob: torch.Tensor, old_log_prob: torch.Tensor, value: torch.Tensor,
                old_value: torch.Tensor, episode_data: Dict[str, Any]) -> torch.Tensor:
        if not self.config.enabled:
            return torch.tensor(0.0, device=state.device)
        if self.baseline_manager.get_centroid() is None:
            return torch.tensor(0.0, device=state.device)
        centroid = self.baseline_manager.get_centroid()
        dist = torch.norm(state - centroid, dim=-1).mean()
        reward = torch.exp(-dist) * self.config.weight
        if self.config.use_scaling:
            reward = self.get_scaled_reward(reward)
        return reward

    def reset(self):
        pass

    def get_state(self) -> Dict[str, Any]:
        return {}

    def set_state(self, state: Dict[str, Any]):
        pass

    def get_config(self) -> ObjectiveConfig:
        return self.config

    def get_scaled_reward(self, raw_reward: torch.Tensor) -> torch.Tensor:
        return raw_reward


class SelectionObjective(ObjectiveModule):
    """Reward for selecting states close to previous ones."""

    def __init__(self, config: ObjectiveConfig):
        self.config = config
        self._prev_state = None

    def validate(self) -> bool:
        return isinstance(self.config, ObjectiveConfig) and self.config.enabled

    def execute(self, state: torch.Tensor, action: torch.Tensor, next_state: torch.Tensor,
                log_prob: torch.Tensor, old_log_prob: torch.Tensor, value: torch.Tensor,
                old_value: torch.Tensor, episode_data: Dict[str, Any]) -> torch.Tensor:
        if not self.config.enabled:
            return torch.tensor(0.0, device=state.device)
        if self._prev_state is None:
            self._prev_state = state.clone()
            return torch.tensor(0.0, device=state.device)
        dist = torch.norm(state - self._prev_state, dim=-1).mean()
        reward = torch.exp(-dist) * self.config.weight
        self._prev_state = state.clone()
        if self.config.use_scaling:
            reward = self.get_scaled_reward(reward)
        return reward

    def reset(self):
        self._prev_state = None

    def get_state(self) -> Dict[str, Any]:
        return {"prev_state": self._prev_state.cpu().numpy() if self._prev_state is not None else None}

    def set_state(self, state: Dict[str, Any]):
        self._prev_state = torch.tensor(state.get("prev_state"), device="cuda") if state.get("prev_state") is not None else None

    def get_config(self) -> ObjectiveConfig:
        return self.config

    def get_scaled_reward(self, raw_reward: torch.Tensor) -> torch.Tensor:
        return raw_reward


class SimilarityObjective(ObjectiveModule):
    """Reward for metric similarity to baseline elements."""

    def __init__(self, config: ObjectiveConfig, baseline_manager: BaselineManager):
        self.config = config
        self.baseline_manager = baseline_manager

    def validate(self) -> bool:
        return isinstance(self.config, ObjectiveConfig) and self.config.enabled and self.baseline_manager is not None

    def execute(self, state: torch.Tensor, action: torch.Tensor, next_state: torch.Tensor,
                log_prob: torch.Tensor, old_log_prob: torch.Tensor, value: torch.Tensor,
                old_value: torch.Tensor, episode_data: Dict[str, Any]) -> torch.Tensor:
        if not self.config.enabled:
            return torch.tensor(0.0, device=state.device)
        if self.baseline_manager.get_centroid() is None:
            return torch.tensor(0.0, device=state.device)
        similarity = self.baseline_manager.get_similarity_score(state)
        reward = similarity * self.config.weight
        if self.config.use_scaling:
            reward = self.get_scaled_reward(reward)
        return reward

    def reset(self):
        pass

    def get_state(self) -> Dict[str, Any]:
        return {}

    def set_state(self, state: Dict[str, Any]):
        pass

    def get_config(self) -> ObjectiveConfig:
        return self.config

    def get_scaled_reward(self, raw_reward: torch.Tensor) -> torch.Tensor:
        return raw_reward


class InterpolationObjective(ObjectiveModule):
    """Reward for interpolating data to find similar trajectories."""

    def __init__(self, config: ObjectiveConfig, baseline_manager: BaselineManager):
        self.config = config
        self.baseline_manager = baseline_manager

    def validate(self) -> bool:
        return isinstance(self.config, ObjectiveConfig) and self.config.enabled and self.baseline_manager is not None

    def execute(self, state: torch.Tensor, action: torch.Tensor, next_state: torch.Tensor,
                log_prob: torch.Tensor, old_log_prob: torch.Tensor, value: torch.Tensor,
                old_value: torch.Tensor, episode_data: Dict[str, Any]) -> torch.Tensor:
        if not self.config.enabled:
            return torch.tensor(0.0, device=state.device)
        if len(self.baseline_manager.get_manifold()) < 2:
            return torch.tensor(0.0, device=state.device)
        # Simple interpolation: reward if state is between two baseline points
        data = self.baseline_manager.get_manifold()
        if len(data) < 2:
            return torch.tensor(0.0, device=state.device)
        a, b = torch.tensor(data[0]), torch.tensor(data[1])
        # Compute proximity to line segment
        v = b - a
        proj = torch.dot(state - a, v) / torch.dot(v, v)
        dist = torch.norm(state - a - proj * v, dim=-1)
        reward = torch.exp(-dist) * self.config.weight
        if self.config.use_scaling:
            reward = self.get_scaled_reward(reward)
        return reward

    def reset(self):
        pass

    def get_state(self) -> Dict[str, Any]:
        return {}

    def set_state(self, state: Dict[str, Any]):
        pass

    def get_config(self) -> ObjectiveConfig:
        return self.config

    def get_scaled_reward(self, raw_reward: torch.Tensor) -> torch.Tensor:
        return raw_reward


class IntegrationObjective(ObjectiveModule):
    """Reward for exploration while integrating data into learned model."""

    def __init__(self, config: ObjectiveConfig):
        self.config = config
        self._integration_buffer = deque(maxlen=self.config.window_size)

    def validate(self) -> bool:
        return isinstance(self.config, ObjectiveConfig) and self.config.enabled

    def execute(self, state: torch.Tensor, action: torch.Tensor, next_state: torch.Tensor,
                log_prob: torch.Tensor, old_log_prob: torch.Tensor, value: torch.Tensor,
                old_value: torch.Tensor, episode_data: Dict[str, Any]) -> torch.Tensor:
        if not self.config.enabled:
            return torch.tensor(0.0, device=state.device)
        # Reward high entropy + high value
        entropy = torch.mean(-log_prob * torch.exp(log_prob))
        value_reward = value.mean()
        reward = (entropy + value_reward) * self.config.weight
        if self.config.use_scaling:
            reward = self.get_scaled_reward(reward)
        return reward

    def reset(self):
        self._integration_buffer.clear()

    def get_state(self) -> Dict[str, Any]:
        return {"integration_buffer": list(self._integration_buffer)}

    def set_state(self, state: Dict[str, Any]):
        self._integration_buffer = deque(state.get("integration_buffer", []), maxlen=self.config.window_size)

    def get_config(self) -> ObjectiveConfig:
        return self.config

    def get_scaled_reward(self, raw_reward: torch.Tensor) -> torch.Tensor:
        return raw_reward


class GatheringMetricObjective(ObjectiveModule):
    """Reward for gathering known data from baseline to match environment."""

    def __init__(self, config: ObjectiveConfig, baseline_manager: BaselineManager):
        self.config = config
        self.baseline_manager = baseline_manager

    def validate(self) -> bool:
        return isinstance(self.config, ObjectiveConfig) and self.config.enabled and self.baseline_manager is not None

    def execute(self, state: torch.Tensor, action: torch.Tensor, next_state: torch.Tensor,
                log_prob: torch.Tensor, old_log_prob: torch.Tensor, value: torch.Tensor,
                old_value: torch.Tensor, episode_data: Dict[str, Any]) -> torch.Tensor:
        if not self.config.enabled:
            return torch.tensor(0.0, device=state.device)
        if self.baseline_manager.get_centroid() is None:
            return torch.tensor(0.0, device=state.device)
        centroid = self.baseline_manager.get_centroid()
        dist = torch.norm(state - centroid, dim=-1).mean()
        reward = torch.exp(-dist) * self.config.weight
        if self.config.use_scaling:
            reward = self.get_scaled_reward(reward)
        return reward

    def reset(self):
        pass

    def get_state(self) -> Dict[str, Any]:
        return {}

    def set_state(self, state: Dict[str, Any]):
        pass

    def get_config(self) -> ObjectiveConfig:
        return self.config

    def get_scaled_reward(self, raw_reward: torch.Tensor) -> torch.Tensor:
        return raw_reward


class ContrastObjective(ObjectiveModule):
    """Reward for spreading using contrastive dynamics from baseline."""

    def __init__(self, config: ObjectiveConfig, baseline_manager: BaselineManager):
        self.config = config
        self.baseline_manager = baseline_manager

    def validate(self) -> bool:
        return isinstance(self.config, ObjectiveConfig) and self.config.enabled and self.baseline_manager is not None

    def execute(self, state: torch.Tensor, action: torch.Tensor, next_state: torch.Tensor,
                log_prob: torch.Tensor, old_log_prob: torch.Tensor, value: torch.Tensor,
                old_value: torch.Tensor, episode_data: Dict[str, Any]) -> torch.Tensor:
        if not self.config.enabled:
            return torch.tensor(0.0, device=state.device)
        if len(self.baseline_manager.get_manifold()) < 2:
            return torch.tensor(0.0, device=state.device)
        data = self.baseline_manager.get_manifold()
        a, b = torch.tensor(data[0]), torch.tensor(data[1])
        dist = torch.norm(state - a, dim=-1) - torch.norm(state - b, dim=-1)
        reward = torch.abs(dist).mean() * self.config.weight
        if self.config.use_scaling:
            reward = self.get_scaled_reward(reward)
        return reward

    def reset(self):
        pass

    def get_state(self) -> Dict[str, Any]:
        return {}

    def set_state(self, state: Dict[str, Any]):
        pass

    def get_config(self) -> ObjectiveConfig:
        return self.config

    def get_scaled_reward(self, raw_reward: torch.Tensor) -> torch.Tensor:
        return raw_reward


class ExtrapolationGatheringObjective(ObjectiveModule):
    """Reward for extrapolating and gathering extrapolated data into baseline."""

    def __init__(self, config: ObjectiveConfig, baseline_manager: BaselineManager):
        self.config = config
        self.baseline_manager = baseline_manager

    def validate(self) -> bool:
        return isinstance(self.config, ObjectiveConfig) and self.config.enabled and self.baseline_manager is not None

    def execute(self, state: torch.Tensor, action: torch.Tensor, next_state: torch.Tensor,
                log_prob: torch.Tensor, old_log_prob: torch.Tensor, value: torch.Tensor,
                old_value: torch.Tensor, episode_data: Dict[str, Any]) -> torch.Tensor:
        if not self.config.enabled:
            return torch.tensor(0.0, device=state.device)
        if self.baseline_manager.get_centroid() is None:
            return torch.tensor(0.0, device=state.device)
        centroid = self.baseline_manager.get_centroid()
        dist = torch.norm(state - centroid, dim=-1).mean()
        reward = torch.exp(-dist) * self.config.weight
        if self.config.use_scaling:
            reward = self.get_scaled_reward(reward)
        return reward

    def reset(self):
        pass

    def get_state(self) -> Dict[str, Any]:
        return {}

    def set_state(self, state: Dict[str, Any]):
        pass

    def get_config(self) -> ObjectiveConfig:
        return self.config

    def get_scaled_reward(self, raw_reward: torch.Tensor) -> torch.Tensor:
        return raw_reward


# ==============================
# 5. Switchable Objective Manager
# ==============================

class SwitchableObjectiveManager:
    """
    Orchestrates dynamic objective switching and composition.
    Supports single and multi-objective modes.
    """

    def __init__(self, registry: ObjectiveRegistry, baseline_manager: BaselineManager, config: Dict[str, ObjectiveConfig]):
        self.registry = registry
        self.baseline_manager = baseline_manager
        self.config = config
        self._active_objectives: List[ObjectiveModule] = []
        self._last_switch_episode = 0
        self._switch_freq = config.get("objective_switching_freq", 10)
        self._mode = config.get("mode", "single")  # "single" or "multi"

    def get_active_objectives(self) -> List[ObjectiveModule]:
        """Return list of active objective instances."""
        return self._active_objectives

    def switch_objective(self, episode: int):
        """Switch to a new objective based on frequency."""
        if episode % self._switch_freq != 0:
            return
        if self._mode == "single":
            names = list(self.config.keys())
            selected_name = random.choice(names)
            obj = self.registry.get(selected_name)
            if obj is not None:
                self._active_objectives = [obj]
        elif self._mode == "multi":
            self._active_objectives = self.registry.get_by_priority(self.config)

    def compute_total_reward(self, state: torch.Tensor, action: torch.Tensor, next_state: torch.Tensor,
                             log_prob: torch.Tensor, old_log_prob: torch.Tensor, value: torch.Tensor,
                             old_value: torch.Tensor, episode_data: Dict[str, Any]) -> torch.Tensor:
        """Compute total reward from all active objectives."""
        total_reward = torch.tensor(0.0, device=state.device)
        for obj in self._active_objectives:
            if not obj.validate():
                continue
            reward = obj.execute(state, action, next_state, log_prob, old_log_prob, value, old_value, episode_data)
            total_reward += reward
        return total_reward

    def reset(self):
        """Reset all active objectives."""
        for obj in self._active_objectives:
            obj.reset()


# ==============================
# 6. PPO Agent with Dynamic Objectives
# ==============================

class PPOAgent:
    """
    PPO agent with dynamic objective switching using modular, pluggable objectives.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.state_dim = config.get("state_dim", 4)
        self.action_dim = config.get("action_dim", 2)
        self.latent_dim = config.get("latent_dim", 64)

        # Initialize modules
        self.actor = Actor(self.state_dim, self.action_dim, self.latent_dim).to(self.device)
        self.critic = Critic(self.state_dim, self.latent_dim).to(self.device)
        self.optimizer_actor = optim.Adam(self.actor.parameters(), lr=config["lr_actor"])
        self.optimizer_critic = optim.Adam(self.critic.parameters(), lr=config["lr_critic"])

        # Initialize baseline manager
        self.baseline_manager = BaselineManager(config)

        # Initialize objective registry and manager
        self.registry = ObjectiveRegistry()
        self._register_default_objectives()
        self.registry.discover_and_load()

        self.objective_manager = SwitchableObjectiveManager(
            registry=self.registry,
            baseline_manager=self.baseline_manager,
            config=config
        )

        # Training state
        self.episode = 0
        self.total_rewards = deque(maxlen=100)
        self.step_count = 0

    def _register_default_objectives(self):
        """Register built-in objectives."""
        self.registry.register("exploration", ExplorationObjective)
        self.registry.register("exploitation", ExploitationObjective)
        self.registry.register("extrapolation", ExtrapolationObjective)
        self.registry.register("spreading", SpreadingObjective)
        self.registry.register("gathering", GatheringObjective)
        self.registry.register("selection", SelectionObjective)
        self.registry.register("similarity", SimilarityObjective)
        self.registry.register("interpolation", InterpolationObjective)
        self.registry.register("integration", IntegrationObjective)
        self.registry.register("gathering_metric", GatheringMetricObjective)
        self.registry.register("contrast", ContrastObjective)
        self.registry.register("extrapolation_gathering", ExtrapolationGatheringObjective)

    def get_action(self, state: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample action and return log probability."""
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            mean, std = self.actor(state_tensor)
            dist = Normal(mean, std)
            action = dist.sample()
            log_prob = dist.log_prob(action).sum(dim=-1)
        return action, log_prob

    def compute_advantages(self, rewards: List[float], values: List[float], dones: List[bool]) -> List[float]:
        """Compute Generalized Advantage Estimation (GAE)."""
        gamma = self.config["gamma"]
        gae_lambda = self.config["gae_lambda"]
        advantages = []
        gae = 0.0
        for i in reversed(range(len(rewards))):
            delta = rewards[i] + gamma * values[i + 1] * (1 - dones[i]) - values[i]
            gae = delta + gamma * gae_lambda * (1 - dones[i]) * gae
            advantages.insert(0, gae)
        return advantages

    def update(self, states: List[torch.Tensor], actions: List[torch.Tensor], log_probs: List[torch.Tensor],
               rewards: List[float], values: List[float], dones: List[bool]):
        """Perform PPO update over multiple epochs."""
        states = torch.cat(states).to(self.device)
        actions = torch.cat(actions).to(self.device)
        old_log_probs = torch.cat(log_probs).to(self.device)
        values = torch.tensor(values, dtype=torch.float32).to(self.device)
        advantages = self.compute_advantages(rewards, values.cpu().numpy().tolist(), dones)

        for _ in range(self.config["epochs"]):
            # Forward pass
            new_log_probs, values_pred = self._forward_pass(states, actions)
            ratio = torch.exp(new_log_probs - old_log_probs)
            clipped_ratio = torch.clamp(ratio, 1 - self.config["clip_epsilon"], 1 + self.config["clip_epsilon"])
            surr1 = ratio * torch.tensor(advantages, device=self.device)
            surr2 = clipped_ratio * torch.tensor(advantages, device=self.device)
            policy_loss = -torch.min(surr1, surr2).mean()

            # Value loss
            value_loss = F.mse_loss(values_pred.squeeze(), torch.tensor(advantages, device=self.device))

            # Entropy bonus
            entropy = -torch.mean(new_log_probs)
            total_loss = policy_loss + self.config["value_loss_coeff"] * value_loss - self.config["entropy_coeff"] * entropy

            # Optimize
            self.optimizer_actor.zero_grad()
            self.optimizer_critic.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
            self.optimizer_actor.step()
            self.optimizer_critic.step()

    def _forward_pass(self, states: torch.Tensor, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass for actor and critic."""
        mean, std = self.actor(states)
        dist = Normal(mean, std)
        log_probs = dist.log_prob(actions).sum(dim=-1)
        values = self.critic(states).squeeze()
        return log_probs, values

    def train(self):
        """Main training loop."""
        env = Environment(self.config)
        episode_rewards = []
        for episode in range(self.config["num_episodes"]):
            state = env.reset()
            states, actions, log_probs, rewards, values, dones = [], [], [], [], [], []
            total_reward = 0.0
            for t in range(self.config["max_steps_per_episode"]):
                action, log_prob = self.get_action(state)
                next_state, reward, done, info = env.step(action.cpu().numpy())
                states.append(torch.tensor(state, dtype=torch.float32))
                actions.append(action)
                log_probs.append(log_prob)
                rewards.append(reward)
                values.append(self.critic(torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)).item())
                dones.append(done)
                state = next_state
                total_reward += reward
                if done:
                    break

            # Update baseline
            if episode % self.config["baseline_update_freq"] == 0:
                self.baseline_manager.add(torch.tensor(state, dtype=torch.float32))

            # Switch objective
            self.objective_manager.switch_objective(episode)

            # Compute total reward
            total_objective_reward = self.objective_manager.compute_total_reward(
                states[-1], actions[-1], states[-1], log_probs[-1], log_probs[-1],
                torch.tensor([values[-1]]), torch.tensor([values[-1]]), {}
            )
            total_reward += total_objective_reward.item()

            # Update
            self.update(states, actions, log_probs, rewards, values, dones)

            # Logging
            episode_rewards.append(total_reward)
            self.total_rewards.append(total_reward)
            if episode % 100 == 0:
                avg_reward = np.mean(self.total_rewards)
                print(f"Episode {episode}, Avg Reward: {avg_reward:.2f}")

            self.episode += 1

        print("Training complete.")


# ==============================
# 7. Supporting Classes (as per original)
# ==============================

class Environment:
    def __init__(self, config):
        self.state_dim = config.get("state_dim", 4)
        self.action_dim = config.get("action_dim", 2)
        self.max_steps = config.get("max_steps_per_episode", 1000)

    def reset(self):
        return np.random.randn(self.state_dim)

    def step(self, action):
        next_state = np.random.randn(self.state_dim)
        reward = np.random.randn()
        done = False
        info = {}
        return next_state, reward, done, info


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, latent_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, action_dim * 2)
        )
        self.action_dim = action_dim

    def forward(self, x):
        x = self.net(x)
        mean = x[:, :self.action_dim]
        std = torch.exp(x[:, self.action_dim:])
        return mean, std


class Critic(nn.Module):
    def __init__(self, state_dim, latent_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, 1)
        )

    def forward(self, x):
        return self.net(x)