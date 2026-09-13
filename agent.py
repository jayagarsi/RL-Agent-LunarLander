"""Legacy zero-arg-contract shim (worker consolidation): adapts an old
direct_v1 Agent() implementation (zero-arg construct, no setup) to the
flexkit protocol by supplying a no-op setup so flexkit's setup() call does
not break the agent. See docs/worker_consolidation_migration.md."""
from legacy_agent_impl import Agent as LegacyAgent


class Agent:
    def __init__(self):
        self._impl = LegacyAgent()

    def setup(self, observation_space, action_space):
        if hasattr(self._impl, "setup"):
            return self._impl.setup(observation_space, action_space)
        return True

    def choose_action(self, observation, reward=0.0, terminated=False,
                      truncated=False, info=None, action_mask=None):
        return self._impl.choose_action(observation, reward, terminated,
                                        truncated, info, action_mask)

    def reset(self, env_player_name, episode_index):
        if hasattr(self._impl, "reset"):
            return self._impl.reset(env_player_name, episode_index)
        return True
