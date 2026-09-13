import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical


def mlp(in_dim, out_dim, hidden):
    return nn.Sequential(
        nn.Linear(in_dim, hidden),
        nn.ReLU(),
        nn.Linear(hidden, out_dim),
    )


class RLModel(nn.Module):
    """Minimal actor-critic model for discrete actions.

    Two 1-hidden-layer MLPs: a policy head producing logits and a value head
    producing V(s). No target network, no replay buffer, no observation stacking,
    no RNN state. Trained with n-step A2C using `loss_a2c`.
    """

    def __init__(self, obs_dim, num_actions, hidden=64, reward_scale=10.0, gamma=0.99):
        super().__init__()
        self.obs_dim = obs_dim
        self.num_actions = num_actions
        self.hidden = hidden
        self.reward_scale = reward_scale
        self.gamma = gamma

        self.policy = mlp(obs_dim, num_actions, hidden)
        self.value = mlp(obs_dim, 1, hidden)

        torch.set_num_threads(3)

    def forward(self, observations):
        """observations: tensor of shape (..., obs_dim).
        Returns logits (..., num_actions) and values (...)."""
        logits = self.policy(observations)
        values = self.value(observations).squeeze(-1)
        return logits, values

    @torch.inference_mode()
    def collect_rollouts(self, env_list, num_steps):
        """Roll out B = len(env_list) envs for num_steps each, auto-resetting on done.

        Continues from where the previous call left off (persistent last-obs
        stored on the model). Action selection samples from the Categorical
        policy; if epsilon > 0, overrides with a uniform-random action
        independently per env.

        Returns a dict of numpy arrays shaped (B, T, ...):
            observations:  (B, T, obs_dim) — obs fed to the policy at step t
            actions:       (B, T)          — action taken at step t
            rewards:       (B, T)          — reward received for that action
            is_done:       (B, T)          — 1.0 if the episode ended at step t
            logits:        (B, T, num_actions) — policy logits at step t
            log_probs_old: (B, T)          — logarithm of the old logits
            last_obs:      (B, obs_dim)    — obs after the last step, for
                                            bootstrapping the n-step return
        """
        B = len(env_list)
        T = num_steps
        device = next(self.parameters()).device

        if not hasattr(self, "_rollout_last_obs") or self._rollout_last_obs.shape[0] != B:
            self._rollout_last_obs = np.stack(
                [env.reset()[0] for env in env_list]
            ).astype(np.float32)

        obs_buf = np.zeros((B, T, self.obs_dim), dtype=np.float32)
        act_buf = np.zeros((B, T), dtype=np.int64)
        rew_buf = np.zeros((B, T), dtype=np.float32)
        done_buf = np.zeros((B, T), dtype=np.float32)
        logit_buf = np.zeros((B, T, self.num_actions), dtype=np.float32)
        log_prob_buf = np.zeros((B, T), dtype=np.float32)

        was_training = self.training
        self.eval()

        for t in range(T):
            # No gradiant descent for these values
            with torch.no_grad():
                obs_t = self._rollout_last_obs
                logits, _ = self.forward(torch.from_numpy(obs_t).to(device))

                # Compute the action index using the model forward function:
                dist = Categorical(logits=logits)
                actions = dist.sample()
                log_probs_old = dist.log_prob(actions)

            next_obs = np.zeros_like(obs_t)
            for i, env in enumerate(env_list):
                o, r, term, trunc, _ = env.step(int(actions[i]))
                done = bool(term or trunc)
                if done:
                    o, _ = env.reset()
                next_obs[i] = o
                rew_buf[i, t] = r
                done_buf[i, t] = float(done)

            obs_buf[:, t] = obs_t
            act_buf[:, t] = actions.cpu().numpy()
            logit_buf[:, t] = logits.cpu().numpy()
            log_prob_buf[:, t] = log_probs_old.cpu().numpy()
            self._rollout_last_obs = next_obs

        if was_training:
            self.train()

        return {
            "observations": obs_buf,
            "actions": act_buf,
            "rewards": rew_buf,
            "is_done": done_buf,
            "logits": logit_buf,
            "log_probs_old": log_prob_buf,
            "last_obs": self._rollout_last_obs.copy(),
        }

    def compute_advantage(self, rollout, gae_lambda=0.95):
        """ Computes the Advantage and the Future Discounted Returns
        for a set of epochs in the PPO algorithm

        Uses GAE to compute the Adantage and the Returns
        Returns a dict with both values.

        """
        device = next(self.parameters()).device
        obs = torch.from_numpy(rollout["observations"]).to(device)
        rewards = torch.from_numpy(rollout["rewards"]).to(device) / self.reward_scale    # (B, T)
        is_done = torch.from_numpy(rollout["is_done"]).to(device)                        # (B, T)
        last_obs = torch.from_numpy(rollout["last_obs"]).to(device)                      # (B, obs_dim)

        B, T = rewards.shape

        logits, values = self.forward(obs)  # (B, T, A), (B, T)
        with torch.no_grad():
            _, last_value = self.forward(last_obs)  # (B,)

            # Use General Advantage Estimations
            # Implement advantage as a trade-off between Monte-Carlo
            # and TD approximations
            # - \delta_t = r_t + \gamma * v(s_{t+1}) - v(s_t)
            # - GAE_t = \delta_t + \gamma * \lambda_{GAE} * I_{not_done}
            advantages = torch.zeros(B, T, device=device)
            gae = torch.zeros(B, device=device)

            for t in reversed(range(T)):   # Build advantages from bottom up
                next_value = last_value if t == T - 1 else values[:, t + 1]   # for last step use last_value
                not_done = 1.0 - is_done[:, t]   # Indicator variable: 0 if episode ends at t, 1 otherwise
                delta = rewards[:, t] + self.gamma * next_value * not_done - values[:, t]
                gae = delta + self.gamma * gae_lambda * not_done * gae
                advantages[:, t] = gae           # Update advantages tensor

            returns = advantages + values       # returns now can be computed as the sum of advantages and values
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)   # normalize advantage value

        return {
            "returns": returns,
            "advantages": advantages
        }

    def loss_a2c(self, rollout, pre_computations, entropy_coef=0.01, value_coef=0.5, clip_eps=0.2):    # 0.05, 0.3, 0.95
        """Compute A2C loss from a rollout dict.

        Uses n-step returns bootstrapped with V(last_obs).
        Returns a dict with the scalar `loss` tensor and detached diagnostics.

        Improvements on the baseline:
          - Use GAE instead of REINFORCE --> stabilize the trade-off between
            variance and bias of the full Montecarlo estimate (gae = 1) and
            the full TD error estimate (gae = 0)
          - Use PPO --> add stability in the learning by minimize huge 
            deviations from previous policies while letting the agents
            improve the actions with not so different improvements
            from previous policies.
          - We have two new hyperparameters:
            - gae_lambda: trade-off between Montecarlo and TD approximation
            - clip_eps: max absolute difference allowed between policies
          - Usage of nn.utils.clip_grad_norm bewteen loss backwards and optimizer
            to avoid large gradient spikes to affect the advantage estimation
          - Execution of multiple gradient steps over the same buffer.
            Since collecting the rollouts is expensive, doing multiple
            runs of the forward pass allows us to get better learning,
            and since the cost is much less than that of the rollouts,
            this is amortized over the whole run.
            The Futer Discounted Returns and advantages are
            computed before each batch and shared between epochs, since
            we are only trying to improve the learning.
        """
        device = next(self.parameters()).device
        obs = torch.from_numpy(rollout["observations"]).to(device)             # (B, T, obs_dim)
        actions = torch.from_numpy(rollout["actions"]).long().to(device)       # (B, T)
        log_probs_old = torch.from_numpy(rollout["log_probs_old"]).to(device)  # (B, T)
        returns = pre_computations["returns"].to(device)
        advantages = pre_computations["advantages"].to(device)

        # Compute the next forward pass
        logits, values = self.forward(obs)  # (B, T, A), (B, T)

        # Compute the log of the probabilities
        # Sample from the logits and compute the log
        dist = Categorical(logits=logits)
        log_probs = dist.log_prob(actions)

        # Ration indicating how much the policy has changed
        ratio = torch.exp(log_probs - log_probs_old)   # only log_probs has grad

        # PPO introduces clipping function to limit
        # probability ration between old policy and new policy
        # The advantage is used to estimate how better or worse the action was
        # and to increase or decrease the probabilities of the actions accordingly
        # This loss is described in https://arxiv.org/pdf/1707.06347
        policy_loss = -torch.min(
            ratio * advantages,
            torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * advantages     # clipping function
        ).mean()
        value_loss = ((values - returns) ** 2).mean()       # K-step + MSE value loss

        # Entropy to take into account the exploitation vs. exploration
        entropy = dist.entropy().mean()

        loss = policy_loss + value_coef * value_loss - entropy_coef * entropy

        return {
            "loss": loss,
            "policy_loss": policy_loss.detach(),
            "value_loss": value_loss.detach(),
            "entropy": entropy.detach(),
            "mean_return": returns.mean().detach(),
        }

    def save(self, path=None):
        if path is None:
            path = "model.pt"
        torch.save(self.state_dict(), path)

    def load(self, path=None):
        if path is None:
            path = "model.pt"
        device = next(self.parameters()).device
        self.load_state_dict(torch.load(path, map_location=device, weights_only=True))
