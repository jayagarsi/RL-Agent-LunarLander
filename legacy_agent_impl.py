import os
import numpy as np
import torch
from torch.distributions import Categorical

from rl_model import RLModel


class Agent:
    def __init__(self, env=None, player_name=None):
        path = "model.pt"
        self.model = RLModel(obs_dim=8, num_actions=4)
        if os.path.exists(path):
            self.model.load(path)
        self.model.eval()

    @torch.inference_mode()
    def choose_action(self, observation, reward=0.0, terminated=False, truncated=False,
                      info=None, action_mask=None):
        obs = torch.from_numpy(np.asarray(observation, dtype=np.float32))[None, :]
        logits, values = self.model(obs)

        ## TODO:
        # Compute the action index using the model forward function:
        # dist = Categorical(logits=logits)
        # action_index = dist.sample() #torch.argmax(logits, dim=1)
        action_index = torch.argmax(logits, dim=1).item()

        return action_index 
