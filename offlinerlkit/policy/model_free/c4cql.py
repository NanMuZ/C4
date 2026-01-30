import numpy as np
import torch
import torch.nn as nn
import gym
from copy import deepcopy
import sys

from torch.nn import functional as F
from typing import Dict, Union, Tuple, Optional, List, Callable
from offlinerlkit.policy import SACPolicy
from offlinerlkit.policy import BasePolicy
from offlinerlkit.utils.noise import GaussianNoise
from offlinerlkit.utils.scaler import StandardScaler

LAMBDA = 1.0  # 0.6
BETA = 1.0
C=False # c
 
class C4CQLPolicy(SACPolicy):
    """
    Conservative Q-Learning <Ref: https://arxiv.org/abs/2006.04779>
    """

    def __init__(
        self,
        actor: nn.Module,
        critic1: nn.Module,
        critic2: nn.Module,
        actor_optim: torch.optim.Optimizer,
        critic1_optim: torch.optim.Optimizer,
        critic2_optim: torch.optim.Optimizer,
        action_space: gym.spaces.Space,
        tau: float = 0.005,
        gamma: float = 0.99,
        alpha: Union[float, Tuple[float, torch.Tensor, torch.optim.Optimizer]] = 0.2,
        cql_weight: float = 1.0,
        temperature: float = 1.0,
        max_q_backup: bool = False,
        deterministic_backup: bool = True,
        with_lagrange: bool = True,
        lagrange_threshold: float = 10.0,
        cql_alpha_lr: float = 1e-4,
        num_repeart_actions: int = 10,
        lmbda: float = 1.0,
        beta: float = 1.0,
        exploration_noise: Callable = GaussianNoise,
        max_action: float = 1.0,
        scaler: StandardScaler = None,
        local: Optional[bool] = True,
    ) -> None:
        super().__init__(
            actor,
            critic1,
            critic2,
            actor_optim,
            critic1_optim,
            critic2_optim,
            tau=tau,
            gamma=gamma,
            alpha=alpha
        )

        self.action_space = action_space
        self._cql_weight = cql_weight
        self._temperature = temperature
        self._max_q_backup = max_q_backup
        self._deterministic_backup = deterministic_backup
        self._with_lagrange = with_lagrange
        self._lagrange_threshold = lagrange_threshold

        self.cql_log_alpha = torch.zeros(1, requires_grad=True, device=self.actor.device)
        self.cql_alpha_optim = torch.optim.Adam([self.cql_log_alpha], lr=cql_alpha_lr)

        self._num_repeat_actions = num_repeart_actions
        self._is_auto_alpha = False
        self.actor_global = deepcopy(actor)
        self.max_exp_scale = 1.0

        self.lmbda = lmbda
        self.beta = beta
        self.exploration_noise = exploration_noise
        self.max_action = max_action
        self.scaler = scaler
        self.local = local

        self.best_index=None

    def get_actor_global_actions(
        self,
        obs: torch.Tensor,
        deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        dist = self.actor_global(obs)
        if deterministic:
            squashed_action, raw_action = dist.mode()
        else:
            squashed_action, raw_action = dist.rsample()
        log_prob = dist.log_prob(squashed_action, raw_action)

        return squashed_action, log_prob, deterministic
    

    def select_action(self, obs: np.ndarray, deterministic: bool = False) -> np.ndarray:
        if self.scaler is not None:
            obs = self.scaler.transform(obs)
        action = super().select_action(obs, deterministic)

        return action




    def calc_pi_values(
        self,
        obs_pi: torch.Tensor,
        obs_to_pred: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        act, log_prob = self.actforward(obs_pi)

        q1 = self.critic1(obs_to_pred, act)
        q2 = self.critic2(obs_to_pred, act)

        return q1 - log_prob.detach(), q2 - log_prob.detach()

    def calc_random_values(
        self,
        obs: torch.Tensor,
        random_act: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        q1 = self.critic1(obs, random_act)
        q2 = self.critic2(obs, random_act)

        log_prob1 = np.log(0.5 ** random_act.shape[-1])
        log_prob2 = np.log(0.5 ** random_act.shape[-1])

        return q1 - log_prob1, q2 - log_prob2
    


    def learn(self, batch: Dict, pmoe_policy: Optional[float] = None, policies: Optional[List] = None) -> Tuple[float, Dict[str, float]]:
        obss, actions, next_obss, rewards, terminals = batch["observations"], batch["actions"], \
            batch["next_observations"], batch["rewards"], batch["terminals"]
        batch_size = obss.shape[0]
        self.actions, self.next_actions = actions, batch["next_actions"]

        
        a, log_probs = self.actforward(obss)
        q1a, q2a = self.critic1(obss, a), self.critic2(obss, a)
        
        dist = self.actor(obss)
        local_log_probs_kl = dist.log_prob(actions)


        a_loss = (self._alpha * log_probs - torch.min(q1a, q2a)).mean() 
        actor_loss = a_loss
        if self.lmbda > 0:
            c_loss = - self.lmbda * (local_log_probs_kl).mean()
            actor_loss = actor_loss + c_loss   # pmoe_ab

        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()

        if self._is_auto_alpha:
            log_probs = log_probs.detach() + self._target_entropy
            alpha_loss = -(self._log_alpha * log_probs).mean()
            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()
            self._alpha = self._log_alpha.detach().exp()



        # update_critic
        if self._max_q_backup:
            with torch.no_grad():
                tmp_next_obss = next_obss.unsqueeze(1) \
                    .repeat(1, self._num_repeat_actions, 1) \
                    .view(batch_size * self._num_repeat_actions, next_obss.shape[-1])
                tmp_next_actions, _ = self.actforward(tmp_next_obss)
                tmp_next_q1 = self.critic1_old(tmp_next_obss, tmp_next_actions) \
                    .view(batch_size, self._num_repeat_actions, 1) \
                    .max(1)[0].view(-1, 1)
                tmp_next_q2 = self.critic2_old(tmp_next_obss, tmp_next_actions) \
                    .view(batch_size, self._num_repeat_actions, 1) \
                    .max(1)[0].view(-1, 1)
                next_q = torch.min(tmp_next_q1, tmp_next_q2)
        else:
            with torch.no_grad():
                next_actions, next_log_probs = self.actforward(next_obss)
                
                next_q = torch.min(
                    self.critic1_old(next_obss, next_actions),
                    self.critic2_old(next_obss, next_actions)
                ) # min



                if not self._deterministic_backup:
                    next_q -= self._alpha * next_log_probs

        target_q = rewards + self._gamma * (1 - terminals) * next_q
        # if self.clamp is True:
        #     target_q = torch.clamp(target_q, min=self.q_min, max=self.q_max)
            # print(self.q_max, self.q_min)
            # sys.exit()

        # print(target_q, target_q.shape)
        # sys.exit()
        q1, q2 = self.critic1(obss, actions), self.critic2(obss, actions)
        critic1_loss = ((q1 - target_q).pow(2)).mean()
        critic2_loss = ((q2 - target_q).pow(2)).mean()
        q1_loss, q2_loss = critic1_loss, critic2_loss


        # CQL
        if self._cql_weight > 0:
            random_actions = torch.FloatTensor(
                batch_size * self._num_repeat_actions, actions.shape[-1]
            ).uniform_(self.action_space.low[0], self.action_space.high[0]).to(self.actor.device)
            
            tmp_obss = obss.unsqueeze(1) \
                .repeat(1, self._num_repeat_actions, 1) \
                .view(batch_size * self._num_repeat_actions, obss.shape[-1])
            tmp_next_obss = next_obss.unsqueeze(1) \
                .repeat(1, self._num_repeat_actions, 1) \
                .view(batch_size * self._num_repeat_actions, obss.shape[-1])
                
            tmp_actions = self.actions.unsqueeze(1) \
                .repeat(1, self._num_repeat_actions, 1) \
                .view(batch_size * self._num_repeat_actions, self.actions.shape[-1])
            tmp_next_actions = self.next_actions.unsqueeze(1) \
                .repeat(1, self._num_repeat_actions, 1) \
                .view(batch_size * self._num_repeat_actions, self.next_actions.shape[-1])
            
            obs_pi_value1, obs_pi_value2 = self.calc_pi_values(tmp_obss, tmp_obss)
            next_obs_pi_value1, next_obs_pi_value2 = self.calc_pi_values(tmp_next_obss, tmp_obss)
            random_value1, random_value2 = self.calc_random_values(tmp_obss, random_actions)

            for value in [
                obs_pi_value1, obs_pi_value2, next_obs_pi_value1, next_obs_pi_value2,
                random_value1, random_value2
            ]:
                value.reshape(batch_size, self._num_repeat_actions, 1)
            
            cat_q1 = torch.cat([obs_pi_value1, next_obs_pi_value1, random_value1], 1)
            cat_q2 = torch.cat([obs_pi_value2, next_obs_pi_value2, random_value2], 1)
            
            conservative_loss1 = \
                torch.logsumexp(cat_q1 / self._temperature, dim=1).mean() * self._cql_weight * self._temperature - \
                q1.mean() * self._cql_weight
            conservative_loss2 = \
                torch.logsumexp(cat_q2 / self._temperature, dim=1).mean() * self._cql_weight * self._temperature - \
                q2.mean() * self._cql_weight
                
            if self._with_lagrange:
                cql_alpha = torch.clamp(self.cql_log_alpha.exp(), 0.0, 1e6)
                conservative_loss1 = cql_alpha * (conservative_loss1 - self._lagrange_threshold)
                conservative_loss2 = cql_alpha * (conservative_loss2 - self._lagrange_threshold)

                self.cql_alpha_optim.zero_grad()
                cql_alpha_loss = -(conservative_loss1 + conservative_loss2) * 0.5
                cql_alpha_loss.backward(retain_graph=True)
                self.cql_alpha_optim.step()

            #     print("hi")
            # sys.exit()
            
            critic1_loss = critic1_loss + conservative_loss1
            critic2_loss = critic2_loss + conservative_loss2

       

        if self.beta > 0:
            # R_cross regularization: Frobenius norm of COV_B(g', g) + beta * (tr C)^2
            # g  = d Q(s,a) / d (s,a)
            # g' = d Q_target(s',a') / d (s',a')
            # Here self.beta is the regularization weight λ, β in the formula is denoted as beta_rc

            if batch_size > 1:  # Avoid division by (B-1) = 0
                obs_dim = obss.shape[-1]
                act_dim = actions.shape[-1]
                beta_rc = 1.0   # β in the paper, unrelated to your self.beta

                # ---------------- critic1's R_cross ----------------
                # Current (s,a) gradient g
                sa1 = torch.cat([obss, actions], dim=-1)
                sa1 = sa1.detach().clone().requires_grad_(True)

                q1_cur = self.critic1(
                    sa1[..., :obs_dim],
                    sa1[..., obs_dim:]
                )
                g1 = torch.autograd.grad(
                    outputs=q1_cur.sum(),
                    inputs=sa1,
                    create_graph=True
                )[0]                      # [B, obs_dim + act_dim]

                # target (s',a') gradient g'
                if self.local:
                    next_act_for_grad = self.next_actions
                else:
                    with torch.no_grad():
                        next_act_for_grad, _ = self.actforward(next_obss)

                sa1_next = torch.cat([next_obss, next_act_for_grad], dim=-1)
                sa1_next = sa1_next.detach().clone().requires_grad_(True)

                q1_tgt = self.critic1_old(
                    sa1_next[..., :obs_dim],
                    sa1_next[..., obs_dim:]
                )
                g1p = torch.autograd.grad(
                    outputs=q1_tgt.sum(),
                    inputs=sa1_next,
                    create_graph=False
                )[0]                      # [B, obs_dim + act_dim]

                # Cross-covariance C = COV_B(g', g)
                g1   = g1.view(batch_size, -1)
                g1p  = g1p.view(batch_size, -1)
                g1c  = g1   - g1.mean(dim=0, keepdim=True)
                g1pc = g1p  - g1p.mean(dim=0, keepdim=True)

                C1 = (g1pc.t() @ g1c) / (batch_size - 1)   # [D, D]

                frob_sq_1 = (C1 ** 2).sum()
                tr_1 = torch.trace(C1)
                # R_cross1 = frob_sq_1 + beta_rc * tr_1.pow(2)

                # critic1_loss = critic1_loss + self.beta * R_cross1

                # ---------------- critic2's R_cross ----------------
                sa2 = torch.cat([obss, actions], dim=-1)
                sa2 = sa2.detach().clone().requires_grad_(True)

                q2_cur = self.critic2(
                    sa2[..., :obs_dim],
                    sa2[..., obs_dim:]
                )
                g2 = torch.autograd.grad(
                    outputs=q2_cur.sum(),
                    inputs=sa2,
                    create_graph=True
                )[0]

                sa2_next = torch.cat([next_obss, next_act_for_grad], dim=-1)
                sa2_next = sa2_next.detach().clone().requires_grad_(True)

                q2_tgt = self.critic2_old(
                    sa2_next[..., :obs_dim],
                    sa2_next[..., obs_dim:]
                )
                g2p = torch.autograd.grad(
                    outputs=q2_tgt.sum(),
                    inputs=sa2_next,
                    create_graph=False
                )[0]

                g2   = g2.view(batch_size, -1)
                g2p  = g2p.view(batch_size, -1)
                g2c  = g2   - g2.mean(dim=0, keepdim=True)
                g2pc = g2p  - g2p.mean(dim=0, keepdim=True)

                C2 = (g2pc.t() @ g2c) / (batch_size - 1)

                frob_sq_2 = (C2 ** 2).sum()
                tr_2 = torch.trace(C2)


                max_rc = self.max_rc


                R_cross1_raw = frob_sq_1 + beta_rc * tr_1.pow(2)
                R_cross1 = max_rc * torch.tanh(R_cross1_raw / max_rc)

                R_cross2_raw = frob_sq_2 + beta_rc * tr_2.pow(2)
                R_cross2 = max_rc * torch.tanh(R_cross2_raw / max_rc)


                inner1_loss = R_cross1 * self.beta
                inner2_loss = R_cross2 * self.beta

                critic1_loss = critic1_loss + inner1_loss
                critic2_loss = critic2_loss + inner2_loss




        if self.beta < 0:
            if self.local == True:
                q1_feature = self.critic1.get_feature(obss, actions)
                q1_feature_next = self.critic1_old.get_feature(next_obss, self.next_actions)
                q2_feature = self.critic2.get_feature(obss, actions)
                q2_feature_next = self.critic2_old.get_feature(next_obss, self.next_actions)
            else:
                next_actions, next_log_probs = self.actforward(next_obss)
                q1_feature = self.critic1.get_feature(obss, actions)
                q1_feature_next = self.critic1_old.get_feature(next_obss, next_actions)
                q2_feature = self.critic2.get_feature(obss, actions)
                q2_feature_next = self.critic2_old.get_feature(next_obss, next_actions)

            inner1_loss = - self.beta * (q1_feature * q1_feature_next).sum(dim=-1).mean()
            inner2_loss = - self.beta * (q2_feature * q2_feature_next).sum(dim=-1).mean()

            critic1_loss = critic1_loss + inner1_loss
            critic2_loss = critic2_loss + inner2_loss



        self.critic1_optim.zero_grad()
        critic1_loss.backward(retain_graph=True)
        self.critic1_optim.step()

        self.critic2_optim.zero_grad()
        critic2_loss.backward()
        self.critic2_optim.step()

        self._sync_weight()


        result = {
            "loss/actor": actor_loss.item(),
            "loss/critic1": critic1_loss.item(),
            "loss/critic2": critic2_loss.item(),
            "loss/q1": q1_loss.item(),
            "loss/q2": q2_loss.item(),
            "loss/aloss": a_loss.item(),
            "TDLoss": (q1_loss.item()+q2_loss.item())/2,
            "loss/conservative1": conservative_loss1.item(),
            "loss/conservative1": conservative_loss2.item(),
        }



        if self.beta > 0:
            result["loss/inner1"] = inner1_loss.item()
            result["loss/inner2"] = inner2_loss.item()

        if self._is_auto_alpha:
            result["loss/alpha"] = alpha_loss.item()
            result["alpha"] = self._alpha.item()
        # else:
        #     result["alpha"] = self._alpha.item()
        if self._with_lagrange:
            result["loss/cql_alpha"] = cql_alpha_loss.item()
            result["cql_alpha"] = cql_alpha.item()
        
        return result
