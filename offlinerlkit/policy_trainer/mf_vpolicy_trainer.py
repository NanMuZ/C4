from copy import deepcopy
import time
import os
import copy
import numpy as np
import torch
# import gym
try:
    import gym
    import gym.vector
except Exception:
    gym = None
import swanlab
import sys
import random

from typing import Optional, Dict, List
from tqdm import tqdm
from collections import deque
from offlinerlkit.buffer import ReplayBuffer
from offlinerlkit.utils.logger import Logger
from offlinerlkit.policy import BasePolicy

LOG = 3
WANDB = True
AGGRE = True
# model-free policy trainer 
class MFVPolicyTrainer: 
    def __init__(
        self,
        policy: BasePolicy,
        eval_env: gym.Env,
        buffers: List[ReplayBuffer],
        logger: Logger,
        epoch: int,
        step_per_epoch: int,
        batch_size: int,
        eval_episodes: int,
        local_num: int,
        local_step_per_epoch: int,
        lr_scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        # local_dataset: Optional[Dict[str, float]] = None,
        buffer_full: Optional[List[ReplayBuffer]] = None,
        cluster: Optional[str] = "kmeans",
        task: Optional[str] = "hopper-expert",
    ) -> None:
        self.policy = policy
        self.eval_env = eval_env
        self.buffers = buffers
        self.logger = logger

        self._epoch = epoch
        self._step_per_epoch = step_per_epoch
        self._batch_size = batch_size
        self._eval_episodes = eval_episodes
        self._local_num = local_num
        self._local_step_per_epoch = local_step_per_epoch
        self.lr_scheduler = lr_scheduler
        self.weights = None
        self.gamma = np.array([0.99 ** n for n in range(2000)])
        self.buffer_full = buffer_full
        self.cluster = cluster
        self.task=task
        self.deterministic=True
        self.visualize=True
        

    def train(self, buffers_new: Optional[List[ReplayBuffer]] = None) -> Dict[str, float]:
        #init parameter
        start_time = time.time()
        num_timesteps = 0
        last_10_performance = deque(maxlen=10)
        valid_indices = list(range(len(self.buffers)))
        label1, label2 = self.best_labels, self.best_labels
        size_buffers = [self.buffer_full._max_size] * self._local_num
        initial_params = None
        device = self.policy.actor.device
        critic_copy = copy.deepcopy(self.policy.critic1)
        critic_old_copy = copy.deepcopy(self.policy.critic1)


        for epoch in range(1, self._epoch + 1):
            start_epoch_time = time.time()
            self.policy.train()

            print(f"Epoch #{epoch}/{self._epoch}")
            print(valid_indices)

            for it in range(self._local_step_per_epoch):
                random_index = random.choice(valid_indices)
                random_index = random.choices(range(len(size_buffers)), weights=size_buffers, k=1)[0]
                batch = self.buffers[random_index].sample_c4(self._batch_size)
                loss =  self.policy.learn(batch)
                for k, v in loss.items():
                    self.logger.logkv_mean(k, v)

            record_loss = loss
            
        
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()
  
            end_epoch_time = time.time()
            second_train = end_epoch_time-start_epoch_time
            print(f"Epoch #{epoch} | Time:  {end_epoch_time-start_epoch_time}")
            start_cluster_time = time.time()

            # Aggregate parameters from critic1 and critic2 to the copy, then assign back to both critics
            if (self.change is True): # and (epoch % self.cluster_change == 1):
                print("if self.change is True and (epoch % self.cluster_change == 1):")
                visualize = True if (epoch % 10 == 1) or epoch <= 3 else False
                # with torch.no_grad():
                if True:
                    critic_copy.load_state_dict({k: ((v + self.policy.critic2.state_dict()[k]) * 0.5) if torch.is_floating_point(v) else v for k, v in self.policy.critic1.state_dict().items()}, strict=True)
                    critic_old_copy.load_state_dict({k: ((v + self.policy.critic2_old.state_dict()[k]) * 0.5) if torch.is_floating_point(v) else v for k, v in self.policy.critic1_old.state_dict().items()}, strict=True)
                    print(self.beta)
                    if self.beta > 0:
                        print(self.beta)
                        batch = self.buffer_full.sample_all()
                        obs, actions, next_obs, next_actions = batch["observations"], batch["actions"], batch["next_observations"], batch["next_actions"]
                        batch_size = obs.shape[0]
                        obs_dim    = obs.shape[-1]

                        
                        print(batch_size, obs_dim)

                        # ---- Current Q(s,a) gradient g = dQ/d(s,a) ----
                        sa = torch.cat([obs, actions], dim=-1)
                        # Create a leaf copy for autograd to compute input gradients
                        sa_for_grad = sa.detach().clone().requires_grad_(True)

                        q_cur = critic_copy(
                            sa_for_grad[..., :obs_dim],
                            sa_for_grad[..., obs_dim:]
                        )  # [N, 1]

                        g = torch.autograd.grad(
                            outputs=q_cur.sum(),       # Scalar
                            inputs=sa_for_grad,
                            create_graph=False         # Used only as feature, no need to backprop through critic
                        )[0]                           # [N, obs_dim + act_dim]

                        q_feature = g                  # Current "feature" is the input gradient

                        # ---- target Q(s',a') gradient g' = dQ_target/d(s',a') ----
                        # Using the same critic_copy as target here,
                        # If you have a separate target critic, replace critic_copy with that network
                        sa_next = torch.cat([next_obs, next_actions], dim=-1)
                        sa_next_for_grad = sa_next.detach().clone().requires_grad_(True)

                        q_tgt = critic_old_copy(
                            sa_next_for_grad[..., :obs_dim],
                            sa_next_for_grad[..., obs_dim:]
                        )  # [N, 1]

                        g_next = torch.autograd.grad(
                            outputs=q_tgt.sum(),
                            inputs=sa_next_for_grad,
                            create_graph=False
                        )[0]                           # [N, obs_dim + act_dim]

                        q_feature_next = g_next        # Target "feature" = input gradient

                        # Stack into [N, 2D] features for subsequent C^4 / clustering logic
                        q_feature = torch.cat([q_feature, q_feature_next], dim=1)

                    else:
                        # Use the copy for buffer structure modification (using only one critic)
                        q_feature = critic_copy.get_feature(self.buffer_full.observations, self.buffer_full.actions)
                        q_feature_next = critic_copy.get_feature(self.buffer_full.next_observations, self.buffer_full.next_actions)
                        q_feature = torch.cat([q_feature, q_feature_next], dim=1)

                    clustered_buffers, size_buffers, label, initial_params = self.buffer_full.cluster_features(
                        feature_matrix=q_feature,
                        method=self.cluster,
                        Algo="CQL",
                        n_clusters=self._local_num,
                        device=device,
                        visualize=(visualize and self.visualize),
                        epoch=epoch,
                        task=f"{self.task}/critic",
                        init_labels=None,
                        initial_params=initial_params,
                    )
                    
                    # Calculate minimum sample size threshold (adjusted to max_size//5, more lenient)
                    min_size_threshold = max(self._batch_size, self.buffer_full._max_size // (max(self._local_num,10)))
                    # Check the size of each buffer, replace with buffer_full copy if insufficient
                    valid_indices = []
                    for i in range(len(clustered_buffers)):
                        if size_buffers[i] < min_size_threshold:
                            clustered_buffers[i] = None
                            size_buffers[i] = 0        # Update to buffer_full size
                        else:
                            valid_indices.append(i)
                    self.buffers = clustered_buffers
                    print("Processed buffer size list:", size_buffers)
                    print(valid_indices)
                    
                    sd = critic_copy.state_dict(); [c.load_state_dict(sd) for c in (self.policy.critic1, self.policy.critic2)]



            second_cluster = time.time() - start_cluster_time
            print(f"second_cluster: {second_cluster}")


            # evaluate   
            # if epoch <=1 or epoch%10==0:
            value_info = self.V_evaluate(self.policy)
            true_reward, esti_value = np.mean(value_info["eval/true_reward"]), np.mean(value_info["eval/esti_value"])

            server_eval_info = self._evaluate(self.policy)
            server_ep_reward, server_ep_reward_std = np.mean(server_eval_info["eval/episode_reward"]), np.std(server_eval_info["eval/episode_length"])
            server_norm_ep_rew_mean = self.eval_env.get_normalized_score(server_ep_reward) * 100
            server_norm_ep_rew_std = self.eval_env.get_normalized_score(server_ep_reward_std) * 100
            



            if True:

                self.logger.logkv("eval/server_episode_reward", server_ep_reward)
                self.logger.logkv("eval/server_episode_reward_std", server_norm_ep_rew_std)
                self.logger.logkv("eval/server_normalized_episode_reward", server_norm_ep_rew_mean)
                self.logger.logkv("eval/True Value", true_reward)
                self.logger.logkv("eval/Estimate Value", esti_value)
                # if self._local_num == 1:
                # print(Cov_result)
                # _ = [self.logger.logkv(f"{k}", float(v.item() if hasattr(v, "item") else v)) for k, v in Cov_result.items()]
                self.logger.set_timestep(num_timesteps)
                self.logger.dumpkvs()
            
            if WANDB is True:


            # if True:
                second = time.time() - start_time  # Directly calculate time difference (float)
                swanlab.log({
                    "global_eval/server_episode_reward": server_ep_reward,
                    "global_eval/server_episode_reward_std": server_norm_ep_rew_std,
                    "global_eval/server_normalized_episode_reward": server_norm_ep_rew_mean,
                    "value_eval/True Value": true_reward,
                    "value_eval/Estimate Value": esti_value,
                    "epoch": num_timesteps,
                    "second": second,
                    "second_cluster": second_cluster,
                    "second_train": second_train,
                    "step": epoch,
                })


                # Pre-calculate sorted results
                sorted_sizes = sorted(size_buffers)

                for idx, size in enumerate(size_buffers):
                    swanlab.log({
                        f"buffer_size/buffer_size_{idx}": size,
                        f"ranked_size/ranked_size_{idx}": sorted_sizes[idx],
                    })
                    

                record_loss_with_prefix = {f"local_loss_eval/num_{key}": value for key, value in record_loss.items()}
                swanlab.log(record_loss_with_prefix)
        
        
        self.logger.log("total time: {:.2f}s".format(time.time() - start_time))
        self.logger.close()

        return {"last_10_performance": np.mean(last_10_performance)}
        


    def _evaluate(self, policy) -> Dict[str, List[float]]:
        policy.eval()# self.policy.eval()
        obs = self.eval_env.reset()
        eval_ep_info_buffer = []
        num_episodes = 0
        episode_reward, episode_length = 0, 0

        while num_episodes < self._eval_episodes:
            action = policy.select_action(obs.reshape(1,-1), deterministic=self.deterministic)# action = self.policy.select_action(obs.reshape(1,-1), deterministic=self.deterministic)
            next_obs, reward, terminal, _ = self.eval_env.step(action.flatten())
            episode_reward += reward
            episode_length += 1

            obs = next_obs

            if terminal:
                eval_ep_info_buffer.append(
                    {"episode_reward": episode_reward, "episode_length": episode_length}
                )
                num_episodes +=1
                episode_reward, episode_length = 0, 0
                obs = self.eval_env.reset()
        
        return {
            "eval/episode_reward": [ep_info["episode_reward"] for ep_info in eval_ep_info_buffer],
            "eval/episode_length": [ep_info["episode_length"] for ep_info in eval_ep_info_buffer]
        }
    
    def V_evaluate(self, policy) -> Dict[str, List[float]]:
        policy.eval()
        obs = self.eval_env.reset()
        esti_value_info_buffer = []
        true_reward_info_buffer = []
        num_episodes = 0
        true_reward, episode_length = 0, 0

        while num_episodes < 3:
            action = policy.select_action(obs.reshape(1, -1), deterministic=True)
            if num_episodes == 0:
                tensor_obs = torch.tensor([obs], dtype=torch.float32, requires_grad=True).to(policy.actor.device)  # Here we set requires_grad=True   
                tensor_a = torch.tensor(action, dtype=torch.float32, requires_grad=True).to(policy.actor.device)  # Here we set requires_grad=True   
                q1 = policy.critic1(tensor_obs, tensor_a)
                q2 = policy.critic2(tensor_obs, tensor_a)
                value = min(q1, q2).item()
                esti_value_info_buffer.append(
                    {"esti_value": value,}
                )
            next_obs, reward, terminal, _ = self.eval_env.step(action.flatten())
            true_reward += self.gamma[episode_length] * reward
            episode_length += 1

            obs = next_obs

            if terminal:
                true_reward_info_buffer.append(
                    {"true_reward": true_reward, "episode_length": episode_length}
                )
                num_episodes += 1
                true_reward, episode_length = 0, 0
                obs = self.eval_env.reset()

        return {
            "eval/true_reward": [ep_info["true_reward"] for ep_info in true_reward_info_buffer],
            "eval/esti_value": [ep_info["esti_value"] for ep_info in esti_value_info_buffer],
        }
    




    def para_evaluate(self, policy) -> Dict[str, List[float]]:
        policy.eval()
        device = getattr(getattr(policy, "actor", policy), "device", torch.device("cpu"))

        # Serial evaluation function
        def eval_serial():
            obs = self.eval_env.reset()
            eval_ep_info_buffer = []
            num_episodes = 0
            episode_reward, episode_length = 0.0, 0
            with torch.no_grad():
                while num_episodes < self._eval_episodes:
                    obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=device).reshape(1, -1)
                    action = policy.select_action(obs_tensor, deterministic=self.deterministic)
                    action_np = action.detach().to("cpu").numpy().reshape(-1)
                    next_obs, reward, terminal, _ = self.eval_env.step(action_np)
                    episode_reward += float(reward)
                    episode_length += 1
                    obs = next_obs
                    if terminal:
                        eval_ep_info_buffer.append(
                            {"episode_reward": episode_reward, "episode_length": episode_length}
                        )
                        num_episodes += 1
                        episode_reward, episode_length = 0.0, 0
                        obs = self.eval_env.reset()
            return eval_ep_info_buffer

        # Parallel evaluation function
        def eval_parallel():
            assert gym is not None, "Requires gym or gymnasium vectorized environment support"

            target_episodes = self._eval_episodes
            # Rule of thumb: open environments in multiples of 8, upper limit can be adjusted as needed
            max_envs = 64
            num_envs = int(min(max_envs, max(8, ((target_episodes + 7) // 8) * 8)))

            # If already a vectorized environment and env count is appropriate, use it directly
            vec_env = None
            if hasattr(self.eval_env, "num_envs") and getattr(self.eval_env, "is_vector_env", False):
                vec_env = self.eval_env
                num_envs = vec_env.num_envs
            else:
                # Prefer SubprocVectorEnv to avoid GIL impact
                make_env_fn = None
                if hasattr(self, "make_eval_env") and callable(self.make_eval_env):
                    # If the class provides an environment factory, use it
                    make_env_fn = self.make_eval_env
                elif hasattr(self.eval_env, "spec") and hasattr(self.eval_env.spec, "id"):
                    env_id = self.eval_env.spec.id
                    def make_env():
                        return gym.make(env_id)
                    make_env_fn = make_env
                else:
                    # Last resort: lambda cloning may be unsafe but we try our best
                    import copy
                    def make_env():
                        return copy.deepcopy(self.eval_env)
                    make_env_fn = make_env

                try:
                    from gym.vector import AsyncVectorEnv
                    VecCls = AsyncVectorEnv
                except Exception:
                    from gym.vector import SyncVectorEnv
                    VecCls = SyncVectorEnv

                vec_env = VecCls([make_env_fn for _ in range(num_envs)])

            obs = vec_env.reset()  # shape [N, obs_dim]
            ep_rewards = np.zeros(num_envs, dtype=np.float64)
            ep_lengths = np.zeros(num_envs, dtype=np.int64)
            finished = []

            with torch.no_grad():
                while len(finished) < target_episodes:
                    obs_np = np.asarray(obs, dtype=np.float32)
                    actions = policy.select_action(obs_np, deterministic=self.deterministic)
                    actions_np = actions
                    next_obs, rewards, dones, infos = vec_env.step(actions_np)

                    ep_rewards += rewards.astype(np.float64)
                    ep_lengths += 1

                    # Handle completion for each environment
                    for i in range(num_envs):
                        if dones[i]:
                            finished.append(
                                {"episode_reward": float(ep_rewards[i]), "episode_length": int(ep_lengths[i])}
                            )
                            ep_rewards[i] = 0.0
                            ep_lengths[i] = 0

                    obs = next_obs

                    # If oversampled, truncate to target count
                    if len(finished) >= target_episodes:
                        finished = finished[:target_episodes]
                        break

            # If we temporarily created a vector environment, close it to release subprocesses
            if vec_env is not self.eval_env and hasattr(vec_env, "close"):
                vec_env.close()

            return finished

        if self._eval_episodes <= 10:
            eval_ep_info_buffer = eval_serial()
        else:
            eval_ep_info_buffer = eval_parallel()

        return {
            "eval/episode_reward": [x["episode_reward"] for x in eval_ep_info_buffer],
            "eval/episode_length": [x["episode_length"] for x in eval_ep_info_buffer],
        }


        
    from typing import Dict
    import torch

    def _power_iteration_sym(A: torch.Tensor, n_iter: int = 200, tol: float = 1e-7) -> torch.Tensor:
        """
        Power iteration for the largest eigenvalue of a real symmetric matrix
        """
        d = A.shape[0]
        v = torch.randn(d, dtype=A.dtype, device=A.device)
        v = v / (v.norm() + 1e-12)
        lam_old = torch.tensor(0.0, dtype=A.dtype, device=A.device)
        for _ in range(n_iter):
            w = A @ v
            nrm = w.norm()
            if nrm == 0:
                return torch.tensor(0.0, dtype=A.dtype, device=A.device)
            v = w / nrm
            lam = v @ (A @ v)
            if torch.abs(lam - lam_old) < tol:
                break
            lam_old = lam
        return lam


    def cross_cov_report_dict(self, 
                            q_feature: torch.Tensor,
                            q_feature_next: torch.Tensor,
                            unbiased: bool = True,
                            eps: float = 1e-8) -> Dict[str, float]:
        """
        Input
        q_feature         [N, D]
        q_feature_next    [N, D]
        Output
        Correlation and covariance statistics, with std provided as Cov_std/* where mean is calculated
        """
        assert q_feature.ndim == 2 and q_feature_next.ndim == 2, "Input must be 2D [N, D]"
        assert q_feature.shape == q_feature_next.shape, "Both inputs must have the same shape"
        N, D = q_feature.shape

        # Decentralize and clean numerical values
        X = q_feature - q_feature.mean(dim=0, keepdim=True)
        Y = q_feature_next - q_feature_next.mean(dim=0, keepdim=True)
        X = torch.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        Y = torch.nan_to_num(Y, nan=0.0, posinf=0.0, neginf=0.0)
        den = N - 1 if unbiased else N

        # Covariance and correlation matrices
        C = (X.T @ Y) / den                              # [D, D]
        sx = X.std(dim=0, unbiased=unbiased).clamp_min(eps)    # [D]
        sy = Y.std(dim=0, unbiased=unbiased).clamp_min(eps)    # [D]
        R = C / (sx.unsqueeze(1) * sy.unsqueeze(0))      # [D, D]

        # Calculate std simultaneously with mean
        absR = R.abs()
        macc = float(absR.mean())
        macc_std = float(absR.std(unbiased=unbiased))

        R2 = R.square()
        rms_corr = float(R2.mean().sqrt())
        rms_corr_std = float(R2.std(unbiased=unbiased))         # Standard deviation of R^2 elements

        diagR = R.diag()
        aligned_mean = float(diagR.mean())
        aligned_std = float(diagR.std(unbiased=unbiased))
        trace_over_D = aligned_mean
        trace_std = aligned_std

        # Spectral norm and maximum eigenvalue
        spec = torch.linalg.svdvals(R.to(torch.float64))[0]
        spectral_norm_over_D = float(spec / D)

        Rsym = 0.5 * (R + R.T)
        Rsym = Rsym.to(torch.float64)
        Rsym = torch.nan_to_num(Rsym, nan=0.0, posinf=0.0, neginf=0.0)
        Rsym = 0.5 * (Rsym + Rsym.T)
        Rsym = Rsym.clamp(min=-1.0, max=1.0)
        Rsym = Rsym + (1e-8 * torch.eye(D, dtype=Rsym.dtype, device=Rsym.device))
        try:
            lam_max = torch.linalg.eigvalsh(Rsym)[-1]
        except Exception:
            lam_max = _power_iteration_sym(Rsym)
        lambda_max_sym_over_D = float(lam_max / D)

        # Mean variance across sample dimensions and std when calculating mean
        var_q_dim = X.var(dim=0, unbiased=unbiased)          # [D]
        var_q = float(var_q_dim.mean())
        var_q_std = float(var_q_dim.std(unbiased=unbiased))

        var_qn_dim = Y.var(dim=0, unbiased=unbiased)         # [D]
        var_q_next = float(var_qn_dim.mean())
        var_q_next_std = float(var_qn_dim.std(unbiased=unbiased))

        return {
            "Cov/macc": macc,
            "Cov_std/macc": macc_std,
            "Cov/rms_corr": rms_corr,
            "Cov_std/rms_corr": rms_corr_std,
            "Cov/aligned_mean": aligned_mean,
            "Cov_std/aligned_mean": aligned_std,
            "Cov/trace_over_D": trace_over_D,
            "Cov_std/trace_over_D": trace_std,
            "Cov/spectral_norm_over_D": spectral_norm_over_D,
            "Cov/lambda_max_sym_over_D": lambda_max_sym_over_D,
            "Cov/var_q_feature_mean": var_q,
            "Cov_std/var_q_feature_mean": var_q_std,
            "Cov/var_q_feature_next_mean": var_q_next,
            "Cov_std/var_q_feature_next_mean": var_q_next_std,
        }
