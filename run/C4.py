import argparse
import random
import sys
import os

import gym
import d4rl
import numpy as np
import torch
import swanlab
import sys

from datetime import datetime

from offlinerlkit.nets import MLP_actor, MLP_critic
from offlinerlkit.modules import ActorProb, Critic, TanhDiagGaussian
from offlinerlkit.utils.load_dataset import qlearning_dataset_c4
from offlinerlkit.buffer import ReplayBuffer
from offlinerlkit.utils.clustering import cluster_and_visualize, find_L1
from offlinerlkit.utils.logger import Logger, make_log_dirs
from offlinerlkit.policy_trainer import MFVPolicyTrainer
from offlinerlkit.policy import C4CQLPolicy
from offlinerlkit.utils.scaler import StandardScaler
from offlinerlkit.utils.noise import GaussianNoise
from offlinerlkit.utils.trajectory_utils import sample, collect_replaytrajs, collect_trajs, extract_and_combine_trajs


def get_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser()
    
    # Basic parameters
    parser.add_argument("--algo-name", type=str, default="C4")
    parser.add_argument("--task", type=str, default="walker2d-medium-expert")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--traj-seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    
    # Network architecture parameters
    parser.add_argument("--hidden-dims", type=int, nargs='*', default=[256, 256, 256])
    parser.add_argument("--actor-lr", type=float, default=1e-4)
    parser.add_argument("--critic-lr", type=float, default=3e-4)
    
    # SAC parameters
    parser.add_argument("--auto-alpha", type=bool, default=False)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--alpha", type=float, default=0.2)
    parser.add_argument("--target-entropy", type=int, default=None)
    parser.add_argument("--alpha-lr", type=float, default=1e-4)

    # CQL parameters
    parser.add_argument("--cql-weight", type=float, default=5.0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--max-q-backup", type=bool, default=False)
    parser.add_argument("--deterministic-backup", type=bool, default=True)
    parser.add_argument("--with-lagrange", type=bool, default=True)
    parser.add_argument("--lagrange-threshold", type=float, default=10.0)
    parser.add_argument("--cql-alpha-lr", type=float, default=3e-4)
    parser.add_argument("--num-repeat-actions", type=int, default=10)
    
    # BC parameters
    parser.add_argument("--lmbda", type=float, default=4.0)
    parser.add_argument("--beta", type=float, default=-0.3)
    
    # Training parameters
    parser.add_argument("--epoch", type=int, default=1000)
    parser.add_argument("--step-per-epoch", type=int, default=1000)
    parser.add_argument("--eval_episodes", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=256)
    
    # Federated learning parameters
    parser.add_argument("--local-num", type=int, default=3)
    parser.add_argument("--local-step-per-epoch", type=int, default=200)
    parser.add_argument("--local-data-size", type=int, default=10000)
    
    # Dataset parameters
    parser.add_argument("--from-full-dataset", type=bool, default=False)
    parser.add_argument("--num-traj", type=int, default=0)
    parser.add_argument("--no-normalized", dest="no_normalized", action="store_false", default=True)
    
    # Clustering parameters
    parser.add_argument("--cluster", type=str, default="cgmm")
    parser.add_argument("--max-k", type=int, default=3)
    parser.add_argument("--extend-traj", type=int, default=0)
    
    # Other parameters
    parser.add_argument("--swanlab", type=str, default="")
    parser.add_argument("--exploration-noise", type=float, default=0.1)


    parser.add_argument("--flag", type=bool, default=True)
    parser.add_argument("--full", type=bool, default=False)
    parser.add_argument("--change", dest="change", action="store_false", default=True)
    parser.add_argument("--shaping", type=float, default=1.0)
    parser.add_argument("--visualize", dest="visualize", action="store_true", default=False)
    parser.add_argument("--cluster-change", type=int, default=1)



    return parser.parse_args()

import json

def train(args=get_args()):
    """Main training function"""
    # Print all configuration parameters (JSON format)
    print("=" * 50)
    print("Training Configuration:")
    print(json.dumps(vars(args), indent=4))
    print("=" * 50)
    # Original training code...
    """Main training function"""
    # Save configuration parameters to JSON file in current directory
    config_path = os.path.join(os.getcwd(), "train_config.json")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=4)
    print(f"Configuration saved to: {config_path}")

    
    # 1. Create environment and dataset
    env = gym.make(args.task)
    # traj_length_list, episode_end_list, dataset = qlearning_dataset_all(
    #     env, dataset=None, terminate_on_end=False, num_traj=args.num_traj
    # )
    traj_length_list, episode_end_list, dataset = qlearning_dataset_c4(env)
    print(episode_end_list[:10])
    print(max(dataset["rewards"]),min(dataset["rewards"]),sum(dataset["rewards"]))
    print(sum(dataset["terminals"]),sum(dataset["timeouts"]))

    if 'antmaze' in args.task:
        # dataset["rewards"] = (dataset["rewards"] - 0.5) * args.shaping
        dataset["rewards"] = (dataset["rewards"]) * args.shaping
    
    # Set environment-related parameters
    args.obs_shape = env.observation_space.shape
    args.action_dim = np.prod(env.action_space.shape)
    args.max_action = env.action_space.high[0]
    

    # 2. Set random seeds
    # random.seed(args.traj_seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    env.seed(args.seed)

    # 3. Set number of local clients
    args.local_num = args.max_k
    local_num = args.local_num

    # 4. Configure SAC alpha parameters
    if args.auto_alpha:
        target_entropy = args.target_entropy if args.target_entropy \
            else -np.prod(env.action_space.shape)
        args.target_entropy = target_entropy
        
        log_alpha = torch.zeros(1, requires_grad=True, device=args.device)
        alpha_optim = torch.optim.Adam([log_alpha], lr=args.alpha_lr)
        alpha = (target_entropy, log_alpha, alpha_optim)
    else:
        alpha = args.alpha

    # 5. Data normalization
    dataset_buffer = ReplayBuffer(
        buffer_size=len(dataset["observations"]),
        obs_shape=args.obs_shape,
        obs_dtype=np.float32,
        action_dim=args.action_dim,
        action_dtype=np.float32,
        device=args.device
    )
    dataset_buffer.load_dataset(dataset)
    
    # Calculate mean and standard deviation of original data
    obs_mean, obs_std = dataset_buffer.normalize_obs()
    if not args.no_normalized:
        obs_mean, obs_std = np.zeros_like(obs_mean), np.ones_like(obs_std)
        obs_Norm = False
    else:
        obs_Norm = True
    
    # Create standard scaler
    scaler = StandardScaler(mu=obs_mean, std=obs_std)
    print(f"Normalization: {'Yes' if args.no_normalized else 'No'} "
          f"(obs_mean={obs_mean.item(0):.2f}, obs_std={obs_std.item(0):.2f})")

    # 6. Create policy list
    # policies = []
    # for i in range(local_num):
        # Create network architecture
    actor_backbone = MLP_actor(input_dim=np.prod(args.obs_shape), hidden_dims=args.hidden_dims)
    critic1_backbone = MLP_critic(input_dim=np.prod(args.obs_shape) + args.action_dim, hidden_dims=args.hidden_dims)
    critic2_backbone = MLP_critic(input_dim=np.prod(args.obs_shape) + args.action_dim, hidden_dims=args.hidden_dims)
    
    # Create action distribution
    dist = TanhDiagGaussian(
        latent_dim=getattr(actor_backbone, "output_dim"),
        output_dim=args.action_dim,
        unbounded=True,
        conditioned_sigma=True
    )
    
    # Create Actor and Critic
    actor = ActorProb(actor_backbone, dist, args.device)
    critic1 = Critic(critic1_backbone, args.device)
    critic2 = Critic(critic2_backbone, args.device)
    
    # Create optimizers
    actor_optim = torch.optim.Adam(actor.parameters(), lr=args.actor_lr)
    critic1_optim = torch.optim.Adam(critic1.parameters(), lr=args.critic_lr)
    critic2_optim = torch.optim.Adam(critic2.parameters(), lr=args.critic_lr)
    
    # Create CQLBC policy

    det_tasks = ("halfcheetah", "hopper", "walker2d", "ant")
    args.deterministic_backup = any(t in args.task.lower() for t in det_tasks) 
    policy = C4CQLPolicy(
        actor,
        critic1,
        critic2,
        actor_optim,
        critic1_optim,
        critic2_optim,
        action_space=env.action_space,
        tau=args.tau,
        gamma=args.gamma,
        alpha=alpha,
        cql_weight=args.cql_weight,
        temperature=args.temperature,
        max_q_backup=args.max_q_backup,
        deterministic_backup=args.deterministic_backup,
        with_lagrange=args.with_lagrange,
        lagrange_threshold=args.lagrange_threshold,
        cql_alpha_lr=args.cql_alpha_lr,
        num_repeart_actions=args.num_repeat_actions,
        lmbda=args.lmbda,
        beta=args.beta,
        exploration_noise=GaussianNoise(sigma=args.exploration_noise),
        max_action=args.max_action,
        scaler=scaler,
        local=args.flag,
    )
        # policies.append(policy)

    # 7. Create local data buffers
    episode_start_list = [end - length + 1 for end, length in zip(episode_end_list, traj_length_list)]
    
    if args.full: 
        args.local_data_size = len(dataset["observations"])
    collected_trajs, total_length = collect_replaytrajs(episode_start_list, episode_end_list, args.local_data_size, traj_seed=args.traj_seed, full=args.full)
    
    # Extract and combine trajectory data
    local_dataset = extract_and_combine_trajs(dataset, collected_trajs)
    print(f"Terminals: {np.where(local_dataset['terminals'])[0]}")
    print(f"Traj: {total_length}: {sum(total_length)}: {len(total_length)}")

    # 8. Data clustering
    if args.full is False: 
        best_k, best_labels, hulls, divided_datasets = cluster_and_visualize(
            local_dataset, method=args.cluster, k=args.max_k, device=args.device, n=args.extend_traj
        )
    else:
        args.local_data_size = len(local_dataset["observations"])
        best_k, best_labels, hulls, divided_datasets = args.max_k, None, None, []
        divided_datasets = [local_dataset for _ in range(best_k)]
        # divided_datasets.append(local_dataset)
    buffer_full = ReplayBuffer(
        buffer_size=len(local_dataset["observations"]),
        obs_shape=args.obs_shape,
        obs_dtype=np.float32,
        action_dim=args.action_dim,
        action_dtype=np.float32,
        device=args.device
    )
    buffer_full.load_dataset(local_dataset)
    buffer_full.load_dataset_c4(local_dataset)
    buffer_full.buffer_normalize_obs(obs_mean=obs_mean, obs_std=obs_std)

    # 9. Create buffers for each cluster
    buffers = []
    buffer_size_arr = []
    local_num = best_k
    
    for i in range(local_num):
        cluster_dataset = divided_datasets[i]
        buffer = ReplayBuffer(
            buffer_size=len(cluster_dataset["observations"]),
            obs_shape=args.obs_shape,
            obs_dtype=np.float32,
            action_dim=args.action_dim,
            action_dtype=np.float32,
            device=args.device
        )
        buffer.load_dataset_c4(cluster_dataset)
        buffer.buffer_normalize_obs(obs_mean=obs_mean, obs_std=obs_std)
        buffers.append(buffer)
        buffer_size_arr.append(len(cluster_dataset["observations"]))
        print(len(cluster_dataset["observations"]))


    # 10. Set up logging
    log_dirs = make_log_dirs(
        args.task, 
        f"{args.algo_name}-lspe{args.local_step_per_epoch}-cqlw_{args.cql_weight}", 
        args.seed, 
        vars(args)
    )
    
    output_config = {
        "consoleout_backup": "stdout",
        "policy_training_progress": "csv",
        "tb": "tensorboard"
    }
    logger = Logger(log_dirs, output_config)
    logger.log_hyperparameters(vars(args))

    # 11. Create policy trainer
    policy_trainers = MFVPolicyTrainer(
        policy=policy,
        eval_env=env,
        buffers=buffers,
        logger=logger,
        epoch=args.epoch,
        step_per_epoch=args.step_per_epoch,
        batch_size=args.batch_size,
        eval_episodes=args.eval_episodes,
        local_num=args.local_num,
        local_step_per_epoch=args.local_step_per_epoch,
        buffer_full=buffer_full,
        cluster=args.cluster,
        task=args.task,
    )

    # 12. Configure experiment tracking
    current_time = datetime.now().strftime("%m%d")
    config = {
        "algo_name": args.algo_name,
        "task": args.task,
        "local_num": args.local_num,
        "local_data_size": args.local_data_size,
        "lmbda": args.lmbda,
        "beta": args.beta,
        "seed": args.seed,
        "cql_weight": args.cql_weight,
        "extend_traj": args.extend_traj,
        "cluster": args.cluster,
        "time": current_time,
    }
    

    experiment_name = (f"{args.algo_name}-{args.task}-ln{args.local_num}-l{args.lmbda}-b{args.beta}-a{args.alpha}-nN{args.no_normalized}"
        f"-c{args.change}-cc{args.cluster_change}-lspe{args.local_step_per_epoch}-lds{args.local_data_size}"
        f"-s{args.traj_seed}-c{args.cql_weight}-wl{args.with_lagrange}-c{args.cluster}"
        f"-f{args.flag}-r{args.shaping}-hd{len(args.hidden_dims)}-oN{obs_Norm}")
    
    swanlab.init(
        project=args.swanlab, 
        entity="", 
        experiment_name=experiment_name, 
        config=config
    )
    
    # 13. Start training
    policy_trainers.best_labels = best_labels
    policy_trainers.change = args.change
    policy_trainers.cluster_change = args.cluster_change
    policy_trainers.visualize = args.visualize
    policy_trainers.beta = args.beta
    policy_trainers.deterministic = any(t in args.task.lower() for t in det_tasks)
    policy_trainers.train()


if __name__ == "__main__":
    train()
