"""Script demonstrating the use of `gym_pybullet_drones`'s Gymnasium interface.

Classes HoverAviary and MultiHoverAviary are used as learning envs for the PPO algorithm.

Example
-------
In a terminal, run as:

    $ python learn.py --multiagent false
    $ python learn.py --multiagent true

Notes
-----
This is a minimal working example integrating `gym-pybullet-drones` with 
reinforcement learning library `stable-baselines3`.

"""
import os
import time
from datetime import datetime
import argparse
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal

from gym_pybullet_drones.utils.Logger import Logger
from gym_pybullet_drones.envs.HoverAviary import HoverAviary
from gym_pybullet_drones.envs.MultiHoverAviary import MultiHoverAviary
from gym_pybullet_drones.utils.utils import sync, str2bool
from gym_pybullet_drones.utils.enums import ObservationType, ActionType

DEFAULT_GUI = True
DEFAULT_RECORD_VIDEO = False
DEFAULT_OUTPUT_FOLDER = 'results'
DEFAULT_COLAB = False
DEFAULT_LOCAL = True

DEFAULT_OBS = ObservationType('kin') # 'kin' or 'rgb'
DEFAULT_ACT = ActionType('one_d_rpm') # 'rpm' or 'pid' or 'vel' or 'one_d_rpm' or 'one_d_pid'
DEFAULT_AGENTS = 2
DEFAULT_MA = False

def run(multiagent=DEFAULT_MA, output_folder=DEFAULT_OUTPUT_FOLDER, gui=DEFAULT_GUI, plot=True, colab=DEFAULT_COLAB, record_video=DEFAULT_RECORD_VIDEO, local=True, eval_only=False, model_path=None):

    # If running evaluation-only and a model_path is provided, use its directory
    if eval_only and model_path:
        if os.path.isfile(model_path):
            filename = os.path.dirname(model_path)
            model_file_provided = model_path
        else:
            filename = model_path
            model_file_provided = None
    else:
        filename = os.path.join(output_folder, 'save-'+datetime.now().strftime("%m.%d.%Y_%H.%M.%S"))
        if not os.path.exists(filename):
            os.makedirs(filename+'/')

    # For single-agent, use a custom PyTorch actor-critic trainer.
    # For multi-agent we fall back to the original SB3 PPO training (not modified here).
    if not multiagent:
        train_env = HoverAviary(gui=False, obs=DEFAULT_OBS, act=DEFAULT_ACT)
        eval_env = HoverAviary(obs=DEFAULT_OBS, act=DEFAULT_ACT)
    else:
        # Keep original behavior for multiagent (PPO) — user can still use the prior script.
        from stable_baselines3 import PPO
        from stable_baselines3.common.env_util import make_vec_env
        from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
        from stable_baselines3.common.evaluation import evaluate_policy

        train_env = make_vec_env(MultiHoverAviary,
                                 env_kwargs=dict(num_drones=DEFAULT_AGENTS, obs=DEFAULT_OBS, act=DEFAULT_ACT),
                                 n_envs=1,
                                 seed=0
                                 )
        eval_env = MultiHoverAviary(num_drones=DEFAULT_AGENTS, obs=DEFAULT_OBS, act=DEFAULT_ACT)

    #### Check the environment's spaces ########################
    print('[INFO] Action space:', train_env.action_space)
    print('[INFO] Observation space:', train_env.observation_space)

    #### Train the model (Actor-Critic for single-agent) #######
    # Skip training if running evaluation-only
    if not eval_only and not multiagent:
        # Build simple actor and critic networks
        obs_sample, _ = train_env.reset()
        # flatten observations
        obs_dim = int(np.prod(obs_sample.shape))
        act_dim = train_env.action_space.shape[0]

        class PolicyNet(nn.Module):
            def __init__(self, obs_dim, act_dim, hidden=256, init_std=0.5):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(obs_dim, hidden), nn.Tanh(),
                    nn.Linear(hidden, hidden), nn.Tanh(),
                )
                self.mean = nn.Linear(hidden, act_dim)
                # log std param
                self.log_std = nn.Parameter(torch.ones(act_dim) * np.log(init_std))
            def forward(self, x):
                h = self.net(x)
                return self.mean(h), self.log_std.exp()

        class ValueNet(nn.Module):
            def __init__(self, obs_dim, hidden=256):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(obs_dim, hidden), nn.Tanh(),
                    nn.Linear(hidden, hidden), nn.Tanh(),
                    nn.Linear(hidden, 1)
                )
            def forward(self, x):
                return self.net(x)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        policy = PolicyNet(obs_dim, act_dim).to(device)
        value = ValueNet(obs_dim).to(device)
        optimizer = optim.Adam(list(policy.parameters()) + list(value.parameters()), lr=3e-4)

        # Training hyperparams
        total_timesteps = int(1e6) if local else int(1e3)
        update_timestep = 2048
        gamma = 0.99
        entropy_coef = 1e-3
        value_coef = 0.5

        def flatten_obs(o):
            return np.asarray(o).ravel()

        timestep = 0
        ep = 0
        rollout = []
        obs, _ = train_env.reset()
        obs = flatten_obs(obs)

        while timestep < total_timesteps:
            # Collect rollout
            states = []
            actions = []
            rewards = []
            dones = []
            logps = []
            values = []
            for _ in range(update_timestep):
                s_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                with torch.no_grad():
                    mean, std = policy(s_t)
                    value_t = value(s_t)
                    dist = Normal(mean, std)
                    a_t = dist.sample()
                    logp = dist.log_prob(a_t).sum(axis=-1)
                a_np = a_t.squeeze(0).cpu().numpy()
                # For single-agent environments the aviary expects actions shaped (NUM_DRONES, action_dim)
                if a_np.ndim == 1:
                    a_env = np.expand_dims(a_np, axis=0)
                else:
                    a_env = a_np
                next_obs, reward, terminated, truncated, info = train_env.step(a_env)
                done = bool(terminated or truncated)
                states.append(obs)
                actions.append(a_np)
                rewards.append(float(reward))
                dones.append(done)
                logps.append(logp.cpu().item())
                values.append(value_t.squeeze(0).cpu().item())
                obs = flatten_obs(next_obs)
                timestep += 1
                if done:
                    ep += 1
                    obs, _ = train_env.reset()
                    obs = flatten_obs(obs)
            # compute returns and advantages
            returns = []
            R = 0.0
            for r, d in zip(reversed(rewards), reversed(dones)):
                if d:
                    R = 0.0
                R = r + gamma * R
                returns.insert(0, R)
            returns = torch.tensor(returns, dtype=torch.float32, device=device).unsqueeze(1)
            values_t = torch.tensor(values, dtype=torch.float32, device=device).unsqueeze(1)
            advantages = returns - values_t

            # Update networks
            states_t = torch.tensor(np.stack(states), dtype=torch.float32, device=device)
            actions_t = torch.tensor(np.stack(actions), dtype=torch.float32, device=device)
            logps_t = torch.tensor(logps, dtype=torch.float32, device=device).unsqueeze(1)

            mean, std = policy(states_t)
            dist = Normal(mean, std)
            new_logps = dist.log_prob(actions_t).sum(axis=-1, keepdim=True)
            entropy = dist.entropy().sum(axis=-1).mean()

            # Policy loss (policy gradient with advantage)
            pg_loss = - (new_logps * advantages.detach()).mean()
            # Value loss
            value_preds = value(states_t)
            v_loss = nn.functional.mse_loss(value_preds, returns)

            loss = pg_loss + value_coef * v_loss - entropy_coef * entropy

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print(f"[TRAIN] timestep {timestep}/{total_timesteps} ep {ep} loss {loss.item():.4f} pg {pg_loss.item():.4f} v {v_loss.item():.4f} ent {entropy.item():.4f}")

        # Save simple actor/critic parameters
        torch.save({
            "policy_state_dict": policy.state_dict(),
            "value_state_dict": value.state_dict(),
            "obs_dim": obs_dim,
            "act_dim": act_dim
        }, filename + '/final_model_actor_critic.pt')
        print("Saved actor-critic model to", filename + '/final_model_actor_critic.pt')
    elif not eval_only and multiagent:
        # If multiagent we used SB3 PPO above: train with that
        model = PPO('MlpPolicy',
                    train_env,
                    verbose=1, device="cpu")
        model.learn(total_timesteps=int(1e7) if local else int(1e2))
        model.save(filename+'/final_model.zip')
        print(filename)
    else:
        # eval_only path: skip training and continue to loading/evaluation
        pass

    #### Print training progression (if available) ################
    eval_file = filename + '/evaluations.npz'
    if os.path.isfile(eval_file):
        try:
            with np.load(eval_file) as data:
                for j in range(data['timesteps'].shape[0]):
                    print(str(data['timesteps'][j])+","+str(data['results'][j][0]))
        except Exception as e:
            print(f"[WARN] could not read evaluations.npz: {e}")
    else:
        print(f"[INFO] No evaluations.npz found at {eval_file}, skipping evaluation progression print.")

    ############################################################
    ############################################################
    ############################################################
    ############################################################
    ############################################################

    if local:
        input("Press Enter to continue...")

    # if os.path.isfile(filename+'/final_model.zip'):
    #     path = filename+'/final_model.zip'
    # Load actor-critic model if present, otherwise try SB3 saved models
    # decide which model file to try to load
    ac_path = None
    if model_path and os.path.isfile(model_path):
        ac_path = model_path
    else:
        ac_path = filename + '/final_model_actor_critic.pt'
    if os.path.isfile(ac_path) and not multiagent:
        # load custom actor-critic
        data = torch.load(ac_path, map_location='cpu')
        # Reconstruct policy for evaluation
        obs_dim = data['obs_dim']; act_dim = data['act_dim']
        class EvalPolicy(nn.Module):
            def __init__(self, obs_dim, act_dim):
                super().__init__()
                self.net = nn.Sequential(nn.Linear(obs_dim, 256), nn.Tanh(), nn.Linear(256, 256), nn.Tanh())
                self.mean = nn.Linear(256, act_dim)
                self.log_std = nn.Parameter(torch.zeros(act_dim))
            def forward(self, x):
                h = self.net(x)
                return self.mean(h), self.log_std.exp()
        policy = EvalPolicy(obs_dim, act_dim)
        policy.load_state_dict(data['policy_state_dict'])
        policy.eval()
        model = policy
    else:
        if os.path.isfile(filename+'/best_model.zip'):
            path = filename+'/best_model.zip'
            from stable_baselines3 import PPO
            model = PPO.load(path)
        else:
            print("[ERROR]: no model under the specified path", filename)
            model = None

    #### Show (and record a video of) the model's performance ##
    if not multiagent:
        test_env = HoverAviary(gui=gui,
                               obs=DEFAULT_OBS,
                               act=DEFAULT_ACT,
                               record=record_video)
        test_env_nogui = HoverAviary(obs=DEFAULT_OBS, act=DEFAULT_ACT)
    else:
        test_env = MultiHoverAviary(gui=gui,
                                        num_drones=DEFAULT_AGENTS,
                                        obs=DEFAULT_OBS,
                                        act=DEFAULT_ACT,
                                        record=record_video)
        test_env_nogui = MultiHoverAviary(num_drones=DEFAULT_AGENTS, obs=DEFAULT_OBS, act=DEFAULT_ACT)
    logger = Logger(logging_freq_hz=int(test_env.CTRL_FREQ),
                num_drones=DEFAULT_AGENTS if multiagent else 1,
                output_folder=output_folder,
                colab=colab
                )

    # Evaluate: if we have a custom policy object, run deterministic episodes
    if model is not None:
        if not multiagent and isinstance(model, torch.nn.Module):
            rewards = []
            for ep_eval in range(5):
                obs, info = test_env_nogui.reset()
                done = False; ep_rew = 0.0
                while not done:
                    obs_arr = torch.tensor(np.asarray(obs).ravel(), dtype=torch.float32)
                    with torch.no_grad():
                        mean, std = model(obs_arr.unsqueeze(0))
                        action = mean.squeeze(0).cpu().numpy()
                    if action.ndim == 1:
                        action_env = np.expand_dims(action, axis=0)
                    else:
                        action_env = action
                    obs, reward, terminated, truncated, info = test_env_nogui.step(action_env)
                    done = bool(terminated or truncated)
                    ep_rew += float(reward)
                rewards.append(ep_rew)
            print("\n\n\nMean reward ", np.mean(rewards), " +- ", np.std(rewards), "\n\n")
        else:
            # fallback to SB3 evaluation if we loaded a SB3 model
            try:
                from stable_baselines3.common.evaluation import evaluate_policy
                mean_reward, std_reward = evaluate_policy(model,
                                                          test_env_nogui,
                                                          n_eval_episodes=10
                                                          )
                print("\n\n\nMean reward ", mean_reward, " +- ", std_reward, "\n\n")
            except Exception:
                print("No evaluation available for the loaded model type.")

    obs, info = test_env.reset(seed=42, options={})
    start = time.time()
    for i in range((test_env.EPISODE_LEN_SEC+2)*test_env.CTRL_FREQ):
        # Model can be a SB3 object (has predict) or a torch.nn.Module (our custom policy)
        if isinstance(model, torch.nn.Module):
            # prepare obs and run forward pass
            obs_arr = torch.tensor(np.asarray(obs).ravel(), dtype=torch.float32)
            with torch.no_grad():
                mean, std = model(obs_arr.unsqueeze(0))
                action_np = mean.squeeze(0).cpu().numpy()
            if action_np.ndim == 1:
                action = np.expand_dims(action_np, axis=0)
            else:
                action = action_np
            _states = None
        else:
            action, _states = model.predict(obs, deterministic=True)

        obs, reward, terminated, truncated, info = test_env.step(action)
        obs2 = obs.squeeze()
        act2 = action.squeeze()
        print("Obs:", obs, "\tAction", action, "\tReward:", reward, "\tTerminated:", terminated, "\tTruncated:", truncated)
        if DEFAULT_OBS == ObservationType.KIN:
            if not multiagent:
                logger.log(drone=0,
                    timestamp=i/test_env.CTRL_FREQ,
                    state=np.hstack([obs2[0:3],
                                        np.zeros(4),
                                        obs2[3:15],
                                        act2
                                        ]),
                    control=np.zeros(12)
                    )
            else:
                for d in range(DEFAULT_AGENTS):
                    logger.log(drone=d,
                        timestamp=i/test_env.CTRL_FREQ,
                        state=np.hstack([obs2[d][0:3],
                                            np.zeros(4),
                                            obs2[d][3:15],
                                            act2[d]
                                            ]),
                        control=np.zeros(12)
                        )
        test_env.render()
        print(terminated)
        sync(i, start, test_env.CTRL_TIMESTEP)
        if terminated:
            obs = test_env.reset(seed=42, options={})
    test_env.close()

    if plot and DEFAULT_OBS == ObservationType.KIN:
        logger.plot()

if __name__ == '__main__':
    #### Define and parse (optional) arguments for the script ##
    parser = argparse.ArgumentParser(description='Single agent reinforcement learning example script')
    parser.add_argument('--multiagent',         default=DEFAULT_MA,            type=str2bool,      help='Whether to use example LeaderFollower instead of Hover (default: False)', metavar='')
    parser.add_argument('--gui',                default=DEFAULT_GUI,           type=str2bool,      help='Whether to use PyBullet GUI (default: True)', metavar='')
    parser.add_argument('--record_video',       default=DEFAULT_RECORD_VIDEO,  type=str2bool,      help='Whether to record a video (default: False)', metavar='')
    parser.add_argument('--output_folder',      default=DEFAULT_OUTPUT_FOLDER, type=str,           help='Folder where to save logs (default: "results")', metavar='')
    parser.add_argument('--colab',              default=DEFAULT_COLAB,         type=bool,          help='Whether example is being run by a notebook (default: "False")', metavar='')
    parser.add_argument('--local',              default=DEFAULT_LOCAL,         type=str2bool,      help='Run locally (long training) or short smoke test (default: True)', metavar='')
    parser.add_argument('--eval',               default=False,                 type=str2bool,      help='Run evaluation only, skip training', metavar='')
    parser.add_argument('--model_path',         default=None,                  type=str,           help='Path to the model file for evaluation', metavar='')
    ARGS = parser.parse_args()

    argsd = vars(ARGS)
    eval_only = argsd.pop('eval', False)
    model_path = argsd.pop('model_path', None)
    # remove local from argsd to avoid passing it twice; we'll pass explicitly
    local_flag = argsd.pop('local', DEFAULT_LOCAL)
    if eval_only:
        # Run with evaluation only: force local=False for shorter runs or test mode
        run(**argsd, local=False, eval_only=True, model_path=model_path)
    else:
        run(**argsd, local=local_flag)
