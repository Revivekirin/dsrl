import torch
import wandb
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
import hydra


class DPPOBasePolicyWrapper:
	def __init__(self, base_policy):
		self.base_policy = base_policy
		
	def __call__(self, obs, initial_noise, return_numpy=True):
		cond = {
			"state": obs,
			"noise_action": initial_noise,
		}
		with torch.no_grad():
			samples = self.base_policy(cond=cond, deterministic=True)
		diffused_actions = (samples.trajectories.detach())
		if return_numpy:
			diffused_actions = diffused_actions.cpu().numpy()
		return diffused_actions	


def load_base_policy(cfg):
	base_policy = hydra.utils.instantiate(cfg.model)
	base_policy = base_policy.eval()
	return DPPOBasePolicyWrapper(base_policy)


class LoggingCallback(BaseCallback):
	def __init__(self, 
		action_chunk=4, 
		log_freq=1000,
		use_wandb=True, 
		eval_env=None, 
		eval_freq=70, 
		eval_episodes=2, 
		verbose=0, 
		rew_offset=0, 
		num_train_env=1,
		num_eval_env=1,
		algorithm='dsrl_sac',
		max_steps=-1,
		deterministic_eval=False,
	):
		super().__init__(verbose)
		self.action_chunk = action_chunk
		self.log_freq = log_freq
		self.episode_rewards = []
		self.episode_lengths = []
		self.use_wandb = use_wandb
		self.eval_env = eval_env
		self.eval_episodes = eval_episodes
		self.eval_freq = eval_freq
		self.log_count = 0
		self.total_reward = 0
		self.rew_offset = rew_offset
		self.total_timesteps = 0
		self.num_train_env = num_train_env
		self.num_eval_env = num_eval_env
		self.episode_success = np.zeros(self.num_train_env)
		self.episode_completed = np.zeros(self.num_train_env)
		self.algorithm = algorithm
		self.max_steps = max_steps
		self.deterministic_eval = deterministic_eval

	def _on_step(self):
		for info in self.locals['infos']:
			if 'episode' in info:
				self.episode_rewards.append(info['episode']['r'])
				self.episode_lengths.append(info['episode']['l'])
		rew = self.locals['rewards']
		self.total_reward += np.mean(rew)
		self.episode_success[rew > -self.rew_offset] = 1
		self.episode_completed[self.locals['dones']] = 1
		self.total_timesteps += self.action_chunk * self.model.n_envs
		if self.n_calls % self.log_freq == 0:
			if len(self.episode_rewards) > 0:
				if self.use_wandb:
					self.log_count += 1
					wandb.log({
						"train/ep_len_mean": np.mean(self.episode_lengths),
						"train/success_rate": np.sum(self.episode_success) / np.sum(self.episode_completed),
						"train/ep_rew_mean": np.mean(self.episode_rewards),
						"train/rew_mean": np.mean(self.total_reward),
						"train/timesteps": self.total_timesteps,
						"train/ent_coef": self.locals['self'].logger.name_to_value['train/ent_coef'],
						"train/actor_loss": self.locals['self'].logger.name_to_value['train/actor_loss'],
						"train/critic_loss": self.locals['self'].logger.name_to_value['train/critic_loss'],
						"train/ent_coef_loss": self.locals['self'].logger.name_to_value['train/ent_coef_loss'],
					}, step=self.log_count)
					if np.sum(self.episode_completed) > 0:
						wandb.log({
							"train/success_rate": np.sum(self.episode_success) / np.sum(self.episode_completed),
						}, step=self.log_count)
					if self.algorithm == 'dsrl_na':
						wandb.log({
							"train/noise_critic_loss": self.locals['self'].logger.name_to_value['train/noise_critic_loss'],
						}, step=self.log_count)
				self.episode_rewards = []
				self.episode_lengths = []
				self.total_reward = 0
				self.episode_success = np.zeros(self.num_train_env)
				self.episode_completed = np.zeros(self.num_train_env)

		if self.n_calls % self.eval_freq == 0:
			self.evaluate(self.locals['self'], deterministic=False)
			if self.deterministic_eval:
				self.evaluate(self.locals['self'], deterministic=True)
		return True
	
	def evaluate(self, agent, deterministic=False):
		if self.eval_episodes > 0:
			env = self.eval_env
			with torch.no_grad():
				success, rews = [], []
				rew_total, total_ep = 0, 0
				rew_ep = np.zeros(self.num_eval_env)
				for i in range(self.eval_episodes):
					obs = env.reset()
					success_i = np.zeros(obs.shape[0])
					r = []
					for _ in range(self.max_steps):
						if self.algorithm == 'dsrl_sac':
							action, _ = agent.predict(obs, deterministic=deterministic)
						elif self.algorithm == 'dsrl_na':
							action, _ = agent.predict_diffused(obs, deterministic=deterministic)
						elif self.algorithm == 'latent_fql':
							action, _ = agent.predict_diffused(obs, deterministic=deterministic)
							pass
						next_obs, reward, done, info = env.step(action)
						obs = next_obs
						rew_ep += reward
						rew_total += sum(rew_ep[done])
						rew_ep[done] = 0 
						total_ep += np.sum(done)
						success_i[reward > -self.rew_offset] = 1
						r.append(reward)
					success.append(success_i.mean())
					rews.append(np.mean(np.array(r)))
					print(f'eval episode {i} at timestep {self.total_timesteps}')
				success_rate = np.mean(success)
				if total_ep > 0:
					avg_rew = rew_total / total_ep
				else:
					avg_rew = 0
				if self.use_wandb:
					name = 'eval'
					if deterministic:
						wandb.log({
							f"{name}/success_rate_deterministic": success_rate,
							f"{name}/reward_deterministic": avg_rew,
						}, step=self.log_count)
					else:
						wandb.log({
							f"{name}/success_rate": success_rate,
							f"{name}/reward": avg_rew,
							f"{name}/timesteps": self.total_timesteps,
						}, step=self.log_count)

	def set_timesteps(self, timesteps):
		self.total_timesteps = timesteps



def collect_rollouts(model, env, num_steps, base_policy, cfg):
	obs = env.reset()
	for i in range(num_steps):
		noise = torch.randn(cfg.env.n_envs, cfg.act_steps, cfg.action_dim).to(device=cfg.device)
		if cfg.algorithm == 'dsrl_sac':
			noise[noise < -cfg.train.action_magnitude] = -cfg.train.action_magnitude
			noise[noise > cfg.train.action_magnitude] = cfg.train.action_magnitude
		action = base_policy(torch.tensor(obs, device=cfg.device, dtype=torch.float32), noise)
		next_obs, reward, done, info = env.step(action)
		if cfg.algorithm == 'dsrl_na':
			action_store = action
		elif cfg.algorithm == 'dsrl_sac':
			action_store = noise.detach().cpu().numpy()
		elif cfg.algorithm == 'latent_fql':
			action_store = action
		action_store = action_store.reshape(-1, action_store.shape[1] * action_store.shape[2])
		
		if cfg.algorithm == 'dsrl_sac':
			action_store = model.policy.scale_action(action_store)
		
		model.replay_buffer.add(
				obs=obs,
				next_obs=next_obs,
				action=action_store,
				reward=reward,
				done=done,
				infos=info,
			)
		obs = next_obs
	model.replay_buffer.final_offline_step()
	


# def load_offline_data(model, offline_data_path, n_env):
# 	# this function should only be applied with dsrl_na
# 	offline_data = np.load(offline_data_path)
# 	obs = offline_data['states']
# 	next_obs = offline_data['states_next']
# 	actions = offline_data['actions']
# 	rewards = offline_data['rewards']
# 	terminals = offline_data['terminals']

# 	for i in range(int(obs.shape[0]/n_env)):
# 		model.replay_buffer.add(
# 					obs=obs[n_env*i:n_env*i+n_env],
# 					next_obs=next_obs[n_env*i:n_env*i+n_env],
# 					action=actions[n_env*i:n_env*i+n_env],
# 					reward=rewards[n_env*i:n_env*i+n_env],
# 					done=terminals[n_env*i:n_env*i+n_env],
# 					infos=[{}] * n_env,
# 				)
# 	model.replay_buffer.final_offline_step()


def load_offline_data(model, offline_data_path, n_env: int):
    offline_data = np.load(offline_data_path)

    obs      = offline_data["states"].astype(np.float32)
    next_obs = offline_data["states_next"].astype(np.float32)
    actions  = offline_data["actions"].astype(np.float32)
    rewards  = offline_data["rewards"]
    terminals = offline_data["terminals"]

    if rewards.ndim == 2 and rewards.shape[1] == 1:
        rewards = rewards.squeeze(-1)
    if terminals.ndim == 2 and terminals.shape[1] == 1:
        terminals = terminals.squeeze(-1)

    rewards   = rewards.astype(np.float32)
    terminals = terminals.astype(np.float32)

    N = obs.shape[0]

    act_steps = getattr(model, "diffusion_act_chunk", None)
    if act_steps is None:
        raise ValueError("model.diffusion_act_chunk is missing")

    primitive_act_dim = actions.shape[1]
    env_action_dim    = model.replay_buffer.action_dim

    if env_action_dim != act_steps * primitive_act_dim:
        raise ValueError("env_action_dim mismatch")

    done_mask = terminals > 0.5
    done_indices = np.where(done_mask)[0]

    episode_ranges = []
    start = 0
    for idx in done_indices:
        episode_ranges.append((start, idx))
        start = idx + 1
    if start < N:
        episode_ranges.append((start, N - 1))

    chunk_states      = []
    chunk_states_next = []
    chunk_actions     = []
    chunk_rewards     = []
    chunk_dones       = []

    for (ep_start, ep_end) in episode_ranges:
        ep_len = ep_end - ep_start + 1
        if ep_len < act_steps:
            continue

        last_start = ep_end - act_steps + 1

        for t in range(ep_start, last_start + 1):
            s_t = obs[t]
            a_chunk = actions[t:t+act_steps].reshape(-1)
            r_chunk = rewards[t:t+act_steps].sum()
            idx_next = t + act_steps - 1
            s_next_chunk = next_obs[idx_next]
            d_chunk = float(np.max(terminals[t:t+act_steps]))

            chunk_states.append(s_t)
            chunk_states_next.append(s_next_chunk)
            chunk_actions.append(a_chunk)
            chunk_rewards.append(r_chunk)
            chunk_dones.append(d_chunk)

    chunk_states      = np.asarray(chunk_states, dtype=np.float32)
    chunk_states_next = np.asarray(chunk_states_next, dtype=np.float32)
    chunk_actions     = np.asarray(chunk_actions, dtype=np.float32)
    chunk_rewards     = np.asarray(chunk_rewards, dtype=np.float32)
    chunk_dones       = np.asarray(chunk_dones, dtype=np.float32)

    M = chunk_states.shape[0]
    print(f"[load_offline_data] chunked transitions = {M}")

    if M % n_env != 0:
        new_M = (M // n_env) * n_env
        print(f"[load_offline_data] drop last {M - new_M} items")
        M = new_M
        chunk_states      = chunk_states[:M]
        chunk_states_next = chunk_states_next[:M]
        chunk_actions     = chunk_actions[:M]
        chunk_rewards     = chunk_rewards[:M]
        chunk_dones       = chunk_dones[:M]

    num_batches = M // n_env
    for i in range(num_batches):
        start = i * n_env
        end   = start + n_env
        model.replay_buffer.add(
            obs=chunk_states[start:end],
            next_obs=chunk_states_next[start:end],
            action=chunk_actions[start:end],
            reward=chunk_rewards[start:end],
            done=chunk_dones[start:end],
            infos=[{}] * n_env,
        )

    if hasattr(model.replay_buffer, "final_offline_step"):
        model.replay_buffer.final_offline_step()
