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

def _ensure_info_container(info, n):
    """VecEnv에서 info가 dict/None/list인지와 무관하게 리스트[dict]로 변환"""
    if isinstance(info, list):
        out = []
        for i in range(n):
            d = info[i] if (i < len(info) and isinstance(info[i], dict)) else {}
            out.append(d)
        return out
    # dict 또는 None이면 모든 env에 동일 사본을 붙인다
    base = info if isinstance(info, dict) else {}
    return [dict(base) for _ in range(n)]


# TODO : 밑의 코드와 비교 수정
def collect_rollouts(model, env, num_steps, base_policy, cfg):
    obs = env.reset()
    n_envs = cfg.env.n_envs
    Ta, Da = cfg.act_steps, cfg.action_dim

    for _ in range(num_steps):
        # -------- 행동 생성 --------
        if cfg.algorithm == 'latent_fql':
            # FQL 모델에서 직접 (action_flat, zprime) 생성
            obs_t = torch.tensor(obs, device=cfg.device, dtype=torch.float32)
            action_flat, zprime = model._act_from_latent(obs_t)         # (N, Ta*Da), (N, Dz)
            action = action_flat.reshape(n_envs, Ta, Da)                 # env 입력용
        else:
            # 기존 DSRL/DSRL-SAC 경로
            noise = torch.randn(n_envs, Ta, Da, device=cfg.device)
            if cfg.algorithm == 'dsrl_sac':
                noise.clamp_(-cfg.train.action_magnitude, cfg.train.action_magnitude)
            action = base_policy(torch.tensor(obs, device=cfg.device, dtype=torch.float32), noise)

        # -------- 환경 step --------
        next_obs, reward, done, info = env.step(action)

        # -------- 버퍼 저장용 가공 --------
        if cfg.algorithm == 'latent_fql':
            action_store = action.reshape(n_envs, -1)  # (N, Ta*Da)
            # info를 리스트[dict]로 표준화하고 z′를 주입
            info_list = _ensure_info_container(info, n_envs)
            z_np = zprime.detach().cpu().numpy()
            for i in range(n_envs):
                info_list[i]["latent"] = z_np[i]
            info = info_list
        elif cfg.algorithm == 'dsrl_sac':
            action_store = model.policy.scale_action(
                action.detach().cpu().numpy().reshape(n_envs, -1)
            )
        else:  # dsrl_na
            action_store = action.reshape(n_envs, -1)

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


# def collect_rollouts(model, env, num_steps, base_policy, cfg):
# 	obs = env.reset()
# 	for i in range(num_steps):
# 		noise = torch.randn(cfg.env.n_envs, cfg.act_steps, cfg.action_dim).to(device=cfg.device)
# 		if cfg.algorithm == 'dsrl_sac':
# 			noise[noise < -cfg.train.action_magnitude] = -cfg.train.action_magnitude
# 			noise[noise > cfg.train.action_magnitude] = cfg.train.action_magnitude
# 		action = base_policy(torch.tensor(obs, device=cfg.device, dtype=torch.float32), noise)
# 		next_obs, reward, done, info = env.step(action)
		
# 		# Store _last_latent in replay buffer for latent_fql
# 		info = info or {}
# 		if cfg.algorithm == 'latent_fql':
# 			print("[DEBUG] last latent ", model._last_latent)
# 			info["latent"] = model._last_latent.detach().cpu().numpy() 

# 		if cfg.algorithm == 'dsrl_na':
# 			action_store = action
# 		elif cfg.algorithm == 'dsrl_sac':
# 			action_store = noise.detach().cpu().numpy()
# 		elif cfg.algorithm == 'latent_fql':
# 			action_store = action

# 		action_store = action_store.reshape(-1, action_store.shape[1] * action_store.shape[2])
# 		if cfg.algorithm == 'dsrl_sac':
# 			action_store = model.policy.scale_action(action_store)
# 		model.replay_buffer.add(
# 				obs=obs,
# 				next_obs=next_obs,
# 				action=action_store,
# 				reward=reward,
# 				done=done,
# 				infos=info,
# 			)
# 		obs = next_obs
# 	model.replay_buffer.final_offline_step()
	


def load_offline_data(model, offline_data_path, n_env):
	# this function should only be applied with dsrl_na
	offline_data = np.load(offline_data_path)
	obs = offline_data['states']
	next_obs = offline_data['states_next']
	actions = offline_data['actions']
	rewards = offline_data['rewards']
	terminals = offline_data['terminals']
	for i in range(int(obs.shape[0]/n_env)):
		model.replay_buffer.add(
					obs=obs[n_env*i:n_env*i+n_env],
					next_obs=next_obs[n_env*i:n_env*i+n_env],
					action=actions[n_env*i:n_env*i+n_env],
					reward=rewards[n_env*i:n_env*i+n_env],
					done=terminals[n_env*i:n_env*i+n_env],
					infos=[{}] * n_env,
				)
	model.replay_buffer.final_offline_step()