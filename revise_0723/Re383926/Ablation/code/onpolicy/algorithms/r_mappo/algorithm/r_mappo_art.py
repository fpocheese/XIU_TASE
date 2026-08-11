import numpy as np
import torch
import torch.nn as nn
from onpolicy.utils.util import get_gard_norm, huber_loss, mse_loss
from onpolicy.utils.valuenorm import ValueNorm
from onpolicy.algorithms.utils.util import check


class R_MAPPO_ART():
    """
    改进的MAPPO训练器类，包含多项先进技术:
    1. Dual-clip PPO: 更好的策略更新稳定性
    2. 自适应KL惩罚: 动态调整策略更新幅度
    3. 价值函数热身: 前期专注训练Critic
    4. 梯度惩罚: 防止梯度爆炸
    5. 改进的优势估计
    """
    def __init__(self, args, policy, device=torch.device("cpu")):

        self.device = device
        self.tpdv = dict(dtype=torch.float32, device=device)
        self.policy = policy

        self.clip_param = args.clip_param
        self.ppo_epoch = args.ppo_epoch
        self.num_mini_batch = args.num_mini_batch
        self.data_chunk_length = args.data_chunk_length
        self.value_loss_coef = args.value_loss_coef
        self.entropy_coef = args.entropy_coef
        self.max_grad_norm = args.max_grad_norm       
        self.huber_delta = args.huber_delta

        self._use_recurrent_policy = args.use_recurrent_policy
        self._use_naive_recurrent = args.use_naive_recurrent_policy
        self._use_max_grad_norm = args.use_max_grad_norm
        self._use_clipped_value_loss = args.use_clipped_value_loss
        self._use_huber_loss = args.use_huber_loss
        self._use_popart = args.use_popart
        self._use_valuenorm = args.use_valuenorm
        self._use_value_active_masks = args.use_value_active_masks
        self._use_policy_active_masks = args.use_policy_active_masks
        
        # 新增的改进选项
        self._use_dual_clip = getattr(args, 'use_dual_clip', True)  # Dual-clip PPO
        self.dual_clip_param = getattr(args, 'dual_clip_param', 3.0)  # Dual clip阈值
        
        self._use_adaptive_kl = getattr(args, 'use_adaptive_kl', True)  # 自适应KL
        self.target_kl = getattr(args, 'target_kl', 0.02)  # 目标KL散度
        self.kl_coef = getattr(args, 'kl_coef', 0.01)  # KL系数（自适应调整）
        self.kl_coef_min = getattr(args, 'kl_coef_min', 1e-4)
        self.kl_coef_max = getattr(args, 'kl_coef_max', 1.0)
        
        self._use_value_warmup = getattr(args, 'use_value_warmup', True)  # 价值函数热身
        self.warmup_episodes = getattr(args, 'warmup_episodes', 100)  # 热身轮数
        self.current_episode = 0
        
        # GAE改进参数
        self._use_gae_norm = getattr(args, 'use_gae_norm', True)  # GAE归一化
        
        # 梯度惩罚
        self._use_grad_penalty = getattr(args, 'use_grad_penalty', False)
        self.grad_penalty_coef = getattr(args, 'grad_penalty_coef', 0.01)

        # Numerically equivalent protection for the PPO importance ratio.
        # Values outside this interval are already far beyond the PPO clipping
        # region; bounding the log-ratio prevents exp() overflow from turning a
        # finite mini-batch into NaN parameters.
        self.log_ratio_clip = float(getattr(args, 'log_ratio_clip', 20.0))
        
        assert (self._use_popart and self._use_valuenorm) == False, (
            "self._use_popart and self._use_valuenorm can not be set True simultaneously")
        
        if self._use_popart:
            self.value_normalizer = self.policy.critic.v_out
        elif self._use_valuenorm:
            self.value_normalizer = ValueNorm(1, device=self.device)
        else:
            self.value_normalizer = None

    def cal_value_loss(self, values, value_preds_batch, return_batch, active_masks_batch):
        """
        计算价值函数损失
        """
        value_pred_clipped = value_preds_batch + (values - value_preds_batch).clamp(
            -self.clip_param, self.clip_param
        )
        
        if self._use_popart or self._use_valuenorm:
            self.value_normalizer.update(return_batch)
            error_clipped = self.value_normalizer.normalize(return_batch) - value_pred_clipped
            error_original = self.value_normalizer.normalize(return_batch) - values
        else:
            error_clipped = return_batch - value_pred_clipped
            error_original = return_batch - values

        if self._use_huber_loss:
            value_loss_clipped = huber_loss(error_clipped, self.huber_delta)
            value_loss_original = huber_loss(error_original, self.huber_delta)
        else:
            value_loss_clipped = mse_loss(error_clipped)
            value_loss_original = mse_loss(error_original)

        if self._use_clipped_value_loss:
            value_loss = torch.max(value_loss_original, value_loss_clipped)
        else:
            value_loss = value_loss_original

        if self._use_value_active_masks:
            value_loss = (value_loss * active_masks_batch).sum() / active_masks_batch.sum().clamp_min(1.0)
        else:
            value_loss = value_loss.mean()

        return value_loss

    def ppo_update(self, sample, update_actor=True):
        """
        改进的PPO更新，包含Dual-clip和自适应KL
        """
        share_obs_batch, obs_batch, rnn_states_batch, rnn_states_critic_batch, actions_batch, \
        value_preds_batch, return_batch, masks_batch, active_masks_batch, old_action_log_probs_batch, \
        adv_targ, available_actions_batch = sample

        old_action_log_probs_batch = check(old_action_log_probs_batch).to(**self.tpdv)
        adv_targ = check(adv_targ).to(**self.tpdv)
        value_preds_batch = check(value_preds_batch).to(**self.tpdv)
        return_batch = check(return_batch).to(**self.tpdv)
        active_masks_batch = check(active_masks_batch).to(**self.tpdv)

        # Evaluate actions
        values, action_log_probs, dist_entropy = self.policy.evaluate_actions(
            share_obs_batch,
            obs_batch, 
            rnn_states_batch, 
            rnn_states_critic_batch, 
            actions_batch, 
            masks_batch, 
            available_actions_batch,
            active_masks_batch
        )
        
        # ==================== Actor Update ====================
        # 计算重要性采样权重
        log_ratio = action_log_probs - old_action_log_probs_batch
        imp_weights = torch.exp(
            torch.clamp(
                log_ratio,
                min=-self.log_ratio_clip,
                max=self.log_ratio_clip,
            )
        )

        # 标准PPO clip
        surr1 = imp_weights * adv_targ
        surr2 = torch.clamp(imp_weights, 1.0 - self.clip_param, 1.0 + self.clip_param) * adv_targ

        # Manuscript Eq. (dual clip): the extra floor is active only for
        # negative-advantage samples.
        clipped_surrogate = torch.min(surr1, surr2)
        if self._use_dual_clip:
            dual_clipped = torch.max(
                clipped_surrogate, self.dual_clip_param * adv_targ
            )
            surrogate = torch.where(
                adv_targ < 0.0, dual_clipped, clipped_surrogate
            )
            policy_action_loss_element = -surrogate
        else:
            policy_action_loss_element = -clipped_surrogate

        if self._use_policy_active_masks:
            policy_action_loss = (
                policy_action_loss_element * active_masks_batch
            ).sum() / active_masks_batch.sum().clamp_min(1.0)
        else:
            policy_action_loss = policy_action_loss_element.mean()

        # 计算KL散度用于自适应调整
        if self._use_policy_active_masks:
            approx_kl_tensor = (
                (old_action_log_probs_batch - action_log_probs)
                * active_masks_batch
            ).sum() / active_masks_batch.sum().clamp_min(1.0)
        else:
            approx_kl_tensor = (
                old_action_log_probs_batch - action_log_probs
            ).mean()
        approx_kl = float(approx_kl_tensor.detach().cpu())
        
        # 自适应KL惩罚
        if self._use_adaptive_kl:
            if approx_kl > self.target_kl * 2.0:
                self.kl_coef = min(self.kl_coef * 1.5, self.kl_coef_max)
            elif approx_kl < self.target_kl / 2.0:
                self.kl_coef = max(self.kl_coef * 0.9, self.kl_coef_min)
            
            # Keep the tensor form in the loss so the KL term has a gradient.
            kl_penalty = self.kl_coef * approx_kl_tensor
            policy_loss = policy_action_loss + kl_penalty
        else:
            policy_loss = policy_action_loss

        # Actor优化
        self.policy.actor_optimizer.zero_grad()

        # 判断是否在热身阶段（热身阶段不更新actor）
        is_warmup = self._use_value_warmup and self.current_episode < self.warmup_episodes
        
        if update_actor and not is_warmup:
            actor_objective = (
                policy_loss - dist_entropy * self.entropy_coef
            )
            if torch.isfinite(actor_objective):
                actor_objective.backward()

                if self._use_max_grad_norm:
                    actor_grad_norm = nn.utils.clip_grad_norm_(
                        self.policy.actor.parameters(),
                        self.max_grad_norm
                    )
                else:
                    actor_grad_norm = get_gard_norm(
                        self.policy.actor.parameters()
                    )

                grad_is_finite = bool(
                    torch.isfinite(
                        torch.as_tensor(actor_grad_norm)
                    ).item()
                )
                if grad_is_finite:
                    self.policy.actor_optimizer.step()
                else:
                    self.policy.actor_optimizer.zero_grad(set_to_none=True)
                    actor_grad_norm = 0.0
                    print(
                        "[NUMERICAL_GUARD] skipped actor step: "
                        "non-finite gradient norm",
                        flush=True,
                    )
            else:
                self.policy.actor_optimizer.zero_grad(set_to_none=True)
                actor_grad_norm = 0.0
                print(
                    "[NUMERICAL_GUARD] skipped actor step: "
                    "non-finite objective",
                    flush=True,
                )
        else:
            actor_grad_norm = 0.0

        # ==================== Critic Update ====================
        value_loss = self.cal_value_loss(values, value_preds_batch, return_batch, active_masks_batch)

        self.policy.critic_optimizer.zero_grad()

        critic_objective = value_loss * self.value_loss_coef
        if torch.isfinite(critic_objective):
            critic_objective.backward()

            if self._use_max_grad_norm:
                critic_grad_norm = nn.utils.clip_grad_norm_(
                    self.policy.critic.parameters(),
                    self.max_grad_norm
                )
            else:
                critic_grad_norm = get_gard_norm(
                    self.policy.critic.parameters()
                )

            grad_is_finite = bool(
                torch.isfinite(
                    torch.as_tensor(critic_grad_norm)
                ).item()
            )
            if grad_is_finite:
                self.policy.critic_optimizer.step()
            else:
                self.policy.critic_optimizer.zero_grad(set_to_none=True)
                critic_grad_norm = 0.0
                print(
                    "[NUMERICAL_GUARD] skipped critic step: "
                    "non-finite gradient norm",
                    flush=True,
                )
        else:
            self.policy.critic_optimizer.zero_grad(set_to_none=True)
            critic_grad_norm = 0.0
            print(
                "[NUMERICAL_GUARD] skipped critic step: "
                "non-finite objective",
                flush=True,
            )

        return value_loss, critic_grad_norm, policy_loss, dist_entropy, actor_grad_norm, imp_weights, approx_kl

    def train(self, buffer, update_actor=True):
        """
        执行训练更新
        """
        # 改进的优势估计归一化
        if self._use_popart or self._use_valuenorm:
            advantages = buffer.returns[:-1] - self.value_normalizer.denormalize(buffer.value_preds[:-1])
        else:
            advantages = buffer.returns[:-1] - buffer.value_preds[:-1]
        
        # 优势归一化
        advantages_copy = advantages.copy()
        advantages_copy[buffer.active_masks[:-1] == 0.0] = np.nan
        mean_advantages = np.nanmean(advantages_copy)
        std_advantages = np.nanstd(advantages_copy)
        advantages = (advantages - mean_advantages) / (std_advantages + 1e-5)

        train_info = {}
        train_info['value_loss'] = 0
        train_info['policy_loss'] = 0
        train_info['dist_entropy'] = 0
        train_info['actor_grad_norm'] = 0
        train_info['critic_grad_norm'] = 0
        train_info['ratio'] = 0
        train_info['kl_divergence'] = 0

        for _ in range(self.ppo_epoch):
            if self._use_recurrent_policy:
                data_generator = buffer.recurrent_generator(advantages, self.num_mini_batch, self.data_chunk_length)
            elif self._use_naive_recurrent:
                data_generator = buffer.naive_recurrent_generator(advantages, self.num_mini_batch)
            else:
                data_generator = buffer.feed_forward_generator(advantages, self.num_mini_batch)

            for sample in data_generator:
                value_loss, critic_grad_norm, policy_loss, dist_entropy, actor_grad_norm, imp_weights, approx_kl \
                    = self.ppo_update(sample, update_actor)

                train_info['value_loss'] += value_loss.item()
                train_info['policy_loss'] += policy_loss.item()
                train_info['dist_entropy'] += dist_entropy.item()
                train_info['actor_grad_norm'] += actor_grad_norm if isinstance(actor_grad_norm, float) else actor_grad_norm.item()
                train_info['critic_grad_norm'] += critic_grad_norm.item()
                train_info['ratio'] += imp_weights.mean().item()
                train_info['kl_divergence'] += approx_kl

        num_updates = self.ppo_epoch * self.num_mini_batch

        for k in train_info.keys():
            train_info[k] /= num_updates
        
        # 更新episode计数（用于热身判断）
        self.current_episode += 1
        
        # 添加额外的训练信息
        train_info['current_episode'] = self.current_episode
        train_info['kl_coef'] = self.kl_coef if self._use_adaptive_kl else 0.0
        train_info['is_warmup'] = self._use_value_warmup and self.current_episode < self.warmup_episodes
 
        return train_info

    def prep_training(self):
        """设置为训练模式"""
        self.policy.actor.train()
        self.policy.critic.train()

    def prep_rollout(self):
        """设置为推理模式"""
        self.policy.actor.eval()
        self.policy.critic.eval()
