# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
import torch
import torch.nn.functional as F
import sys
import os
import deepspeed
import numpy as np
from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

from utils.utils import print_rank_0


def print_all_ranks(tag, value, rank):
    world_size = torch.distributed.get_world_size()
    all_tensor = torch.zeros(world_size, dtype=torch.float32).cuda()
    all_tensor[rank] = value
    torch.distributed.all_reduce(all_tensor, op=torch.distributed.ReduceOp.SUM)
    print_rank_0(f'{tag} {all_tensor}', rank)


def get_model_norm(model):
    with torch.no_grad():
        total = 0.0
        for param in model.parameters():
            should_gather = hasattr(
                param,
                'ds_id') and param.ds_status == ZeroParamStatus.NOT_AVAILABLE
            with deepspeed.zero.GatheredParameters(param,
                                                   enabled=should_gather):
                total += float(param.float().norm())

    return total


def gather_log_probs(logits, labels):
    log_probs = F.log_softmax(logits, dim=-1)
    log_probs_labels = log_probs.gather(dim=-1, index=labels.unsqueeze(-1))
    return log_probs_labels.squeeze(-1)


class AdaptiveKLController:
    """
    Adaptive KL controller described in the paper:
    https://arxiv.org/pdf/1909.08593.pdf
    """

    def __init__(self, init_kl_coef, target, horizon):
        self.value = init_kl_coef
        self.target = target
        self.horizon = horizon

    def update(self, current, n_steps):
        target = self.target
        proportional_error = np.clip(current / target - 1, -0.2, 0.2)
        mult = 1 + proportional_error * n_steps / self.horizon
        self.value *= mult
class FixedKLController:
    """Fixed KL controller."""

    def __init__(self, kl_coef):
        self.value = kl_coef

    def update(self, current, n_steps):
        pass
class DeepSpeedPPOTrainer():

    def __init__(self, rlhf_engine, args):
        self.rlhf_engine = rlhf_engine
        self.actor_model = self.rlhf_engine.actor
        self.critic_model = self.rlhf_engine.critic
        self.ref_model = self.rlhf_engine.ref
        self.reward_model = self.rlhf_engine.reward
        self.tokenizer = self.rlhf_engine.actor_tokenizer
        self.reward_tokenizer = self.rlhf_engine.reward_tokenizer
        self.args = args
        self.max_answer_seq_len = args.max_answer_seq_len
        # self.end_of_conversation_token_id = self.tokenizer(
        #     args.end_of_conversation_token)['input_ids'][-1]
        self.end_of_conversation_token_id = self.tokenizer.encode(args.end_of_conversation_token)[-1]
        # Those value can be changed
        # self.kl_ctl = 0.05 # 0.02
        self.device_num = torch.cuda.device_count()
        self.target = 6
        self.init_kl_coef = 0.4
        self.horizon = 10000
        # self.kl_ctl = AdaptiveKLController(self.init_kl_coef, self.target, self.horizon)
        self.kl_ctl = FixedKLController(self.init_kl_coef)
        self.clip_reward_value = 5
        self.cliprange = 0.2
        self.cliprange_value = 0.2
        self.gamma = 1.0
        self.lam = 0.95

    def _generate_sequence(self, prompts,attention_mask):

        with torch.no_grad():
            seq = self.actor_model.module.generate(prompts,
                                                   attention_mask = attention_mask,
                                                   max_length= self.max_answer_seq_len + prompts.shape[1],
                                                   # min_length=max_min_length,
                                                   num_beams=3,
                                                   do_sample=True,
                                                   temperature=0.95,
                                                   synced_gpus=True,
                                                   )

        # Filter out seq with no asnwers (or very short). This happens when users directly use the pre-training ckpt without supervised finetuning
        # NOTE: this will causes each GPU has different number of examples
        batch_size = seq.shape[0]


        prompt_length = prompts.shape[1]
        ans = seq[:, prompt_length:]
        self.prompt_length = prompt_length
        valid_ans_len = (ans != self.tokenizer.pad_token_id).sum(dim=-1)
        out_seq = []
        for i in range(batch_size):
            if valid_ans_len[i] <= 1:  # if the answer is shorter than 1 token, drop it
                continue
            else:
                out_seq.append(seq[i:i + 1])
        try:
            out_seq = torch.cat(out_seq, dim=0)  # concate output in the batch dim
        except Exception as e: # 容易出现out_seq为空的情况？
            print_rank_0(valid_ans_len)
            print_rank_0(seq[0])
            print_rank_0(seq)
            exit()
        return out_seq

    def generate_experience(self, prompts,attention_mask, args):
        self.eval()
        seq = self._generate_sequence(prompts,attention_mask)
        self.train()

        pad_token_id = self.tokenizer.pad_token_id
        attention_mask = seq.not_equal(pad_token_id).long()

        with torch.no_grad():
            output = self.actor_model(seq.long(), attention_mask=attention_mask.long())
            output_ref = self.ref_model(seq.long(), attention_mask=attention_mask.long())
            reward_score = self.reward_model.forward_value(
                seq.long(), attention_mask.long())['chosen_end_scores'].detach(
                )

            # values = self.critic_model.forward_value(
            #     reward_input_ids.long(), reward_attention_mask.long(), return_value_only=True).detach()[:, :-1]
            # critic 和 actor 同一个初始化
            values = self.critic_model.forward_value(
                seq, attention_mask.long(), return_value_only=True).detach()

        # if args.global_rank == 0:
        #     # print(decoded)
        #     # from IPython import embed; embed(header = '')
        #     import IPython;import sys; IPython.embed(header = f'file:\n{__file__}\nline:{sys._getframe().f_lineno}')
            
        logits = output.logits
        logits_ref = output_ref.logits

        return {
            'prompts': prompts,
            'logprobs': gather_log_probs(logits[:, :-1, :], seq[:, 1:]),
            'ref_logprobs': gather_log_probs(logits_ref[:, :-1, :], seq[:,
                                                                        1:]),
            'value': values,
            'rewards': reward_score,
            'input_ids': seq,
            "attention_mask": attention_mask,
        }

    def compute_rewards(self, prompts, log_probs, ref_log_probs, reward_score,
                        action_mask):
        # kl = torch.exp(log_probs) 直接计算不行
        try:
            kl_divergence_estimate = -self.kl_ctl.value * torch.clamp(log_probs - ref_log_probs, min=0.0) # * torch.min(log_probs - ref_log_probs,0)
        except Exception as e:
            print(torch.clamp(log_probs - ref_log_probs, min=0.0))
            print(self.kl_ctl.value)
            exit()

        # exit()
        batch_kl_mean = (kl_divergence_estimate * action_mask).sum(axis=-1).mean()
        rewards = kl_divergence_estimate # batch_size * seq_len
        start = prompts.shape[1] - 1
        ends = start + action_mask[:, start:].sum(1)
        reward_clip = torch.clamp(reward_score, -self.clip_reward_value,
                                  self.clip_reward_value)
        batch_size = log_probs.shape[0]
        for j in range(batch_size):
            rewards[j, start:ends[j]][-1] += reward_clip[j]

        return rewards,batch_kl_mean

    def masked_mean(self,values, mask, axis=None):
        """Compute mean of tensor with a masked values."""
        if axis is not None:
            return (values * mask).sum(axis=axis) / mask.sum(axis=axis)
        else:
            return (values * mask).sum() / mask.sum()

    def masked_var(self,values, mask, unbiased=True):
        """Compute variance of tensor with masked values."""
        mean = self.masked_mean(values, mask)
        centered_values = values - mean
        variance = self.masked_mean(centered_values ** 2, mask)
        if unbiased:
            mask_sum = mask.sum()
            if mask_sum == 0:
                raise ValueError(
                    "The sum of the mask is zero, which can happen when `mini_batch_size=1`;"
                    "try increase the `mini_batch_size` or `gradient_accumulation_steps`"
                )
            # note that if mask_sum == 1, then there is a division by zero issue
            # to avoid it you just need to use a larger minibatch_size
            bessel_correction = mask_sum / (mask_sum - 1)
            variance = variance * bessel_correction
        return variance

    def masked_whiten(self,values, mask, shift_mean=True):
        """Whiten values with masked values."""
        mean, var = self.masked_mean(values, mask), self.masked_var(values, mask)
        whitened = (values - mean) * torch.rsqrt(var + 1e-8)
        if not shift_mean:
            whitened += mean
        return whitened

    def whiten(self,values, shift_mean=True):
        """Whiten values."""
        mean, var = torch.mean(values), torch.var(values)
        whitened = (values - mean) * torch.rsqrt(var + 1e-8)
        if not shift_mean:
            whitened += mean
        return whitened

    def train_rlhf(self, inputs):
        # train the rlhf mode here
        ### process the old outputs
        prompts = inputs['prompts']
        log_probs = inputs['logprobs']
        ref_log_probs = inputs['ref_logprobs']
        reward_score = inputs['rewards'] # detached
        values = inputs['value'] # detached
        attention_mask = inputs['attention_mask']
        seq = inputs['input_ids']

        start = prompts.size()[-1] - 1
        action_mask = attention_mask[:, 1:]

        old_values = values
        with torch.no_grad(): # KL penalty detached
            old_rewards,batch_kl_mean = self.compute_rewards(prompts, log_probs,
                                               ref_log_probs, reward_score,
                                               action_mask)
            advantages, returns = self.get_advantages_and_returns(
                old_values, old_rewards, start, attention_mask)
        # advantages = self.masked_whiten(advantages,attention_mask[start: old_rewards.size(-1)])
        ### process the new outputs
        batch = {'input_ids': seq, "attention_mask": attention_mask}
        actor_prob = self.actor_model(**batch, use_cache=False).logits
        actor_log_prob = gather_log_probs(actor_prob[:, :-1, :],
                                          seq[:, 1:])

        actor_loss = self.actor_loss_fn(actor_log_prob[:, start:],
                                        log_probs[:, start:],
                                        advantages,
                                        action_mask[:, start:])
        self.actor_model.backward(actor_loss)
        self.actor_model.step()

        # value = self.critic_model.forward_value(**reward_batch,
        #                                         return_value_only=True,
        #                                         use_cache=False)[:, :-1]


        value = self.critic_model.forward_value(**batch,
                                                return_value_only=True,
                                                use_cache=False)

        critic_loss = self.critic_loss_fn(value[:, start:],
                                          old_values[:,start:],
                                          returns,
                                          action_mask[:, start:])
        self.critic_model.backward(critic_loss)
        self.critic_model.step()
        self.kl_ctl.update(batch_kl_mean.float().cpu().numpy(), self.args.per_device_train_batch_size * self.device_num)
        return actor_loss, critic_loss,torch.mean(advantages),batch_kl_mean

    def actor_loss_fn(self, logprobs, old_logprobs, advantages, mask):
        ## policy gradient loss
        log_ratio = (logprobs - old_logprobs) * mask
        ratio = torch.exp(log_ratio)
        pg_loss1 = -advantages * ratio
        pg_loss2 = -advantages * torch.clamp(ratio, 1.0 - self.cliprange,
                                             1.0 + self.cliprange)
        pg_loss = torch.sum(torch.max(pg_loss1, pg_loss2) * mask) / mask.sum()
        return pg_loss

    def critic_loss_fn(self, values, old_values, returns, mask):
        ## value loss
        values_clipped = torch.clamp(
            values,
            old_values - self.cliprange_value,
            old_values + self.cliprange_value,
        )
        length = returns.size()[-1]
        values = values[:,:length]
        values_clipped = values_clipped[:,:length]
        vf_loss1 = (values - returns)**2
        vf_loss2 = (values_clipped - returns)**2
        vf_loss = 0.5 * torch.sum(
            torch.max(vf_loss1, vf_loss2) * mask) / mask.sum()
        return vf_loss

    def get_advantages_and_returns(self, values, rewards, start,mask):
        # Adopted from https://github.com/CarperAI/trlx/blob/main/trlx/models/modeling_ppo.py#L134
        lastgaelam = 0
        advantages_reversed = []
        values = values * mask
        rewards = rewards * mask[:,1:]
        length = rewards.size()[-1] # 这里的length是batch的最后一个,后续多余的padding是用mask掩盖了的
        for t in reversed(range(start, length)):
            nextvalues = values[:, t + 1] if t < length - 1 else 0.0
            delta = rewards[:, t] + self.gamma * nextvalues - values[:, t]
            lastgaelam = delta + self.gamma * self.lam * lastgaelam
            advantages_reversed.append(lastgaelam)
        advantages = torch.stack(advantages_reversed[::-1], dim=1) # b * t
        returns = advantages + values[:, start: length]
        return advantages.detach(), returns

    def _validate_training_mode(self):
        assert self.actor_model.module.training
        assert self.critic_model.module.training

    def _validate_evaluation_mode(self):
        assert not self.actor_model.module.training
        assert not self.critic_model.module.training
        assert not self.ref_model.module.training
        assert not self.reward_model.module.training

    def train(self):
        self.actor_model.train()
        self.critic_model.train()

    def eval(self):
        self.actor_model.eval()
        self.critic_model.eval()
        self.reward_model.eval()
        self.ref_model.eval()

    def dump_model_norms(self, tag):
        actor_model_norm = get_model_norm(self.actor_model)
        ref_model_norm = get_model_norm(self.ref_model)
        critic_model_norm = get_model_norm(self.critic_model)
        reward_model_norm = get_model_norm(self.reward_model)
        print_all_ranks(f'{tag} global_actor_model_norm', actor_model_norm,
                        self.args.local_rank)
        print_all_ranks(f'{tag} global_ref_model_norm', ref_model_norm,
                        self.args.local_rank)
        print_all_ranks(f'{tag} global_critic_model_norm', critic_model_norm,
                        self.args.local_rank)
        print_all_ranks(f'{tag} global_reward_model_norm', reward_model_norm,
                        self.args.local_rank)


class DeepSpeedPPOTrainerUnsupervised(DeepSpeedPPOTrainer):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def train_unsupervised(self, inputs, unsup_coef):
        # Train the unsupervised model here
        self._validate_training_mode()

        outputs = self.actor_model(**inputs, use_cache=False)
        loss = outputs.loss
        self.actor_model.backward(unsup_coef * loss)
        self.actor_model.step()

        return loss
