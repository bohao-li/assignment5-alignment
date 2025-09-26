import torch
from typing import Callable
from typing import Literal

def compute_group_normalized_rewards(
    reward_fn: Callable,
    rollout_responses: list[str],
    repeated_ground_truths: list[str],
    group_size: int,
    advantage_eps: float,
    normalize_by_std: bool,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    """
    Computes rewards for each group of rollout responses, normalized by the group size.

    Args:
        reward_fn: Callable[[str, str], dict[str, float]] Scores the rollout responses against
            the ground truths, producing a dict with keys "reward", "format_reward", and
            "answer_reward".
        rollout_responses: list[str] Rollouts from the policy. The length of this list is
            rollout_batch_size = n_prompts_per_rollout_batch * group_size.
        repeated_ground_truths: list[str] The ground truths for the examples. The length of this
            list is rollout_batch_size, because the ground truth for each example is repeated
            group_size times.
        group_size: int Number of responses per question (group).
        advantage_eps: float Small constant to avoid division by zero in normalization.
        normalize_by_std: bool If True, divide by the per-group standard deviation; otherwise
            subtract only the group mean.

    Returns:
        tuple[torch.Tensor, torch.Tensor, dict[str, float]].
        advantages shape (rollout_batch_size,). Group-normalized rewards for each rollout
            response.
        raw_rewards shape (rollout_batch_size,). Unnormalized rewards for each rollout
            response.
        metadata your choice of other statistics to log (e.g. mean, std, max/min of rewards).
    """
    raw_rewards = []
    # Calculate raw rewards for all responses
    for response, ground_truth in zip(rollout_responses, repeated_ground_truths):
        score_dict = reward_fn(response, ground_truth)
        raw_rewards.append(score_dict["reward"])
    
    raw_rewards = torch.tensor(raw_rewards, dtype=torch.float32)
    rollout_batch_size = len(rollout_responses)
    
    # Reshape rewards into groups
    raw_rewards_reshaped = raw_rewards.view(-1, group_size)
    
    # Calculate group means and standard deviations
    group_means = raw_rewards_reshaped.mean(dim=-1, keepdim=True)
    group_stds = raw_rewards_reshaped.std(dim=-1, keepdim=True)

    # Normalize rewards within each group
    advantages = raw_rewards_reshaped - group_means
    
    if normalize_by_std:
        advantages = advantages / (group_stds + advantage_eps)
    
    # Flatten advantages back to original shape
    advantages = advantages.view(-1)
    
    # Prepare metadata for logging
    metadata = {
        "mean_raw_reward": raw_rewards.mean().item(),
        "std_raw_reward": raw_rewards.std().item(),
        "max_raw_reward": raw_rewards.max().item(),
        "min_raw_reward": raw_rewards.min().item(),
    }
    
    return advantages, raw_rewards, metadata


def compute_naive_policy_gradient_loss(
    raw_rewards_or_advantages: torch.Tensor,
    policy_log_probs: torch.Tensor,
) -> torch.Tensor:
    """
    raw_rewards_or_advantages: torch.Tensor Shape (batch_size, 1), scalar
        reward/advantage for each rollout response.
    policy_log_probs: torch.Tensor Shape (batch_size, sequence_length), logprobs for
        each token.
    Returns:
        torch.Tensor Shape (batch_size, sequence_length), the per-token policy
            gradient loss.
    """
    sequence_length = policy_log_probs.shape[1]
    broadcasted_rewards = raw_rewards_or_advantages.expand(-1, sequence_length)
    return -policy_log_probs * broadcasted_rewards

def compute_grpo_clip_loss(
    advantages: torch.Tensor,
    policy_log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    cliprange: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """
    Compute the GRPO-Clip loss.

    Args:
        advantages: torch.Tensor Shape (batch_size, 1), the advantages for each rollout response.
        policy_log_probs: torch.Tensor Shape (batch_size, sequence_length), the log-probs of the policy.
        old_log_probs: torch.Tensor Shape (batch_size, sequence_length), the log-probs of the old policy.
        cliprange: float, the clip range for the ratio.

    Returns:
        tuple[torch.Tensor, dict[str, torch.Tensor]]:
            torch.Tensor Shape (batch_size, sequence_length), the GRPO-Clip per-token loss.
            dict[str, torch.Tensor]: metadata for the GRPO-Clip loss (used to compute clip fraction).
    """
    broadcasted_advantages = advantages.expand(-1, policy_log_probs.shape[1])

    # Compute the ratio
    ratio = torch.exp(policy_log_probs - old_log_probs)

    surr1 = ratio * broadcasted_advantages
    surr2 = torch.clamp(ratio, 1.0 - cliprange, 1.0 + cliprange) * broadcasted_advantages

    per_token_advantages = torch.min(surr1, surr2)
    loss = -per_token_advantages
    
    approx_kl = ((ratio - 1) - (policy_log_probs - old_log_probs)).mean()
    metadata = {
        "mean_ratio": ratio.mean(),
        "approx_kl": approx_kl,
    }

    return loss, metadata


def compute_policy_gradient_loss(
    policy_log_probs: torch.Tensor,
    loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip"],
    raw_rewards: torch.Tensor | None = None,
    advantages: torch.Tensor | None = None,
    old_log_probs: torch.Tensor | None = None,
    cliprange: float | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """
    Selects and computes the desired policy-gradient loss.

    Args:
        policy_log_probs (batch_size, sequence_length), per-token log-probabilities from the
            policy being trained.
        loss_type: One of "no_baseline", "reinforce_with_baseline", or "grpo_clip".
        raw_rewards: Required if loss_type == "no_baseline"; shape (batch_size, 1).
        advantages: Required for "reinforce_with_baseline" and "grpo_clip"; shape
            (batch_size, 1).
        old_log_probs: Required for "grpo_clip"; shape (batch_size, sequence_length).
        cliprange: Required for "grpo_clip"; scalar ϵ used for clipping.

    Returns:
        tuple[torch.Tensor, dict[str, torch.Tensor]].
        loss: (batch_size, sequence_length), per-token loss.
        metadata: dict, statistics from the underlying routine (e.g., clip fraction for GRPO-Clip).
    """
    metadata = {}

    if loss_type == "no_baseline":
        assert raw_rewards is not None, "raw_rewards is required for 'no_baseline' loss_type."
        loss = compute_naive_policy_gradient_loss(raw_rewards, policy_log_probs)

    elif loss_type == "reinforce_with_baseline":
        assert advantages is not None, "advantages is required for 'reinforce_with_baseline' loss_type."
        loss = compute_naive_policy_gradient_loss(advantages, policy_log_probs)

    elif loss_type == "grpo_clip":
        assert advantages is not None, "advantages is required for 'grpo_clip' loss_type."
        assert old_log_probs is not None, "old_log_probs is required for 'grpo_clip' loss_type."
        assert cliprange is not None, "cliprange is required for 'grpo_clip' loss_type."
        
        loss, grpo_metadata = compute_grpo_clip_loss(
            advantages=advantages,
            policy_log_probs=policy_log_probs,
            old_log_probs=old_log_probs,
            cliprange=cliprange,
        )
        metadata.update(grpo_metadata)

    else:
        raise ValueError(f"Unknown loss_type: {loss_type}")

    return loss, metadata


def masked_mean(
    tensor: torch.Tensor,
    mask: torch.Tensor,
    dim: int | None= None,
) -> torch.Tensor:
    masked_tensor = tensor * mask
    if dim is None:
        return masked_tensor.sum() / mask.sum()
    else:
        return masked_tensor.sum(dim=dim) / mask.sum(dim=dim)


def grpo_microbatch_train_step(
    policy_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    gradient_accumulation_steps: int,
    loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip"],
    raw_rewards: torch.Tensor | None= None,
    advantages: torch.Tensor | None= None,
    old_log_probs: torch.Tensor | None= None,
    cliprange: float | None= None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    # Compute the loss and metadata
    loss, metadata = compute_policy_gradient_loss(
        policy_log_probs=policy_log_probs,
        loss_type=loss_type,
        raw_rewards=raw_rewards,
        advantages=advantages,
        old_log_probs=old_log_probs,
        cliprange=cliprange,
    )

    loss = masked_mean(
        tensor=loss,
        mask=response_mask,
        dim=None,
    )

    loss = loss / gradient_accumulation_steps

    # Backpropagate the loss
    loss.backward()

    return loss, metadata