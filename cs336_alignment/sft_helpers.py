import torch
import torch.nn.functional as F
from transformers import PreTrainedTokenizer
from transformers import PreTrainedModel

def tokenize_prompt_and_output(
    prompt_strs: list[str],
    output_strs: list[str],
    tokenizer: PreTrainedTokenizer
) -> dict[str, torch.Tensor]:
    """
    Tokenize the prompt and output strings, and construct a mask that is 1 for the
    response tokens and 0 for other tokens (prompt or padding).
    
    Args:
        prompt_strs: list[str] List of prompt strings.
        output_strs: list[str] List of output strings.
        tokenizer: PreTrainedTokenizer Tokenizer to use for tokenization.
    
    Returns:
        dict[str, torch.Tensor].
    """
    # 1. Tokenize prompts and outputs separately to get lengths
    tokenized_prompts = tokenizer(
        prompt_strs,
        add_special_tokens=False,
        padding=False,
        truncation=False
    )
    tokenized_outputs = tokenizer(
        output_strs,
        add_special_tokens=False,
        padding=False,
        truncation=False
    )

    # 2. Combine the tokenized inputs
    # Add a special start token if the tokenizer supports it
    input_ids = []
    response_mask = []
    
    # We need to handle the case where a tokenizer doesn't have a bos_token
    bos_token_id = tokenizer.bos_token_id
    has_bos = bos_token_id is not None
    
    prompt_and_output_lens = []

    for prompt_ids, output_ids in zip(tokenized_prompts['input_ids'], tokenized_outputs['input_ids']):
        # Combine prompt and output ids
        if has_bos:
            combined_ids = [bos_token_id] + prompt_ids + output_ids
        else:
            combined_ids = prompt_ids + output_ids

        input_ids.append(combined_ids)
        
        # Calculate mask: 0 for prompt/padding, 1 for output
        mask = [0] * len(prompt_ids) + [1] * len(output_ids)
        if has_bos:
            mask = [0] + mask
        
        response_mask.append(mask)
        prompt_and_output_lens.append(len(combined_ids))

    # 3. Pad the combined sequences to the max length
    max_len = max(prompt_and_output_lens)
    
    padded_input_ids = []
    padded_response_mask = []
    
    for ids, mask in zip(input_ids, response_mask):
        padding_len = max_len - len(ids)
        padded_ids = ids + [tokenizer.pad_token_id] * padding_len
        padded_input_ids.append(padded_ids)
        
        padded_mask = mask + [0] * padding_len
        padded_response_mask.append(padded_mask)

    # Convert to tensors
    padded_input_ids = torch.tensor(padded_input_ids)
    padded_response_mask = torch.tensor(padded_response_mask)

    # 4. Slice tensors as required by the return specification
    # input_ids: without the last token
    final_input_ids = padded_input_ids[:, :-1]

    # labels: shifted input ids (i.e., without the first token)
    labels = padded_input_ids[:, 1:]

    # response_mask: mask on the labels, so it also needs to be shifted
    final_response_mask = padded_response_mask[:, 1:]

    return {
        'input_ids': final_input_ids,
        'labels': labels,
        'response_mask': final_response_mask,
    }

def compute_entropy(logits: torch.Tensor) -> torch.Tensor:
    """
    Compute the entropy of a categorical distribution given its logits.
    
    Args:
        logits: torch.Tensor of shape (batch_size, seq_length, vocab_size)
            The logits for each token in the vocabulary.
    
    Returns:
        torch.Tensor of shape (batch_size, seq_length)
            The entropy for each token position in the sequence.
    """
    # Compute probabilities
    probs = torch.softmax(logits, dim=-1)  # Shape: (batch_size, seq_length, vocab_size)
    
    # Compute log probabilities
    log_probs = torch.log_softmax(logits, dim=-1)  # Shape: (batch_size, seq_length, vocab_size)
    
    # Compute entropy
    entropy = -torch.sum(probs * log_probs, dim=-1)  # Shape: (batch_size, seq_length)
    
    return entropy

def compute_entropy(log_probs: torch.Tensor) -> torch.Tensor:
    """
    Computes the per-token entropy from a tensor of log-probabilities.
    
    Args:
        log_probs: torch.Tensor of shape (batch_size, sequence_length, vocab_size).
    
    Returns:
        torch.Tensor of shape (batch_size, sequence_length).
    """
    # The formula for entropy H(p) = -sum(p(x) * log(p(x)))
    # We can use the log-probabilities directly to avoid numerical issues
    # H(p) = -sum(exp(log_p) * log_p)
    probs = torch.exp(log_probs)
    entropy = -torch.sum(probs * log_probs, dim=-1)
    return entropy

def get_response_log_probs(
    model: PreTrainedModel,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    return_token_entropy: bool = False,
) -> dict[str, torch.Tensor]:
    """
    Obtains conditional log-probabilities and optional per-token entropy for a sequence.
    
    Args:
        model: HuggingFace model.
        input_ids: torch.Tensor shape (batch_size, sequence_length).
        labels: torch.Tensor shape (batch_size, sequence_length).
        return_token_entropy: bool If True, also return per-token entropy.
    
    Returns:
        dict[str, torch.Tensor] containing "log_probs" and optionally "token_entropy".
    """
    # Get logits from the model
    # The model output is typically a tuple or a dataclass, with logits as the first element.
    # We explicitly access the .logits attribute.
    model_output = model(input_ids)
    logits = model_output.logits

    # Calculate log-probabilities for the entire vocabulary at each position
    # The `log_softmax` function is used for numerical stability
    log_probs_all_tokens = F.log_softmax(logits, dim=-1)

    # Use torch.gather to select the log-probability of the correct label token at each position
    # The `gather` function requires the label tensor to have the same number of dimensions as the log_probs tensor.
    log_probs = torch.gather(
        log_probs_all_tokens, 
        dim=2, 
        index=labels.unsqueeze(-1)
    ).squeeze(-1)
    
    results = {
        "log_probs": log_probs
    }
    
    # Optionally compute and add the per-token entropy
    if return_token_entropy:
        # We need to compute entropy on the full log-probability distribution before gathering.
        token_entropy = compute_entropy(log_probs_all_tokens)
        results["token_entropy"] = token_entropy
        
    return results


