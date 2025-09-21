#!/usr/bin/env python3
"""
Script to evaluate Qwen 2.5 Math 1.5B zero-shot performance on MATH dataset.
"""

import json
import os
import re
from datetime import datetime
from typing import Any, Callable, Dict, List, Union

from bohao.assignment5.drgrpo_grader import r1_zero_reward_fn
from vllm import LLM, SamplingParams


def extract_answer(text):
    """Extract answer from \\boxed{} format."""
    pattern = r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}"
    match = re.search(pattern, text)
    if match:
        return match.group(1)
    return None


def load_math_validation_data(file_path: str) -> List[Dict[str, Any]]:
    """Load MATH validation examples from JSONL file."""
    examples = []
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    examples.append(json.loads(line))
        print(f"Loaded {len(examples)} validation examples from {file_path}")
        return examples
    except FileNotFoundError:
        print(f"Error: File {file_path} not found.")
        return []
    except Exception as e:
        print(f"Error loading validation data: {e}")
        return []


def format_prompts(examples: List[Dict[str, Any]]) -> List[str]:
    """Format examples as R1 zero-shot prompts."""
    prompts = []
    for example in examples:
        problem = example.get("problem", "")
        # Create the prompt with thinking placeholder
        prompt = f"""A conversation between User and Assistant. The User asks a question, and the Assistant
→
solves it. The Assistant first thinks about the reasoning process in the mind and
→
then provides the User with the answer. The reasoning process is enclosed within
i.e., <think> reasoning process here </think> <answer> answer here </answer>.
→
<think> </think> and answer is enclosed within <answer> </answer> tags, respectively,
→
User: {problem}
Assistant: <think>"""
        prompts.append(prompt)

    return prompts


def evaluate_vllm(
    vllm_model: LLM,
    reward_fn: Callable[[str, str], Dict[str, float]],
    examples: List[Dict[str, Any]],
    prompts: List[str],
    eval_sampling_params: SamplingParams,
    output_file: str = "",
) -> Dict[str, Union[float, str]]:
    """
    Evaluate a language model on a list of prompts,
    compute evaluation metrics, and serialize results to disk.
    """
    print(f"Evaluating model on {len(prompts)} examples...")

    # Generate responses
    outputs = vllm_model.generate(prompts, eval_sampling_params)

    # Collect results
    results = []
    total_reward = 0
    total_format_reward = 0
    total_answer_reward = 0

    for i, output in enumerate(outputs):
        prompt = output.prompt
        full_response = output.outputs[0].text

        # Get ground truth
        ground_truth = examples[i].get("solution", "")

        if len(ground_truth) == 0:
            print(f"Warning: No ground truth found for example {i}. Skipping...")
            continue

        # Evaluate response
        scores = reward_fn(full_response, ground_truth)

        # Store result
        result = {
            "example_id": i,
            "problem": examples[i].get("problem", ""),
            "ground_truth": ground_truth,
            "prompt": prompt,
            "full_response": full_response,
            "scores": scores,
            "subject": examples[i].get("subject", ""),
            "level": examples[i].get("level", ""),
        }
        results.append(result)

        # Accumulate metrics
        total_reward += scores["reward"]
        total_format_reward += scores["format_reward"]
        total_answer_reward += scores["answer_reward"]

        if i % 10 == 0:
            print(f"Processed {i+1}/{len(prompts)} examples...")

    # Calculate final metrics
    num_examples = len(results)
    metrics = {
        "total_examples": num_examples,
        "average_reward": total_reward / num_examples,
        "format_accuracy": total_format_reward / num_examples,
        "answer_accuracy": total_answer_reward / num_examples,
        "evaluation_timestamp": datetime.now().isoformat(),
    }

    # Save results to disk
    if output_file:
        output_data = {"metrics": metrics, "results": results}

        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)

        print(f"Results saved to {output_file}")

    # Print summary
    print("\n" + "=" * 50)
    print("EVALUATION SUMMARY")
    print("=" * 50)
    print(f"Total Examples: {metrics['total_examples']}")
    print(f"Average Reward: {metrics['average_reward']:.4f}")
    print(f"Format Accuracy: {metrics['format_accuracy']:.4f}")
    print(f"Answer Accuracy: {metrics['answer_accuracy']:.4f}")
    print("=" * 50)

    return metrics


def main():
    # Configuration
    data_output_dir = "/home/bohao/persistent/private-90d/MATH"
    validation_file_path = os.path.join(data_output_dir, "validation.jsonl")
    model_path = "/home/bohao/persistent/private-90d/qwen"
    output_file = "results/math_evaluation_results.json"

    # Load validation examples
    examples = load_math_validation_data(validation_file_path)
    if not examples:
        print("No validation examples loaded. Exiting.")
        return

    # Limit to smaller subset for testing (remove this line for full evaluation)
    # examples = examples[:50]  # Uncomment to test with smaller subset

    # Format prompts
    prompts = format_prompts(examples)
    print(f"Created {len(prompts)} prompts")

    # Create sampling parameters
    sampling_params = SamplingParams(
        temperature=0.0,  # Deterministic for evaluation
        top_p=1.0,
        max_tokens=2048,
        stop=["</answer>"],
    )
    sampling_params.include_stop_str_in_output = True

    # Load model
    print(f"Loading model from {model_path}...")
    try:
        llm = LLM(model=model_path, trust_remote_code=True)
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Run evaluation
    metrics = evaluate_vllm(
        vllm_model=llm,
        reward_fn=r1_zero_reward_fn,
        examples=examples,
        prompts=prompts,
        eval_sampling_params=sampling_params,
        output_file=output_file,
    )

    print("Evaluation completed!")
    print(f"Metrics: {metrics}")


if __name__ == "__main__":
    main()
