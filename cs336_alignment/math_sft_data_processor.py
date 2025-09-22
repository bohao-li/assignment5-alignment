import json
import re
from typing import Dict, Any, List, Optional

from drgrpo_grader import r1_zero_reward_fn

def grade(model_answer, ground_truth, fast=True):
    """
    Placeholder for grade function.
    Replace with actual implementation.
    """
    # Simple string comparison for now
    return str(model_answer).strip() == str(ground_truth).strip()

def is_complete_response(response: str) -> bool:
    """
    Check if a response is complete based on required format.
    
    Args:
        response: The model response to check
        
    Returns:
        bool: True if response appears complete, False otherwise
    """
    # Check for required format elements
    has_think_tags = "</think>" in response
    has_answer_tags = "<answer>" in response and "</answer>" in response
    
    # Basic completeness checks
    if not (has_think_tags and has_answer_tags):
        return False
    
    # Extract answer content
    try:
        answer_content = response.split("<answer>")[-1].replace("</answer>", "").strip()
        # Check if answer is not empty
        if not answer_content or "Incomplete response" in answer_content:
            return False
    except:
        return False
    
    return True

def extract_ground_truth(item: Dict[str, Any]) -> Optional[str]:
    """
    Extract ground truth answer from the dataset item.
    
    Args:
        item: Dataset item containing original_metadata
        
    Returns:
        Ground truth solution or None if not found
    """
    try:
        if 'original_metadata' in item and 'solution' in item['original_metadata']:
            return item['original_metadata']['solution']
        return None
    except:
        return None

def process_jsonl_dataset(input_file: str, output_file: str = None) -> Dict[str, Any]:
    """
    Process JSONL dataset to filter incomplete responses and evaluate correctness.
    
    Args:
        input_file: Path to input JSONL file
        output_file: Optional path to save filtered dataset
        
    Returns:
        Dictionary containing statistics and results
    """
    total_items = 0
    complete_responses = 0
    correct_answers = 0
    filtered_data = []
    evaluation_results = []
    
    with open(input_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            try:
                item = json.loads(line.strip())
                total_items += 1
                
                # Extract the assistant's response from the last message
                messages = item.get('messages', [])
                if not messages:
                    continue
                
                # Find the last assistant response
                assistant_response = None
                for msg in reversed(messages):
                    if msg.get('role') == 'assistant':
                        assistant_response = msg.get('content', '')
                        break
                
                if not assistant_response:
                    continue
                
                # Check if response is complete
                if not is_complete_response(assistant_response):
                    continue
                
                complete_responses += 1
                
                # Extract ground truth
                ground_truth = extract_ground_truth(item)
                if ground_truth is None:
                    continue
                
                # Evaluate correctness
                eval_result = r1_zero_reward_fn(assistant_response, ground_truth, fast=True)
                evaluation_results.append({
                    'line_number': line_num,
                    'evaluation': eval_result,
                    'ground_truth': ground_truth
                })
                
                if eval_result.get('reward', 0.0) > 0.0:
                    correct_answers += 1
                
                    # Add to filtered dataset
                    filtered_item = item.copy()
                    filtered_item['evaluation'] = eval_result
                    filtered_data.append(filtered_item)
                
            except json.JSONDecodeError as e:
                print(f"Error parsing line {line_num}: {e}")
                continue
            except Exception as e:
                print(f"Error processing line {line_num}: {e}")
                continue
    
    # Save filtered dataset if output file specified
    if output_file:
        with open(output_file, 'w', encoding='utf-8') as f:
            for item in filtered_data:
                f.write(json.dumps(item) + '\n')
    
    # Calculate statistics
    stats = {
        'total_items': total_items,
        'complete_responses': complete_responses,
        'correct_answers': correct_answers,
        'completion_rate': complete_responses / total_items if total_items > 0 else 0,
        'accuracy_rate': correct_answers / complete_responses if complete_responses > 0 else 0,
        'overall_success_rate': correct_answers / total_items if total_items > 0 else 0,
        'filtered_data_size': len(filtered_data)
    }
    
    return {
        'statistics': stats,
        'filtered_data': filtered_data,
        'evaluation_results': evaluation_results
    }

def print_statistics(stats: Dict[str, Any]):
    """Print formatted statistics."""
    print("=" * 60)
    print("DATASET PROCESSING STATISTICS")
    print("=" * 60)
    print(f"Total items processed: {stats['total_items']}")
    print(f"Complete responses: {stats['complete_responses']}")
    print(f"Correct answers: {stats['correct_answers']}")
    print(f"Completion rate: {stats['completion_rate']:.2%}")
    print(f"Accuracy rate (correct/complete): {stats['accuracy_rate']:.2%}")
    print(f"Overall success rate: {stats['overall_success_rate']:.2%}")
    print(f"Items in filtered dataset: {stats['filtered_data_size']}")
    print("=" * 60)

def analyze_evaluation_results(evaluation_results: List[Dict]) -> Dict[str, Any]:
    """Analyze evaluation results in detail."""
    format_rewards = [r['evaluation']['format_reward'] for r in evaluation_results]
    answer_rewards = [r['evaluation']['answer_reward'] for r in evaluation_results]
    total_rewards = [r['evaluation']['reward'] for r in evaluation_results]
    
    analysis = {
        'format_reward_avg': sum(format_rewards) / len(format_rewards) if format_rewards else 0,
        'answer_reward_avg': sum(answer_rewards) / len(answer_rewards) if answer_rewards else 0,
        'total_reward_avg': sum(total_rewards) / len(total_rewards) if total_rewards else 0,
        'perfect_format_count': sum(1 for r in format_rewards if r == 1.0),
        'perfect_answer_count': sum(1 for r in answer_rewards if r == 1.0),
        'perfect_total_count': sum(1 for r in total_rewards if r == 1.0),
    }
    
    return analysis

def main():
    """Main function to run the dataset processing."""
    # Configuration
    input_file = 'data/MATH/merged_sft_dataset.jsonl'  # Change this to your input file path
    output_file = 'filtered_dataset.jsonl'  # Change this to your desired output path
    
    print("Processing dataset...")
    results = process_jsonl_dataset(input_file, output_file)
    
    # Print statistics
    print_statistics(results['statistics'])
    
    # Detailed analysis
    if results['evaluation_results']:
        print("\nDETAILED EVALUATION ANALYSIS")
        print("-" * 40)
        analysis = analyze_evaluation_results(results['evaluation_results'])
        print(f"Average format reward: {analysis['format_reward_avg']:.3f}")
        print(f"Average answer reward: {analysis['answer_reward_avg']:.3f}")
        print(f"Average total reward: {analysis['total_reward_avg']:.3f}")
        print(f"Perfect format responses: {analysis['perfect_format_count']}")
        print(f"Perfect answer responses: {analysis['perfect_answer_count']}")
        print(f"Perfect total responses: {analysis['perfect_total_count']}")
    
    print(f"\nFiltered dataset saved to: {output_file}")
    print("Processing complete!")

if __name__ == "__main__":
    main()