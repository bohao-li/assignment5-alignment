# To run this code you need to install the following dependencies:
# pip install google-genai
import json
import time
import os
import base64
from typing import List, Dict
from google import genai
from google.genai import types

class GeminiMathGenerator:
    def __init__(self, api_key: str = None):
        # Use provided API key or environment variable
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("API key must be provided either as parameter or GEMINI_API_KEY environment variable")
        
        self.client = genai.Client(api_key=self.api_key)
        self.model = "gemini-2.5-flash-lite"
    
    def load_problems_from_jsonl(self, file_path: str) -> List[Dict]:
        """Load problems from a JSONL file and return structured data"""
        problems = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if line:
                        try:
                            data = json.loads(line)
                            problems.append(data)
                        except json.JSONDecodeError as e:
                            print(f"Warning: Invalid JSON on line {line_num}: {e}")
                            continue
            
            print(f"Successfully loaded {len(problems)} problems from {file_path}")
            return problems
            
        except FileNotFoundError:
            print(f"Error: File {file_path} not found")
            return []
        except Exception as e:
            print(f"Error loading file: {e}")
            return []
    
    def preview_problems(self, problems: List[Dict], num_samples: int = 3):
        """Preview the loaded problems to verify format"""
        if not problems:
            print("No problems to preview")
            return
        
        print(f"\n{'='*60}")
        print(f"DATA PREVIEW - Showing first {min(num_samples, len(problems))} problems:")
        print(f"{'='*60}")
        
        for i, problem_data in enumerate(problems[:num_samples]):
            print(f"\nProblem {i+1}:")
            print(f"Keys available: {list(problem_data.keys())}")
            
            # Display the actual problem text
            problem_text = problem_data.get('problem', 'No problem text found')
            print(f"Problem text: {problem_text[:200]}{'...' if len(problem_text) > 200 else ''}")
            
            # Display other metadata if available
            if 'level' in problem_data:
                print(f"Level: {problem_data['level']}")
            if 'type' in problem_data:
                print(f"Type: {problem_data['type']}")
            if 'solution' in problem_data:
                solution = problem_data['solution']
                print(f"Solution preview: {solution[:100]}{'...' if len(solution) > 100 else ''}")
            
            print("-" * 40)
        
        print(f"\nTotal problems loaded: {len(problems)}")
        
        # Check for consistency in data structure
        all_keys = set()
        for prob in problems[:10]:  # Check first 10 for consistency
            all_keys.update(prob.keys())
        
        print(f"All keys found in first 10 problems: {sorted(all_keys)}")
        
    def get_system_prompt(self) -> str:
        return """You are an expert mathematics tutor. Your task is to solve mathematical problems with clear, step-by-step explanations that students can follow and learn from.
            FORMATTING REQUIREMENTS:
            - Always structure your response as: <think>reasoning</think> <answer>final answer</answer>
            - Use proper mathematical notation and LaTeX when needed
            - Show ALL work and intermediate steps
            - Explain the reasoning behind each step
            - State the final answer clearly in the <answer> tags

            SOLUTION GUIDELINES:
            - Begin by identifying what type of problem this is
            - State any relevant formulas, theorems, or concepts
            - Work through the solution methodically
            - Verify your answer when possible
            - Keep explanations concise but complete (aim for 500 words total)

            QUALITY STANDARDS:
            - Solutions must be mathematically correct
            - Steps should be logically connected
            - Use clear, educational language
            - Avoid unnecessary verbosity
            - Always complete the solution - never leave it unfinished

            Remember: You're teaching, not just solving. Make your reasoning clear and educational."""
    
    def generate_response(self, math_problem: str, max_retries: int = 3) -> Dict:
        """Generate a single response for a math problem using Gemini Flash Lite"""
        
        prompt = f"""A conversation between User and Assistant. The User asks a question, and the Assistant solves it. The Assistant first thinks about the reasoning process in the mind and then provides the User with the answer. The reasoning process is enclosed within <think> reasoning process here </think> <answer> answer here </answer>.

User: {math_problem}
Assistant: <think>"""
        
        for attempt in range(max_retries):
            try:
                contents = [
                    types.Content(
                        role="user",
                        parts=[
                            types.Part.from_text(text=prompt),
                        ],
                    ),
                ]
                
                # Optional: Add Google Search tool if needed for complex problems
                tools = [
                    types.Tool(googleSearch=types.GoogleSearch()),
                ]
                
                generate_content_config = types.GenerateContentConfig(
                    thinking_config=types.ThinkingConfig(
                        thinking_budget=0,  # No additional thinking budget
                    ),
                    tools=tools,
                    temperature=0.7,
                    max_output_tokens=1024,
                )
                generate_content_config = types.GenerateContentConfig(
                    thinking_config = types.ThinkingConfig(
                        thinking_budget=-1,
                    ),
                    tools=tools,
                    system_instruction=[
                        types.Part.from_text(text=self.get_system_prompt()),
                    ],
                )
                
                # Collect the streamed response
                generated_text = ""
                for chunk in self.client.models.generate_content_stream(
                    model=self.model,
                    contents=contents,
                    config=generate_content_config,
                ):
                    if chunk.text:
                        generated_text += chunk.text
                
                # Complete the response (add </think> if missing and ensure proper format)
                if not generated_text.strip().endswith('</answer>'):
                    # Handle case where response might be incomplete
                    if '</think>' not in generated_text:
                        generated_text += '\n</think>\n\n<answer>\nIncomplete response\n</answer>'
                    elif '</answer>' not in generated_text:
                        generated_text += '\n</answer>'
                
                return {
                    "success": True,
                    "response": generated_text,
                    "usage": {
                        # Gemini API doesn't provide detailed token usage in the same way
                        "prompt_tokens": len(prompt.split()),  # Approximate
                        "completion_tokens": len(generated_text.split()),  # Approximate
                        "total_tokens": len(prompt.split()) + len(generated_text.split())  # Approximate
                    },
                    "problem": math_problem
                }
                
            except Exception as e:
                print(f"Request failed (attempt {attempt + 1}): {str(e)}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
        
        return {
            "success": False,
            "error": "Failed after all retries",
            "problem": math_problem
        }
    
    def process_single_problem(self, problem_text: str, metadata: Dict = None) -> Dict:
        """Process a single math problem and return formatted result"""
        result = self.generate_response(problem_text)
        
        if result["success"]:
            # Format for SFT training with original metadata preserved
            sft_entry = {
                "messages": [
                    {
                        "role": "user",
                        "content": problem_text
                    },
                    {
                        "role": "assistant",
                        "content": result["response"]
                    }
                ]
            }
            
            # Add original metadata if provided
            if metadata:
                sft_entry["original_metadata"] = metadata
            
            return {
                "success": True,
                "sft_entry": sft_entry,
                "usage": result.get("usage", {})
            }
        else:
            return result
    
    def process_problems_from_jsonl(self, input_file: str, output_file: str = "sft_dataset.jsonl", 
                                  max_problems: int = None, start_from: int = 0):
        """Process problems from JSONL file and save to new JSONL file"""
        
        # Load problems
        problems_data = self.load_problems_from_jsonl(input_file)
        
        if not problems_data:
            print("No problems loaded. Exiting.")
            return
        
        # Preview the data
        self.preview_problems(problems_data)
        
        # Ask for confirmation before proceeding
        user_input = input(f"\nDo you want to proceed with processing {len(problems_data)} problems? (y/n): ")
        if user_input.lower() != 'y':
            print("Processing cancelled.")
            return
        
        # Slice data if needed
        if max_problems:
            problems_data = problems_data[start_from:start_from + max_problems]
            print(f"Processing {len(problems_data)} problems (from index {start_from} to {start_from + len(problems_data)})")
        
        successful = 0
        failed = 0
        total_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        
        with open(output_file, 'w', encoding='utf-8') as f:
            for i, problem_data in enumerate(problems_data):
                print(f"Processing problem {i+1}/{len(problems_data)}")
                
                # Extract problem text (adjust key name based on your data structure)
                problem_text = problem_data.get('problem', '')
                if not problem_text:
                    print(f"Warning: No 'problem' field found in entry {i+1}")
                    failed += 1
                    continue
                
                # Create metadata dict without the problem text
                metadata = {k: v for k, v in problem_data.items() if k != 'problem'}
                
                result = self.process_single_problem(problem_text, metadata)
                
                if result["success"]:
                    # Write to JSONL file
                    json.dump(result["sft_entry"], f, ensure_ascii=False)
                    f.write('\n')
                    
                    successful += 1
                    
                    # Track usage
                    usage = result.get("usage", {})
                    for key in total_usage:
                        total_usage[key] += usage.get(key, 0)
                    
                    print(f"✓ Success - Total successful: {successful}")
                else:
                    failed += 1
                    print(f"✗ Failed - Total failed: {failed}")
                    print(f"Error: {result.get('error', 'Unknown error')}")
                
                # Rate limiting - be respectful to the API
                time.sleep(1)
                
                # Save progress every 50 problems
                if (i + 1) % 50 == 0:
                    print(f"Progress: {i+1}/{len(problems_data)} completed")
                    print(f"Success rate: {successful/(successful+failed)*100:.1f}%")
                    print(f"Token usage so far: {total_usage}")
        
        print(f"\nFinal Results:")
        print(f"Successful: {successful}")
        print(f"Failed: {failed}")
        print(f"Success rate: {successful/(successful+failed)*100:.1f}%" if (successful+failed) > 0 else "No attempts made")
        print(f"Total token usage: {total_usage}")
        print(f"Output saved to: {output_file}")

    def test_connection(self):
        """Test the API connection with a simple request"""
        try:
            contents = [
                types.Content(
                    role="user",
                    parts=[
                        types.Part.from_text(text="Hello, can you solve a simple math problem: 2 + 2 = ?"),
                    ],
                ),
            ]
            
            generate_content_config = types.GenerateContentConfig(
                thinking_config=types.ThinkingConfig(
                    thinking_budget=0,
                ),
                temperature=0.1,
                max_output_tokens=100,
            )
            
            response_text = ""
            for chunk in self.client.models.generate_content_stream(
                model=self.model,
                contents=contents,
                config=generate_content_config,
            ):
                if chunk.text:
                    response_text += chunk.text
            
            print("✓ API connection successful!")
            print(f"Response: {response_text}")
            return True
        except Exception as e:
            print(f"✗ API connection failed: {e}")
            return False

# Example usage
def main():
    # Set your API key here or via environment variable
    api_key = "AIzaSyAwsymkmI5n03ORYL_ECpJLmB2ljjkjGvk"
    
    if not api_key:
        print("Please set your API key as GEMINI_API_KEY environment variable")
        return
    
    # Initialize the generator
    try:
        generator = GeminiMathGenerator(api_key)
    except ValueError as e:
        print(f"Error: {e}")
        return
    
    # Test connection first
    print("Testing API connection...")
    if not generator.test_connection():
        print("Please check your API key and try again.")
        return
    
    # Load and preview your JSONL data
    input_file = "data/MATH/train.jsonl"  # Update this path
    output_file = "processed_sft_dataset.jsonl"
    
    # For testing, you might want to process only a few problems first
    print(f"\nProcessing problems from {input_file}...")
    generator.process_problems_from_jsonl(
        input_file=input_file,
        output_file=output_file,
        start_from=0
    )

if __name__ == "__main__":
    main()