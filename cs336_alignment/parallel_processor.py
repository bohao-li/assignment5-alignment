# To run this code you need to install the following dependencies:
# pip install google-genai tqdm
import json
import time
import os
import sys
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List, Dict, Tuple
from pathlib import Path
from tqdm import tqdm
import argparse
from datetime import datetime

# Import your generator (make sure the original file is in the same directory or pythonpath)
try:
    from gemini_math_generator import GeminiMathGenerator
except ImportError:
    print("Error: Could not import GeminiMathGenerator. Make sure the original script is available.")
    sys.exit(1)


class BatchProcessor:
    def __init__(self, api_key: str, input_file: str, output_dir: str = "output_batches"):
        self.api_key = api_key
        self.input_file = input_file
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Create status directory
        self.status_dir = self.output_dir / "status"
        self.status_dir.mkdir(exist_ok=True)
        
        # Create logs directory
        self.logs_dir = self.output_dir / "logs"
        self.logs_dir.mkdir(exist_ok=True)
    
    def load_and_split_data(self, num_batches: int = 10) -> List[Tuple[int, List[Dict], str, str]]:
        """Load data and split into batches"""
        print(f"Loading data from {self.input_file}...")
        
        # Load all problems
        problems = []
        try:
            with open(self.input_file, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if line:
                        try:
                            data = json.loads(line)
                            problems.append(data)
                        except json.JSONDecodeError as e:
                            print(f"Warning: Invalid JSON on line {line_num}: {e}")
                            continue
        except FileNotFoundError:
            print(f"Error: File {self.input_file} not found")
            return []
        
        print(f"Loaded {len(problems)} problems")
        
        # Split into batches
        batch_size = len(problems) // num_batches
        batches = []
        
        for i in range(num_batches):
            start_idx = i * batch_size
            if i == num_batches - 1:  # Last batch gets remaining items
                end_idx = len(problems)
            else:
                end_idx = start_idx + batch_size
            
            batch_data = problems[start_idx:end_idx]
            output_file = str(self.output_dir / f"batch_{i:02d}.jsonl")
            log_file = str(self.logs_dir / f"batch_{i:02d}.log")
            
            batches.append((i, batch_data, output_file, log_file))
            print(f"Batch {i:02d}: {len(batch_data)} problems (indices {start_idx}-{end_idx-1})")
        
        return batches
    
    def create_status_file(self, batch_id: int, total_problems: int):
        """Create initial status file for a batch"""
        status_file = self.status_dir / f"batch_{batch_id:02d}_status.json"
        status = {
            "batch_id": batch_id,
            "total_problems": total_problems,
            "completed": 0,
            "successful": 0,
            "failed": 0,
            "start_time": datetime.now().isoformat(),
            "last_update": datetime.now().isoformat(),
            "status": "starting",
            "current_problem": 0,
            "progress_percent": 0.0,
            "estimated_completion": None,
            "errors": []
        }
        
        with open(status_file, 'w') as f:
            json.dump(status, f, indent=2)
    
    def update_status_file(self, batch_id: int, **updates):
        """Update status file for a batch"""
        status_file = self.status_dir / f"batch_{batch_id:02d}_status.json"
        
        # Load current status
        try:
            with open(status_file, 'r') as f:
                status = json.load(f)
        except:
            return  # Skip if file doesn't exist
        
        # Update fields
        for key, value in updates.items():
            status[key] = value
        
        status["last_update"] = datetime.now().isoformat()
        
        # Calculate progress
        if "completed" in updates and status["total_problems"] > 0:
            status["progress_percent"] = (status["completed"] / status["total_problems"]) * 100
            
            # Estimate completion time
            if status["completed"] > 0:
                start_time = datetime.fromisoformat(status["start_time"])
                elapsed = datetime.now() - start_time
                rate = status["completed"] / elapsed.total_seconds()
                remaining_seconds = (status["total_problems"] - status["completed"]) / rate
                estimated_completion = datetime.now().timestamp() + remaining_seconds
                status["estimated_completion"] = datetime.fromtimestamp(estimated_completion).isoformat()
        
        with open(status_file, 'w') as f:
            json.dump(status, f, indent=2)


def process_single_batch(args: Tuple[int, List[Dict], str, str, str]) -> Dict:
    """Process a single batch of problems (runs in separate process)"""
    batch_id, batch_data, output_file, log_file, api_key = args
    
    # Set up logging to file
    import logging
    
    # Clear any existing handlers to avoid conflicts
    logging.getLogger().handlers.clear()
    
    # Create a custom formatter that handles missing batch_id
    class BatchFormatter(logging.Formatter):
        def format(self, record):
            if not hasattr(record, 'batch_id'):
                record.batch_id = batch_id
            return super().format(record)
    
    formatter = BatchFormatter('%(asctime)s - Batch %(batch_id)02d - %(levelname)s - %(message)s')
    
    # Set up file handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.INFO)
    
    # Set up console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.setLevel(logging.INFO)
    
    # Configure root logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    # Also configure the google.genai logger to use our format
    genai_logger = logging.getLogger('google.genai')
    genai_logger.setLevel(logging.WARNING)  # Reduce verbosity of genai logs
    
    try:
        # Initialize generator for this process
        generator = GeminiMathGenerator(api_key)
        
        # Create processor instance for status updates
        processor = BatchProcessor(api_key, "", Path(output_file).parent)
        processor.create_status_file(batch_id, len(batch_data))
        processor.update_status_file(batch_id, status="running")
        
        logger.info(f"Starting batch {batch_id} with {len(batch_data)} problems")
        
        successful = 0
        failed = 0
        total_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        errors = []
        
        with open(output_file, 'w', encoding='utf-8') as f:
            for i, problem_data in enumerate(batch_data):
                try:
                    # Extract problem text
                    problem_text = problem_data.get('problem', '')
                    if not problem_text:
                        logger.warning(f"No 'problem' field found in entry {i+1}")
                        failed += 1
                        continue
                    
                    # Create metadata dict without the problem text
                    metadata = {k: v for k, v in problem_data.items() if k != 'problem'}
                    
                    result = generator.process_single_problem(problem_text, metadata)
                    
                    if result["success"]:
                        # Write to JSONL file
                        json.dump(result["sft_entry"], f, ensure_ascii=False)
                        f.write('\n')
                        f.flush()  # Ensure data is written immediately
                        
                        successful += 1
                        
                        # Track usage
                        usage = result.get("usage", {})
                        for key in total_usage:
                            total_usage[key] += usage.get(key, 0)
                        
                    else:
                        failed += 1
                        error_msg = result.get('error', 'Unknown error')
                        errors.append(f"Problem {i+1}: {error_msg}")
                        logger.error(f"Failed to process problem {i+1}: {error_msg}")
                    
                    # Update status every problem
                    processor.update_status_file(
                        batch_id,
                        completed=i + 1,
                        successful=successful,
                        failed=failed,
                        current_problem=i + 1,
                        errors=errors[-10:]  # Keep last 10 errors
                    )
                    
                    # Rate limiting
                    time.sleep(1)
                    
                    # Progress logging every 10 problems
                    if (i + 1) % 10 == 0:
                        logger.info(f"Progress: {i+1}/{len(batch_data)} completed. "
                                  f"Success: {successful}, Failed: {failed}")
                
                except Exception as e:
                    failed += 1
                    error_msg = f"Unexpected error processing problem {i+1}: {str(e)}"
                    errors.append(error_msg)
                    logger.error(error_msg)
        
        # Final status update
        processor.update_status_file(
            batch_id,
            status="completed",
            completed=len(batch_data),
            successful=successful,
            failed=failed,
            errors=errors
        )
        
        logger.info(f"Batch {batch_id} completed. Success: {successful}, Failed: {failed}")
        
        return {
            "batch_id": batch_id,
            "successful": successful,
            "failed": failed,
            "total_usage": total_usage,
            "output_file": output_file
        }
    
    except Exception as e:
        error_msg = f"Critical error in batch {batch_id}: {str(e)}"
        logger.error(error_msg)
        
        # Update status with error
        try:
            processor.update_status_file(batch_id, status="error", errors=[error_msg])
        except:
            pass
        
        return {
            "batch_id": batch_id,
            "successful": 0,
            "failed": len(batch_data),
            "error": error_msg,
            "output_file": output_file
        }


def monitor_status(status_dir: Path, num_batches: int, update_interval: int = 30):
    """Monitor and display status of all batches"""
    print(f"\n{'='*80}")
    print("BATCH PROCESSING STATUS MONITOR")
    print(f"{'='*80}")
    print("Press Ctrl+C to stop monitoring (processing will continue)")
    
    try:
        while True:
            print(f"\n--- Status Update: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ---")
            
            total_completed = 0
            total_successful = 0
            total_failed = 0
            total_problems = 0
            
            for i in range(num_batches):
                status_file = status_dir / f"batch_{i:02d}_status.json"
                
                try:
                    with open(status_file, 'r') as f:
                        status = json.load(f)
                    
                    total_completed += status.get("completed", 0)
                    total_successful += status.get("successful", 0)
                    total_failed += status.get("failed", 0)
                    total_problems += status.get("total_problems", 0)
                    
                    print(f"Batch {i:02d}: {status.get('progress_percent', 0):.1f}% "
                          f"({status.get('completed', 0)}/{status.get('total_problems', 0)}) "
                          f"✓{status.get('successful', 0)} ✗{status.get('failed', 0)} "
                          f"[{status.get('status', 'unknown')}]")
                    
                except FileNotFoundError:
                    print(f"Batch {i:02d}: Not started")
                except Exception as e:
                    print(f"Batch {i:02d}: Status error - {e}")
            
            # Overall progress
            overall_progress = (total_completed / total_problems * 100) if total_problems > 0 else 0
            print(f"\nOVERALL: {overall_progress:.1f}% "
                  f"({total_completed}/{total_problems}) "
                  f"✓{total_successful} ✗{total_failed}")
            
            if total_completed >= total_problems and total_problems > 0:
                print("\n🎉 ALL BATCHES COMPLETED!")
                break
            
            time.sleep(update_interval)
            
    except KeyboardInterrupt:
        print("\nMonitoring stopped. Batches continue processing in background.")


def main():
    parser = argparse.ArgumentParser(description='Process math problems in parallel batches')
    parser.add_argument('--input', '-i', required=True, help='Input JSONL file path')
    parser.add_argument('--output-dir', '-o', default='output_batches', help='Output directory')
    parser.add_argument('--num-batches', '-n', type=int, default=10, help='Number of batches')
    parser.add_argument('--num-workers', '-w', type=int, default=None, help='Number of parallel workers (default: num_batches)')
    parser.add_argument('--api-key', '-k', default=None, help='Gemini API key (or set GEMINI_API_KEY env var)')
    parser.add_argument('--monitor-only', '-m', action='store_true', help='Only monitor existing batches')
    parser.add_argument('--monitor-interval', type=int, default=30, help='Status monitor update interval in seconds')
    
    args = parser.parse_args()
    
    # Get API key
    api_key = args.api_key or os.environ.get("GEMINI_API_KEY")
    if not api_key and not args.monitor_only:
        print("Error: API key must be provided via --api-key or GEMINI_API_KEY environment variable")
        return
    
    # Initialize processor
    processor = BatchProcessor(api_key or "", args.input, args.output_dir)
    
    if args.monitor_only:
        # Only monitor existing batches
        monitor_status(processor.status_dir, args.num_batches, args.monitor_interval)
        return
    
    # Test API connection
    if api_key:
        print("Testing API connection...")
        try:
            test_generator = GeminiMathGenerator(api_key)
            if not test_generator.test_connection():
                print("API connection failed. Please check your API key.")
                return
        except Exception as e:
            print(f"API connection error: {e}")
            return
    
    # Load and split data
    batches = processor.load_and_split_data(args.num_batches)
    if not batches:
        print("No data to process. Exiting.")
        return
    
    # Confirm processing
    total_problems = sum(len(batch[1]) for batch in batches)
    print(f"\nReady to process {total_problems} problems in {len(batches)} batches")
    print(f"Output directory: {processor.output_dir}")
    print(f"Each batch will be saved to a separate .jsonl file")
    
    user_input = input("Do you want to proceed? (y/n): ")
    if user_input.lower() != 'y':
        print("Processing cancelled.")
        return
    
    # Prepare arguments for multiprocessing
    num_workers = args.num_workers or args.num_batches
    batch_args = [(batch_id, batch_data, output_file, log_file, api_key) 
                  for batch_id, batch_data, output_file, log_file in batches]
    
    print(f"\nStarting processing with {num_workers} parallel workers...")
    print(f"Monitor status files in: {processor.status_dir}")
    print(f"Monitor logs in: {processor.logs_dir}")
    
    # Start monitoring in a separate process
    monitor_process = mp.Process(
        target=monitor_status, 
        args=(processor.status_dir, args.num_batches, args.monitor_interval)
    )
    monitor_process.start()
    
    try:
        # Process batches in parallel
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            # Submit all jobs
            future_to_batch = {
                executor.submit(process_single_batch, batch_arg): batch_arg[0] 
                for batch_arg in batch_args
            }
            
            # Wait for completion
            completed_batches = []
            for future in as_completed(future_to_batch):
                batch_id = future_to_batch[future]
                try:
                    result = future.result()
                    completed_batches.append(result)
                    print(f"\n✓ Batch {batch_id} finished: "
                          f"✓{result['successful']} ✗{result['failed']}")
                except Exception as e:
                    print(f"\n✗ Batch {batch_id} failed with exception: {e}")
        
        # Stop monitoring
        monitor_process.terminate()
        monitor_process.join()
        
        # Final summary
        total_successful = sum(batch['successful'] for batch in completed_batches)
        total_failed = sum(batch['failed'] for batch in completed_batches)
        
        print(f"\n{'='*80}")
        print("FINAL RESULTS")
        print(f"{'='*80}")
        print(f"Total successful: {total_successful}")
        print(f"Total failed: {total_failed}")
        print(f"Success rate: {total_successful/(total_successful+total_failed)*100:.1f}%")
        print(f"Output files saved in: {processor.output_dir}")
        print(f"Status files saved in: {processor.status_dir}")
        print(f"Log files saved in: {processor.logs_dir}")
        
        # Merge all batch files into one final file
        final_output = processor.output_dir / "merged_sft_dataset.jsonl"
        print(f"\nMerging all batches into: {final_output}")
        
        with open(final_output, 'w', encoding='utf-8') as outf:
            for batch in completed_batches:
                batch_file = batch['output_file']
                if os.path.exists(batch_file):
                    with open(batch_file, 'r', encoding='utf-8') as inf:
                        outf.write(inf.read())
        
        print("✓ All batches merged successfully!")
        
    except KeyboardInterrupt:
        print("\n\nProcessing interrupted! Stopping monitor...")
        monitor_process.terminate()
        monitor_process.join()
        print("Some batches may still be running in background.")
        print(f"Check status files in: {processor.status_dir}")


if __name__ == "__main__":
    main()