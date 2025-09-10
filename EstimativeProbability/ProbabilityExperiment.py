"""
This is based on "Perception of Probability words" https://waf.cs.illinois.edu/visualizations/Perception-of-Probability-Words/

We will define a list of probability words and a list of models.
We will then ask the model to estimate the numerical probability of the word.
"""

import csv
import re
import argparse
import time
import threading
from queue import Queue, PriorityQueue
from datetime import datetime
from dataclasses import dataclass, field
from typing import Any
from openrouter_client import OpenRouterClient

paper_words = [
    "Almost Certain",
    "Highly Likely",
    "Very Good Chance",
    "We Believe",
    "Likely",
    "Probable",
    "Probably",
    "Better than Even",
    "About Even",
    "Probably Not",
    "We Doubt",
    "Unlikely",
    "Improbable",
    "Chances are Slight",
    "Little Chance",
    "Highly Unlikely",
    "Almost No Chance",
]

words = [
    "Definite",
    "Almost certain",
    "Highly probable",
    "A good chance",
    "Likely",
    "Quite likely",
    "Better than even",
    "Probable",
    "Possible",
    "Improbable",
    "Highly unlikely",
    "Unlikely",
    "Seldom",
    "Impossible",
    "Rare",
]


free_models = [
    "nvidia/nemotron-nano-9b-v2",
    "openrouter/sonoma-dusk-alpha",
    "openrouter/sonoma-sky-alpha",
    "deepseek/deepseek-chat-v3.1:free",
    "openai/gpt-oss-120b:free",
    "openai/gpt-oss-20b:free",
    "z-ai/glm-4.5-air:free",
    "qwen/qwen3-coder:free",
    "moonshotai/kimi-k2:free",
    "cognitivecomputations/dolphin-mistral-24b-venice-edition:free",
    "google/gemma-3n-e2b-it:free",
    "tencent/hunyuan-a13b-instruct:free",
    "tngtech/deepseek-r1t2-chimera:free",
    "mistralai/mistral-small-3.2-24b-instruct:free",
    "moonshotai/kimi-dev-72b:free",
    "deepseek/deepseek-r1-0528-qwen3-8b:free",
    "deepseek/deepseek-r1-0528:free",
    "mistralai/devstral-small-2505:free",
    "google/gemma-3n-e4b-it:free",
    "meta-llama/llama-3.3-8b-instruct:free",
    "qwen/qwen3-4b:free",
    "qwen/qwen3-30b-a3b:free",
    "qwen/qwen3-8b:free",
    "qwen/qwen3-14b:free",
    "qwen/qwen3-235b-a22b:free",
    "tngtech/deepseek-r1t-chimera:free",
    "microsoft/mai-ds-r1:free",
    "shisa-ai/shisa-v2-llama3.3-70b:free",
    "arliai/qwq-32b-arliai-rpr-v1:free",
    "agentica-org/deepcoder-14b-preview:free",
    "moonshotai/kimi-vl-a3b-thinking:free",
    "nvidia/llama-3.1-nemotron-ultra-253b-v1:free",
    "meta-llama/llama-4-maverick:free",
    "meta-llama/llama-4-scout:free",
    "qwen/qwen2.5-vl-32b-instruct:free",
    "deepseek/deepseek-chat-v3-0324:free",
    "mistralai/mistral-small-3.1-24b-instruct:free",
    "google/gemma-3-4b-it:free",
    "google/gemma-3-12b-it:free",
    "rekaai/reka-flash-3:free",
    "google/gemma-3-27b-it:free",
    "qwen/qwq-32b:free",
    "nousresearch/deephermes-3-llama-3-8b-preview:free",
    "cognitivecomputations/dolphin3.0-r1-mistral-24b:free",
    "cognitivecomputations/dolphin3.0-mistral-24b:free",
    "qwen/qwen2.5-vl-72b-instruct:free",
    "mistralai/mistral-small-24b-instruct-2501:free",
    "deepseek/deepseek-r1-distill-qwen-14b:free",
    "deepseek/deepseek-r1-distill-llama-70b:free",
    "deepseek/deepseek-r1:free",
    "google/gemini-2.0-flash-exp:free",
    "meta-llama/llama-3.3-70b-instruct:free",
]

paid_models = [
    "google/gemini-2.5-flash-lite-preview-06-17",
    "google/gemini-2.5-flash",
    "google/gemini-2.5-pro",
    "x-ai/grok-4",
    "x-ai/grok-3",

    "anthropic/claude-opus-4.1",
    "openai/gpt-oss-20b",
    "openai/gpt-oss-120b",
    "openai/gpt-5-nano",
    "openai/gpt-5-mini",
    "openai/gpt-5-chat",
    "deepseek/deepseek-chat-v3.1",
    "moonshotai/kimi-k2-0905",
    "openrouter/sonoma-sky-alpha",
    "openrouter/sonoma-dusk-alpha",
    "mistralai/mistral-medium-3",
    "anthropic/claude-3.7-sonnet",
    "deepseek/deepseek-chat",
    "google/gemma-3-4b-it",
    "anthropic/claude-3.7-sonnet:thinking",

]

prompt = """You will be given a probability word. Your task is to estimate the numerical probability that word conveys on a scale of 0 to 1.

CRITICAL INSTRUCTIONS:
- You MUST respond with ONLY a single decimal number between 0 and 1
- Do NOT include ANY other text, words, explanations, or punctuation
- Do NOT use ranges (e.g., "0.7-0.8") 
- Do NOT use percentages (e.g., "70%")
- Do NOT add words like "approximately", "about", "around", etc.
- Your entire response must be EXACTLY one number and nothing else

Valid response examples: 0.5, 0.85, 0.1, 0.95, 0.0, 1.0
INVALID response examples: "0.7", "approximately 0.7", "0.6-0.8", "70%", "I think 0.5", "0.5 (50%)"

Your response must be parseable as a float between 0 and 1. Any other format will be discarded.
"""

def extract_number(response_text):
    """
    Extract number only if response contains EXACTLY one number and nothing else.
    This enforces our strict instruction that models should respond with only a number.
    """
    text = response_text.strip()
    
    # Check if the response is EXACTLY a number (and nothing else)
    # This pattern matches: 0.5, .5, 1.0, 0, 1 etc. but rejects any text around it
    exact_number_pattern = r'^(\d*\.?\d+)$'
    match = re.match(exact_number_pattern, text)
    
    if match:
        try:
            num = float(match.group(1))
            if 0 <= num <= 1:
                return num
        except ValueError:
            pass
    
    # If response doesn't match our strict format, return None
    return None


@dataclass
class RequestItem:
    """A request item for the retry queue."""
    model: str
    word: str
    repeat: int
    retry_count: int = 0
    priority: int = field(init=False)
    _creation_time: float = field(init=False, default_factory=time.time)
    
    def __post_init__(self):
        # Lower numbers = higher priority. Base priority on retry count.
        self.priority = self.retry_count
    
    def __lt__(self, other):
        # For PriorityQueue comparison when priorities are equal
        if self.priority != other.priority:
            return self.priority < other.priority
        # If same priority, use creation time (FIFO)
        return self._creation_time < other._creation_time


class ModelRateLimits:
    """Manages rate limit information for OpenRouter models."""
    
    def __init__(self, client: OpenRouterClient):
        self.client = client
        self.global_limits = self._check_global_limits()
        
    def _check_global_limits(self):
        """Check global account limits via OpenRouter API."""
        try:
            import requests
            response = requests.get(
                "https://openrouter.ai/api/v1/auth/key",
                headers={"Authorization": f"Bearer {self.client.api_key}"}
            )
            
            if response.status_code == 200:
                data = response.json()
                limits = {
                    "requests_per_minute": 20,  # Standard for free models
                    "requests_per_day": 50 if data.get("usage", {}).get("credits", 0) < 10 else 1000,
                    "current_usage": data.get("usage", {}),
                    "balance": data.get("usage", {}).get("credits", 0)
                }
                print(f"💳 Account balance: ${limits['balance']:.2f}")
                print(f"📊 Daily limit: {limits['requests_per_day']} requests")
                return limits
            else:
                print(f"⚠️  Could not check account limits: {response.status_code}")
                return {"requests_per_minute": 20, "requests_per_day": 50}
                
        except Exception as e:
            print(f"⚠️  Error checking account limits: {e}")
            return {"requests_per_minute": 20, "requests_per_day": 50}
    
    def get_suggested_delay(self, retry_count: int) -> float:
        """Get suggested delay based on retry count."""
        # Exponential backoff: 1s, 2s, 4s, 8s, then 10s max
        return min(2 ** retry_count, 10.0)


def smart_api_worker(task_queue: PriorityQueue, results_queue: Queue, rate_limits: ModelRateLimits, 
                     progress_lock: threading.Lock, progress_counter: list, total_tasks: int, max_retries: int = 3):
    """Smart worker function with retry logic and exponential backoff."""
    client = OpenRouterClient()
    
    while True:
        try:
            # Get next task from priority queue (timeout prevents hanging)
            priority, request_item = task_queue.get(timeout=1)
            if request_item is None:  # Poison pill to stop worker
                break
            
            # Update progress
            with progress_lock:
                progress_counter[0] += 1
                current_progress = progress_counter[0]
            
            # Wait based on retry count (exponential backoff)
            if request_item.retry_count > 0:
                delay = rate_limits.get_suggested_delay(request_item.retry_count)
                time.sleep(delay)
            
            try:
                # Create the full prompt
                full_prompt = f"{prompt}\n\nProbability word: \"{request_item.word}\""
                
                # Get response from model
                messages = [{"role": "user", "content": full_prompt}]
                response = client.chat_completion(messages, model=request_item.model)
                raw_response = response["choices"][0]["message"]["content"]
                
                # Extract number from response
                probability = extract_number(raw_response)
                
                # Success! Record result
                result = {
                    "model": request_item.model,
                    "word": request_item.word,
                    "repeat": request_item.repeat,
                    "raw_response": raw_response.strip(),
                    "extracted_probability": probability,
                    "timestamp": datetime.now().isoformat(),
                    "thread_id": threading.current_thread().name,
                    "retry_count": request_item.retry_count
                }
                
                status = "✅" if probability is not None else "❌"
                retry_info = f" (retry {request_item.retry_count})" if request_item.retry_count > 0 else ""
                print(f"    [{current_progress:4d}/{total_tasks}] {status} {request_item.model[:20]:<20} | {request_item.word[:15]:<15} | {probability if probability else 'REJECTED'}{retry_info}")
                
                results_queue.put(result)
                
            except Exception as e:
                error_type = type(e).__name__
                error_message = str(e)
                
                # Determine if this is retryable
                is_retryable = False
                if "429" in error_message or "rate limit" in error_message.lower():
                    is_retryable = True
                    status_msg = "⏳ RATE LIMITED"
                elif "502" in error_message or "503" in error_message or "timeout" in error_message.lower():
                    is_retryable = True
                    status_msg = "🔄 SERVER ERROR"
                elif "402" in error_message or "credit" in error_message.lower():
                    status_msg = "💳 CREDIT ERROR"
                else:
                    status_msg = f"❌ ERROR: {error_type}"
                
                # Retry logic
                if is_retryable and request_item.retry_count < max_retries:
                    request_item.retry_count += 1
                    request_item.priority = request_item.retry_count  # Lower priority for retries
                    task_queue.put((request_item.priority, request_item))
                    retry_info = f" (will retry {request_item.retry_count}/{max_retries})"
                else:
                    # Give up, record failure
                    result = {
                        "model": request_item.model,
                        "word": request_item.word,
                        "repeat": request_item.repeat,
                        "raw_response": f"ERROR ({error_type}): {e}",
                        "extracted_probability": None,
                        "timestamp": datetime.now().isoformat(),
                        "thread_id": threading.current_thread().name,
                        "retry_count": request_item.retry_count
                    }
                    results_queue.put(result)
                    retry_info = f" (gave up after {request_item.retry_count} retries)" if request_item.retry_count > 0 else ""
                
                print(f"    [{current_progress:4d}/{total_tasks}] {status_msg} {request_item.model[:20]:<20} | {request_item.word[:15]:<15}{retry_info}")
            
            task_queue.task_done()
            
        except:  # Queue timeout or other issues
            break


def run_experiment(words_to_test, models_to_test, n_repeats=3, output_filename=None, max_threads=10, max_retries=3):
    """Run the probability estimation experiment with smart retry logic."""
    
    # Generate filename with current date if not provided
    if output_filename is None:
        date_str = datetime.now().strftime("%Y%m%d")
        filename = f"{n_repeats}-ModelSurvey-{date_str}.csv"
    else:
        filename = output_filename
    
    print(f"Running experiment with {len(models_to_test)} models and {len(words_to_test)} words...")
    print(f"Repeating each combination {n_repeats} times")
    print(f"Using {max_threads} threads with smart retry logic")
    print(f"Max retries per request: {max_retries}")
    print(f"Results will be saved to: {filename}")
    
    # Initialize rate limit checker
    client = OpenRouterClient()
    rate_limits = ModelRateLimits(client)
    
    # Create priority task queue and results queue
    task_queue = PriorityQueue()
    results_queue = Queue()
    
    # Add all tasks to priority queue
    total_tasks = 0
    for model in models_to_test:
        for word in words_to_test:
            for repeat in range(1, n_repeats + 1):
                request_item = RequestItem(model=model, word=word, repeat=repeat)
                task_queue.put((request_item.priority, request_item))
                total_tasks += 1
    
    print(f"Total initial tasks: {total_tasks}")
    print(f"\nProgress format: [processed/total] status model | word | result")
    print("=" * 80)
    
    # Progress tracking
    progress_lock = threading.Lock()
    progress_counter = [0]  # Use list for mutable reference
    
    # Start smart worker threads
    threads = []
    for i in range(max_threads):
        thread = threading.Thread(
            target=smart_api_worker,
            args=(task_queue, results_queue, rate_limits, progress_lock, progress_counter, total_tasks, max_retries),
            name=f"Worker-{i+1}"
        )
        thread.daemon = True
        thread.start()
        threads.append(thread)
    
    # Wait for all tasks to complete
    task_queue.join()
    
    # Stop worker threads
    for _ in threads:
        task_queue.put((0, None))  # Poison pill with priority 0
    
    for thread in threads:
        thread.join(timeout=5)
    
    # Collect all results
    results = []
    while not results_queue.empty():
        results.append(results_queue.get())
    
    # Save results to CSV
    print(f"\n{'='*80}")
    print(f"Saving {len(results)} results to {filename}...")
    with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ["model", "word", "repeat", "raw_response", "extracted_probability", "timestamp", "thread_id", "retry_count"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        for result in results:
            writer.writerow(result)
    
    # Print retry statistics
    retry_stats = {}
    for result in results:
        retry_count = result.get("retry_count", 0)
        retry_stats[retry_count] = retry_stats.get(retry_count, 0) + 1
    
    print(f"\n📊 Retry Statistics:")
    for retry_count in sorted(retry_stats.keys()):
        print(f"   {retry_count} retries: {retry_stats[retry_count]} requests")
    
    print(f"Experiment complete! Results saved to {filename}")
    return results


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run probability estimation experiment with AI models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 ProbabilityExperiment.py                     # Run full experiment with free models
  python3 ProbabilityExperiment.py --paid              # Use paid models (costs money!)
  python3 ProbabilityExperiment.py --models 5          # Test first 5 models only
  python3 ProbabilityExperiment.py --words 10          # Test first 10 words only
  python3 ProbabilityExperiment.py --threads 20        # Use 20 worker threads
  python3 ProbabilityExperiment.py --retries 5         # Allow up to 5 retries per request
  python3 ProbabilityExperiment.py --paid --models 3 --words 5 --repeats 1 --threads 5 --retries 2 --output test.csv
        """
    )
    
    parser.add_argument(
        '--models', '-m', 
        type=int, 
        default=None,
        help='Number of models to test (default: all models)'
    )
    
    parser.add_argument(
        '--words', '-w',
        type=int,
        default=None, 
        help='Number of words to test (default: all words)'
    )
    
    parser.add_argument(
        '--repeats', '-p',
        type=int,
        default=3,
        help='Number of repeats per model/word combination (default: 3)'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        default=None,
        help='Output filename (default: N-ModelSurvey-YYYYMMDD.csv)'
    )
    
    parser.add_argument(
        '--threads', '-t',
        type=int,
        default=10,
        help='Number of worker threads (default: 10)'
    )
    
    parser.add_argument(
        '--retries', '-r',
        type=int,
        default=3,
        help='Maximum number of retries per failed request (default: 3)'
    )
    
    parser.add_argument(
        '--paid',
        action='store_true',
        help='Use paid models instead of free models (costs money!)'
    )
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Use full lists by default, or limit to first N
    test_words = words[:args.words] if args.words else words
    
    # Choose model set based on --paid flag
    model_set = paid_models if args.paid else free_models
    test_models = model_set[:args.models] if args.models else model_set
    n_repeats = args.repeats
    
    total_calls = len(test_models) * len(test_words) * n_repeats
    
    print("=== Probability Estimation Experiment ===")
    model_type = "💰 PAID" if args.paid else "🆓 FREE"
    print(f"Model set: {model_type} ({len(test_models)} models)")
    print(f"Testing {len(test_words)} words: {test_words[:3]}{'...' if len(test_words) > 3 else ''}")
    print(f"Testing {len(test_models)} models: {test_models[:3]}{'...' if len(test_models) > 3 else ''}")
    print(f"Repeats per combination: {n_repeats}")
    print(f"Worker threads: {args.threads}")
    print(f"Total API calls: {total_calls}")
    
    # Usage limits warning
    if args.paid:
        print(f"\n💰 WARNING: Using PAID models - this will cost money!")
        print(f"   • Each API call will charge your account balance")
        print(f"   • {total_calls} requests may cost significant money")
        print(f"   • Check model pricing at https://openrouter.ai/")
    elif total_calls > 50:
        print(f"\n📋 OpenRouter Free Model Limits:")
        print(f"   • 50 requests/day (with <$10 credits)")
        print(f"   • 1000 requests/day (with ≥$10 credits)")
        print(f"   • 20 requests/minute rate limit")
        print(f"   Current experiment: {total_calls} requests")
        
        if total_calls > 1000:
            print(f"\n⚠️  This exceeds the maximum daily free limit!")
        elif total_calls > 50:
            print(f"\n⚠️  This may require ≥$10 credits in your account!")
    
    if total_calls > 100:
        response = input(f"\n⚠️  This will make {total_calls} API calls. Continue? (y/N): ")
        if response.lower() != 'y':
            print("Experiment cancelled.")
            return
    
    results = run_experiment(test_words, test_models, n_repeats, args.output, args.threads, args.retries)
    
    # Print summary
    successful_extractions = sum(1 for r in results if r["extracted_probability"] is not None)
    total_attempts = len(results)
    success_rate = (successful_extractions / total_attempts) * 100
    
    print(f"\n=== Summary ===")
    print(f"Total queries: {total_attempts}")
    print(f"Successful number extractions: {successful_extractions}")
    print(f"Success rate: {success_rate:.1f}%")


if __name__ == "__main__":
    main()