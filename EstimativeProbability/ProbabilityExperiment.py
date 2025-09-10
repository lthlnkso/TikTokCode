"""
This is based on "Perception of Probability words" https://waf.cs.illinois.edu/visualizations/Perception-of-Probability-Words/

We will define a list of probability words and a list of models.
We will then ask the model to estimate the numerical probability of the word.
"""

import csv
import re
import argparse
import time
from datetime import datetime
from openrouter_client import OpenRouterClient

words = [
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


models = [
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


def run_experiment(words_to_test, models_to_test, n_repeats=3, output_filename=None):
    """Run the probability estimation experiment."""
    client = OpenRouterClient()
    
    # Generate filename with current date if not provided
    if output_filename is None:
        date_str = datetime.now().strftime("%Y%m%d")
        filename = f"{n_repeats}-ModelSurvey-{date_str}.csv"
    else:
        filename = output_filename
    
    results = []
    
    print(f"Running experiment with {len(models_to_test)} models and {len(words_to_test)} words...")
    print(f"Repeating each combination {n_repeats} times")
    print(f"Results will be saved to: {filename}")
    
    total_queries = len(models_to_test) * len(words_to_test) * n_repeats
    query_count = 0
    
    for model in models_to_test:
        print(f"\nTesting model: {model}")
        
        for word in words_to_test:
            print(f"  Word: '{word}'")
            
            for repeat in range(n_repeats):
                query_count += 1
                print(f"    Attempt {repeat + 1}/{n_repeats} ({query_count}/{total_queries})")
                
                try:
                    # Create the full prompt
                    full_prompt = f"{prompt}\n\nProbability word: \"{word}\""
                    
                    # Get response from model
                    messages = [{"role": "user", "content": full_prompt}]
                    response = client.chat_completion(messages, model=model)
                    raw_response = response["choices"][0]["message"]["content"]
                    
                    # Extract number from response
                    probability = extract_number(raw_response)
                    
                    # Record result
                    result = {
                        "model": model,
                        "word": word,
                        "repeat": repeat + 1,
                        "raw_response": raw_response.strip(),
                        "extracted_probability": probability,
                        "timestamp": datetime.now().isoformat()
                    }
                    results.append(result)
                    
                    if probability is not None:
                        print(f"      → ✅ {probability}")
                    else:
                        print(f"      → ❌ REJECTED: '{raw_response[:50]}{'...' if len(raw_response) > 50 else ''}'")
                        
                except Exception as e:
                    error_type = type(e).__name__
                    error_message = str(e)
                    
                    # Handle rate limits specifically
                    if "429" in error_message or "rate limit" in error_message.lower():
                        print(f"      → ⏳ RATE LIMITED: Waiting 60 seconds...")
                        time.sleep(60)
                        # Could retry here, but for now just log and continue
                    elif "402" in error_message or "credit" in error_message.lower():
                        print(f"      → 💳 CREDIT ERROR: {e}")
                        print("      → You may need to add credits to your OpenRouter account")
                    else:
                        print(f"      → ❌ ERROR ({error_type}): {e}")
                    
                    result = {
                        "model": model,
                        "word": word,
                        "repeat": repeat + 1,
                        "raw_response": f"ERROR ({error_type}): {e}",
                        "extracted_probability": None,
                        "timestamp": datetime.now().isoformat()
                    }
                    results.append(result)
                
                # Small delay between API calls to respect rate limits
                # (20 requests/minute = 3 seconds between requests for safety)
                time.sleep(0.5)
    
    # Save results to CSV
    print(f"\nSaving results to {filename}...")
    with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ["model", "word", "repeat", "raw_response", "extracted_probability", "timestamp"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        for result in results:
            writer.writerow(result)
    
    print(f"Experiment complete! Results saved to {filename}")
    return results


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run probability estimation experiment with AI models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 ProbabilityExperiment.py                     # Run full experiment (all models and words)
  python3 ProbabilityExperiment.py --models 5          # Test first 5 models only
  python3 ProbabilityExperiment.py --words 10          # Test first 10 words only
  python3 ProbabilityExperiment.py --models 3 --words 5 --repeats 1 --output test.csv
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
        '--repeats', '-r',
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
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Use full lists by default, or limit to first N
    test_words = words[:args.words] if args.words else words
    test_models = models[:args.models] if args.models else models
    n_repeats = args.repeats
    
    total_calls = len(test_models) * len(test_words) * n_repeats
    
    print("=== Probability Estimation Experiment ===")
    print(f"Testing {len(test_words)} words: {test_words[:3]}{'...' if len(test_words) > 3 else ''}")
    print(f"Testing {len(test_models)} models: {test_models[:3]}{'...' if len(test_models) > 3 else ''}")
    print(f"Repeats per combination: {n_repeats}")
    print(f"Total API calls: {total_calls}")
    
    # Usage limits warning
    if total_calls > 50:
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
    
    results = run_experiment(test_words, test_models, n_repeats, args.output)
    
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