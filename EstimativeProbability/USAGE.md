# How to Run the Probability Estimation Experiment

## Setup

1. **Get your OpenRouter API key:**
   - Sign up at https://openrouter.ai/
   - Get your API key from the dashboard

2. **Set up environment:**
   ```bash
   cd EstimativeProbability
   ./setup.sh
   source venv/bin/activate
   ```

3. **Configure API key:**
   ```bash
   cp env.example .env
   # Edit .env and add your OPENROUTER_API_KEY
   ```

## Running the Experiment

### Quick Test (5 words × 5 models × 3 repeats = 75 API calls)
```bash
source venv/bin/activate
python3 ProbabilityExperiment.py
```

This will test the first 5 probability words with the first 5 models, repeating each combination 3 times.

### Output

The experiment creates a CSV file named `N-ModelSurvey-YYYYMMDD.csv` where:
- `N` = number of repeats per combination
- `YYYYMMDD` = current date

**CSV Columns:**
- `model`: The AI model used
- `word`: The probability word tested  
- `repeat`: Which repetition (1, 2, or 3)
- `raw_response`: Exactly what the model returned
- `extracted_probability`: The number we extracted (0-1), or None if failed
- `timestamp`: When this test was run

## Testing the Number Extraction

Test how well our number extraction works:
```bash
python3 test_extract_number.py
```

## Customizing the Experiment

Edit `ProbabilityExperiment.py` to:
- Test different words: modify `test_words = words[:5]`
- Test different models: modify `test_models = models[:5]`  
- Change repetitions: modify `n_repeats = 3`

## Example Results

The experiment will output progress like:
```
=== Probability Estimation Experiment ===
Testing words: ['Almost Certain', 'Highly Likely', 'Very Good Chance', 'We Believe', 'Likely']
Testing models: ['nvidia/nemotron-nano-9b-v2', 'openrouter/sonoma-dusk-alpha', ...]

Testing model: nvidia/nemotron-nano-9b-v2
  Word: 'Almost Certain'
    Attempt 1/3 (1/75)
      → 0.95
    Attempt 2/3 (2/75) 
      → 0.92
    Attempt 3/3 (3/75)
      → 0.96
```

And save detailed results to CSV for analysis!
