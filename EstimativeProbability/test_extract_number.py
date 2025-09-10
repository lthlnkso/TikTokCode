#!/usr/bin/env python3
"""
Test script for the number extraction function.
This tests how well our regex works at extracting probabilities from various response formats.
"""

from ProbabilityExperiment import extract_number

# Test cases simulating various model responses (now with STRICT validation)
test_responses = [
    "0.8",                           # ✅ Perfect response
    "0.75",                          # ✅ Perfect response  
    "The probability is 0.6",        # ❌ Response with explanation (SHOULD BE REJECTED)
    "I estimate approximately 0.85", # ❌ Response with qualifier (SHOULD BE REJECTED)
    "Around 0.4 or so",             # ❌ Casual response (SHOULD BE REJECTED)
    "Between 0.7 and 0.8",          # ❌ Range (SHOULD BE REJECTED)
    "85%",                          # ❌ Percentage format (invalid)
    "0.9 (90%)",                    # ❌ Mixed format (SHOULD BE REJECTED)
    "Roughly 70%",                  # ❌ Percentage only (invalid)
    "I think it's about 0.55",      # ❌ Conversational (SHOULD BE REJECTED)
    "0.12345",                      # ✅ High precision
    "1.0",                          # ✅ Edge case: exactly 1
    "0.0",                          # ✅ Edge case: exactly 0
    "1.5",                          # ❌ Invalid: > 1 (should be rejected)
    "-0.3",                         # ❌ Invalid: < 0 (should be rejected)
    "No idea",                      # ❌ No number
    "",                             # ❌ Empty response
    "approximately 0.7-0.8",        # ❌ Range with qualifier (SHOULD BE REJECTED)
    ".5",                           # ✅ Decimal without leading zero
    "fifty percent",                # ❌ Written out (invalid)
    "0.5 ",                         # ✅ Number with trailing space (strip() handles this)
    " 0.5",                         # ✅ Number with leading space (strip() handles this) 
    "0.5\n",                        # ✅ Number with newline (strip() handles this)
    "1",                            # ✅ Integer 1
    "0",                            # ✅ Integer 0  
]

print("=== Testing Number Extraction ===\n")

for i, response in enumerate(test_responses, 1):
    extracted = extract_number(response)
    status = "✅" if extracted is not None else "❌"
    print(f"{i:2d}. {status} '{response}' → {extracted}")

print(f"\n=== Summary ===")
successful = sum(1 for response in test_responses if extract_number(response) is not None)
print(f"Successfully extracted: {successful}/{len(test_responses)} ({successful/len(test_responses)*100:.1f}%)")
