"""
Chain-of-Thought Data Generation Script for AI Telco Troubleshooting Challenge
Designed for Kaggle execution with Qwen3-32B as teacher model

Usage on Kaggle:
1. Enable GPU (H100/A100 recommended)
2. Add huggingface token as secret
3. Run all cells
"""

import os
import json
import torch
import pandas as pd
from tqdm import tqdm
from typing import List, Dict, Tuple
from dataclasses import dataclass
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

# Import expert examples (copy this file to your Kaggle notebook)
from expert_examples import (
    EXPERT_EXAMPLES, 
    SYSTEM_PROMPT, 
    get_few_shot_examples,
    build_generation_prompt
)


@dataclass
class GenerationConfig:
    """Configuration for CoT generation"""
    model_name: str = "Qwen/Qwen3-32B"
    max_new_tokens: int = 1024
    temperature_low: float = 0.3
    temperature_high: float = 0.7
    top_p: float = 0.9
    batch_size: int = 4
    traces_per_question: int = 2
    output_dir: str = "./cot_data"
    
    # Quality filters
    min_reasoning_length: int = 100
    require_boxed_answer: bool = True


def load_telelogs_data(split: str = "train") -> pd.DataFrame:
    """
    Load TeleLogs dataset from HuggingFace.
    Requires accepting the dataset terms first.
    """
    print(f"Loading TeleLogs {split} data...")
    
    # Load dataset
    dataset = load_dataset("netop/TeleLogs", split=split)
    
    # Convert to DataFrame
    df = pd.DataFrame(dataset)
    
    print(f"Loaded {len(df)} samples")
    return df


def load_local_data(filepath: str) -> pd.DataFrame:
    """
    Alternative: Load data from local CSV file.
    Use this if HuggingFace access is problematic.
    """
    print(f"Loading data from {filepath}...")
    df = pd.read_csv(filepath)
    print(f"Loaded {len(df)} samples")
    return df


def setup_model(config: GenerationConfig) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """
    Load Qwen3-32B model with INT8 quantization for memory efficiency.
    """
    print(f"Loading {config.model_name}...")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name,
        trust_remote_code=True
    )
    
    # Ensure pad token is set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model with INT8 quantization
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        load_in_8bit=True  # Enable INT8 for memory efficiency
    )
    
    model.eval()
    print(f"Model loaded on {next(model.parameters()).device}")
    
    return model, tokenizer


def generate_single_trace(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt: str,
    temperature: float,
    config: GenerationConfig
) -> str:
    """
    Generate a single reasoning trace.
    """
    inputs = tokenizer(
        prompt, 
        return_tensors="pt",
        truncation=True,
        max_length=4096  # Leave room for generation
    ).to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=config.max_new_tokens,
            temperature=temperature,
            top_p=config.top_p,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    
    # Decode only the new tokens
    generated = tokenizer.decode(
        outputs[0][inputs.input_ids.shape[1]:], 
        skip_special_tokens=True
    )
    
    return generated


def validate_trace(trace: str, expected_answer: str, config: GenerationConfig) -> bool:
    """
    Check if generated trace meets quality criteria.
    """
    # Check for boxed answer
    if config.require_boxed_answer:
        if f"\\boxed{{{expected_answer}}}" not in trace:
            return False
    
    # Check minimum length
    if len(trace) < config.min_reasoning_length:
        return False
    
    # Check for reasoning content
    if "<reasoning>" not in trace.lower() and "reasoning" not in trace.lower():
        # Allow if it has numbered steps
        if not any(f"{i}." in trace for i in range(1, 6)):
            return False
    
    return True


def extract_reasoning(trace: str) -> str:
    """
    Extract and clean the reasoning portion from generated text.
    """
    # Try to extract content between <reasoning> tags
    if "<reasoning>" in trace.lower():
        start = trace.lower().find("<reasoning>") + len("<reasoning>")
        end = trace.lower().find("</reasoning>")
        if end > start:
            trace = trace[start:end].strip()
    
    # Ensure it ends with boxed answer
    if "\\boxed" not in trace:
        # Find if the answer is somewhere and format it
        for i in range(1, 9):
            if f"C{i}" in trace[-50:]:  # Check last 50 chars
                trace = trace.rstrip() + f"\n\n\\boxed{{C{i}}}"
                break
    
    return trace


def generate_cot_dataset(
    data: pd.DataFrame,
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    config: GenerationConfig,
    question_col: str = "question",
    answer_col: str = "answer"
) -> List[Dict]:
    """
    Generate CoT traces for entire dataset.
    """
    results = []
    failed = 0
    
    os.makedirs(config.output_dir, exist_ok=True)
    
    temperatures = [config.temperature_low, config.temperature_high]
    
    for idx, row in tqdm(data.iterrows(), total=len(data), desc="Generating CoT"):
        question = row[question_col]
        answer = row[answer_col]
        
        # Build prompt with few-shot examples
        prompt = build_generation_prompt(question, answer)
        
        for temp_idx, temp in enumerate(temperatures[:config.traces_per_question]):
            try:
                # Generate trace
                trace = generate_single_trace(
                    model, tokenizer, prompt, temp, config
                )
                
                # Clean and extract reasoning
                cleaned_trace = extract_reasoning(trace)
                
                # Validate
                if validate_trace(cleaned_trace, answer, config):
                    results.append({
                        "id": f"{idx}_{temp_idx}",
                        "question": question,
                        "reasoning": cleaned_trace,
                        "answer": answer,
                        "temperature": temp,
                        "valid": True
                    })
                else:
                    # Still save but mark as invalid
                    results.append({
                        "id": f"{idx}_{temp_idx}",
                        "question": question,
                        "reasoning": cleaned_trace,
                        "answer": answer,
                        "temperature": temp,
                        "valid": False
                    })
                    failed += 1
                    
            except Exception as e:
                print(f"Error at index {idx}: {e}")
                failed += 1
        
        # Save checkpoints every 100 questions
        if (idx + 1) % 100 == 0:
            checkpoint_path = os.path.join(config.output_dir, f"checkpoint_{idx+1}.json")
            with open(checkpoint_path, "w") as f:
                json.dump(results, f, indent=2)
            print(f"Checkpoint saved: {len(results)} traces ({failed} failed)")
    
    # Final save
    output_path = os.path.join(config.output_dir, "cot_dataset_full.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    
    # Also save as JSONL for easier processing
    jsonl_path = os.path.join(config.output_dir, "cot_dataset.jsonl")
    with open(jsonl_path, "w") as f:
        for item in results:
            f.write(json.dumps(item) + "\n")
    
    print(f"\nGeneration complete!")
    print(f"Total traces: {len(results)}")
    print(f"Valid traces: {len([r for r in results if r['valid']])}")
    print(f"Invalid traces: {failed}")
    print(f"Saved to: {output_path}")
    
    return results


def prepare_sft_dataset(results: List[Dict], output_path: str = "./sft_data.jsonl"):
    """
    Convert generated CoT data to SFT training format.
    Only includes valid traces.
    """
    sft_data = []
    
    for item in results:
        if not item["valid"]:
            continue
            
        # Format for instruction tuning
        sft_item = {
            "instruction": item["question"],
            "input": "",
            "output": f"<reasoning>\n{item['reasoning']}\n</reasoning>\n\n\\boxed{{{item['answer']}}}"
        }
        sft_data.append(sft_item)
    
    # Save
    with open(output_path, "w") as f:
        for item in sft_data:
            f.write(json.dumps(item) + "\n")
    
    print(f"SFT dataset saved: {len(sft_data)} samples to {output_path}")
    return sft_data


# ============== KAGGLE EXECUTION ==============

def main():
    """
    Main execution function for Kaggle.
    """
    # Configuration
    config = GenerationConfig(
        model_name="Qwen/Qwen3-32B",
        max_new_tokens=1024,
        temperature_low=0.3,
        temperature_high=0.7,
        traces_per_question=2,
        output_dir="/kaggle/working/cot_data"
    )
    
    # Option 1: Load from HuggingFace
    # data = load_telelogs_data("train")
    
    # Option 2: Load from local file (if you uploaded CSV)
    # data = load_local_data("/kaggle/input/telelogs/train.csv")
    
    # For testing: use a small sample first
    # data = data.head(10)
    
    # Setup model
    model, tokenizer = setup_model(config)
    
    # Generate CoT traces
    results = generate_cot_dataset(
        data=data,
        model=model,
        tokenizer=tokenizer,
        config=config,
        question_col="question",  # Adjust based on actual column names
        answer_col="answer"
    )
    
    # Prepare SFT dataset
    sft_data = prepare_sft_dataset(
        results, 
        output_path="/kaggle/working/sft_data.jsonl"
    )
    
    print("\n=== Generation Complete ===")
    print(f"CoT traces: {len(results)}")
    print(f"SFT samples: {len(sft_data)}")
    
    return results, sft_data


# For Kaggle notebook execution
if __name__ == "__main__":
    # Uncomment to run
    # results, sft_data = main()
    
    # For testing the expert examples
    print("Testing expert examples...")
    from expert_examples import EXPERT_EXAMPLES, get_few_shot_examples
    
    for cause in ["C1", "C4", "C6", "C8"]:
        examples = get_few_shot_examples(cause)
        print(f"Few-shot for {cause}: {len(examples)} chars, {examples.count('boxed')} answers")
