import dspy
import os
import sys
from dotenv import load_dotenv
from datasets import load_dataset


load_dotenv()
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")
ENDPOINT_URL = os.environ.get("ENDPOINT_URL")

# Configuration for the model
HUGGINGFACE_MODEL_NAME = "Qwen/Qwen3-8B"

# Configure DSPy LM for text generation
lm = dspy.LM(f"openai/{HUGGINGFACE_MODEL_NAME}",
             api_base=os.environ.get("ENDPOINT_URL"),
             api_key="local", 
             model_type="chat")
dspy.configure(lm=lm)


def load_train_set(num_examples=100):
    """Load training data from HuggingFace dataset."""
    print(f"Loading {num_examples} examples from HuggingFace...")
    dataset = load_dataset("dmitva/human_ai_generated_text", split="train").select(range(num_examples))
    
    train_examples = []
    for example in dataset:
        train_examples.append(
            dspy.Example({
                "prompts": example["instructions"],
                "human_text": example["human_text"],
                "ai_text": example["ai_text"],
            }).with_inputs("prompts")
        )
    return train_examples


class GenerateText(dspy.Signature):
    """Generate text based on given instructions."""
    prompts = dspy.InputField(desc="Task instructions for text generation")
    generated_text = dspy.OutputField(desc="Generated text following the instructions")


class AIDetection(dspy.Signature):
    """Detect whether a text is AI-generated or human-written."""
    text = dspy.InputField(desc="Text to analyze")
    ai_probability = dspy.OutputField(desc="Probability (0-100) that this text is AI-generated, where 0 is definitely human and 100 is definitely AI")

class HumanTextGenerator(dspy.Module):
    """Module to generate human-like text based on instructions."""
    
    def __init__(self):
        super().__init__()
        self.generate = dspy.Predict(GenerateText)
    
    def forward(self, prompts):
        result = self.generate(prompts=prompts)
        return dspy.Prediction(generated_text=result.generated_text)
import re
import math
def extract_probability(result):
    """Helper to extract probability from detector result."""
    try:
        return float(result.ai_probability)
    except (ValueError, AttributeError):
        match = re.search(r'\d+\.?\d*', str(result.ai_probability))
        return float(match.group()) if match else 50.0
        
def gepa_answer_metric(gold, pred, trace=None, pred_name=None, pred_trace=None):
    generated_text = pred.generated_text
    with dspy.context(lm=detector_lm):
        ai_detector = dspy.Predict(AIDetection)
        generated_detection_result = ai_detector(text=generated_text)
        generated_ai_prob = extract_probability(generated_detection_result)
    generated_ai_prob = max(0, min(100, generated_ai_prob)) / 100
    temperature = 0.5
    raw_score = 1.0 - generated_ai_prob
    logit = math.log(raw_score + 1e-8) - math.log(1 - raw_score + 1e-8)
    scaled_logit = logit / temperature
    score = 1.0 / (1.0 + math.exp(-scaled_logit))
    return score
 
if __name__ == "__main__":
    # Check server configuration
    print(f"Endpoint URL: {ENDPOINT_URL}")
    print(f"Model: {HUGGINGFACE_MODEL_NAME}")
    
    # Load training data
    print("\nLoading training data...")
    trainset = load_train_set(num_examples=50)  # Start with smaller set for faster iteration
    print(f"Loaded {len(trainset)} training examples")
    
    # Create the text generator module
    print("\nInitializing HumanTextGenerator module...")
    text_generator = HumanTextGenerator()
    
    # Test the module on first example
    print("\nTesting module on first example...")
    first_example = trainset[0]
    print(f"Prompts: {first_example.prompts[:100]}...")
    result = text_generator(prompts=first_example.prompts)
    print(f"Generated: {result.generated_text[:200]}...")
    
    # Create reflection LM for GEPA (same as detector_lm)
    print("\nSetting up GEPA optimizer...")
    print("Using Claude Sonnet 4.5 for AI detection and reflection...")
    reflection_lm = dspy.LM(
        "anthropic/claude-sonnet-4-5-20250929",
        api_key=ANTHROPIC_API_KEY,
        temperature=1.0,
        thinking={
            "type": "enabled",
            "budget_tokens": 10000
        },
        max_tokens=32000,
    )
    
    # Run GEPA optimization
    print("\nStarting GEPA optimization...")
    print("This will optimize the prompts to generate text that appears human-written...")
    print("Metric: AI detection model judges whether text appears AI-generated or human-written")
    tp = dspy.GEPA(
        metric=answer_metric,
        auto="medium",
        num_threads=4,  # Reduced for stability
        reflection_lm=reflection_lm,
    )
    
    optimized_generator = tp.compile(text_generator, trainset=trainset)
    
    # Save the optimized program
    output_path = "/home/stephenx/code/ghostwriter/optimized_human_text_generator.json"
    optimized_generator.save(output_path)
    print(f"\nOptimized generator saved to {output_path}")
    
    # Test optimized version
    print("\nTesting optimized module...")
    result_optimized = optimized_generator(prompts=first_example.prompts)
    print(f"Optimized generated: {result_optimized.generated_text[:200]}...")