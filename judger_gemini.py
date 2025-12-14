import os
import json
import google.generativeai as genai
from typing import List

# Load your Gemini API key from environment variables
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("Please set the GEMINI_API_KEY environment variable.")

# Configure the Gemini client
genai.configure(api_key=api_key)
client = genai.Client()

# Define your model
MODEL_NAME = "gemini-2.5-flash"  # Replace with your desired model

def load_json(file_path: str) -> dict:
    """Load JSON data from a file."""
    with open(file_path, "r") as file:
        return json.load(file)

def generate_code(prompt: str) -> str:
    """Generate code using the Gemini model."""
    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt
    )
    return response.text

def evaluate_task(task: dict) -> dict:
    """Evaluate a task by generating code and comparing it to the expected result."""
    prompt = task["prompt"]
    expected_code = task["result"][0]["result"]
    generated_code = generate_code(prompt)
    
    # Compare generated code with expected code
    is_correct = generated_code.strip() == expected_code.strip()
    
    return {
        "task_id": task["task_id"],
        "is_correct": is_correct,
        "generated_code": generated_code,
        "expected_code": expected_code
    }

def main():
    # Load the task data
    task_data = load_json("generated_outputs/exp1/codeLlamaResults/result.json")
    
    # Process each task
    results = []
    for task in task_data:
        result = evaluate_task(task)
        results.append(result)
    
    # Output the results
    for result in results:
        print(f"Task ID: {result['task_id']}")
        print(f"Correct: {result['is_correct']}")
        print(f"Generated Code:\n{result['generated_code']}")
        print(f"Expected Code:\n{result['expected_code']}")
        print("-" * 40)

if __name__ == "__main__":
    main()
