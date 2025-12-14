#### Begin of users emails' extraction ####


GITHUB_TOKEN = "PUT YOUR GITHUB TOKEN"
HEADERS = {"Authorization": f"token {GITHUB_TOKEN}"}
keywords = [
    # Coplot
    'by GitHub Copilot',
    'use GitHub Copilot',
    'with GitHub Copilot',
    
    # Chatgpt 
    'by ChatGPT',
    'use ChatGPT',
    'with ChatGPT',
    
    # Chatgpt 
    'by Codex',
    'use Codex',
    'with Codex',
    
    # CodeLlama
    'by CodeLlama',
    'use CodeLlama',
    'with CodeLlama',
    
    # Llama
    'by Llama',
    'use Llama',
    'with Llama',
    
    # Llama2
    'by Llama-2',
    'use Llama-2',
    'with Llama-2',
]
languages = ["Python"]  # Adjust based on study focus
results_per_page = 30
max_pages = 5  # Number of pages to retrieve for each search query

# GitHub Search API endpoint
SEARCH_URL = "https://api.github.com/search/code"

#### End of users emails' extraction ####



#### Begin of Judger ####
# SYSTEM PROMPT
JUDGE_SYSTEM_PROMPT = """
You are an expert Python programmer and code reviewer. 
Your task is to carefully evaluate a single Python code snippet in response to a prompt. 
Assess the code for correctness, efficiency, readability, and security practices. 
Provide detailed reasoning for your evaluation. 
Output ONLY a JSON object as follows:

{
    "score": <integer between 0 and 100>,
    "reasoning": "<detailed explanation of your assessment>"
}
"""

# USER PROMPT
JUDGE_USER_PROMPT = """
Please evaluate the following Python code snippet for the task described in the prompt:

Prompt:
{}

Code:
{}

Respond with the JSON object as described in the system instructions.
"""

# === Security Evaluation Prompts ===
SECURITY_SYSTEM_PROMPT = """
You are an expert in Python security and secure coding practices.
Analyze the given Python code for potential security vulnerabilities.
Focus on real risks that could be exploited in production environments.
Provide results in JSON format with the following keys:
- 'vulnerability_type': type/category of issue
- 'description': explanation of the vulnerability
- 'severity': low, medium, or high
Respond ONLY with JSON.
"""

SECURITY_USER_PROMPT = "Analyze the following Python code for security issues:\nCode:\n{}"



JUDGE_MODEL_NAME = "gemini-2.5-flash-lite"
# JUDGE_MODEL_NAME = "gemini-2.5-flash"
# JUDGE_MODEL_NAME = "gemini-1.5-flash"

JUDGE_TEMPERATURE = 0.0 
JUDGE_MAX_TOKENS = 5120
JUDGE_TOP_P = 1
JUDGE_FREQUENCY_PENALITY = 0
JUDGE_PRESENCE_PENALITY = 0


DO_SAMPLE = True
TOP_K = 10
TEMPERATURE = 0.1
TOP_P = 0.95
MAX_LENGTH = 200
MAX_NEW_TOKENS = 512

HUMAN_EVAL_DATASET = "HumanEval"
CLASS_EVAL_DATASET = "ClassEval"



#### End of Judger ####