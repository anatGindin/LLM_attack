import os
import uuid
import json
from tqdm import tqdm
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline
from huggingface_hub import login
from dotenv import load_dotenv
from constants import (
    DO_SAMPLE, TOP_K, TEMPERATURE, TOP_P, MAX_LENGTH, MAX_NEW_TOKENS,
    HUMAN_EVAL_DATASET, CLASS_EVAL_DATASET
)
from utils import save_json, read_jsonl, add_jsonl_completion, add_jsonl_row, human_eval_generate_prompt, class_eval_generate_prompt, read_json

# --------------------
# 🔑 Environment setup
# --------------------
load_dotenv()
hf_token = os.getenv('HF_TOEKN')
if hf_token:
    login(hf_token)
else:
    print("⚠️  Missing HF_TOEKN in .env — continuing without login")

# --------------------
# ⚙️ Config
# --------------------
BATCH_SIZE = 2  # adjust as needed (4–8 works best for 7B models)
SLEEP_BETWEEN_BATCHES = 0.3

# --------------------
# 🧠 Model Setup
# --------------------
# model_name = "google/codegemma-7b"
# model_name = "codellama/CodeLlama-7b-hf"
model_name = "deepseek-ai/deepseek-coder-6.7b-base"

bnb_config = BitsAndBytesConfig(
    load_in_8bit=True,
    llm_int8_enable_fp32_cpu_offload=True,
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch.float16,
)

tokenizer = AutoTokenizer.from_pretrained(model_name)

pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=MAX_NEW_TOKENS,
    do_sample=DO_SAMPLE,
    top_k=TOP_K,
    top_p=TOP_P,
    temperature=TEMPERATURE,
)
# ✅ Fix for tokenizer without padding token
if pipe.tokenizer.pad_token_id is None:
    pipe.tokenizer.pad_token_id = pipe.model.config.eos_token_id

if torch.cuda.is_available():
    print(f"✅ Using GPU: {torch.cuda.get_device_name(0)}")
else:
    print("⚠️ Using CPU (expect slower generation)")

# --------------------
# 💡 Helper function
# --------------------
def complete_batch(prompts):
    """Generate completions for a list of prompts in one batch."""
    try:
        outputs = pipe(
            prompts,
            batch_size=len(prompts),
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=DO_SAMPLE,
            top_k=TOP_K,
            top_p=TOP_P,
            temperature=TEMPERATURE,
            eos_token_id=tokenizer.eos_token_id,
        )
        # HuggingFace pipeline returns list of dicts
        return [out[0]["generated_text"] if isinstance(out, list) else out["generated_text"] for out in outputs]
    except Exception as e:
        print(f"❌ Error during batch generation: {e}")
        return [f"ERROR: {e}" for _ in prompts]

# --------------------
# 🧩 Dataset setup
# --------------------
datasets = ["HumanEval.jsonl"]
result_folder_base = os.path.join(os.path.dirname(__file__), "generated_outputs", "exp2", "deepSeekResults")

# --------------------
# 🚀 Main Loop
# --------------------
for dataset in datasets:
    exp_result = os.path.join(result_folder_base, str(uuid.uuid4()))
    os.makedirs(exp_result, exist_ok=True)
    config_file = os.path.join(exp_result, 'config.json')
    result_file = os.path.join(exp_result, 'result.json')

    exp_config = {
        'dataset': dataset,
        'task': 'code_completion',
        'generation_config': {
            'do_sample': DO_SAMPLE,
            'top_k': TOP_K,
            'temperature': TEMPERATURE,
            'top_p': TOP_P,
            'max_length': MAX_LENGTH,
            'max_new_tokens': MAX_NEW_TOKENS
        }
    }
    save_json(save_file=config_file, object=exp_config)

    # Choose generator and loader
    if HUMAN_EVAL_DATASET in dataset or dataset.endswith("HumanEval.jsonl"):
        generate_prompt = human_eval_generate_prompt
        data_inputs = read_jsonl(dataset)
        task_id_field = 'task_id'
    elif CLASS_EVAL_DATASET in dataset:
        generate_prompt = class_eval_generate_prompt
        data_inputs = read_json(dataset)
        task_id_field = 'task_id'
    else:
        print("Unknown dataset type")
        exit()

    print(f"🧾 Loaded {len(data_inputs)} tasks from {dataset}")

    # Iterate and generate in batches
    for start in tqdm(range(0, len(data_inputs), BATCH_SIZE), desc="Generating"):
        batch_inputs = data_inputs[start:start + BATCH_SIZE]
        prompts = []
        task_ids = []

        for inp in batch_inputs:
            task_id = inp[task_id_field]
            for p in generate_prompt(inp):
                prompts.append(p)
                task_ids.append(task_id)
                # pre-save row so we can later add completion
                add_jsonl_row(result_file, {"task_id": task_id, "prompt": p, "result": []})

        completions = complete_batch(prompts)
        for task_id, prompt, completion in zip(task_ids, prompts, completions):
            add_jsonl_completion(result_file, model=model_name, completion=completion)

        torch.cuda.empty_cache()
        import time; time.sleep(SLEEP_BETWEEN_BATCHES)

print("✅ Done generating all completions.")
