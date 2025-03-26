import json
import torch
import transformers
from huggingface_hub import snapshot_download, login

with open("api.json") as f:
    login(token=json.load(f)["HF_TOKEN"])

device = "cuda" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
local_dir = snapshot_download(repo_id="meta-llama/Llama-3.2-3B-Instruct")
# pipeline: transformers.pipeline = transformers.pipeline("text-generation", model=local_dir, device_map="auto", model_kwargs={"torch_dtype": torch.float16, "low_cpu_mem_usage": True})
tokenizer = transformers.AutoTokenizer.from_pretrained(local_dir)
model = transformers.AutoModelForCausalLM.from_pretrained(
    local_dir, 
    device_map="auto",   # Automatically assign layers to GPU
    torch_dtype=torch_dtype, 
    low_cpu_mem_usage=True
)
pipeline = transformers.pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
)
terminators = [pipeline.tokenizer.eos_token_id, pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")]

def response_from_msg_list(msg_list: list[dict], max_tokens=2048, temperature=0.6, top_p=0.9):
    user_prompt = msg_list
    prompt = pipeline.tokenizer.apply_chat_template(
        user_prompt, tokenize=False, add_generation_prompt=True
    )
    outputs = pipeline(
        prompt,
        max_new_tokens=max_tokens,
        eos_token_id=terminators,
        do_sample=True,
        temperature=temperature,
        top_p=top_p
    )
    response = outputs[0]["generated_text"][len(prompt):]
    return response