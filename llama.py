import json
import torch
import transformers
from huggingface_hub import snapshot_download, login

with open("api.json") as f:
    login(token=json.load(f)["HF_TOKEN"])

local_dir = snapshot_download(repo_id="meta-llama/Llama-3.2-3B-Instruct")

pipeline: transformers.pipeline = transformers.pipeline("text-generation", model=local_dir, device_map="auto", model_kwargs={"torch_dtype": torch.float16, "low_cpu_mem_usage": True})
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