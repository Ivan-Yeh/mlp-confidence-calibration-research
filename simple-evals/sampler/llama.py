import json
import torch
import transformers
from huggingface_hub import snapshot_download, login
import os

login(token="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")

local_dir = snapshot_download(repo_id="meta-llama/Llama-3.2-3B-Instruct")

pipeline: transformers.pipeline = transformers.pipeline("text-generation", model=local_dir, device_map="auto", model_kwargs={"torch_dtype": torch.float16, "low_cpu_mem_usage": True})
terminators = [pipeline.tokenizer.eos_token_id, pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")]

# takes a str query and returns str responses
def get_response(query, message_history=[], max_tokens=2048, temperature=0.6, top_p=0.9, confidence=False):
    user_prompt = message_history + [{"role": "user", "content": query}]
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
    print(response)
    return response

# take a list of messages (dict[str, any] -> role, content) and a response
def response_from_msg_list(msg_list: list[dict], message_history=[], max_tokens=2048, temperature=0.6, top_p=0.9, confidence=False):
    if confidence:
        for x in range(len(msg_list)):
            msg_list[x]["content"] += " If you dont't know, say I don't know. At the end of your response, add your confidence in percentage in the correctness of your response in square brackets, percentage only."
    user_prompt = msg_list
    print("message list:", user_prompt)
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
    print(response)
    return response