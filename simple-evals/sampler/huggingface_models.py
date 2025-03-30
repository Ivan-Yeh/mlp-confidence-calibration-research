import transformers
from huggingface_hub import snapshot_download, login


# takes a str query and returns str responses
def get_response(query, pipeline: transformers.pipeline, terminators: list, max_tokens=2048, temperature=0.6, top_p=0.9):
    user_prompt = [{"role": "user", "content": query}]
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


# takes a list of messages (dict[str, any] -> role, content) returns a str response
def response_from_msg_list(msg_list: list[dict], pipeline: transformers.pipeline, terminators: list, max_tokens=2048, temperature=0.6, top_p=0.9):
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