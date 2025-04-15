import transformers
import torch
import numpy as np
from transformers import LlamaForCausalLM, AutoTokenizer


# takes a str query and returns str responses
def response_pipeline(query, pipeline: transformers.pipeline, terminators: list, max_tokens=2048, temperature=0.6, top_p=0.9):
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
def response_pipeline_msg_lst(msg_list: list[dict], pipeline: transformers.pipeline, terminators: list, max_tokens=2048, temperature=0.6, top_p=0.9):
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


def response_tokeniser_with_sgc(message_lst: list[dict], tokeniser: AutoTokenizer, model: LlamaForCausalLM, max_new_tokens = 5) -> tuple:
    """_summary_

    Args:
        message_lst (list[dict]): 
        tokeniser (AutoTokenizer): 
        model (LlamaForCausalLM): 
        max_new_tokens (int, optional): Defaults to 5.
    Returns:
        tuple: (str, float) response, single generation confidence
    """
    flat_msg_list = []
    for x in range(len(message_lst)): flat_msg_list += [{"role":k , "content": v} for k, v in message_lst[x].items()]
    inputs = tokeniser(tokeniser.apply_chat_template(flat_msg_list, tokenize=False, add_generation_prompt=True), return_tensors="pt")
    with torch.no_grad():
        outputs = model.generate(
            inputs['input_ids'], 
            attention_mask=inputs["attention_mask"],
            max_new_tokens=max_new_tokens, 
            return_dict_in_generate=True, 
            output_logits=True,
            output_scores=True,
            top_p =0.9,
            pad_token_id=tokeniser.eos_token_id
        )
    return "".join(tokeniser.batch_decode(outputs.sequences[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)), float(np.exp(model.compute_transition_scores(outputs.sequences, outputs.scores, normalize_logits=True).numpy(force=True)).mean())