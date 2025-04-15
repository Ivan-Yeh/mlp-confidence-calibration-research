import base64
import time
from typing import Any

from ..types import MessageList, SamplerBase

from transformers import LlamaForCausalLM, AutoTokenizer
import torch
import numpy as np
from openai import OpenAI
from together import Together

# Hugging Face Sampler
class HFSampler(SamplerBase):
    """
    Sample from huggingface_models using tokeniser
    """

    def __init__(
        self,
        model: LlamaForCausalLM,
        tokeniser: AutoTokenizer, 
        model_name: str = "HF Tokeniser",
        system_message: str | None = None,
        temperature: float = 0.5,
        max_new_tokens: int = 10):

        self.model: LlamaForCausalLM = model
        self.tokeniser: AutoTokenizer = tokeniser
        self.model_name: str = str(model_name).split("/")[-1]
        self.system_message = system_message
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self.image_format = "url"

    def _handle_image(
        self, image: str, encoding: str = "base64", format: str = "png", fovea: int = 768
    ):
        new_image = {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/{format};{encoding},{image}",
            },
        }
        return new_image

    def _handle_text(self, text: str):
        return {"type": "text", "text": text}

    def _pack_message(self, role: str, content: Any):
        return {"role": str(role), "content": content}

    def __call__(self, message_list: MessageList) -> str:
        if self.system_message:
            message_list = [self._pack_message("system", self.system_message)] + message_list
        trial = 0
        while True:
            try:
                flat_msg_list = []
                for x in range(len(message_list)): flat_msg_list += [{"role": k , "content": v} for k, v in message_list[x].items()]
                inputs = self.tokeniser(self.tokeniser.apply_chat_template(flat_msg_list, tokenize=False, add_generation_prompt=True), return_tensors="pt")
                with torch.no_grad():
                    outputs = self.model.generate(
                        inputs['input_ids'], 
                        attention_mask=inputs["attention_mask"],
                        max_new_tokens=self.max_new_tokens, 
                        return_dict_in_generate=True, 
                        output_logits=True,
                        output_scores=True,
                        top_p =0.9,
                        pad_token_id=self.tokeniser.eos_token_id
                    )
                # return response_text and logprob based confidence
                return "".join(self.tokeniser.batch_decode(outputs.sequences[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)), float(np.exp(self.model.compute_transition_scores(outputs.sequences, outputs.scores, normalize_logits=True).numpy(force=True)).mean())

            # NOTE: BadRequestError is triggered once for MMMU, please uncomment if you are reruning MMMU
            except Exception as e:
                exception_backoff = 2**trial  # expontial back off
                print(
                    f"Rate limit exception so wait and retry {trial} after {exception_backoff} sec",
                    e,
                )
                time.sleep(exception_backoff)
                trial += 1
            # unknown error shall throw exception

# Databricks Sampler
class DBSampler(SamplerBase):
    """
    Sample from Databricks model
    """

    def __init__(
        self,
        api_key: str,
        model_url: str,
        model_name: str = "Databricks",
        system_message: str | None = None,
        temperature: float = 0.5,
        max_tokens: int = 128):

        self.api_key: str = api_key
        self.model_url: str = model_url,
        self.model_name: str = str(model_name).split("/")[-1]
        self.system_message = system_message
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.image_format = "url"

    def _handle_image(
        self, image: str, encoding: str = "base64", format: str = "png", fovea: int = 768
    ):
        new_image = {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/{format};{encoding},{image}",
            },
        }
        return new_image

    def _handle_text(self, text: str):
        return {"type": "text", "text": text}

    def _pack_message(self, role: str, content: Any):
        return {"role": str(role), "content": content}

    def __call__(self, message_list: MessageList) -> str:
        if self.system_message:
            message_list = [self._pack_message("system", self.system_message)] + message_list
        trial = 0
        while True:
            try:
                client = OpenAI(api_key=self.api_key, base_url=(self.model_url[0] if isinstance(self.model_url, tuple) else self.model_url))
                chat_completion = client.chat.completions.create(messages=message_list, model=self.model_name, max_tokens=self.max_tokens, logprobs=True)

                # return response_text and logprob based confidence
                return chat_completion.choices[0].message.content, float(np.exp(np.array([t.logprob for t in chat_completion.choices[0].logprobs.content])).mean())
            # NOTE: BadRequestError is triggered once for MMMU, please uncomment if you are reruning MMMU
            except Exception as e:
                exception_backoff = 2**trial  # expontial back off
                print(
                    f"Rate limit exception so wait and retry {trial} after {exception_backoff} sec",
                    e,
                )
                time.sleep(exception_backoff)
                trial += 1
            # unknown error shall throw exception



# Together AI Sampler
class TogetherSampler(SamplerBase):
    """
    Sample from Together AI model
    """

    def __init__(
        self,
        api_key: str,
        model_id: str,
        model_name: str = "Together AI",
        system_message: str | None = None,
        temperature: float = 0.5,
        max_tokens: int = 128):

        self.api_key: str = api_key
        self.model_id: str = model_id,
        self.model_name: str = str(model_name).split("/")[-1]
        self.system_message = system_message
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.image_format = "url"

    def _handle_image(
        self, image: str, encoding: str = "base64", format: str = "png", fovea: int = 768
    ):
        new_image = {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/{format};{encoding},{image}",
            },
        }
        return new_image

    def _handle_text(self, text: str):
        return {"type": "text", "text": text}

    def _pack_message(self, role: str, content: Any):
        return {"role": str(role), "content": content}

    def __call__(self, message_list: MessageList) -> str:
        if self.system_message:
            message_list = [self._pack_message("system", self.system_message)] + message_list
        trial = 0
        while True:
            try:
                client = Together(api_key=self.api_key)
                chat_completion = client.chat.completions.create(messages=message_list, model=self.model_id, max_tokens=self.max_tokens, logprobs=1)

                # return response_text and logprob based confidence
                return chat_completion.choices[0].message.content, float(np.exp(chat_completion.choices[0].logprobs.token_logprobs).mean())
            # NOTE: BadRequestError is triggered once for MMMU, please uncomment if you are reruning MMMU
            except Exception as e:
                exception_backoff = 2**trial  # expontial back off
                print(
                    f"Rate limit exception so wait and retry {trial} after {exception_backoff} sec",
                    e,
                )
                time.sleep(exception_backoff)
                trial += 1
            # unknown error shall throw exception