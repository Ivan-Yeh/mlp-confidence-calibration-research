import base64
import time
from typing import Any

from .hf_response_generator import response_pipeline, response_pipeline_msg_lst, response_tokeniser_with_sgc

from ..types import MessageList, SamplerBase

from transformers import LlamaForCausalLM, AutoTokenizer
import torch
import numpy as np
import os
from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering
from collections import Counter


class HFSamplerPipeline(SamplerBase):
    """
    Sample from huggingface_models using pipeline
    """

    def __init__(
        self,
        pipeline,
        terminators, 
        model_name: str = "HF Pipeline",
        system_message: str | None = None,
        temperature: float = 0.5,
        max_tokens: int = 2048):

        self.pipeline = pipeline
        self.terminators = terminators
        self.model_name = str(model_name).split("/")[-1]
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
                response = response_pipeline_msg_lst(message_list, self.pipeline, self.terminators, self.max_tokens, self.temperature, 0.9)
                return response
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


class HFSamplerTokeniser(SamplerBase):
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

        self.model = model
        self.tokeniser = tokeniser
        self.model_name = str(model_name).split("/")[-1]
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
                response = response_tokeniser_with_sgc(message_list, self.tokeniser, self.model, self.max_new_tokens)
                return response
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
