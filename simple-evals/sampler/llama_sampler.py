import time
from transformers import pipeline
from huggingface_hub import snapshot_download, login
from ..types import MessageList, SamplerBase
from typing import Any
from .llama import response_from_msg_list

class Llama3BSample(SamplerBase):
    def __init__(
        self,
        system_message: str = "As concisely as possible and your confidence as a percentage in parentheses at the end. If you don't know, return 'I don't know'.",
        temperature: float = 0.6,
        max_tokens: int = 2048,
    ):

        self.top_p = 0.9
        self.system_message = system_message
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.image_format = "png"

    # def _handle_image(
    #     self, image: str, encoding: str = "base64", format: str = "png", fovea: int = 768
    # ):
    #     new_image = {
    #         "type": "image_url",
    #         "image_url": {
    #             "url": f"data:image/{format};{encoding},{image}",
    #         },
    #     }
    #     return new_image

    def _handle_text(self, text: str):
        return {"type": "text", "text": text}

    def _pack_message(self, role: str, content: Any):
        return {"role": str(role), "content": content}
    
    def __call__(self, message_list:MessageList) -> str:
        if self.system_message:
            message_list = [self._pack_message("system", self.system_message)] + message_list
        trial = 0
        while True:
            try:
                response = response_from_msg_list(message_list)
                return response
            # NOTE: BadRequestError is triggered once for MMMU, please uncomment if you are reruning MMMU
            except Exception as e:
                exception_backoff = 2**trial
                print("Bad Request Error", e)
                time.sleep(exception_backoff)
                trial += 1

    