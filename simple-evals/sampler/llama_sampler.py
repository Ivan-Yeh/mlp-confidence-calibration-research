import base64
import time
from typing import Any

from .llama import response_from_msg_list, get_response

from ..types import MessageList, SamplerBase


class LlamaSampler(SamplerBase):
    """
    Sample from llama-3.2-3b-instruct
    """

    def __init__(
        self,
        model: str = "llama-3.2-3b-instruct",
        system_message: str | None = None,
        temperature: float = 0.5,
        max_tokens: int = 1024,
        confidence = False):

        self.confidence = confidence
        self.model = model
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
                response = response_from_msg_list(message_list, [], self.max_tokens, self.temperature, 0.9, self.confidence)
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
