import logging
from collections.abc import Iterator

import ollama

logger = logging.getLogger(__name__)


class OllamaGenerationError(RuntimeError):
    """Raised when the local Ollama server can't be reached or fails to respond."""


class OllamaClient:
    def __init__(self, host: str, model: str, temperature: float, timeout_seconds: float) -> None:
        self._client = ollama.Client(host=host, timeout=timeout_seconds)
        self._model = model
        self._temperature = temperature

    def chat(self, system_prompt: str, user_message: str) -> str:
        try:
            response = self._client.chat(
                model=self._model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message},
                ],
                options={"temperature": self._temperature},
            )
        except Exception as exc:
            logger.exception("Ollama chat request failed")
            raise OllamaGenerationError(
                f"Could not get a response from Ollama model '{self._model}' at "
                f"the configured host. Is `ollama serve` running and is the model pulled?"
            ) from exc

        return response["message"]["content"]

    def chat_stream(self, system_prompt: str, user_message: str) -> Iterator[str]:
        try:
            stream = self._client.chat(
                model=self._model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message},
                ],
                options={"temperature": self._temperature},
                stream=True,
            )
            for part in stream:
                yield part["message"]["content"]
        except Exception as exc:
            logger.exception("Ollama streaming chat request failed")
            raise OllamaGenerationError(
                f"Could not stream a response from Ollama model '{self._model}'."
            ) from exc
