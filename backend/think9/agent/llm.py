"""The only place that talks to a model. OpenAI-compatible, so it points at Groq."""

from openai import OpenAI

from think9.config import get_settings


class LLM:
    def __init__(self) -> None:
        settings = get_settings()
        self._settings = settings
        self._client = OpenAI(api_key=settings.llm_api_key, base_url=settings.llm_base_url)

    def complete(self, system: str, user: str, model: str | None = None) -> str:
        response = self._client.chat.completions.create(
            model=model or self._settings.llm_model,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            temperature=0,
        )
        return (response.choices[0].message.content or "").strip()
