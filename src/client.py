import os
from contextvars import ContextVar
from langchain_gigachat.chat_models import GigaChat
from dotenv import load_dotenv

load_dotenv()

LLM_SOURCE = "GigaChat-2-Pro"

_credentials: ContextVar[str | None] = ContextVar("gigachat_credentials", default=None)
_scope: ContextVar[str | None] = ContextVar("gigachat_scope", default=None)


def set_credentials(key: str | None, scope: str | None = None) -> None:
    _credentials.set(key.strip() if key else None)
    _scope.set(scope)


def get_client(temperature = 0.1):
    key = _credentials.get() or os.getenv("AUTH_KEY")
    if not key:
        raise RuntimeError("Не задан ключ GigaChat: введите его в боковой панели или в AUTH_KEY.")
    
    return GigaChat(
        credentials = key,
        model=LLM_SOURCE,
        scope="GIGACHAT_API_PERS",
        # ca_bundle_file=os.getenv("CERTIFI_PATH"),
        verify_ssl_certs=False,
        temperature=temperature,
        max_tokens=600
    )
