import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_gigachat.chat_models import GigaChat

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env", override=True)


def get_client(temperature=0.1):
    auth_key = (os.getenv("AUTH_KEY") or "").strip()
    if not auth_key:
        raise RuntimeError(f"AUTH_KEY не найден. Проверь файл {ROOT / '.env'}")

    return GigaChat(
        credentials=auth_key,
        model="GigaChat-2",
        base_url="https://gigachat.devices.sberbank.ru/api/v1",
        scope="GIGACHAT_API_PERS",
        verify_ssl_certs=False,
        temperature=temperature,
        max_tokens=600,
    )