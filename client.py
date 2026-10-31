import os
from langchain_gigachat.chat_models import GigaChat
from dotenv import load_dotenv

load_dotenv()

def get_client(temperature = 0.1):
    return GigaChat(
        credentials = os.getenv("AUTH_KEY"),
        model="GigaChat-2",
        scope="GIGACHAT_API_PERS",
        # ca_bundle_file=os.getenv("CERTIFI_PATH"),
        verify_ssl_certs=False,
        temperature=temperature,
        max_tokens=600,
        # allow_any_tool_choice_fallback=True
    )
