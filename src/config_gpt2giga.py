import os

import gpt2giga.api_server as api_server


def start_proxy(
    base_url: str,
    model: str,
    host: str = "127.0.0.1",
    port: str = "8090",
    verify_ssl_certs: str = "False",
) -> None:
    os.environ["GPT2GIGA_LOG_FILENAME"] = "/dev/null"
    os.environ["GPT2GIGA_MODE"] = "DEV"
    os.environ["GPT2GIGA_HOST"] = host
    os.environ["GPT2GIGA_PORT"] = port

    os.environ["GIGACHAT_BASE_URL"] = base_url
    os.environ["GIGACHAT_VERIFY_SSL_CERTS"] = verify_ssl_certs
    os.environ["GIGACHAT_MODEL"] = model

    os.environ["GPT2GIGA_MAX_REQUEST_BODY_BYTES"] = "104857600"
    api_server.run()
