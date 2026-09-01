from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path
from types import ModuleType

from packaging.requirements import Requirement


NODE_ROOT = Path(__file__).resolve().parents[1]


class FakeOpenAI:
    def __init__(self, *, base_url, api_key):
        self.base_url = base_url
        self.api_key = api_key


def _load_main(monkeypatch):
    openai = ModuleType("openai")
    openai.OpenAI = FakeOpenAI
    openai.RateLimitError = type("RateLimitError", (Exception,), {})
    openai.APIError = type("APIError", (Exception,), {})
    monkeypatch.setitem(sys.modules, "openai", openai)

    gpt2giga = ModuleType("gpt2giga")
    gpt2giga.__path__ = []
    api_server = ModuleType("gpt2giga.api_server")
    api_server.run = lambda: None
    monkeypatch.setitem(sys.modules, "gpt2giga", gpt2giga)
    monkeypatch.setitem(sys.modules, "gpt2giga.api_server", api_server)

    for name in tuple(sys.modules):
        if name == "src" or name.startswith("src."):
            monkeypatch.delitem(sys.modules, name)

    parsing = ModuleType("src.parsing")
    parsing.__path__ = []
    parser = ModuleType("src.parsing.parser")
    parser.parse_file = lambda *_args, **_kwargs: None
    config = ModuleType("src.config_gpt2giga")
    config.start_proxy = lambda **_kwargs: None
    report_browser = ModuleType("src.report_browser")
    report_browser.ReportBrowser = type("ReportBrowser", (), {})
    agents = ModuleType("src.agents")
    agents.__path__ = []
    extraction = ModuleType("src.agents.extraction_agent")
    extraction.extraction_agent = lambda **_kwargs: {}
    for name, module in {
        "src.parsing": parsing,
        "src.parsing.parser": parser,
        "src.config_gpt2giga": config,
        "src.report_browser": report_browser,
        "src.agents": agents,
        "src.agents.extraction_agent": extraction,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)
    return importlib.import_module("src.main")


def test_non_giga_client_uses_configured_gateway(monkeypatch):
    module = _load_main(monkeypatch)
    monkeypatch.setenv("AI_GATEWAY_URL", "http://configured-gateway/")
    monkeypatch.setenv("AI_GATEWAY_API_KEY", "test-key")

    client, process = module.configure_client("minimax-m2.5")

    assert client.base_url == "http://configured-gateway/api/v1"
    assert client.api_key == "test-key"
    assert process is None

    monkeypatch.setenv("AI_GATEWAY_URL", "http://configured-gateway/api/v1")
    client, _ = module.configure_client("minimax-m2.5")
    assert client.base_url == "http://configured-gateway/api/v1"


def test_json_test_data_and_extraction_model_are_honored(monkeypatch):
    module = _load_main(monkeypatch)
    frame = module._convert_test_data(
        '[{"session_id":"s1","dialogue":"[(1, \\"q\\", \\"a\\")]"}]'
    )
    captured = {}
    monkeypatch.setattr(module, "configure_client", lambda _model: (object(), None))

    def fake_extraction_agent(**kwargs):
        captured.update(kwargs)
        return {}

    monkeypatch.setattr(module, "extraction_agent", fake_extraction_agent)

    result = module.main(
        report_dict={},
        test_data=frame,
        current_card="готовая карточка",
        llm_selector="main-model",
        pr_extraction_model="extraction-model",
    )

    assert frame["session_id"].tolist() == ["s1"]
    assert captured["model"] == "extraction-model"
    assert result["all_results"]["bp_card"] == "готовая карточка"


def test_proxy_process_is_stopped_after_main(monkeypatch):
    module = _load_main(monkeypatch)

    class FakeProcess:
        def __init__(self):
            self.terminated = False
            self.join_timeout = None

        def terminate(self):
            self.terminated = True

        def join(self, timeout):
            self.join_timeout = timeout

    process = FakeProcess()
    monkeypatch.setattr(module, "configure_client", lambda _model: (object(), process))
    monkeypatch.setattr(module, "extraction_agent", lambda **_kwargs: {})

    module.main(report_dict={}, current_card="карточка")

    assert process.terminated is True
    assert process.join_timeout == 5


def test_descriptor_contains_runtime_sources_and_wiring_shapes():
    descriptor = json.loads((NODE_ROOT / "descriptor.json").read_text(encoding="utf-8"))
    sources = set(descriptor["script"]["runConfiguration"]["sourceFiles"])
    python_sources = {
        str(path.relative_to(NODE_ROOT))
        for path in (NODE_ROOT / "src").rglob("*.py")
    }

    assert python_sources <= sources
    assert "src/metadata/task_types.json" in sources
    assert all((NODE_ROOT / source).is_file() for source in sources)

    ports = {port["name"]: port for port in descriptor["ports"]}
    assert ports["test_data"]["type"] == "dataframe"
    assert ports["test_data"]["shape"] == "shape_dataframe"
    assert ports["extracted_fields"]["shape"] == "shape_model"


def test_requirements_are_individually_parseable():
    lines = (NODE_ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines()

    assert lines
    assert all(Requirement(line) for line in lines if line and not line.startswith("#"))
