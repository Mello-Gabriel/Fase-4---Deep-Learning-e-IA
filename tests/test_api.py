"""
============================================================================
ARQUIVO: tests/test_api.py
PAPEL  : "Smoke tests" da API — testes rápidos que verificam se as rotas
         principais respondem o esperado. Não substituem testes mais
         detalhados, mas pegam regressões grosseiras (rota quebrada,
         schema fora do contrato etc.).
============================================================================

O QUE É UM "SMOKE TEST"?
------------------------
Termo emprestado da engenharia: "ligue a máquina e veja se sai fumaça".
São testes superficiais que confirmam que NADA óbvio está pegando fogo.
Aqui: sobe a API, chama cada rota uma vez e checa o status e o formato.

POR QUE PRECISA DO MODELO TREINADO?
-----------------------------------
A API depende dos arquivos em `models/`. Sem eles, o `/health` vira
"degraded" e o `/predict` vira 503. O fixture `client` pula os testes
(via `pytest.skip`) quando o modelo não existe, em vez de quebrar — bom
para rodar em ambiente recém clonado, sem treinar antes.
"""

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
from fastapi.testclient import TestClient  # cliente "fake" para testar FastAPI sem rede.


@pytest.fixture(scope="module")
def client() -> Iterator[TestClient]:
    """
    "Fixture" do pytest: uma função que prepara o ambiente para os testes
    que pedirem `client` como parâmetro.

    `scope="module"` significa: criar o cliente UMA vez por arquivo de teste,
    reutilizando entre as funções. Bem mais rápido do que recriar a cada teste.

    O `with TestClient(app) as c` força o FastAPI a rodar o `lifespan`
    (que carrega o modelo) e a limpar tudo no final.
    """
    if not (Path("models") / "lstm_stock.keras").exists():
        pytest.skip("Modelo não treinado ainda; rode `uv run python -m src.train` antes.")
    from api.main import app

    with TestClient(app) as c:
        yield c


def _fake_yf_df(closes: list[float], multi_index: bool = False) -> pd.DataFrame:
    """Monta um DataFrame parecido com o que `yfinance.download` devolve.

    Args:
        closes: Valores a colocar na coluna ``Close``.
        multi_index: Se ``True``, devolve com colunas em ``MultiIndex``
            (formato das versões recentes do ``yfinance``).
    """
    idx = pd.date_range("2025-01-01", periods=len(closes), freq="B")
    if multi_index:
        cols = pd.MultiIndex.from_tuples([("Close", "SUZB3.SA")])
        return pd.DataFrame(closes, index=idx, columns=cols)
    return pd.DataFrame({"Close": closes}, index=idx)


def test_health_ok(client: TestClient) -> None:
    """
    Com modelo carregado, `/health` deve responder 200 com status="ok"
    e `model_loaded=True`. Esse é o "tudo certo".
    """
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True


def test_metadata_ok(client: TestClient) -> None:
    """
    `/metadata` deve devolver um corpo JSON com pelo menos `symbol`,
    `window` (≥ 1) e `metrics`. Não validamos o valor das métricas (varia
    por treino), só a presença das chaves.
    """
    r = client.get("/metadata")
    assert r.status_code == 200
    body = r.json()
    assert body["symbol"]
    assert body["window"] >= 1
    assert "metrics" in body


def test_predict_ok(client: TestClient) -> None:
    """
    Caminho feliz do `/predict`:
        - Pega a janela de exemplo (`last_window_real`) salva no metadata.
        - Manda como `closes`.
        - Espera 200 com a previsão como float, e `window_used` igual ao
          window do metadata.
        - Confere que `inference_ms >= 0` (timer estranho seria sinal ruim).
    """
    meta = json.loads(Path("models/metadata.json").read_text())
    closes = meta["last_window_real"]
    r = client.post("/predict", json={"closes": closes})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["window_used"] == meta["window"]
    assert isinstance(body["predicted_close"], float)
    assert body["inference_ms"] >= 0


def test_predict_short_returns_422(client: TestClient) -> None:
    """
    Caminho de erro do cliente: lista curta demais.
    Esperado: 422 (Unprocessable Entity) — NÃO 500. 422 indica claramente
    que o problema é "dado enviado errado", não bug no servidor.
    """
    r = client.post("/predict", json={"closes": [10.0, 11.0]})
    assert r.status_code == 422


def test_predict_missing_closes_returns_422(client: TestClient) -> None:
    """Pydantic deve rejeitar payload sem o campo obrigatório ``closes``."""
    r = client.post("/predict", json={})
    assert r.status_code == 422


def test_predict_empty_closes_returns_422(client: TestClient) -> None:
    """Lista vazia bate na restrição ``min_length=1`` do schema."""
    r = client.post("/predict", json={"closes": []})
    assert r.status_code == 422


def test_predict_non_numeric_returns_422(client: TestClient) -> None:
    """Tipos não conversíveis em float devem ser barrados pelo Pydantic."""
    r = client.post("/predict", json={"closes": ["a", "b", "c"]})
    assert r.status_code == 422


def test_metadata_full_contract(client: TestClient) -> None:
    """Garante que o ``/metadata`` devolve todos os campos do contrato."""
    r = client.get("/metadata")
    assert r.status_code == 200
    body = r.json()
    expected_keys = {
        "symbol",
        "window",
        "features",
        "train_start",
        "train_end",
        "metrics",
        "saved_at",
        "framework",
    }
    assert expected_keys.issubset(body.keys()), f"faltam campos: {expected_keys - body.keys()}"
    assert isinstance(body["features"], list) and body["features"]
    assert isinstance(body["metrics"], dict) and body["metrics"]


def test_metrics_endpoint_returns_prometheus(client: TestClient) -> None:
    """``/metrics`` deve responder 200 no formato texto do Prometheus."""
    r = client.get("/metrics")
    assert r.status_code == 200
    ctype = r.headers.get("content-type", "")
    assert "text/plain" in ctype
    body = r.text
    assert "# HELP" in body or "# TYPE" in body, "esperado cabeçalho do Prometheus"


def test_predict_symbol_ok(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    """Caminho feliz do ``GET /predict/{symbol}`` com ``yfinance`` mockado.

    Usa exatamente os 60 fechamentos salvos no ``metadata.json`` para que a
    previsão volte determinística e fora-de-rede.
    """
    meta = json.loads(Path("models/metadata.json").read_text())
    fake_df = _fake_yf_df(meta["last_window_real"])

    def fake_download(*_args: Any, **_kwargs: Any) -> pd.DataFrame:
        return fake_df

    from api import main as api_main

    monkeypatch.setattr(api_main.yf, "download", fake_download)

    r = client.get(f"/predict/{meta['symbol']}")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["symbol"] == meta["symbol"]
    assert body["window_used"] == meta["window"]
    assert isinstance(body["predicted_close"], float)
    assert body["last_close"] == round(float(meta["last_window_real"][-1]), 4)
    assert body["inference_ms"] >= 0

    # O fake_df começa em 2025-01-01 (uma quarta) com 60 dias úteis;
    # o último cai em 2025-03-25 (terça). Próximo dia útil: 2025-03-26.
    assert body["last_close_date"] == "2025-03-25"
    assert body["predicted_for_date"] == "2025-03-26"


def test_predict_symbol_handles_multiindex(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Versões recentes do ``yfinance`` devolvem colunas em ``MultiIndex``.

    O handler precisa achatar para não receber um DataFrame onde esperava
    uma Series; sem o achatamento, ``df["Close"].dropna().astype(float)``
    quebra com ``TypeError`` (chamada em DataFrame, não Series).
    """
    meta = json.loads(Path("models/metadata.json").read_text())
    fake_df = _fake_yf_df(meta["last_window_real"], multi_index=True)

    def fake_download(*_args: Any, **_kwargs: Any) -> pd.DataFrame:
        return fake_df

    from api import main as api_main

    monkeypatch.setattr(api_main.yf, "download", fake_download)

    r = client.get(f"/predict/{meta['symbol']}")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["symbol"] == meta["symbol"]
    assert isinstance(body["predicted_close"], float)


def test_predict_symbol_empty_returns_404(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """yfinance vazio (símbolo inexistente) deve virar 404 — não 500."""

    def fake_download(*_args: Any, **_kwargs: Any) -> pd.DataFrame:
        return pd.DataFrame()

    from api import main as api_main

    monkeypatch.setattr(api_main.yf, "download", fake_download)

    r = client.get("/predict/INVALIDO.SA")
    assert r.status_code == 404


def test_predict_symbol_few_points_returns_422(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Período curto demais (menos que ``window`` pontos) → 422."""
    fake_df = _fake_yf_df([10.0, 11.0, 12.0])

    def fake_download(*_args: Any, **_kwargs: Any) -> pd.DataFrame:
        return fake_df

    from api import main as api_main

    monkeypatch.setattr(api_main.yf, "download", fake_download)

    r = client.get("/predict/SUZB3.SA?period=5d")
    assert r.status_code == 422
