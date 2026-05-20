"""
============================================================================
ARQUIVO: api/main.py
PAPEL  : Define o serviço web que expõe o modelo treinado para o mundo.
         Usa FastAPI (framework) + Uvicorn (servidor) + Prometheus (métricas).
============================================================================

O QUE É UMA "API"?
------------------
Sigla de "Application Programming Interface", em português "Interface de
Programação de Aplicações". É um conjunto de endereços (URLs) por onde um
programa fala com outro. Aqui ela atende:

    GET  /health               → "você está viva?"
    GET  /metadata             → "me conte sobre o modelo treinado"
    POST /predict              → "aqui vai uma lista de 60 preços, me diga o próximo"
    GET  /predict/{symbol}     → "baixe sozinho do Yahoo e me diga o próximo"
    GET  /metrics              → métricas internas para Prometheus consumir
    GET  /docs                 → documentação interativa (Swagger UI) — vem grátis

POR QUE FASTAPI?
----------------
- Tipagem forte de Python + Pydantic = validação automática.
- Documentação Swagger gerada sozinha em /docs.
- Compatível com programação assíncrona (async/await), boa para alta carga.

POR QUE CARREGAR O MODELO NO "LIFESPAN"?
----------------------------------------
"Lifespan" é um gancho do FastAPI que roda UMA vez na subida e UMA vez na
descida do servidor. A gente aproveita a subida para já chamar
`_load_artifacts()` e deixar o modelo na memória. Assim, a primeira
requisição já é rápida (não tem o "ônibus parado", custo de carregar do
disco no meio do request).

POR QUE EXPOR /metrics?
-----------------------
Para integrar com Prometheus (sistema de monitoramento). Ele coleta
contadores (quantas requisições), histogramas (tempo de resposta), e
deixa tudo bonito no Grafana ou no painel do Coolify. Isso atende o
requisito de "monitoramento" da Fase 4.
"""

from __future__ import annotations

import logging
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import pandas as pd
import yfinance as yf
from fastapi import FastAPI, HTTPException
from pandas.tseries.offsets import BDay
from prometheus_fastapi_instrumentator import Instrumentator

from api.schemas import (
    HealthResponse,
    MetadataResponse,
    PredictRequest,
    PredictResponse,
    SymbolPredictResponse,
)
from src.predict import _load_artifacts, get_metadata, predict_next

# Configura logging com formato legível: data, nível, módulo e mensagem.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s | %(message)s",
)
log = logging.getLogger("api")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """
    Gerencia o ciclo de vida da aplicação.

    O que está ANTES do `yield` roda quando o servidor sobe.
    O que está DEPOIS do `yield` roda quando ele desce.

    Aqui só usamos o "antes": tentamos carregar os artefatos do modelo.
    Se eles não existirem, logamos um aviso e seguimos — a API sobe em
    modo "degradado". Isso permite testar o /health antes mesmo de
    treinar, o que é útil em pipeline de deploy.
    """
    try:
        _load_artifacts()  # carrega modelo + scaler + metadata na memória
        meta = get_metadata()
        log.info("Artefatos carregados | symbol=%s window=%s", meta["symbol"], meta["window"])
    except FileNotFoundError as e:
        log.warning("Subindo sem modelo: %s", e)
    yield
    # (nada para limpar na descida; o Python coletará o que precisar)


# Cria a aplicação FastAPI. Esses metadados aparecem no /docs.
app = FastAPI(
    title="Stock LSTM API — SUZB3",
    description="Prevê o próximo preço de fechamento via LSTM treinada na Suzano (SUZB3.SA).",
    version="1.0.0",
    lifespan=lifespan,
)

# Instala o instrumentador Prometheus e expõe a rota /metrics.
# A partir daí, qualquer request passa por contadores e histogramas
# que ficam acessíveis no formato que o Prometheus entende.
Instrumentator().instrument(app).expose(app, endpoint="/metrics")


def _meta_or_503() -> dict[str, Any]:
    """
    Atalho usado pelos endpoints que EXIGEM modelo carregado.

    Se o modelo está ausente (a API subiu em modo degradado), responde com
    status HTTP 503 (Service Unavailable = "serviço indisponível") em vez
    de estourar uma exceção genérica 500. O 503 deixa claro para quem
    consome a API que o problema é de configuração, não um bug.
    """
    try:
        return get_metadata()
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e)) from e


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """
    Endpoint de saúde. NÃO falha mesmo sem modelo, porque o Docker e o
    Coolify usam essa rota para decidir se o container está vivo. Se ela
    quebrasse na ausência do modelo, o orquestrador ficaria reiniciando
    o container para sempre.

    Retorna `status="ok"` se modelo carregado, `status="degraded"` se não.
    """
    try:
        meta = get_metadata()
        return HealthResponse(
            status="ok",
            symbol=meta["symbol"],
            window=int(meta["window"]),
            model_loaded=True,
        )
    except FileNotFoundError:
        return HealthResponse(status="degraded", symbol="-", window=0, model_loaded=False)


@app.get("/metadata", response_model=MetadataResponse)
def metadata() -> MetadataResponse:
    """
    Devolve a carteira de identidade do modelo treinado. Inclui métricas
    finais (MAE, RMSE, MAPE), hiperparâmetros e quando foi salvo.
    Bom para auditar versão em produção.
    """
    meta = _meta_or_503()
    return MetadataResponse(
        symbol=meta["symbol"],
        window=int(meta["window"]),
        features=meta.get("features", ["Close"]),
        train_start=meta["train_start"],
        train_end=meta["train_end"],
        metrics=meta["metrics"],
        baseline_naive=meta.get("baseline_naive"),
        hyperparameters=meta.get("hyperparameters"),
        saved_at=meta["saved_at"],
        framework=meta.get("framework", "keras"),
    )


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest) -> PredictResponse:
    """
    Endpoint principal: recebe uma lista de preços de fechamento reais
    (em reais, R$) e devolve a previsão do próximo dia.

    Códigos de erro:
        422 Unprocessable Entity → cliente mandou dado inválido (lista
                                   curta, número inválido etc.).
        500 Internal Server Error → erro inesperado no servidor. NÃO
                                    vazamos stack trace na resposta (boa
                                    prática de segurança).
        503 Service Unavailable  → modelo ainda não foi treinado/disponível.
    """
    meta = _meta_or_503()
    window = int(meta["window"])

    # Confere o tamanho da lista contra o `window` do modelo treinado.
    if len(req.closes) < window:
        raise HTTPException(
            status_code=422,
            detail=f"Forneça pelo menos {window} preços (recebido {len(req.closes)}).",
        )

    # Cronometra a inferência para reportar no campo `inference_ms`.
    t0 = time.perf_counter()
    try:
        y = predict_next(req.closes)
    except ValueError as e:
        # Erro de validação que veio de dentro de `predict_next`. Devolvemos
        # como 422 para ser consistente com o resto da API.
        raise HTTPException(status_code=422, detail=str(e)) from e
    except Exception as e:
        # Qualquer outro erro: log completo no servidor, mensagem genérica
        # para o cliente (não expor detalhes internos).
        log.exception("erro na inferência")
        raise HTTPException(status_code=500, detail="erro interno na inferência") from e

    dt_ms = (time.perf_counter() - t0) * 1000
    log.info("predict ok | n=%d | %.2fms", len(req.closes), dt_ms)

    return PredictResponse(
        symbol=meta["symbol"],
        predicted_close=round(y, 4),
        window_used=window,
        inference_ms=round(dt_ms, 2),
        n_closes_received=len(req.closes),
    )


@app.get("/predict/{symbol}", response_model=SymbolPredictResponse)
def predict_symbol(symbol: str, period: str = "6mo") -> SymbolPredictResponse:
    """
    Atalho de demonstração: a API mesma busca o histórico no yfinance.

    O cliente só precisa passar o símbolo (e opcionalmente um `period`,
    tipo "6mo" para 6 meses, "1y" para 1 ano). A API:
        1. Baixa o fechamento do período.
        2. Garante que veio pelo menos `window` pontos.
        3. Aplica o mesmo `predict_next`.
        4. Devolve previsão + último preço observado.

    AVISO: usa o scaler treinado em SUZB3. Se o símbolo tiver outra faixa
    de preços (ex.: BVMF3 a 4 reais vs. BVMF3 a 100 reais), o resultado
    é apenas demonstrativo. Para produção, treinar um modelo por papel.
    """
    meta = _meta_or_503()
    window = int(meta["window"])

    df = yf.download(symbol, period=period, progress=False, auto_adjust=False)
    if df.empty:
        raise HTTPException(status_code=404, detail=f"yfinance vazio para {symbol}")

    # Versões recentes do yfinance devolvem colunas em MultiIndex
    # (ex.: ("Close", "SUZB3.SA")). Achata para um nível só, igual ao
    # tratamento de `src/data.py`, para que `df["Close"]` volte como Series.
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # `dropna(subset=["Close"])` mantém o índice de datas alinhado com os
    # closes — se aplicássemos `.dropna()` direto na Series, perderíamos
    # a referência para qual data é o último ponto.
    df = df.dropna(subset=["Close"])
    closes = df["Close"].astype(float).tolist()
    if len(closes) < window:
        raise HTTPException(
            status_code=422,
            detail=f"yfinance retornou só {len(closes)} pontos (precisa {window}).",
        )

    last_dt = df.index[-1]
    # `BDay(1)` pula só fins de semana; feriados da B3 não são tratados
    # — documentado no schema para o consumidor saber.
    predicted_dt = last_dt + BDay(1)

    t0 = time.perf_counter()
    y = predict_next(closes)
    dt_ms = (time.perf_counter() - t0) * 1000

    return SymbolPredictResponse(
        symbol=symbol,
        period=period,
        predicted_close=round(y, 4),
        window_used=window,
        last_close=round(closes[-1], 4),
        last_close_date=last_dt.strftime("%Y-%m-%d"),
        predicted_for_date=predicted_dt.strftime("%Y-%m-%d"),
        inference_ms=round(dt_ms, 2),
    )
