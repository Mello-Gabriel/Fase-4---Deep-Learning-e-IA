"""
============================================================================
ARQUIVO: api/schemas.py
PAPEL  : Definir os "formatos" de entrada e saída da API web usando Pydantic.
============================================================================

O QUE É PYDANTIC E POR QUE USAMOS?
----------------------------------
Pydantic é uma biblioteca que valida dados a partir de classes. A gente
declara "este endpoint recebe um JSON com a chave `closes` que é uma lista
de números" e o Pydantic:

    - Recusa requisições no formato errado (com erro 422 automaticamente).
    - Converte tipos quando dá (string "12.5" vira float 12.5).
    - Gera a documentação interativa do FastAPI (no Swagger em /docs).

Em resumo: descreve UMA vez o formato, ganha validação + documentação
de graça.

CADA CLASSE AQUI = UM "CONTRATO" DA API
---------------------------------------
Para cada rota da API, há dois contratos possíveis:
    - "Request"  : o que a rota ESPERA receber (no corpo do POST).
    - "Response" : o que a rota PROMETE devolver (em JSON).

Quem chama a API só precisa olhar essas classes para saber o formato.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    """
    Corpo aceito pelo endpoint POST /predict.

    Exemplo de JSON válido:
        {"closes": [50.7, 49.5, 48.89, ...]}   # 60 ou mais números

    A regra "60 ou mais" é validada DEPOIS, dentro do handler, porque
    depende do `window` que o modelo treinado declarou no metadata.json.
    Aqui no Pydantic só exigimos pelo menos 1 elemento — proteção contra
    lista vazia.
    """

    closes: list[float] = Field(
        ...,  # ... = "obrigatório, sem valor padrão"
        min_length=1,
        description="Lista de preços de fechamento reais (R$). Tamanho ≥ window do modelo.",
    )


class PredictResponse(BaseModel):
    """
    Resposta do POST /predict.

    - symbol            : qual ação o modelo foi treinado para prever.
    - predicted_close   : o preço previsto para o próximo fechamento (R$).
    - window_used       : tamanho da janela usada (sempre o do treino).
    - inference_ms      : quanto tempo a previsão levou, em milissegundos.
                          Útil para monitorar latência.
    - n_closes_received : quantos preços vieram na requisição (para log).
    """

    symbol: str
    predicted_close: float
    window_used: int
    inference_ms: float
    n_closes_received: int


class HealthResponse(BaseModel):
    """
    Resposta do GET /health. Usada pelo Docker e por monitoramento externo
    para saber se a API está viva E com modelo carregado.

    - status        : "ok" se tudo certo; "degraded" se subiu sem modelo
                      (situação anormal, mas dá para diagnosticar).
    - symbol        : símbolo do modelo carregado (ou "-" se nenhum).
    - window        : janela do modelo (ou 0 se nenhum).
    - model_loaded  : booleano simples para checagens rápidas.
    """

    status: str
    symbol: str
    window: int
    model_loaded: bool


class SymbolPredictResponse(BaseModel):
    """
    Resposta do GET /predict/{symbol}, que é uma rota "quick-test":
    em vez de o usuário enviar 60 preços, ele só passa o símbolo e a API
    baixa o histórico recente do Yahoo Finance e prevê.

    Sobre os campos de data:
        - ``last_close_date``    : data do último fechamento usado (formato
          ``YYYY-MM-DD``). É o dia em que o preço ``last_close`` foi cotado.
        - ``predicted_for_date`` : dia que a previsão ``predicted_close``
          representa. É o próximo dia útil (segunda a sexta) depois de
          ``last_close_date``. Atenção: feriados da B3 não são considerados;
          se o próximo dia útil real for feriado, a previsão se refere ao
          pregão seguinte ao informado.

    ATENÇÃO conceitual: o modelo foi treinado SÓ na Suzano (SUZB3.SA).
    Usá-lo em outro símbolo é mais um exercício de demonstração — o
    scaler está calibrado para a faixa de preço da Suzano.
    """

    symbol: str
    period: str
    predicted_close: float
    window_used: int
    last_close: float
    last_close_date: str
    predicted_for_date: str
    inference_ms: float


class MetadataResponse(BaseModel):
    """
    Resposta do GET /metadata. Devolve a "carteira de identidade" do
    modelo treinado: data de treino, métricas, hiperparâmetros.

    Útil para auditar qual versão do modelo está em produção sem precisar
    descer no servidor.
    """

    symbol: str
    window: int
    features: list[str]
    train_start: str
    train_end: str
    metrics: dict[str, float]
    baseline_naive: dict[str, float] | None = None
    hyperparameters: dict[str, Any] | None = None
    saved_at: str
    framework: str
