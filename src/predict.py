"""
============================================================================
ARQUIVO: src/predict.py
PAPEL  : Função reutilizável de inferência (= "usar o modelo para prever").
         É chamada tanto pela linha de comando quanto pela API web.
============================================================================

O QUE É "INFERÊNCIA"?
---------------------
É o oposto de "treino". No treino, a rede ajusta os pesos. Na inferência,
os pesos já estão prontos — a gente só passa a entrada e lê a saída.

POR QUE ISOLAR ESSA LÓGICA EM UM ARQUIVO?
-----------------------------------------
- Evita duplicar código entre a linha de comando e a API.
- Centraliza o carregamento dos artefatos do modelo (assim a gente paga
  o custo de leitura UMA vez e reaproveita).
- Facilita testar: dá para escrever testes só sobre `predict_next` sem
  precisar subir servidor web.

CACHE DE CARREGAMENTO COM `lru_cache`
-------------------------------------
Carregar o modelo Keras do disco demora alguns segundos. Se a API
recarregasse o modelo a cada requisição, cada `/predict` levaria
eternidade. Por isso usamos `@lru_cache(maxsize=1)`: a primeira chamada
realmente lê o disco; as chamadas seguintes recebem o objeto já em
memória, instantaneamente. (LRU significa "Least Recently Used", o
algoritmo do cache. `maxsize=1` quer dizer "guardar só o último resultado".)
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model

# Pasta padrão onde o `train.py` deixou os artefatos.
MODELS_DIR = Path("models")
MODEL_PATH = MODELS_DIR / "lstm_stock.keras"
SCALER_PATH = MODELS_DIR / "scaler.pkl"
META_PATH = MODELS_DIR / "metadata.json"

# `Any` para o modelo: o Keras não expõe stubs de tipo úteis em `load_model`
# (e o módulo está marcado com `ignore_missing_imports` no mypy). Usar `Any`
# aqui é a saída pragmática para manter o resto do projeto em modo strict.
Artifacts = tuple[Any, MinMaxScaler, dict[str, Any]]


@lru_cache(maxsize=1)
def _load_artifacts() -> Artifacts:
    """
    Lê os três artefatos do disco e devolve em uma tupla. Cacheado.

    Por que retornar uma tupla em vez de globais? Para evitar variáveis
    globais mutáveis e para o `lru_cache` controlar a vida útil em um lugar só.

    Cuidado: o cache é POR PROCESSO. Se o servidor reiniciar, ele carrega
    de novo. Se trocar o modelo no disco em pleno ar, é preciso reiniciar
    para a API reler.
    """
    # Confere que tudo existe antes de tentar carregar. Mensagem clara
    # ajuda quem clonou o repositório e ainda não rodou o treino.
    if not MODEL_PATH.exists() or not SCALER_PATH.exists() or not META_PATH.exists():
        raise FileNotFoundError(
            "Artefatos do modelo ausentes. Rode `uv run python -m src.train` primeiro."
        )

    # Carrega cada um com a ferramenta adequada:
    #   - .keras → load_model do Keras
    #   - .pkl   → joblib.load (objetos do scikit-learn)
    #   - .json  → json.loads + leitura como texto
    model: Any = load_model(MODEL_PATH)
    scaler: MinMaxScaler = joblib.load(SCALER_PATH)
    meta: dict[str, Any] = json.loads(META_PATH.read_text())
    return model, scaler, meta


def get_metadata() -> dict[str, Any]:
    """Atalho para pegar só o dicionário de metadata, sem expor o resto."""
    return _load_artifacts()[2]


def predict_next(closes: list[float] | np.ndarray) -> float:
    """
    Recebe uma lista com os ÚLTIMOS preços reais (em reais, R$) e devolve
    a previsão do próximo fechamento, também em R$.

    Passo a passo:
        1. Carrega modelo + scaler + metadata (do cache, se já carregado).
        2. Confere se a lista tem pelo menos `window` valores (60 por padrão).
        3. Pega só os últimos `window`.
        4. Aplica a MESMA normalização que foi usada no treino.
        5. Coloca no formato 3D que a LSTM espera: (1 amostra, window, 1 feature).
        6. Pede ao modelo para prever.
        7. Desfaz a normalização para retornar em R$.

    Parâmetros:
        closes : lista (ou array numpy) de preços reais, do mais antigo
                 ao mais recente, do tipo float. Tamanho ≥ `window`.

    Retorna:
        Um único `float` com o preço previsto do próximo dia útil.

    Levanta:
        ValueError se a lista for menor que `window`.
        FileNotFoundError se os artefatos não estiverem em `models/`.
    """
    model, scaler, meta = _load_artifacts()
    window = int(meta["window"])

    # Garante array numpy 1D para evitar surpresas de formato.
    closes = np.asarray(closes, dtype=np.float32).ravel()

    if closes.size < window:
        raise ValueError(f"Forneça pelo menos {window} preços de fechamento.")

    # Fica só com os últimos `window` valores; o restante é histórico que
    # já passou da janela útil.
    last = closes[-window:].reshape(-1, 1)

    # Normaliza com o MESMO scaler do treino (importantíssimo).
    scaled = scaler.transform(last)

    # Formato exigido pela LSTM: (n_amostras, n_passos, n_features).
    # Aqui passamos UMA amostra de 60 passos com 1 valor cada.
    x = scaled.reshape(1, window, 1)

    # `verbose=0` para não imprimir barra de progresso a cada chamada.
    pred_scaled = model.predict(x, verbose=0)

    # Desnormaliza para R$. `[0, 0]` pega o único valor da matriz (1x1).
    return float(scaler.inverse_transform(pred_scaled)[0, 0])


# Atalho útil de depuração:
#   uv run python -m src.predict
# Roda uma previsão usando os últimos 60 preços que o treino salvou
# dentro do metadata.json. Não precisa de internet nem de input do usuário.
if __name__ == "__main__":
    meta = get_metadata()
    sample = meta.get("last_window_real")
    if sample is None:
        raise SystemExit("metadata.json sem last_window_real; rode `src.train` de novo.")
    print(f"symbol={meta['symbol']} window={meta['window']}")
    print(f"próximo close previsto: {predict_next(sample):.4f}")
