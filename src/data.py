"""
============================================================================
ARQUIVO: src/data.py
PAPEL  : Coletar o preço de fechamento da ação na Bolsa e transformar essa
         série de números em "exercícios prontos" que a rede neural consegue
         estudar.
============================================================================

POR QUE ESTE ARQUIVO EXISTE
---------------------------
A rede neural que vamos treinar (no arquivo `model.py`) não entende "ação",
"preço", nem "data". Ela só entende números organizados em formato de
tabela tridimensional. Este arquivo é o tradutor:

    Mundo real  →  números brutos  →  números normalizados  →  janelas
    (Bolsa)        (1 coluna)         (entre 0 e 1)             (cubo 3D)

Ao final, a função `build_dataset()` devolve um objeto com tudo que o
treino precisa: entradas de treino, saídas de treino, entradas de teste,
saídas de teste, e o "escalonador" que sabe desfazer a normalização para
voltarmos a ver o número em reais (R$).

VOCABULÁRIO TRADUZIDO (sem siglas soltas)
-----------------------------------------
- "Série temporal"  = uma lista de números na ordem do tempo (um por dia útil).
- "Fechamento"      = o último preço negociado naquele dia ("Close" em inglês).
- "Normalização"    = encolher todos os números para caberem entre 0 e 1.
                      A rede aprende melhor com números pequenos e parecidos.
- "Janela"          = um pedaço da série; aqui pegamos 60 dias consecutivos
                      como "pergunta" e o próximo dia como "resposta".
- "Treino x Teste"  = parte da série serve para a rede APRENDER; outra parte,
                      mais nova, serve para checar se ela aprendeu de verdade.
                      Em série temporal, o teste é SEMPRE o trecho mais novo
                      (futuro do treino) — embaralhar quebraria a ordem.
- "Vazamento de dados" = quando o teste contamina o treino. Aqui evitamos
                          ajustando o escalonador SÓ no treino.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler

# Notas dos imports acima:
#   - `dataclass`     : declara classe só de dados, sem escrever `__init__`.
#   - `numpy`         : aritmética rápida com vetores e matrizes.
#   - `pandas`        : tabelas com índice de data (planilha em Python).
#   - `yfinance`      : baixa cotações grátis do Yahoo Finance.
#   - `MinMaxScaler`  : encolhe números para [0, 1] e sabe desfazer a conta.


# ---------------------------------------------------------------------------
# Constantes padrão (valores usados quando ninguém passa nada diferente).
# Foram fixados no plano de estudos: Suzano, série de 2018 em diante,
# janela de 60 dias, 80% para treino.
# ---------------------------------------------------------------------------
DEFAULT_SYMBOL = "SUZB3.SA"  # Código da Suzano Papel e Celulose na B3.
DEFAULT_START = "2018-01-01"  # Data inicial do histórico que baixamos.
DEFAULT_END = "2025-12-31"  # Data final.
DEFAULT_WINDOW = 60  # Quantos dias passados a rede olha para chutar o próximo.
DEFAULT_SPLIT = 0.8  # 80% dos pontos vão para treinar, 20% para testar.


@dataclass
class Dataset:
    """
    Pacote de dados pronto para o treino.

    Pense nisso como uma "mochila" devolvida por `build_dataset()` que
    carrega tudo que o restante do programa vai precisar.

    Campos:
        symbol      : qual ação foi usada (ex.: "SUZB3.SA").
        window      : quantos dias formam cada janela de entrada.
        train_start : data inicial usada na coleta.
        train_end   : data final.
        X_train     : entradas de treino. Formato: (n_amostras, 60, 1).
                      É uma pilha de 60 dias x 1 coluna (só o fechamento).
        y_train     : respostas certas correspondentes. Formato: (n_amostras,).
                      Cada elemento é o preço do dia seguinte à janela.
        X_test      : entradas de teste (período mais novo, não visto no treino).
        y_test      : respostas certas do teste.
        scaler      : o objeto que sabe normalizar (e desnormalizar). A gente
                      salva ele junto do modelo para a API usar a MESMA escala.
        raw_close   : a série original em reais (R$), antes de qualquer ajuste.
        split_index : posição em que cortamos treino e teste.
    """

    symbol: str
    window: int
    train_start: str
    train_end: str
    X_train: np.ndarray
    y_train: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    scaler: MinMaxScaler
    raw_close: pd.Series
    split_index: int


def download_close(
    symbol: str = DEFAULT_SYMBOL,
    start: str = DEFAULT_START,
    end: str = DEFAULT_END,
    cache_dir: str | Path | None = "data",
) -> pd.Series:
    """
    Baixa o histórico do Yahoo Finance e devolve SÓ a coluna de fechamento.

    Por que só o fechamento?
        O plano é fazer um modelo "univariado", ou seja, que aprende a partir
        de UMA variável só. Isso simplifica bastante e já é suficiente para o
        objetivo do Tech Challenge.

    Por que existe um cache em arquivo?
        Baixar do Yahoo é lento e às vezes falha (rate limit, internet caindo,
        feriado etc). Guardamos o resultado em um CSV dentro de `data/` para
        que, na próxima execução, o programa apenas leia o arquivo.

    Parâmetros:
        symbol    : código da ação (ex.: "SUZB3.SA"). O ".SA" indica Bovespa.
        start/end : intervalo de datas no formato "YYYY-MM-DD".
        cache_dir : pasta onde guardar/buscar o CSV. Passe `None` para sempre
                    baixar de novo (útil em teste, quando você quer dados frescos).

    Retorna:
        Uma série do pandas (vetor com índice de datas) contendo só o
        preço de fechamento, sem dias faltantes.
    """
    cache_path: Path | None = None
    if cache_dir is not None:
        # Substituímos "." por "_" no nome do arquivo porque o ponto poderia
        # confundir com extensão. O resultado fica algo como
        # `data/SUZB3_SA_2018-01-01_2025-12-31.csv`.
        cache_path = Path(cache_dir) / f"{symbol.replace('.', '_')}_{start}_{end}.csv"
        if cache_path.exists():
            # Já temos o CSV — só lê e retorna, sem rede.
            df_cached = pd.read_csv(cache_path, index_col=0, parse_dates=True)
            series_cached: pd.Series = df_cached["Close"].dropna()
            return series_cached

    # Caso o cache não exista, baixa do Yahoo Finance.
    # `progress=False` evita a barrinha de progresso (poluindo o log do treino).
    # `auto_adjust=False` mantém o preço bruto, sem ajustes automáticos de
    # dividendos/desdobramentos — o plano de estudo decidiu por essa simplicidade.
    # O yfinance pode retornar `None` em falha extrema (ex.: ticker sem dados);
    # tratamos esse caso e o caso de DataFrame vazio juntos.
    raw = yf.download(symbol, start=start, end=end, progress=False, auto_adjust=False)
    if raw is None or raw.empty:
        raise RuntimeError(f"yfinance vazio para {symbol} ({start} → {end})")

    # Em algumas versões do yfinance a tabela vem com colunas em dois níveis
    # (tipo "Close > SUZB3.SA"). Esse `if` achata para um nível só.
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    # Pega só a coluna `Close`, joga fora as linhas vazias (feriados, suspensão
    # de pregão etc.) e dá um nome explícito.
    closes: pd.Series = raw["Close"].dropna()
    closes.name = "Close"

    # Salva o CSV para reutilizar na próxima rodada.
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        closes.to_frame().to_csv(cache_path)
    return closes


def make_windows(series: np.ndarray, window: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Transforma uma série em "perguntas e respostas".

    Idéia da janela deslizante (sliding window), passo a passo:

        Suponha série = [10, 11, 12, 13, 14, 15] e window = 3.

        Pergunta 1: [10, 11, 12]  → Resposta 1: 13
        Pergunta 2: [11, 12, 13]  → Resposta 2: 14
        Pergunta 3: [12, 13, 14]  → Resposta 3: 15

    A rede vai aprender exatamente esse padrão: dada uma sequência de
    `window` valores, qual é o próximo?

    Por que o resultado tem forma (n, window, 1)?
        A camada LSTM (a rede que vamos usar) exige três dimensões:
            - 1ª: quantos exemplos existem (n).
            - 2ª: quantos passos no tempo cada exemplo tem (window=60).
            - 3ª: quantas variáveis por passo (aqui só 1 = "Close").

    Parâmetros:
        series : array no formato (N, 1) com os preços já normalizados.
        window : tamanho do "olho para o passado" (60 no padrão).

    Retorna:
        X : array no formato (n_amostras, window, 1) — as perguntas.
        y : array no formato (n_amostras,)         — as respostas.
    """
    if len(series) <= window:
        # Precisamos de pelo menos `window + 1` pontos: window para a pergunta
        # e mais um para a resposta seguinte.
        raise ValueError(f"Série precisa de mais que {window} pontos, tem {len(series)}.")

    X, y = [], []
    for i in range(window, len(series)):
        # Pega os `window` valores anteriores ao índice atual como entrada,
        # e o valor do índice atual como rótulo (gabarito) a aprender.
        X.append(series[i - window : i, 0])  # fatia [i-60, i-59, ..., i-1]
        y.append(series[i, 0])  # valor exato no índice `i`

    # Converte as listas Python para arrays numpy 32-bit (float32 é o formato
    # padrão das redes neurais, mais rápido e menor que float64).
    X_arr = np.asarray(X, dtype=np.float32).reshape(-1, window, 1)
    y_arr = np.asarray(y, dtype=np.float32)
    return X_arr, y_arr


def build_dataset(
    symbol: str = DEFAULT_SYMBOL,
    start: str = DEFAULT_START,
    end: str = DEFAULT_END,
    window: int = DEFAULT_WINDOW,
    split: float = DEFAULT_SPLIT,
    cache_dir: str | Path | None = "data",
) -> Dataset:
    """
    Pipeline completo de dados, do download até as janelas prontas.

    Ordem dos passos (importante!):
        1. Baixa fechamento.
        2. Separa em treino (80%) e teste (20%) ANTES de normalizar.
        3. Ajusta o escalonador SÓ no treino (`fit_transform`).
        4. Aplica o mesmo escalonador no teste (`transform`, sem `fit`).
        5. Faz janelas no treino normalizado.
        6. Para fazer janelas no teste, "empresta" os últimos `window`
           pontos do treino, senão perderíamos os primeiros `window` exemplos
           do teste por falta de histórico.

    Por que tanto cuidado em 3 e 4?
        Se ajustássemos o `MinMaxScaler` em TODA a série, o modelo saberia
        antecipadamente o maior e o menor valor que apareceria no futuro.
        Isso é "vazamento de dados" (data leakage) — a métrica de teste sairia
        otimista de mentira. Fazer fit só no treino imita a vida real, onde
        o futuro é desconhecido.

    Parâmetros: ver `Dataset` e `download_close` acima.

    Retorna:
        Uma `Dataset` pronta para entregar ao treinamento.
    """
    # --- Passo 1: baixar série em escala original (reais) ---
    closes = download_close(symbol, start=start, end=end, cache_dir=cache_dir)

    # `values` vira matriz (N, 1): N linhas, 1 coluna (o fechamento).
    # Esse formato é o que o `MinMaxScaler` espera.
    values = np.asarray(closes, dtype=np.float32).reshape(-1, 1)

    # --- Passo 2: divisão temporal (NÃO embaralhar!) ---
    # Em séries temporais, embaralhar destrói a ordem e o modelo ficaria
    # "vendo o futuro" durante o treino. Sempre cortamos no índice.
    split_idx = int(len(values) * split)
    train_raw = values[:split_idx]  # parte antiga → treino
    test_raw = values[split_idx:]  # parte recente → teste

    # --- Passo 3 e 4: normalização sem vazamento ---
    scaler = MinMaxScaler(feature_range=(0, 1))
    train_scaled = scaler.fit_transform(train_raw)  # ajusta + aplica no treino
    test_scaled = scaler.transform(test_raw)  # SÓ aplica no teste

    # --- Passo 5: janelas do treino ---
    X_train, y_train = make_windows(train_scaled, window)

    # --- Passo 6: janelas do teste, com contexto emprestado do treino ---
    # Sem isso, o primeiro exemplo do teste exigiria 60 valores e nós
    # estaríamos jogando fora os primeiros 60 dias de teste (que NÃO têm
    # 60 dias de história dentro do próprio teste). Concatenamos treino e
    # teste e cortamos a partir de `split_idx - window` para resolver isso.
    full_scaled = np.concatenate([train_scaled, test_scaled])
    test_input = full_scaled[split_idx - window :]
    X_test, y_test = make_windows(test_input, window)

    return Dataset(
        symbol=symbol,
        window=window,
        train_start=start,
        train_end=end,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        scaler=scaler,
        raw_close=closes,
        split_index=split_idx,
    )


if __name__ == "__main__":
    dataset = build_dataset()
    print(dataset)
