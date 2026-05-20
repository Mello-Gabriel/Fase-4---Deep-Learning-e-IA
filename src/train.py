"""
============================================================================
ARQUIVO: src/train.py
PAPEL  : Programa principal de treino. Cola TUDO de ponta a ponta:
         baixa dados → monta a rede → treina → mede erro → salva os
         artefatos que a API vai usar depois.
============================================================================

COMO RODAR
----------
Pela linha de comando, dentro da pasta do projeto:

    uv run python -m src.train

Pode passar argumentos para mudar o padrão, por exemplo:

    uv run python -m src.train --symbol PETR4.SA --epochs 30 --no-cache

O QUE ESTE PROGRAMA PRODUZ
--------------------------
Ao final, três arquivos vão parar na pasta `models/`:

    models/lstm_stock.keras  → o modelo treinado (a rede com os pesos finais)
    models/best.keras        → o "melhor checkpoint" salvo durante o treino
    models/scaler.pkl        → o objeto que normaliza/desnormaliza preços
    models/metadata.json     → carteira de identidade do modelo

E mais dois gráficos PNG em `reports/`:

    reports/learning_curve.png → curva de aprendizado (loss por época)
    reports/pred_vs_real.png   → previsão vs. valor real no teste

POR QUE PRECISAMOS DO SCALER JUNTO DO MODELO?
---------------------------------------------
O modelo foi treinado vendo números entre 0 e 1, não em reais. Quando a
API recebe `closes` em reais, ela precisa aplicar a MESMA normalização
que foi usada no treino, senão o modelo não reconhece o input. Por isso
salvamos o scaler como `joblib` e a API o carrega no startup.

ROTEIRO DESTE ARQUIVO (5 ETAPAS)
--------------------------------
    [1/5] Baixa série e prepara janelas (chama `build_dataset` do data.py).
    [2/5] Monta a LSTM (chama `build_lstm` do model.py).
    [3/5] Treina com EarlyStopping (para automaticamente se parar de melhorar).
    [4/5] Avalia no conjunto de teste (desnormalizando para reais).
    [5/5] Salva modelo, scaler, metadata e gráficos.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path
from typing import Any

import joblib
import matplotlib

# IMPORTANTE: usamos backend "Agg" (sem janela gráfica). Isso é obrigatório
# em servidor ou container Docker, que não têm tela. Tem que ser feito ANTES
# de importar `matplotlib.pyplot`. Por isso o import abaixo vem fora de ordem
# (e silenciamos `E402` localmente).
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from numpy.typing import NDArray
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# Notas dos imports:
#   - `argparse` lê argumentos da linha de comando.
#   - `datetime` grava a data/hora de quando o modelo foi salvo.
#   - `joblib`   serializa objetos Python (vamos usar para o scaler).
#   - Callbacks Keras interceptam o treino entre épocas:
#       * EarlyStopping   : para quando a perda de validação não cai há N épocas.
#       * ModelCheckpoint : salva em disco o modelo da MELHOR época vista até agora.
from src.data import (
    DEFAULT_END,
    DEFAULT_SPLIT,
    DEFAULT_START,
    DEFAULT_SYMBOL,
    DEFAULT_WINDOW,
    build_dataset,
)
from src.evaluate import naive_baseline, regression_metrics
from src.model import build_lstm

# Pastas de saída. Criamos com `mkdir(parents=True, exist_ok=True)` mais embaixo
# para não dar erro caso já existam.
MODELS_DIR = Path("models")
REPORTS_DIR = Path("reports")


def parse_args() -> argparse.Namespace:
    """
    Declara e lê os argumentos aceitos pela linha de comando.

    Cada `add_argument` vira uma flag, com seu valor padrão herdado das
    constantes do `data.py`. Isso facilita rodar variações sem editar código.
    """
    parser = argparse.ArgumentParser(description="Treina LSTM para preço de fechamento.")
    parser.add_argument(
        "--symbol",
        default=DEFAULT_SYMBOL,
        help="Código da ação no Yahoo Finance (ex.: SUZB3.SA, PETR4.SA).",
    )
    parser.add_argument(
        "--start", default=DEFAULT_START, help="Data inicial da coleta (YYYY-MM-DD)."
    )
    parser.add_argument("--end", default=DEFAULT_END, help="Data final da coleta (YYYY-MM-DD).")
    parser.add_argument(
        "--window",
        type=int,
        default=DEFAULT_WINDOW,
        help="Tamanho da janela em dias (entrada da rede).",
    )
    parser.add_argument(
        "--split",
        type=float,
        default=DEFAULT_SPLIT,
        help="Fração do começo da série que vai para treino (0.8 = 80%%).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Número máximo de épocas. EarlyStopping pode parar antes.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Quantas amostras a rede vê por passo de ajuste de pesos.",
    )
    parser.add_argument("--units", type=int, default=50, help="Neurônios em cada camada LSTM.")
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.2,
        help="Fração de neurônios desligados aleatoriamente (regularização).",
    )
    parser.add_argument(
        "--lr", type=float, default=1e-3, help="Taxa de aprendizado do otimizador Adam."
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=10,
        help="Quantas épocas sem melhora antes de o EarlyStopping cortar.",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Não usar cache CSV de dados; sempre baixar do yfinance.",
    )
    return parser.parse_args()


def plot_history(history: Any, path: Path) -> None:
    """
    Gera a "curva de aprendizado": duas linhas mostrando como a perda
    diminuiu (ou não) ao longo das épocas, no treino e na validação.

    Como ler o gráfico:
        - Ambas as linhas caindo juntas → modelo aprendendo bem.
        - Linha de treino caindo, mas a de validação subindo → "decoreba"
          (overfitting); está aprendendo o treino, mas não generalizando.
        - As duas planas e altas → não está aprendendo (taxa de
          aprendizado errada, dados ruins, modelo simples demais etc.).
    """
    plt.figure(figsize=(10, 4))
    plt.plot(history.history["loss"], label="treino")
    plt.plot(history.history["val_loss"], label="validação")
    plt.title("Curva de aprendizado (loss)")
    plt.xlabel("Época")
    plt.ylabel("MSE (escala normalizada)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()


def plot_predictions(
    y_true: NDArray[Any],
    y_pred: NDArray[Any],
    symbol: str,
    mae: float,
    path: Path,
) -> None:
    """
    Gera o gráfico "Real vs. Previsto" para o conjunto de teste.

    Cada ponto no eixo X é um dia (do trecho de teste); o eixo Y é o preço
    em reais. Se as duas linhas andam coladas, o modelo está acertando.
    """
    plt.figure(figsize=(12, 5))
    plt.plot(y_true, label="Real")
    plt.plot(y_pred, label="Previsto")
    plt.title(f"{symbol} — Close real vs previsto | MAE={mae}")
    plt.xlabel("Dias (teste)")
    plt.ylabel("Preço (R$)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()


def main() -> None:
    """Função principal: orquestra as 5 etapas do treino."""
    args = parse_args()
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    # ----------------------------------------------------------------------
    # [1/5] Dados
    # ----------------------------------------------------------------------
    # Chama o pipeline definido em data.py. Devolve uma `Dataset` completa
    # com X_train, y_train, X_test, y_test, scaler e a série bruta.
    print(f"[1/5] Baixando e preparando dados de {args.symbol}…")
    dataset = build_dataset(
        symbol=args.symbol,
        start=args.start,
        end=args.end,
        window=args.window,
        split=args.split,
        cache_dir=None if args.no_cache else "data",
    )
    print(
        f"  série completa: {len(dataset.raw_close)} pontos | "
        f"X_train={dataset.X_train.shape} | X_test={dataset.X_test.shape}"
    )

    # ----------------------------------------------------------------------
    # [2/5] Modelo
    # ----------------------------------------------------------------------
    # Monta a arquitetura LSTM descrita em model.py.
    print("[2/5] Construindo modelo LSTM…")
    model = build_lstm(
        window=args.window,
        units=args.units,
        dropout=args.dropout,
        learning_rate=args.lr,
    )
    # `summary()` imprime uma tabela com cada camada e o número de pesos
    # ajustáveis. Bom para conferir se a forma de entrada está certa.
    model.summary()

    # ----------------------------------------------------------------------
    # [3/5] Treino
    # ----------------------------------------------------------------------
    # Onde o "melhor modelo até agora" será salvo durante o treino.
    best_path = MODELS_DIR / "best.keras"

    callbacks = [
        # EarlyStopping: para o treino quando a `val_loss` não melhora por
        # `patience` épocas seguidas, restaurando os pesos da MELHOR época.
        # Isso evita ficar treinando à toa e protege contra overfitting.
        EarlyStopping(monitor="val_loss", patience=args.patience, restore_best_weights=True),
        # ModelCheckpoint: a cada época, se a `val_loss` for a melhor já vista,
        # grava o modelo em disco. Garante que, se algo crashar, ainda temos
        # o melhor checkpoint salvo.
        ModelCheckpoint(str(best_path), monitor="val_loss", save_best_only=True),
    ]

    print("[3/5] Treinando…")
    history = model.fit(
        dataset.X_train,
        dataset.y_train,
        # `validation_split=0.1`: separa OS ÚLTIMOS 10% do `X_train` para
        # validar a cada época. Como estamos em série temporal e
        # `shuffle=False`, esses 10% são justamente os mais recentes do
        # treino — uma validação respeitosa com a ordem do tempo.
        validation_split=0.1,
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=callbacks,
        # `verbose=2`: imprime uma linha por época, sem barra de progresso.
        # Bom para log limpo, especialmente quando treinando em servidor.
        verbose=2,
        # `shuffle=False`: NÃO embaralhar! Mantém a ordem temporal
        # tanto dentro do treino quanto na separação da validação.
        shuffle=False,
    )

    # ----------------------------------------------------------------------
    # [4/5] Avaliação no teste (com desnormalização)
    # ----------------------------------------------------------------------
    print("[4/5] Avaliando no teste (escala real)…")

    # O modelo cospe números entre ~0 e ~1 (escala normalizada).
    pred_scaled = model.predict(dataset.X_test, verbose=0)

    # Desfaz a normalização para voltar para reais (R$).
    # `inverse_transform` espera matriz 2D, daí o `reshape(-1,1)`.
    # `ravel()` no fim achata para 1D para facilitar nas métricas.
    y_pred = dataset.scaler.inverse_transform(pred_scaled).ravel()
    y_true = dataset.scaler.inverse_transform(dataset.y_test.reshape(-1, 1)).ravel()

    # Métricas do modelo (MAE, RMSE, MAPE) e do baseline ingênuo.
    metrics = regression_metrics(y_true, y_pred)
    baseline = naive_baseline(y_true)
    print(f"  LSTM:     {metrics}")
    print(f"  Baseline: {baseline}")

    # Salva os dois gráficos PNG em reports/.
    plot_history(history, REPORTS_DIR / "learning_curve.png")
    plot_predictions(y_true, y_pred, args.symbol, metrics["MAE"], REPORTS_DIR / "pred_vs_real.png")

    # ----------------------------------------------------------------------
    # [5/5] Persistência dos artefatos
    # ----------------------------------------------------------------------
    print("[5/5] Salvando artefatos…")
    model_path = MODELS_DIR / "lstm_stock.keras"
    scaler_path = MODELS_DIR / "scaler.pkl"
    meta_path = MODELS_DIR / "metadata.json"

    # Modelo Keras nativo: extensão .keras (formato novo, recomendado).
    model.save(model_path)
    # Scaler: usamos joblib porque ele lida bem com objetos do scikit-learn.
    joblib.dump(dataset.scaler, scaler_path)

    # Guardamos no metadata uma "janela de exemplo" — os últimos 60 preços
    # reais usados, para que a gente consiga rodar a inferência de teste
    # rapidamente, sem precisar baixar dados de novo.
    last_window_real = dataset.raw_close.tail(args.window).astype(float).tolist()

    # `metadata.json` é uma espécie de "carteira de identidade" do modelo.
    # A API carrega ele no startup para responder ao endpoint `/metadata`,
    # validar o tamanho da janela em `/predict`, e contar a história do treino.
    meta: dict[str, Any] = {
        "symbol": args.symbol,
        "window": args.window,
        "features": ["Close"],
        "train_start": args.start,
        "train_end": args.end,
        "split": args.split,
        "hyperparameters": {
            "units": args.units,
            "dropout": args.dropout,
            "learning_rate": args.lr,
            "batch_size": args.batch_size,
            "epochs_max": args.epochs,
            "patience": args.patience,
        },
        "metrics": metrics,
        "baseline_naive": baseline,
        "framework": "keras",
        "n_train_samples": int(dataset.X_train.shape[0]),
        "n_test_samples": int(dataset.X_test.shape[0]),
        "last_window_real": last_window_real,
        # Data/hora em UTC (universal); fica óbvio quando foi treinado.
        "saved_at": dt.datetime.now(dt.UTC).isoformat(),
    }
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False))

    # Resumo final no log para conferência rápida.
    print(f"  modelo:   {model_path}")
    print(f"  scaler:   {scaler_path}")
    print(f"  metadata: {meta_path}")
    print("  plots:    reports/learning_curve.png, reports/pred_vs_real.png")
    print(f"Métricas finais: {metrics}")


# Bloco que executa só quando o arquivo é chamado direto pelo Python.
# Quando outro módulo IMPORTA este, `__name__` não é "__main__", então o
# `main()` não dispara — só as funções ficam disponíveis.
if __name__ == "__main__":
    main()
