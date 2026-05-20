# Tech Challenge Fase 4 — LSTM para previsão do fechamento da Suzano (SUZB3.SA)

Pipeline completo end-to-end: coleta de dados → pré-processamento → treino de LSTM → avaliação → salvamento de artefatos → API REST (FastAPI) → Docker → deploy → monitoramento.

Ação alvo: **SUZB3.SA** (Suzano Papel e Celulose, B3).

---

## 1. Estrutura

```
.
├── src/
│   ├── data.py        # download yfinance, MinMaxScaler, janela deslizante (N=60)
│   ├── model.py       # arquitetura LSTM (2× LSTM(50) + Dropout + Dense)
│   ├── train.py       # CLI de treino → models/, reports/
│   ├── evaluate.py    # MAE / RMSE / MAPE + baseline naive
│   └── predict.py     # função reutilizável de inferência
├── api/
│   ├── main.py        # FastAPI app (lifespan, /health, /metadata, /predict, /metrics)
│   └── schemas.py     # Pydantic
├── models/            # lstm_stock.keras + scaler.pkl + metadata.json (geradas pelo train)
├── reports/           # learning_curve.png + pred_vs_real.png
├── tests/             # pytest (data, evaluate, api)
├── Dockerfile
├── docker-compose.yml
└── pyproject.toml
```

## 2. Stack

- **Python 3.12+**, dependências gerenciadas por **uv** (lockfile commitado).
- **TensorFlow / Keras 3** (`tensorflow>=2.21`) — LSTM.
- **scikit-learn** — `MinMaxScaler`.
- **yfinance** — coleta histórica gratuita do Yahoo Finance.
- **FastAPI + Uvicorn** — API.
- **prometheus-fastapi-instrumentator** — métricas `/metrics`.
- **pytest + httpx** — testes.

## 3. Como rodar

### 3.1 Instalar dependências

```bash
uv sync
```

### 3.2 Treinar o modelo (gera `models/` e `reports/`)

```bash
uv run python -m src.train --symbol SUZB3.SA --epochs 50
```

Hiperparâmetros expostos por CLI (`--window`, `--units`, `--dropout`, `--lr`, `--batch-size`, `--patience`, `--split`). Default reproduz a configuração base do desafio.

### 3.3 Subir a API local

```bash
uv run uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
# Swagger interativo: http://localhost:8000/docs
```

### 3.4 Testes

```bash
uv run pytest -q
```

### 3.5 Docker

```bash
docker compose up --build
curl http://localhost:8000/health
```

## 4. Pipeline (Requisitos 1–3 do PDF)

| Etapa | Arquivo | Decisões |
|-------|---------|----------|
| Coleta | `src/data.py::download_close` | `yf.download("SUZB3.SA", 2018‑01‑01 → 2025‑12‑31)`, só `Close`, `dropna`, cache em `data/`. |
| Normalização | `MinMaxScaler(0,1)` | `fit_transform` **só no treino**; `transform` no teste. Sem vazamento. |
| Split | `build_dataset` | **Temporal** (80/20), sem `shuffle`. Teste inclui os últimos 60 pontos do treino para alimentar a primeira janela. |
| Janela | `make_windows`, N=60 | Cada amostra = 60 dias passados → próximo close. Shape `(samples, 60, 1)`. |
| Modelo | `src/model.py::build_lstm` | `Sequential([LSTM(50, return_sequences=True), Dropout(0.2), LSTM(50), Dropout(0.2), Dense(25, relu), Dense(1)])` + Adam + MSE. |
| Treino | `src/train.py` | `validation_split=0.1` (últimos 10% do treino, sem shuffle), `EarlyStopping(patience=10, restore_best_weights=True)`, `ModelCheckpoint(save_best_only)`. |
| Avaliação | `src/evaluate.py` | MAE, RMSE, MAPE **na escala real** (inverte o scaler) + baseline ingênuo (`ŷ_t = y_{t-1}`). |
| Artefatos | `models/` | `lstm_stock.keras` + `scaler.pkl` + `metadata.json` (symbol, window, features, datas, métricas, hiperparâmetros, `last_window_real` p/ demo). |

### 4.1 Métricas obtidas no teste

> Resultado de uma execução de referência (treino com 50 epochs no histórico 2018‑01‑01 → 2025‑12‑31). Reproduza rodando `src.train` — números variam levemente por inicialização.

| Modelo | MAE (R$) | RMSE (R$) | MAPE (%) |
|--------|----------|-----------|----------|
| LSTM (2× 50 + Dropout) | 1.15 | 1.54 | 2.08 |
| Baseline ingênuo (amanhã = hoje) | 0.55 | 0.80 | 1.02 |

> **Observação honesta para a defesa**: em séries de preço diário, o baseline "amanhã ≈ hoje" é dificílimo de bater porque o melhor preditor de muito curto prazo é o próprio último valor (random walk). O LSTM ainda entrega previsão estável com MAPE ≈ 2%, mas para o vídeo vale citar essa limitação: melhorias plausíveis envolvem horizonte maior (5/10 dias à frente em vez de 1), features adicionais (volume, indicadores técnicos) e janela maior.

### 4.2 Plots (em `reports/`)

- `learning_curve.png` — loss de treino vs validação por época.
- `pred_vs_real.png` — preço real vs previsto no conjunto de teste.

## 5. API (Requisito 4)

Modelo + scaler + metadata são carregados **uma vez no startup** via `lifespan` do FastAPI. Cada request reusa o modelo carregado em memória.

| Endpoint | Método | Descrição |
|----------|--------|-----------|
| `/health` | GET | status + symbol + window + flag `model_loaded`. |
| `/metadata` | GET | symbol, window, datas de treino, métricas, hiperparâmetros. |
| `/predict` | POST | body: `{"closes": [floats com ≥60 valores em R$]}` → próximo close previsto. |
| `/predict/{symbol}` | GET | baixa histórico do símbolo via yfinance e roda predict (demo). |
| `/metrics` | GET | métricas Prometheus (latência, contadores, GC, processo). |
| `/docs` | GET | Swagger interativo (ótimo p/ vídeo). |

### Exemplo

```bash
curl -X POST http://localhost:8000/predict \
  -H 'Content-Type: application/json' \
  -d "$(jq -c '{closes: .last_window_real}' models/metadata.json)"
```

Resposta esperada:

```json
{
  "symbol": "SUZB3.SA",
  "predicted_close": 51.16,
  "window_used": 60,
  "inference_ms": 18.4,
  "n_closes_received": 60
}
```

Validação:
- `< window` valores → HTTP 422 (`Pydantic` + checagem extra contra `meta.window`).
- Erro interno na inferência → HTTP 500 sem vazar stack trace.

## 6. Docker + Deploy (Requisito 4 — deploy)

`Dockerfile` baseado em `python:3.12-slim`, instala via `uv sync --frozen --no-dev`, copia `src/`, `api/`, `models/`. Healthcheck nativo do Docker bate `/health`. Comando final: `uvicorn ... --workers 2`.

`docker-compose.yml` expõe `8000:8000`, `restart: unless-stopped`, healthcheck.

### Coolify (Hostinger) — deploy alvo

1. Push do repo para o GitHub.
2. No Coolify (Hostinger), criar nova *Application* tipo **Docker Compose**.
3. Apontar para o repo, branch `main`, arquivo `docker-compose.yml`.
4. Porta exposta: 8000. Domínio: configurar no painel do Hostinger.
5. Deploy → URL pública (gravar no README e mostrar no vídeo).

> **Importante p/ o vídeo**: o estudo (`07-DOCKER-E-DEPLOY-NUVEM.md`) cita Render/Fly só como exemplo genérico — a decisão deste projeto é **Coolify rodando na VPS Hostinger**, escolhida por dar controle total sobre o container e domínio próprio.

## 7. Monitoramento e escalabilidade (Requisito 5)

- **Tempo de resposta**: cada response do `/predict` traz `inference_ms` calculado no servidor. Histograma agregado em `/metrics` (`http_request_duration_seconds_*`).
- **Recursos**: `/metrics` expõe `process_cpu_seconds_total`, `process_resident_memory_bytes` — Coolify mostra CPU/RAM do container no painel.
- **Tráfego/erros**: contadores `http_requests_total{status=...}` por endpoint.
- **Logging**: `logging` estruturado, cada `/predict` emite `predict ok | n=60 | 18.4ms`.
- **Escalabilidade**: container stateless → escala horizontal; `uvicorn --workers 2` no Dockerfile; modelo carregado 1× por worker; healthcheck p/ orquestrador reiniciar.

Para dashboards completos, basta apontar um Prometheus para `/metrics` e plugar Grafana — config exemplo em `ESTUDO-TECH-CHALLENGE/08-MONITORAMENTO.md`.

## 8. Roteiro do vídeo (5–10 min)

1. **Abertura (30s)** — apresentar você + objetivo: "LSTM para prever o fechamento da Suzano (SUZB3) servida via API."
2. **Dados e janela (~1min)** — abrir `src/data.py`, mostrar yfinance, `MinMaxScaler` apenas no treino (citar vazamento), `make_windows(N=60)` → `(samples, 60, 1)`.
3. **Modelo (~1min)** — `src/model.py`: 2 camadas LSTM(50) + Dropout(0.2) + Dense(25) + Dense(1), Adam + MSE. Por que LSTM: gates resolvem vanishing gradient da RNN simples.
4. **Treino e métricas (~1min)** — rodar (ou mostrar log já capturado) `uv run python -m src.train --symbol SUZB3.SA`. Exibir `reports/learning_curve.png` e `reports/pred_vs_real.png`. Citar MAE ≈ R$1.15, MAPE ≈ 2% e comparar honestamente com o baseline ingênuo.
5. **API ao vivo (~2min)** — `uv run uvicorn api.main:app --reload`. Abrir `http://localhost:8000/docs`. Disparar `GET /health`, `GET /metadata` (mostra métricas treinadas), `POST /predict` colando os 60 últimos closes (ou o array de `last_window_real` do `metadata.json`). Mostrar `inference_ms` na resposta.
6. **Monitoramento (~30s)** — `GET /metrics`, apontar `http_request_duration_seconds`, `process_resident_memory_bytes`.
7. **Docker / deploy (~1min)** — abrir `Dockerfile` e `docker-compose.yml`, citar `--workers 2`, healthcheck. Mostrar URL pública do Coolify + curl no `/health` em produção.
8. **Encerramento (30s)** — limitações (1 dia à frente, univariado, baseline forte) e próximos passos (multivariado, horizonte maior, monitorar drift).

## 9. Mapa rápido para a banca

| Requisito do PDF | Onde está |
|------------------|-----------|
| 1. Coleta e pré-processamento | `src/data.py`, cache `data/` |
| 2. Modelo LSTM + tuning + métricas | `src/model.py`, `src/train.py`, `src/evaluate.py`, `reports/*.png` |
| 3. Salvar modelo | `models/lstm_stock.keras` + `models/scaler.pkl` + `models/metadata.json` |
| 4. API + Docker + deploy | `api/main.py`, `Dockerfile`, `docker-compose.yml`, Coolify (Hostinger) |
| 5. Monitoramento + escalabilidade | `/metrics` (Prometheus), `inference_ms`, `--workers 2`, healthcheck, app stateless |
