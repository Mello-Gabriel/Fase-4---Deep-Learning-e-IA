# Tech Challenge Fase 4 — LSTM para previsão do fechamento da Suzano (SUZB3.SA)

Pipeline completo end-to-end: coleta de dados → pré-processamento → treino de LSTM → avaliação → salvamento de artefatos → API REST (FastAPI) → Docker → deploy → monitoramento.

Ação alvo: **SUZB3.SA** (Suzano Papel e Celulose, B3).

---

## 1. Estrutura

```
.
├── MLET4TC.ipynb      # notebook principal: dados, modelo, treino, avaliação, artefatos
├── api/
│   ├── main.py        # FastAPI app (lifespan, /health, /metadata, /predict, /metrics)
│   └── schemas.py     # Pydantic
├── src/
│   └── predict.py     # função reutilizável de inferência (usada pela API)
├── models/            # lstm_stock.keras + scaler.pkl + metadata.json (gerados pelo notebook)
├── reports/           # learning_curve.png + pred_vs_real.png (gerados pelo notebook)
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

### 3.2 Treinar o modelo (executar o notebook)

Abrir e executar todas as células do **`MLET4TC.ipynb`** — o notebook realiza o pipeline completo e salva os artefatos em `models/` e os gráficos em `reports/`.

```bash
uv run jupyter lab MLET4TC.ipynb
# ou via VS Code: abrir o arquivo e "Run All"
```

O notebook usa `EarlyStopping(patience=10)` com limite de 100 épocas; em CPU a execução converge em torno de 36 épocas (~1 min).

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

Todo o pipeline de ML está no `MLET4TC.ipynb`, organizado nas seções abaixo:

| Etapa | Seção do notebook | Decisões |
|-------|-------------------|----------|
| Coleta | §4 `download_close` | `yf.download("SUZB3.SA", 2018‑01‑01 → 2025‑12‑31)`, só `Close`, `dropna`, cache em `data/`. |
| Normalização | §6 `build_dataset` | `MinMaxScaler(0,1)` — `fit_transform` **só no treino**; `transform` no teste. Sem vazamento. |
| Split | §6 `build_dataset` | **Temporal** (80/20), sem `shuffle`. Teste herda os últimos 60 pontos do treino para alimentar a primeira janela. |
| Janela deslizante | §5 `make_windows`, N=60 | Cada amostra = 60 dias passados → próximo close. Shape `(samples, 60, 1)`. |
| Modelo | §7 `build_lstm` | `Sequential([LSTM(50, return_sequences=True), Dropout(0.2), LSTM(50), Dropout(0.2), Dense(25, relu), Dense(1)])` + Adam + MSE. |
| Treino | §9.1–9.2 | `validation_split=0.1`, `EarlyStopping(patience=10, restore_best_weights=True)`, `ModelCheckpoint(save_best_only)`. Max 100 épocas; early stop em ~36. |
| Avaliação | §8 + §9.3 | MAE, RMSE, MAPE **na escala real** (inverte o scaler) + baseline ingênuo (`ŷ_t = y_{t-1}`). Plots salvos em `reports/`. |
| Artefatos | §9.4 | `lstm_stock.keras` + `scaler.pkl` + `metadata.json` (symbol, window, features, datas, métricas, hiperparâmetros, `last_window_real` p/ demo). |

### 4.1 Métricas obtidas no teste

> Resultado da execução de referência do `MLET4TC.ipynb` (histórico 2018‑01‑01 → 2025‑12‑31, EarlyStopping em época 26). Números variam levemente por inicialização aleatória.

| Modelo | MAE (R$) | RMSE (R$) | MAPE (%) |
|--------|----------|-----------|----------|
| LSTM (2× 50 + Dropout) | 1.1595 | 1.5877 | 2.0998 |
| Baseline ingênuo (amanhã = hoje) | 0.55 | 0.80 | 1.02 |



### 4.2 Plots (em `reports/`)

- `learning_curve.png` — loss de treino vs validação por época.
- `pred_vs_real.png` — preço real vs previsto no conjunto de teste.

## 5. API (Requisito 4)

Modelo + scaler + metadata são carregados **uma vez no startup** via `lifespan` do FastAPI. Cada request reusa o modelo carregado em memória. A inferência é implementada em `src/predict.py`, reutilizada pela API.

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


## 7. Monitoramento e escalabilidade (Requisito 5)

- **Tempo de resposta**: cada response do `/predict` traz `inference_ms` calculado no servidor. Histograma agregado em `/metrics` (`http_request_duration_seconds_*`).
- **Recursos**: `/metrics` expõe `process_cpu_seconds_total`, `process_resident_memory_bytes` — Coolify mostra CPU/RAM do container no painel.
- **Tráfego/erros**: contadores `http_requests_total{status=...}` por endpoint.
- **Logging**: `logging` estruturado, cada `/predict` emite `predict ok | n=60 | 18.4ms`.
- **Escalabilidade**: container stateless → escala horizontal; `uvicorn --workers 2` no Dockerfile; modelo carregado 1× por worker; healthcheck p/ orquestrador reiniciar.

Para dashboards completos, basta apontar um Prometheus para `/metrics` e plugar Grafana.
