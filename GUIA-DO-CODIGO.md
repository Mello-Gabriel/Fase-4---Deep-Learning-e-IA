# Guia do Código — Tech Challenge Fase 4

> Documento didático para entender, do zero, **o que cada arquivo faz, por que ele faz, e como tudo se conecta**.
> Foi escrito para ser lido na ordem, sem pressa. Cada sigla é explicada na primeira aparição. Se você ficar perdido, volte aqui.

---

## 0. O que estamos construindo, em uma frase

Um robô que olha os **últimos 60 dias** de preço de fechamento de uma ação na Bolsa e tenta chutar **qual vai ser o preço do próximo dia**. Esse robô é uma **rede neural** chamada **LSTM** (vamos definir já já), e ele fica disponível por uma **API web** (uma URL) que qualquer um pode chamar.

Tudo isso roda dentro de um **container Docker** para ficar fácil de mover entre máquinas e subir na nuvem (Coolify rodando em servidor Hostinger).

---

## 1. Vocabulário — leia isso antes de qualquer código

Você disse que não gosta de siglas. Aqui vai a tradução de TODAS as que aparecem no projeto:

| Termo / Sigla | Significado completo | Tradução amigável |
|---|---|---|
| **LSTM** | Long Short-Term Memory | "Memória de Curto e Longo Prazo". Um tipo de rede neural especialista em sequências (texto, áudio, série de preço). |
| **Rede neural** | — | Uma cadeia de operações matemáticas com "botões" ajustáveis. Aprende ajustando os botões para errar menos. |
| **API** | Application Programming Interface | "Interface de Programação de Aplicações". Uma URL pela qual outro programa fala com o nosso. |
| **HTTP 200 / 422 / 500 / 503** | — | Códigos de resposta da web: 200 = "deu certo", 422 = "dado enviado errado", 500 = "bug interno", 503 = "serviço fora do ar". |
| **Pydantic** | — | Biblioteca que valida formatos JSON automaticamente. |
| **FastAPI** | — | Framework Python para criar APIs web. |
| **Uvicorn** | — | O servidor que roda a FastAPI (o "Apache" deste mundinho). |
| **MAE** | Mean Absolute Error | Erro Absoluto Médio. Quanto você erra, em média, em valor absoluto. Lê na unidade do dado (R$). |
| **RMSE** | Root Mean Squared Error | Raiz do Erro Quadrático Médio. Igual ao MAE, mas castiga erros grandes mais. |
| **MAPE** | Mean Absolute Percentage Error | Erro Percentual Absoluto Médio. Mesma idéia, em porcentagem. |
| **MinMaxScaler** | — | Ferramenta que encolhe números para o intervalo entre 0 e 1, e sabe desfazer a conta. |
| **Sliding window** | — | Janela deslizante. Forma de quebrar a série temporal em "perguntas de 60 dias → resposta do dia 61". |
| **Overfitting** | — | "Decoreba". A rede vai bem no treino mas mal em dados novos. |
| **Dropout** | — | Técnica que desliga aleatoriamente alguns neurônios durante o treino, evitando "decoreba". |
| **EarlyStopping** | — | "Parar cedo". Para o treino quando a rede parou de melhorar, evitando gastar tempo à toa. |
| **Checkpoint** | — | "Foto" do modelo no melhor ponto do treino, salva em disco. |
| **Inferência** | — | Usar o modelo treinado para prever (oposto de "treinar"). |
| **Endpoint** | — | Uma rota específica da API (ex.: `/health`, `/predict`). |
| **Swagger / OpenAPI** | — | Página interativa em `/docs` que documenta a API sozinha. |
| **Prometheus** | — | Sistema de monitoramento que coleta métricas de serviços. |
| **Grafana** | — | Painel visual que desenha gráficos a partir do Prometheus. |
| **Docker** | — | Tecnologia que empacota o programa + dependências em uma "caixinha" (container) que roda igual em qualquer lugar. |
| **Coolify** | — | Plataforma que gerencia deploys de Docker numa máquina sua (no nosso caso, na Hostinger). |
| **yfinance** | Yahoo Finance | Biblioteca Python que baixa cotações grátis do Yahoo. |
| **B3** | — | A Bolsa de Valores do Brasil. Ações brasileiras terminam em `.SA` no Yahoo (ex.: `SUZB3.SA`). |
| **Keras / TensorFlow** | — | O conjunto de bibliotecas que usamos para montar e treinar a rede neural. Keras é a "casca amigável", TensorFlow é o motor. |
| **uv** | — | Gerenciador de dependências Python rápido. Substitui o `pip`. |

---

## 2. Mapa da pasta — o que mora onde

```
Fase 4 - Deep Learning e IA/
├── src/                          ← código de Machine Learning (treino + inferência)
│   ├── data.py                   coleta + pré-processamento + janelas
│   ├── model.py                  arquitetura da rede LSTM
│   ├── train.py                  programa principal de treino
│   ├── evaluate.py               métricas (MAE, RMSE, MAPE) + baseline
│   └── predict.py                função reutilizável para prever
├── api/                          ← serviço web
│   ├── main.py                   define os endpoints HTTP
│   └── schemas.py                define os formatos JSON esperados
├── tests/                        ← testes automatizados
│   ├── test_data.py              testa o pré-processamento
│   ├── test_evaluate.py          testa as métricas
│   └── test_api.py               testa as rotas da API
├── models/                       ← artefatos GERADOS pelo treino
│   ├── lstm_stock.keras          o modelo final treinado
│   ├── best.keras                melhor checkpoint visto no treino
│   ├── scaler.pkl                o normalizador (vai junto do modelo!)
│   └── metadata.json             "carteira de identidade" do modelo
├── data/                         ← cache local dos CSV baixados do Yahoo
├── reports/                      ← gráficos PNG gerados no treino
│   ├── learning_curve.png        loss por época
│   └── pred_vs_real.png          previsão vs. valor real no teste
├── Dockerfile                    como construir o container Docker
├── docker-compose.yml            como subir o container localmente
├── pyproject.toml                lista de dependências (Python)
├── uv.lock                       trava de versões (NÃO edite)
├── CLAUDE.md                     instruções para o assistente IA
├── README.md                     resumo público do projeto
├── GUIA-DO-CODIGO.md             este arquivo
└── ESTUDO-TECH-CHALLENGE/        material de estudo (NÃO faz parte do código)
```

---

## 3. Fluxo geral — o filme inteiro em uma página

```
                  ┌────────────────────────────────────────────┐
                  │                MUNDO REAL                  │
                  │  Bolsa B3 publica o preço diário da Suzano │
                  └────────────────────────────────────────────┘
                                       │
                              yfinance baixa
                                       ▼
                  ┌────────────────────────────────────────────┐
                  │              src/data.py                   │
                  │  • baixa "Close" (fechamento)              │
                  │  • divide 80% treino / 20% teste           │
                  │  • normaliza (0 a 1) só com treino         │
                  │  • cria janelas de 60 dias → próximo dia   │
                  └────────────────────────────────────────────┘
                                       │
                          devolve um objeto "Dataset"
                                       ▼
                  ┌────────────────────────────────────────────┐
                  │              src/model.py                  │
                  │  • LSTM(50) + Dropout 20%                  │
                  │  • LSTM(50) + Dropout 20%                  │
                  │  • Dense 25 (ReLU) + Dense 1               │
                  └────────────────────────────────────────────┘
                                       │
                              rede neural pronta
                                       ▼
                  ┌────────────────────────────────────────────┐
                  │              src/train.py                  │
                  │  • treina com EarlyStopping + Checkpoint   │
                  │  • avalia no teste (desnormalizado em R$)  │
                  │  • compara com baseline "amanhã=hoje"      │
                  │  • salva 3 artefatos em models/            │
                  │  • gera 2 gráficos em reports/             │
                  └────────────────────────────────────────────┘
                                       │
                  arquivos: lstm_stock.keras, scaler.pkl, metadata.json
                                       ▼
                  ┌────────────────────────────────────────────┐
                  │              api/main.py                   │
                  │  Sobe e carrega os 3 artefatos UMA vez.    │
                  │  Atende:                                   │
                  │    GET  /health                            │
                  │    GET  /metadata                          │
                  │    POST /predict                           │
                  │    GET  /predict/{symbol}                  │
                  │    GET  /metrics  (Prometheus)             │
                  │    GET  /docs     (Swagger)                │
                  └────────────────────────────────────────────┘
                                       │
                            internet → seu navegador
                                       ▼
                  ┌────────────────────────────────────────────┐
                  │   Container Docker rodando em Coolify     │
                  │              (Hostinger)                   │
                  └────────────────────────────────────────────┘
```

---

## 4. Passo a passo de cada arquivo

### 4.1 `src/data.py` — o tradutor entre mundo real e rede neural

**O problema:** a rede neural não entende "preço da Suzano em 15/03/2024". Ela só aceita números bonitinhos entre 0 e 1, organizados em formato 3D.

**O que o arquivo faz, na ordem em que faz:**

1. **`download_close(...)`** — chama o `yfinance` para baixar o histórico da Bolsa. Guarda em CSV dentro da pasta `data/` para não baixar de novo na próxima execução.
2. **Divide treino e teste** — 80% dos dados mais antigos viram treino, 20% mais novos viram teste. **Não embaralhamos!** Em série temporal embaralhar destruiria a ordem do tempo (a rede ficaria "vendo o futuro").
3. **Normaliza com `MinMaxScaler`** — encolhe todos os preços para o intervalo [0, 1]. Faz `fit_transform` **só no treino** e `transform` no teste. Se ajustasse no conjunto inteiro, o modelo conheceria de antemão o preço máximo e mínimo do "futuro" (= vazamento de dados, métrica de teste sairia otimista falsamente).
4. **`make_windows(...)`** — fatiamento em janelas de 60 dias com a resposta sendo o dia 61. É a "lição de casa" da rede: dadas 60 perguntas, qual é a resposta?

**O que devolve:** uma `Dataset` (uma classe simples que segura `X_train`, `y_train`, `X_test`, `y_test`, `scaler`, etc.).

> 💡 **Por que guardamos o `scaler`?** Porque o modelo aprendeu a falar a língua "0 a 1". Quando a API receber preços em reais no futuro, precisamos aplicar a MESMA normalização. Sem o scaler, o modelo recebe o input em escala errada e cospe lixo.

---

### 4.2 `src/model.py` — a arquitetura da rede

Aqui a gente DESENHA a rede, sem treinar ainda. Pense numa planta de casa: definimos as paredes, mas a casa ainda está vazia.

```
Entrada (60 dias × 1 valor)
    ↓
LSTM(50, devolve a sequência inteira)  ← aprende padrões curtos
    ↓
Dropout 20%                            ← desliga 20% dos sinais (anti-decoreba)
    ↓
LSTM(50, devolve só o último estado)   ← aprende padrões mais largos
    ↓
Dropout 20%
    ↓
Dense(25, ReLU)                        ← "misturador"
    ↓
Dense(1, sem ativação)                 ← cospe um número (preço normalizado)
```

**Por que duas LSTMs?** A primeira lida com padrões de curto prazo (ruído do dia a dia). A segunda recebe esse resumo e procura padrões mais longos (tendências de semanas). Empilhar costuma ajudar nesse tipo de problema.

**Por que `Dense(1)` no final sem ativação?** Estamos prevendo um número real (preço). Qualquer ativação ali (como ReLU ou sigmoid) limitaria o intervalo de saída e atrapalharia.

**`compile`** diz à rede COMO treinar: usar o otimizador Adam (algoritmo que ajusta os pesos), minimizar o erro quadrático médio (loss), e acompanhar o MAE como métrica de leitura humana.

---

### 4.3 `src/train.py` — o programa que cola tudo

É o "maestro". Quando você roda `uv run python -m src.train`, este arquivo executa cinco passos:

| Passo | O que acontece | Quem faz o trabalho |
|---|---|---|
| **1/5** | Baixa série e cria janelas | `src/data.py` |
| **2/5** | Monta a rede LSTM | `src/model.py` |
| **3/5** | Treina (com EarlyStopping + Checkpoint) | Keras |
| **4/5** | Avalia em R$ (desnormalizando) + compara com baseline | `src/evaluate.py` |
| **5/5** | Salva modelo + scaler + metadata + 2 gráficos | (disco) |

**`EarlyStopping(patience=10)`** observa a `val_loss` (erro no pedacinho de validação) a cada época. Se ela não cair por **10 épocas seguidas**, a rede pára e RESTAURA os melhores pesos vistos. Garante o melhor modelo sem desperdiçar tempo.

**`ModelCheckpoint`** salva em disco o melhor checkpoint visto. Redundância barata: se algo crashar no meio, o modelo bom está salvo.

**`validation_split=0.1`** com `shuffle=False` separa os 10% MAIS RECENTES do treino para validar. Isso respeita a ordem do tempo (validação = "futuro próximo" do treino).

**Ao terminar**, gera:
- `models/lstm_stock.keras` — o modelo treinado.
- `models/best.keras` — o melhor checkpoint.
- `models/scaler.pkl` — o normalizador (joblib).
- `models/metadata.json` — carteira de identidade (símbolo, datas, métricas, hiperparâmetros, janela de exemplo).
- `reports/learning_curve.png` — gráfico do erro por época.
- `reports/pred_vs_real.png` — real vs. previsto no teste.

---

### 4.4 `src/evaluate.py` — as três métricas + baseline

Treinar é fácil. Saber se ficou BOM é o difícil. Para isso, três medidores:

- **MAE (Erro Absoluto Médio)** — média de `|previsto − real|`. Lê em R$. "Em média erro R$ 1,15."
- **RMSE (Raiz do Erro Quadrático Médio)** — castiga erros grandes mais. Sempre ≥ MAE. Diferença grande entre os dois = previsões com variância alta.
- **MAPE (Erro Percentual)** — `|erro| / real`, em porcentagem. "Em média erro 2% do preço."

E um **baseline ingênuo**: prever que "amanhã = hoje". Esse é o piso. Se a LSTM ficar PIOR que o palpite preguiçoso, ela não aprendeu nada útil. Em ações que oscilam pouco no dia, esse baseline é forte e nem sempre é fácil bater — e isso está documentado.

---

### 4.5 `src/predict.py` — a função reutilizável de inferência

Tanto a linha de comando quanto a API web querem usar o modelo treinado. Em vez de duplicar código, isolamos a lógica aqui.

**`_load_artifacts()`** carrega modelo + scaler + metadata UMA vez por processo (cacheado com `lru_cache`, sigla para Least Recently Used). Sem cache, cada `/predict` levaria segundos só lendo o disco.

**`predict_next(closes)`** é o coração:
1. Recebe os preços em R$.
2. Pega os últimos 60.
3. Aplica o MESMO scaler.
4. Coloca no formato 3D que a LSTM espera: `(1, 60, 1)`.
5. Pede previsão.
6. Desfaz a normalização para devolver em R$.

---

### 4.6 `api/schemas.py` — os contratos JSON

Cada classe Pydantic descreve o formato de UMA mensagem da API:

- **`PredictRequest`** — o que o cliente manda no `POST /predict`. Só uma chave: `closes`, uma lista de números (≥ 1 elemento).
- **`PredictResponse`** — o que a API devolve: símbolo, preço previsto, janela usada, tempo de inferência, quantos preços vieram.
- **`HealthResponse`** — formato do `GET /health`.
- **`MetadataResponse`** — formato do `GET /metadata`.
- **`SymbolPredictResponse`** — formato do `GET /predict/{symbol}`.

> 💡 **De graça com Pydantic**: validação automática (recusa request com formato errado e devolve HTTP 422), conversão de tipo (string `"12.5"` vira `float`), e documentação Swagger gerada sozinha.

---

### 4.7 `api/main.py` — o serviço web

Define os endpoints. Pontos a entender:

**O `lifespan` carrega o modelo na SUBIDA do servidor**, não a cada request. Isso é o que faz a API ser rápida — a primeira requisição já chega com o modelo na memória.

**`/health`** NUNCA falha. Mesmo sem modelo carregado, devolve 200 com `status="degraded"`. Por quê? Porque o Docker e o Coolify usam essa rota para decidir se o container está vivo; se quebrasse na ausência do modelo, eles ficariam reiniciando para sempre.

**`/predict`** é o caminho principal:
- Valida o tamanho da lista (≥ window).
- Cronometra para reportar `inference_ms`.
- Captura `ValueError` (vira HTTP 422) e qualquer outro erro (vira HTTP 500 com mensagem genérica — **não vazamos stack trace** por segurança).

**`/predict/{symbol}`** é um atalho de demonstração: você passa só o ticker e a API mesma baixa do Yahoo. Útil para testar no navegador. Cuidado conceitual: o scaler foi treinado na faixa de preço da Suzano; usar em outra ação dá um número, mas é só demonstrativo.

**`/metrics`** é instalado pelo `prometheus-fastapi-instrumentator`. Coleta contagem de requisições e histogramas de latência para o Prometheus ler depois.

---

### 4.8 `tests/` — a rede de segurança

Os testes são pequenos programas que chamam o código real com entradas conhecidas e checam se a saída bate. Se você muda algo e quebra, eles avisam.

Para rodar:

```bash
uv run pytest                              # todos
uv run pytest tests/test_data.py           # só um arquivo
uv run pytest tests/test_api.py::test_predict_ok   # um teste específico
```

Os testes da API pulam-se sozinhos (`pytest.skip`) se o modelo ainda não tiver sido treinado. Isso evita falsa falha em ambiente recém-clonado.

---

### 4.9 `Dockerfile` e `docker-compose.yml` — empacotamento

**`Dockerfile`** é a receita de bolo do container:
1. Parte de uma imagem Python 3.12 enxuta.
2. Instala `uv`.
3. Copia `pyproject.toml` + `uv.lock` e roda `uv sync --frozen --no-dev` (instala dependências exatamente nas versões travadas).
4. Copia `src/`, `api/`, `models/`.
5. Expõe a porta 8000.
6. Define um `HEALTHCHECK` apontando para `/health` — assim o Docker sabe se o container está saudável.
7. Sobe `uvicorn` com `--workers 2` (dois processos servindo em paralelo).

**`docker-compose.yml`** é o atalho local: `docker compose up --build` constrói e sobe a API mapeando `localhost:8000`.

---

## 5. Como rodar tudo localmente — passo a passo

```bash
# 1. Instalar dependências
uv sync

# 2. Treinar o modelo (gera arquivos em models/ e reports/)
uv run python -m src.train

# 3. (opcional) Rodar uma previsão por linha de comando, sem subir API
uv run python -m src.predict

# 4. Subir a API local
uv run uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
# → abra http://localhost:8000/docs no navegador para o Swagger interativo

# 5. Rodar todos os testes
uv run pytest

# 6. Subir tudo via Docker (opcional)
docker compose up --build
```

**Exemplo de chamada manual** (após subir a API):

```bash
# /health
curl http://localhost:8000/health

# /predict — usando a janela de exemplo salva no metadata
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d "$(python -c 'import json; d=json.load(open("models/metadata.json")); print(json.dumps({"closes": d["last_window_real"]}))')"
```

---

## 6. Como o treino "aprende" — explicação para destravar a intuição

Imagine que a rede tem milhares de "botões" (pesos). No começo, eles estão em valores aleatórios e a previsão é lixo.

O treino é um loop:

1. Mostra uma janela de 60 dias para a rede.
2. Ela cospe um chute.
3. A gente compara com a resposta certa do dia 61. A diferença é o **erro**.
4. Um algoritmo de cálculo (derivadas, regra da cadeia, etc. — gerenciado pelo otimizador **Adam**) descobre **qual direção girar cada botão** para reduzir o erro um pouquinho.
5. Gira tudo um passo pequeno. Volta ao item 1 com a próxima janela.

Uma **época** é uma passada por TODOS os exemplos de treino. Pode levar várias épocas até a rede começar a acertar. O **EarlyStopping** decide quando parar.

A cada época, a gente mede o erro também no pedacinho de **validação** (10% mais recente do treino). Se o erro de treino cai mas o de validação sobe, é sinal de **decoreba** (overfitting): a rede memorizou exemplos específicos em vez de aprender o padrão. O dropout e o EarlyStopping são as armas contra isso.

---

## 7. Onde o "para que" se cruza com cada decisão

| Decisão no código | Por quê em uma frase |
|---|---|
| Janela = 60 dias | Ler ~3 meses de história costuma ser o ponto doce para ações no estudo de referência. |
| Só `Close` (univariado) | Reduz complexidade; suficiente para o objetivo didático da Fase 4. |
| MinMaxScaler em [0, 1] | LSTMs treinam melhor com números pequenos e parecidos entre si. |
| Fit do scaler só no treino | Evitar vazamento de dados (data leakage). |
| Sem shuffle | Série temporal: a ordem do tempo é informação. |
| Duas LSTMs empilhadas | Curto + longo prazo na mesma rede. |
| Dropout 0.2 | Combate overfitting com pouca degradação. |
| Adam + lr=1e-3 | Combinação padrão que "funciona bem por default". |
| EarlyStopping patience=10 | Dá margem para flutuações antes de cortar. |
| Salvar scaler como `.pkl` | A API precisa aplicar exatamente a mesma normalização. |
| `_load_artifacts` com `lru_cache` | Carregar o modelo a cada request mataria a latência. |
| `/health` nunca falha | O orquestrador (Docker/Coolify) depende disso para não reiniciar em loop. |
| HTTP 422 para erro do cliente | Distingue claramente "você mandou errado" de "o servidor bugou". |
| Prometheus em `/metrics` | Requisito de monitoramento da Fase 4. |
| Docker com `--workers 2` | Dois processos em paralelo aumentam throughput sem dobrar memória. |

---

## 8. Quando algo der errado — diagnóstico rápido

- **`uv run python -m src.train` falha no download** → sem internet ou ticker errado. Tente `--symbol SUZB3.SA`.
- **Treino fica horas sem terminar** → diminua `--epochs` ou aumente `--batch-size`.
- **`/health` retorna `degraded`** → a pasta `models/` está sem os três arquivos. Rode o treino.
- **`/predict` retorna 422** → a lista `closes` tem menos elementos que o `window` (60 por padrão).
- **`/predict` retorna 500** → bug interno. Olhe o log do servidor (`uvicorn` no terminal ou `docker logs`).
- **Métrica do modelo é pior que o baseline** → ação que oscila pouco no dia. O baseline "amanhã=hoje" é forte. Tente treinar com mais dados ou outro símbolo.

---

## 9. O que ler em seguida

- **`Pos_Tech - MLET - Tech Challenge Fase 4.pdf`** — a especificação oficial do trabalho.
- **`ESTUDO-TECH-CHALLENGE/PLANO-ESTUDO-TECH-CHALLENGE-FASE4.md`** — o checklist mestre com todas as decisões já tomadas (janela, normalização, formato dos artefatos).
- **`ESTUDO-TECH-CHALLENGE/06-API-FASTAPI.md`** — referência detalhada da API.
- **`ESTUDO-TECH-CHALLENGE/00-ROADMAP.md`** — o plano em ordem cronológica.
- **README.md** — resumo rápido (público) do projeto.

> Toda dúvida pontual: comece pelo arquivo `.py` correspondente. Os comentários foram reescritos para explicar **o quê e o porquê** linha a linha. Quando o `.py` não bastar, volte aqui no GUIA-DO-CODIGO.md, parte 3 (fluxo geral) ou parte 4 (passo a passo).
