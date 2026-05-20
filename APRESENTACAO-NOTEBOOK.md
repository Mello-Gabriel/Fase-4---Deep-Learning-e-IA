# Apresentação do Notebook `MLET4TC.ipynb` — Guia para NotebookLM

> Documento de apoio para o NotebookLM gerar uma apresentação didática sobre o notebook `MLET4TC.ipynb`.
> Linguagem em português, sem siglas soltas (toda sigla é explicada na primeira aparição).
> Público-alvo da apresentação: banca da Pós-Tech FIAP MLET — Fase 4 (Deep Learning e Inteligência Artificial).

---

## 0. Contexto do projeto

- **Curso:** Pós-Tech em Machine Learning Engineering — FIAP.
- **Fase 4:** "Deep Learning e Inteligência Artificial".
- **Tech Challenge:** vale 90% da nota da fase.
- **Tarefa:** construir uma rede neural do tipo **Memória de Longo e Curto Prazo** (em inglês, *Long Short-Term Memory*, abreviada como LSTM) que **prevê o preço de fechamento do próximo dia** de uma ação da bolsa.
- **Entrega final:** pipeline ponta a ponta — coleta → pré-processamento → treino → avaliação → salvamento → Interface de Programação de Aplicações (em inglês, *Application Programming Interface*, abreviada como API) em FastAPI → contêiner Docker → publicação na nuvem em Coolify (sobre infraestrutura Hostinger) → monitoramento via Prometheus.
- **Ação escolhida:** `SUZB3.SA` (Suzano S.A., negociada na bolsa brasileira B3).
- **Período da série:** 2018-01-01 até 2025-12-31 (1988 dias úteis).
- **Resultado final do modelo no conjunto de teste:**
  - Erro Médio Absoluto (em inglês, *Mean Absolute Error*, abreviado como MAE): **R$ 1,16**.
  - Raiz do Erro Quadrático Médio (em inglês, *Root Mean Squared Error*, abreviada como RMSE): **R$ 1,59**.
  - Erro Médio Percentual Absoluto (em inglês, *Mean Absolute Percentage Error*, abreviado como MAPE): **2,10%**.

O notebook foi montado especificamente para a apresentação oral: cada seção é independente, comentada em português, e formatada com a ferramenta Ruff para manter o estilo consistente.

---

## 1. Estrutura do notebook em 9 seções

O notebook tem 9 seções numeradas. Esta é a "espinha dorsal" da apresentação:

1. **Imports** — todas as bibliotecas em uma só célula no topo.
2. **Constantes padrão** — escolhas fixas do projeto (ação, datas, janela, split).
3. **Estrutura do `Dataset`** — um pacote único que carrega tudo do treino.
4. **Download da série de preços** — busca e cache dos dados.
5. **Janelas deslizantes** — recorte da série temporal.
6. **Pipeline `build_dataset`** — pipeline completo numa só função.
7. **Modelo LSTM** — construção e compilação da rede.
8. **Métricas de avaliação** — MAE, RMSE, MAPE e baseline ingênuo.
9. **Treinamento** — divido em 4 subpassos:
   - 9.1 Hiperparâmetros e diretórios de saída.
   - 9.2 Treino com paradas inteligentes.
   - 9.3 Avaliação no teste e gráficos.
   - 9.4 Persistência dos artefatos.

---

## 2. Seção 1 — Imports

### O que acontece
Toda dependência usada no notebook é importada de uma vez só na primeira célula.

### Por que isso importa para a apresentação
- Sinaliza disciplina de engenharia: o leitor sabe, em 20 segundos, todas as bibliotecas envolvidas.
- Evita "imports espalhados" que poluem a leitura linear do notebook.

### Pontos de fala
- **`from __future__ import annotations`**: deixa o Python tratar todas as anotações de tipo como texto, evitando custo de avaliação em tempo de execução. Útil mesmo em Python 3.12.
- **`tensorflow.keras`**: o framework de aprendizado profundo escolhido. Keras é a interface de alto nível do TensorFlow.
- **`scikit-learn`**: usado apenas para o normalizador `MinMaxScaler` e para calcular MAE/RMSE.
- **`yfinance`**: cliente do Yahoo Finance, fonte gratuita de cotações históricas.
- **`joblib`**: serializa o normalizador para disco (a Interface de Programação de Aplicações precisa reusar o mesmo objeto em produção).
- **`matplotlib`**: gera os dois gráficos finais (curva de aprendizado e real-vs-previsto).

---

## 3. Seção 2 — Constantes padrão

```python
DEFAULT_SYMBOL = "SUZB3.SA"
DEFAULT_START = "2018-01-01"
DEFAULT_END = "2025-12-31"
DEFAULT_WINDOW = 60
DEFAULT_SPLIT = 0.8
```

### O que acontece
São fixadas as escolhas do projeto.

### Decisões e justificativas
- **`SUZB3.SA`**: ação líquida da B3, séries históricas confiáveis no Yahoo Finance.
- **8 anos de histórico** (2018–2025): suficiente para a rede ver vários regimes de mercado (crescimento, queda da pandemia, juros altos, recuperação).
- **`window=60`**: a rede recebe 60 dias úteis (~3 meses corridos) para prever o próximo. Esse é um valor consagrado em tutoriais de séries financeiras com LSTM.
- **`split=0.8`**: 80% iniciais para treino, 20% finais para teste. A divisão é **cronológica** (e não aleatória), por razões explicadas na Seção 6.

---

## 4. Seção 3 — Estrutura do `Dataset`

### O que acontece
Define-se uma classe de dados (em inglês, *dataclass*) chamada `Dataset`. Ela é apenas um "envelope" que guarda, num só lugar:
- a ação e a janela usadas;
- as entradas e rótulos de treino e teste (`X_train`, `y_train`, `X_test`, `y_test`);
- o normalizador ajustado;
- a série bruta original (em reais);
- o índice onde treino e teste foram separados.

### Por que isso importa
- Evita passar 10 variáveis soltas pelas funções.
- A Interface de Programação de Aplicações em produção vai precisar do **mesmo normalizador** que foi ajustado no treino — guardá-lo no `Dataset` deixa esse acoplamento explícito.

### Ponto de fala chave
> "Eu coloco tudo num só objeto porque o normalizador e os dados não podem ser separados — usar um normalizador novo em produção é o erro número um nesse tipo de projeto."

---

## 5. Seção 4 — Download da série de preços

### O que acontece
A função `download_close` baixa as cotações pelo `yfinance` e mantém um cache em arquivo de valores separados por vírgula (em inglês, *Comma-Separated Values*, abreviado como CSV) na pasta `data/`.

### Decisões importantes
- **Cache em CSV**: evita rebaixar dados toda vez que o notebook é executado. Importante durante a apresentação, onde re-executar custa tempo.
- **`auto_adjust=False`**: usa o preço de fechamento "bruto" (sem ajuste por dividendos), porque é o que aparece nos gráficos de mercado que a banca vai reconhecer.
- **`dropna()`**: remove dias sem cotação (feriados eventuais que o Yahoo Finance possa registrar como vazio).
- **`MultiIndex` colapsado**: versões recentes do `yfinance` devolvem colunas hierárquicas; o código achata para uma única dimensão.

### Por que isso importa
- Mostra cuidado com **reprodutibilidade**: a mesma chamada na semana que vem trará exatamente os mesmos dados se o cache existir.
- A função pode ser usada igualmente em treino offline e na futura Interface de Programação de Aplicações (basta passar `cache_dir=None`).

---

## 6. Seção 5 — Janelas deslizantes

### O que acontece
A função `make_windows` transforma a série em pares **(entrada, rótulo)**:
- **Entrada:** um vetor com os 60 preços mais recentes.
- **Rótulo:** o preço do dia seguinte.

Para uma série de 1000 dias com janela 60, sobram **940 janelas** de treino.

### Forma esperada pela LSTM
A camada LSTM exige entradas em três dimensões:

```
(número_de_amostras, número_de_passos_no_tempo, número_de_variáveis_por_passo)
= (940, 60, 1)
```

O `1` no final representa "uma única variável por dia" — neste projeto, apenas o preço de fechamento. A esse tipo de série dá-se o nome de **univariada**.

### Pontos de fala
- **Por que janelas e não toda a série?** Uma LSTM aprende padrões locais melhor com janelas curtas; a memória interna dela cuida da dependência de longo prazo dentro da janela.
- **Por que 60 e não 30 ou 90?** Trade-off: janelas maiores capturam mais contexto mas geram menos amostras de treino e tornam o gradiente mais difícil. 60 é a faixa consagrada em estudos de previsão diária de fechamento.

---

## 7. Seção 6 — Pipeline `build_dataset`

### O que a função faz, em ordem
1. **Baixa** a série de fechamento.
2. **Divide** em treino (80% inicial) e teste (20% final) — **antes** de normalizar.
3. **Ajusta o normalizador** apenas no trecho de treino (`fit_transform`).
4. **Aplica** o normalizador (já ajustado) no trecho de teste (`transform`).
5. **Gera as janelas** de treino e teste.
6. **Empacota** tudo num `Dataset`.

### A decisão mais importante do notebook: evitar vazamento de dados

> **Vazamento de dados** (em inglês, *data leakage*) acontece quando informação do conjunto de teste contamina o treino. Em séries temporais, o erro clássico é normalizar a série inteira de uma vez: o mínimo e o máximo do futuro "vazam" para o passado, e o modelo aparenta ter performance que ele não tem no mundo real.

A defesa contra isso é dupla:
- **Split temporal**, não aleatório: nada de `shuffle=True`, nada de `train_test_split` da scikit-learn. O teste é o **futuro** do treino.
- **Normalização ajustada apenas no treino**: o `MinMaxScaler` aprende a faixa olhando só os 80% iniciais. Para gerar as janelas de teste, são concatenados os últimos 60 dias normalizados de treino com o teste normalizado, garantindo histórico inicial.

### Ponto de fala forte
> "Se eu usasse `train_test_split` com embaralhamento, meu modelo veria pedaços de 2024 enquanto treinava para prever 2020 — isso é trapaça. Por isso eu separo no tempo e só depois normalizo."

---

## 8. Seção 7 — Modelo LSTM

### O que é uma rede LSTM, em uma frase
É um tipo de rede neural recorrente que tem uma **memória interna** capaz de carregar informação de muitos passos atrás, usando três comportas internas (de **esquecimento**, de **entrada** e de **saída**) que decidem o que guardar, o que atualizar e o que devolver.

Esse desenho resolve um problema clássico das redes recorrentes simples: o **desvanecimento do gradiente** (em inglês, *vanishing gradient*), em que o sinal de aprendizado some quando a sequência é longa.

### Arquitetura escolhida

```
Input(60, 1)
  ↓
LSTM(50, return_sequences=True)
  ↓
Dropout(0.2)
  ↓
LSTM(50, return_sequences=False)
  ↓
Dropout(0.2)
  ↓
Dense(25, activation="relu")
  ↓
Dense(1)
```

Total de parâmetros treináveis: **31.901** (~124 KB).

### Por que cada peça está aí

| Camada | Função | Justificativa |
|---|---|---|
| `Input(60, 1)` | Define o formato esperado | Faz a forma do tensor ficar explícita; ajuda o Keras a gerar o resumo. |
| `LSTM(50, return_sequences=True)` | Primeira camada de memória | `return_sequences=True` devolve uma saída por passo de tempo, permitindo que **outra LSTM** receba uma sequência completa. |
| `Dropout(0.2)` | Regularização | Desliga 20% dos neurônios a cada passo de treino. Combate sobreajuste (em inglês, *overfitting*). |
| `LSTM(50, return_sequences=False)` | Segunda camada de memória | Sintetiza a sequência num único vetor. |
| `Dropout(0.2)` | Regularização | Mesmo motivo da primeira. |
| `Dense(25, activation="relu")` | Camada totalmente conectada com não-linearidade | Combina as features extraídas. **Unidade Linear Retificada** (em inglês, *Rectified Linear Unit*, abreviada como ReLU) é a função de ativação. |
| `Dense(1)` | Saída | **Sem ativação** — é um problema de regressão. Quero um número real, não uma probabilidade. |

### Compilação
- **Otimizador:** `Adam`, taxa de aprendizado `1e-3`. Padrão de mercado para LSTM.
- **Função de perda:** `mean_squared_error` (erro quadrático médio) — penaliza erros grandes mais do que MAE, ajudando a empurrar previsões fora da curva de volta para perto da realidade.
- **Métrica acompanhada:** `mae` — mais legível durante o treino, expressa em unidades da saída.

### Ponto de fala recorrente da banca
> **"Por que duas LSTMs empilhadas?"**
> R: A primeira camada aprende padrões locais (volatilidade de curtíssimo prazo, ruído); a segunda recebe essa representação já refinada e captura padrões mais abstratos sobre toda a janela. Empilhar é o equivalente, para redes recorrentes, do que fazer várias camadas convolucionais é para imagens.

> **"Por que não tem ativação no Dense final?"**
> R: O alvo é um número real (preço de fechamento na escala normalizada). Aplicar `sigmoid` limitaria a saída entre 0 e 1 sem nenhuma vantagem; aplicar `relu` impediria de prever valores ligeiramente negativos durante o aprendizado.

> **"Por que dropout só de 0.2?"**
> R: A série não é muito longa (~1.500 janelas de treino). Dropout alto demais subtreina o modelo. 0.2 é um meio-termo conservador.

---

## 9. Seção 8 — Métricas de avaliação

### As três métricas usadas

- **MAE (Mean Absolute Error / Erro Médio Absoluto)**
  - Fórmula: média do valor absoluto dos erros.
  - Lê-se em reais. **Quanto menor, melhor.**
- **RMSE (Root Mean Squared Error / Raiz do Erro Quadrático Médio)**
  - Fórmula: raiz da média dos erros ao quadrado.
  - Lê-se em reais. **Penaliza mais erros grandes.**
- **MAPE (Mean Absolute Percentage Error / Erro Médio Percentual Absoluto)**
  - Fórmula: média do valor absoluto de `(real − previsto) / real`, em porcentagem.
  - Lê-se como percentual. **Não tem escala** — facilita comparar entre ações de preços diferentes.

### O baseline ingênuo

A função `naive_baseline` calcula as mesmas métricas para uma regra trivial: **"o preço de amanhã será igual ao de hoje"**.

Esse é o **mínimo legal**: se a LSTM treinada errar **mais** que esse baseline, o modelo não está agregando informação — está apenas gastando energia.

### Por que isso é determinante na apresentação
> "Apresentar uma rede neural sem mostrar o baseline é como dizer que um carro é rápido sem mencionar a velocidade dos outros."

Comparar o resultado do modelo (MAPE ~2,10%) contra o resultado do baseline (a banca pode pedir) demonstra que **a previsão é melhor do que apenas repetir o último valor**.

---

## 10. Seção 9 — Treinamento

A seção é fragmentada em 4 subcélulas, cada uma com função pedagógica distinta.

### 10.1 — Hiperparâmetros e diretórios

- **`EPOCHS = 100`** — limite máximo de épocas. Na prática, a parada antecipada interrompe antes.
- **`BATCH_SIZE = 32`** — quantas janelas o modelo vê antes de atualizar os pesos. 32 equilibra estabilidade e velocidade.
- **`PATIENCE = 10`** — quantas épocas consecutivas sem melhora antes da rede parar.
- **`MODELS_DIR = Path("models")`** — onde os artefatos serão salvos.
- **`REPORTS_DIR = Path("reports")`** — onde os gráficos serão salvos como imagens.

### 10.2 — Treino com paradas inteligentes

```python
callbacks = [
    EarlyStopping(monitor="val_loss", patience=PATIENCE, restore_best_weights=True),
    ModelCheckpoint(str(best_path), monitor="val_loss", save_best_only=True),
]

history = model.fit(
    dataset.X_train,
    dataset.y_train,
    validation_split=0.1,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=callbacks,
    verbose=2,
    shuffle=False,
)
```

#### Dois mecanismos defensivos, em paralelo

- **`EarlyStopping`** — observa a perda de validação. Se ela não melhorar por 10 épocas seguidas, interrompe o treino **e restaura os pesos da melhor época**. Defende contra sobreajuste e desperdício de tempo.
- **`ModelCheckpoint`** — a cada época onde a validação melhora, salva o modelo em disco. Defende contra perda de progresso em caso de falha.

#### Detalhes técnicos importantes

- **`validation_split=0.1`** — o Keras separa os **últimos 10%** dos dados de treino para validar. Como a separação é pelo final, **continua preservando a ordem temporal**.
- **`shuffle=False`** — embaralhar quebraria a ordem temporal dentro do batch. Em séries temporais isso é proibido.
- **`verbose=2`** — uma linha de log por época. Limpo o suficiente para a apresentação e detalhado o suficiente para diagnosticar.

#### Leitura da curva de aprendizado real
- **Épocas 1–6:** ambas as perdas caindo, modelo ainda muito subajustado.
- **Épocas 10–20:** perda de validação caindo rápido, abaixo da de treino (sinal saudável, fruto do dropout).
- **Épocas 25 em diante:** perda estagnada em torno de 8–10 × 10⁻⁴. A parada antecipada disparou por volta da época 36.

### 10.3 — Avaliação no teste e gráficos

```python
pred_scaled = model.predict(dataset.X_test, verbose=0)
y_pred = dataset.scaler.inverse_transform(pred_scaled).ravel()
y_true = dataset.scaler.inverse_transform(dataset.y_test.reshape(-1, 1)).ravel()
```

#### Por que inverter o normalizador antes de pontuar
O modelo prevê em escala 0–1 (porque foi treinado nessa escala). Para reportar erros **em reais**, é necessário aplicar a operação inversa do `MinMaxScaler` — voltar de "0,72" para "R$ 38,15".

#### Os dois gráficos
1. **Curva de aprendizado** — duas linhas (treino vs validação) ao longo das épocas. Mostra se o modelo convergiu e se não está sobreajustando.
2. **Real vs previsto** — duas linhas no eixo temporal do teste. Visualmente, a previsão "segue" a curva real com um pequeno atraso.

Ambos são salvos em `reports/` como arquivos PNG, para serem reutilizados no slide.

### 10.4 — Persistência dos artefatos

Esta é a **ponte** entre o notebook e a Interface de Programação de Aplicações.

São salvos **três** arquivos em `models/`:

1. **`lstm_stock.keras`** — o modelo treinado completo (arquitetura + pesos).
2. **`scaler.pkl`** — o normalizador serializado com `joblib`. **Sem ele, a API não consegue normalizar a janela recebida do usuário.**
3. **`metadata.json`** — um cartão de identidade do modelo, contendo:
   - Ação treinada e período.
   - Tamanho da janela e lista de features.
   - Hiperparâmetros usados.
   - Métricas finais (MAE, RMSE, MAPE).
   - Métricas do baseline ingênuo, para comparação.
   - Quantas amostras de treino e teste foram usadas.
   - Os **últimos 60 dias reais**, úteis para o cliente testar a Interface de Programação de Aplicações sem coletar dados.
   - Carimbo de tempo da última gravação.

### Ponto de fala estratégico
> "Os três arquivos viajam juntos. A Interface de Programação de Aplicações carrega os três em sua inicialização, uma única vez, usando cache. Cada requisição de previsão depende apenas do modelo já em memória."

---

## 11. Como tudo se conecta com o resto da entrega

```
Notebook (treino)
   ↓ salva
models/lstm_stock.keras
models/scaler.pkl
models/metadata.json
   ↓ carrega em startup
api/main.py  (FastAPI + Pydantic)
   ↓ recebe POST /predict com 60 closes
   ↓ normaliza, prevê, desnormaliza
   ↓ retorna preço em reais
   ↓ instrumentado com Prometheus em /metrics
   ↓
Dockerfile + docker-compose.yml
   ↓ build & push
Coolify (Hostinger)
   ↓ deploy
URL pública + monitoramento
```

A Interface de Programação de Aplicações expõe três rotas:

- **`GET /health`** — verifica se está viva.
- **`GET /metadata`** — devolve o `metadata.json` para o cliente saber em que ação o modelo foi treinado.
- **`POST /predict`** — recebe uma lista de 60 ou mais preços de fechamento e devolve a previsão do próximo dia.

---

## 12. Possíveis perguntas da banca e respostas curtas

### Pergunta 1: "Por que LSTM e não outro modelo?"
LSTM é o padrão consagrado para séries temporais univariadas porque carrega memória de longo prazo sem sofrer com o desvanecimento do gradiente. Modelos mais simples como ARIMA não capturam não-linearidades; modelos mais novos (Transformer) exigem volume de dados muito maior.

### Pergunta 2: "Por que apenas o preço de fechamento?"
A especificação do Tech Challenge define a tarefa como univariada. Incluir abertura, máxima, mínima e volume melhoraria pouco e aumentaria o risco de sobreajuste com apenas 2 mil dias.

### Pergunta 3: "Como você evita vazamento de dados?"
Split temporal (nunca aleatório) antes de qualquer transformação, e normalizador ajustado **apenas** no trecho de treino — confirmado pelo método `fit_transform` chamado uma única vez sobre `train_raw`.

### Pergunta 4: "O que prova que o modelo aprendeu algo?"
A comparação com o baseline ingênuo "amanhã = hoje". Se o MAPE da LSTM é menor que o do baseline, há informação sendo extraída.

### Pergunta 5: "Por que não usar dados em tempo real na API?"
A Interface de Programação de Aplicações é desacoplada da fonte de dados: o cliente envia a janela. Isso evita dependência do Yahoo Finance estar disponível e mantém a previsão determinística. Para casos em que o cliente não tem os dados, o `metadata.json` traz os últimos 60 fechamentos reais.

### Pergunta 6: "Como você monitora em produção?"
A Interface de Programação de Aplicações é instrumentada com a biblioteca `prometheus-fastapi-instrumentator`, que expõe latência por rota e contadores HTTP na rota `/metrics`. Métricas de uso de processador e memória são observadas diretamente no painel do Coolify.

### Pergunta 7: "Como você escolheria outra ação?"
Trocando a constante `DEFAULT_SYMBOL` e re-executando o notebook. A janela de 60 dias e a divisão 80/20 funcionam bem para qualquer ação líquida da B3 com pelo menos 5 anos de histórico.

### Pergunta 8: "Quanto tempo leva o treino?"
Cerca de 30 a 60 segundos em uma máquina sem placa de vídeo dedicada. A parada antecipada disparou por volta da época 36 nesta execução.

### Pergunta 9: "O modelo prevê amanhã ou vários dias à frente?"
Apenas o próximo dia. Para vários dias, seria necessária previsão multi-passo (autoregressiva, realimentando a saída) — não está no escopo desta entrega.

### Pergunta 10: "E se o usuário enviar menos de 60 valores?"
A Interface de Programação de Aplicações valida com Pydantic e devolve erro HTTP 422 (entidade não processável) antes de tocar no modelo.

---

## 13. Roteiro sugerido de apresentação (10 minutos)

| Tempo | Bloco | Conteúdo |
|---|---|---|
| 0:00–0:45 | Abertura | Tech Challenge, ação escolhida, resultado final em uma frase |
| 0:45–2:00 | Coleta e cache | Seções 4, 5 e 6 |
| 2:00–3:30 | Anti-vazamento | Por que split temporal + normalização só no treino |
| 3:30–5:30 | Arquitetura LSTM | Seção 7 + tabela das camadas |
| 5:30–6:30 | Treino | Seção 10.2, curva de aprendizado, parada antecipada |
| 6:30–7:30 | Métricas vs baseline | Seção 9 com os números do experimento |
| 7:30–8:30 | Artefatos e API | Como o notebook alimenta o serviço |
| 8:30–9:30 | Demo da API (se possível) | `POST /predict` com a janela do `metadata.json` |
| 9:30–10:00 | Encerramento | "Próximos passos": multivariada, multi-passo, retraining automático |

---

## 14. Glossário rápido (para revisão de última hora)

- **API (Application Programming Interface / Interface de Programação de Aplicações):** porta de entrada pela qual outros programas conversam com o modelo.
- **Baseline ingênuo:** previsão trivial usada como teto mínimo de qualidade. Aqui: "amanhã = hoje".
- **Cache:** cópia local que evita refazer trabalho caro.
- **Dropout:** técnica que desliga aleatoriamente uma fração de neurônios durante o treino para reduzir sobreajuste.
- **Early stopping (parada antecipada):** interrompe o treino quando a validação para de melhorar.
- **Hiperparâmetro:** valor escolhido **antes** do treino (ex.: número de épocas, tamanho do lote).
- **LSTM (Long Short-Term Memory / Memória de Longo e Curto Prazo):** tipo de rede recorrente com comportas que controlam memória.
- **MAE (Mean Absolute Error):** erro médio absoluto em unidades reais.
- **MAPE (Mean Absolute Percentage Error):** erro médio em porcentagem.
- **MinMaxScaler:** normalizador que mapeia valores para o intervalo 0–1.
- **Pipeline:** sequência ordenada de transformações dos dados.
- **ReLU (Rectified Linear Unit / Unidade Linear Retificada):** função de ativação que devolve `max(0, x)`.
- **RMSE (Root Mean Squared Error):** raiz do erro quadrático médio, em unidades reais.
- **Sliding window (janela deslizante):** recorte de uma série temporal em sub-sequências fixas.
- **Sobreajuste (overfitting):** quando o modelo memoriza o treino e generaliza mal.
- **Vazamento de dados (data leakage):** contaminação do treino com informação do teste.

---

## 15. Mensagem-chave da apresentação

> Construí uma rede neural do tipo Memória de Longo e Curto Prazo que prevê o preço de fechamento do dia seguinte da ação SUZB3.SA com erro médio de R$ 1,16 (cerca de 2% do preço). O pipeline é reprodutível, não tem vazamento de dados, é monitorado em produção e está empacotado em três arquivos — modelo, normalizador e cartão de identidade — que viajam juntos do notebook para a Interface de Programação de Aplicações.
