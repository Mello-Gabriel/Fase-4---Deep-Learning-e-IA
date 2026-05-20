"""
============================================================================
ARQUIVO: src/model.py
PAPEL  : Montar a "arquitetura" da rede neural (a forma dela), pronta para
         ser treinada por `train.py`.
============================================================================

O QUE É UMA "REDE NEURAL"?
--------------------------
É um empilhamento de operações matemáticas com botões ajustáveis (os
"pesos"). A gente apresenta exemplos (entrada → resposta certa), ela
chuta, compara com o gabarito, e vai girando esses botões para errar
menos da próxima vez. Esse processo é o "treino".

POR QUE LSTM?
-------------
"LSTM" é a sigla em inglês de "Long Short-Term Memory" — em português,
algo como "Memória de Curto e Longo Prazo". É um tipo especial de rede
recorrente (que processa sequências passo a passo) com uma memória
interna que decide o que guardar e o que esquecer. Funciona bem com:

    - texto (palavras em sequência),
    - áudio,
    - SÉRIES TEMPORAIS, como o nosso preço diário.

Para previsão de preço, ela é uma das primeiras escolhas didáticas: vê
60 dias passados como uma fita, lembra padrões úteis (tendências, ciclos
curtos), e cospe um chute para o próximo dia.

DICIONÁRIO RÁPIDO (sem siglas soltas)
-------------------------------------
- "Camada"          = um andar do prédio que é a rede. Os dados entram em
                      cima, passam por cada andar, e saem embaixo.
- "Sequential"      = "andar de cima para baixo, em ordem". É a forma mais
                      simples de empilhar camadas, sem desvios nem ramos.
- "Dropout"         = durante o treino, desliga aleatoriamente uma fração
                      dos neurônios (aqui 20%). Isso obriga a rede a não
                      depender de poucos atalhos e ajuda contra o "decoreba"
                      (overfitting).
- "Dense"           = a camada clássica "totalmente conectada": cada
                      neurônio dela enxerga TODOS os neurônios da camada
                      anterior.
- "ReLU"            = uma função de ativação simples: se o número for
                      negativo, vira zero; se for positivo, fica igual.
                      Mantém só a parte "positiva" do sinal.
- "Adam"            = o otimizador. Quem decide o tamanho do passo na hora
                      de ajustar os botões da rede. É o padrão "que funciona
                      bem na maioria dos casos sem mexer muito".
- "Erro quadrático médio" (mean_squared_error) = soma dos quadrados das
                      diferenças entre previsão e gabarito, dividido pelo
                      número de exemplos. Penaliza erros grandes com mais
                      força que erros pequenos.
- "Erro absoluto médio" (MAE)  = média das diferenças em valor absoluto.
                      Métrica acompanhada durante o treino só para ler com
                      mais facilidade que o erro quadrático.
"""

from __future__ import annotations

# Importes da biblioteca Keras (que vem dentro do TensorFlow).
# Keras é uma "interface amigável" para montar redes neurais.
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam


def build_lstm(
    window: int = 60,
    n_features: int = 1,
    units: int = 50,
    dropout: float = 0.2,
    learning_rate: float = 1e-3,
) -> Sequential:
    """
    Constrói a rede neural e devolve o objeto pronto para treinar.

    Esquema da arquitetura (de cima para baixo):

        Entrada (janela de 60 dias x 1 valor por dia)
              │
              ▼
        ┌──────────────────────────────────────┐
        │ LSTM com 50 unidades, devolvendo a   │
        │ sequência inteira (return_sequences) │
        └──────────────────────────────────────┘
              │
              ▼
        Dropout 20%   ← desliga 20% dos neurônios no treino
              │
              ▼
        ┌──────────────────────────────────────┐
        │ Segunda LSTM com 50 unidades,        │
        │ devolvendo SÓ o último estado        │
        │ (return_sequences=False)             │
        └──────────────────────────────────────┘
              │
              ▼
        Dropout 20%
              │
              ▼
        Dense de 25 neurônios + ReLU  ← mistura final
              │
              ▼
        Dense de 1 neurônio (sem ativação)  ← o número previsto

    Por que duas LSTMs empilhadas?
        A primeira aprende padrões "perto" (mudanças do dia, fim de
        semana, ruído curto). A segunda recebe a saída da primeira já
        digerida e tem espaço para aprender padrões "mais largos"
        (tendência ao longo de semanas). Empilhar costuma melhorar para
        séries não muito curtas.

    Por que `return_sequences=True` só na primeira?
        A próxima LSTM precisa receber a sequência inteira (60 passos)
        para poder processar passo a passo. Já a segunda LSTM, como
        depois dela vem uma camada Dense, só precisa devolver o último
        estado — um único vetor.

    Por que o último Dense não tem ativação?
        Estamos prevendo um número contínuo (preço). Qualquer ativação
        ali (como ReLU ou sigmoid) limitaria o intervalo de saída.
        Sem ativação, a rede pode prever qualquer número real.

    Parâmetros:
        window        : tamanho da janela de entrada (60 dias por padrão).
        n_features    : variáveis por passo (1 = só fechamento).
        units         : quantos "neurônios" cada LSTM tem. 50 é o tamanho
                        sugerido no plano de estudo.
        dropout       : fração de neurônios desligados (0.2 = 20%).
        learning_rate : tamanho do passo do otimizador. 0.001 é o padrão
                        do Adam e funciona bem aqui.

    Retorna:
        Um `Sequential` Keras já compilado, ou seja, pronto para receber
        `.fit(...)` no `train.py`.
    """
    # `Sequential([...])` recebe uma lista de camadas e as empilha em ordem.
    model = Sequential(
        [
            # Camada de entrada: declaramos o formato esperado por amostra.
            # `(window, n_features)` significa "60 passos no tempo, 1 valor por passo".
            # A 1ª dimensão (n_amostras) fica de fora — o Keras descobre sozinho
            # quantos exemplos a gente passa em cada lote (batch).
            Input(shape=(window, n_features)),
            # 1ª LSTM. `return_sequences=True` faz ela devolver os 60 estados
            # intermediários (um por dia), permitindo empilhar outra LSTM depois.
            LSTM(units, return_sequences=True),
            # "Apaga" 20% dos sinais nesta passagem (só durante o treino).
            # No tempo de inferência (previsão real), o dropout é desativado
            # automaticamente pelo Keras.
            Dropout(dropout),
            # 2ª LSTM. `return_sequences=False` (padrão) → devolve apenas o
            # último estado, um vetor de 50 números resumindo a janela inteira.
            LSTM(units, return_sequences=False),
            Dropout(dropout),
            # Camada totalmente conectada de 25 neurônios com ReLU. Funciona
            # como um "misturador" entre a saída da LSTM e o número final.
            Dense(25, activation="relu"),
            # Saída: 1 número sem ativação. Esse é o "preço previsto" ainda
            # na escala normalizada [0, 1]. O `train.py` desnormaliza depois.
            Dense(1),
        ]
    )

    # `compile` diz ao Keras COMO treinar:
    #   - optimizer="adam"  → algoritmo de atualização dos pesos.
    #   - loss="mean_squared_error" → função objetivo (queremos minimizar).
    #   - metrics=["mae"]  → métrica extra mostrada por época, sem afetar o treino.
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss="mean_squared_error",
        metrics=["mae"],
    )
    return model
