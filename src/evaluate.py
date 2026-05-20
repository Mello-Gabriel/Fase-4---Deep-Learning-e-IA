"""
============================================================================
ARQUIVO: src/evaluate.py
PAPEL  : Calcular as métricas de erro (na escala real, em reais) para
         saber se o modelo está acertando ou errando muito.
============================================================================

POR QUE ESTE ARQUIVO EXISTE
---------------------------
Treinar é fácil; saber se ficou BOM é o difícil. Aqui ficam as três
métricas que o plano de estudo decidiu acompanhar, mais um "baseline"
ingênuo para comparar.

AS TRÊS MÉTRICAS
----------------
1) "Erro absoluto médio" (em inglês: Mean Absolute Error, abreviado MAE)
   - Pega cada erro (previsão menos valor real), tira o sinal e faz média.
   - Lê-se na mesma unidade do dado. Se MAE = 1.15, em média erramos
     R$ 1,15 por previsão.
   - Vantagem: fácil de entender.
   - Limitação: dá o mesmo peso a erros pequenos e grandes.

2) "Raiz do erro quadrático médio" (Root Mean Squared Error, RMSE)
   - Eleva cada erro ao quadrado, faz média, tira a raiz quadrada.
   - Também sai na unidade do dado (R$).
   - Castiga mais os erros grandes do que pequenos.
   - Sempre maior ou igual ao MAE; quanto maior a diferença entre os
     dois, maior a variância dos erros.

3) "Erro percentual absoluto médio" (Mean Absolute Percentage Error,
   MAPE)
   - Calcula o erro de cada previsão como percentual do valor real, e
     faz média.
   - Sai em PORCENTAGEM, independente da escala da ação. 2% de MAPE
     significa que, em média, erramos 2% do preço.

O QUE É O "BASELINE INGÊNUO"?
-----------------------------
É a regra mais boba possível: "amanhã vai ser igual a hoje". Serve como
piso de comparação. Se o nosso modelo treinado erra MAIS que essa regra
boba, ele não está aprendendo nada útil. (No caso atual, a Suzano oscila
pouco no dia, então o baseline ingênuo é forte e pode até bater o modelo
em alguns períodos — isso é normal nesse tipo de série e está documentado.)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

if TYPE_CHECKING:
    from numpy.typing import ArrayLike


def regression_metrics(y_true: ArrayLike, y_pred: ArrayLike) -> dict[str, float]:
    """
    Calcula MAE, RMSE e MAPE comparando duas listas (gabarito vs. previsão).

    Importante: as duas listas devem estar na MESMA escala. No projeto, as
    chamamos com os preços já desnormalizados (em reais), para que a leitura
    faça sentido financeiro.

    Parâmetros:
        y_true : valores reais (gabarito), em R$.
        y_pred : valores previstos pelo modelo, em R$.

    Retorna:
        Dicionário com três chaves:
            "MAE"   → erro absoluto médio (R$)
            "RMSE"  → raiz do erro quadrático médio (R$)
            "MAPE_%"→ erro percentual absoluto médio (%)
    """
    # `ravel()` achata para 1D, evitando dor de cabeça com formatos (n,1) vs (n,).
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()

    # Erro absoluto médio: já vem pronto do scikit-learn.
    mae = mean_absolute_error(y_true, y_pred)

    # Raiz do erro quadrático médio. `mean_squared_error` devolve a média
    # dos quadrados, então tiramos a raiz para voltar à escala de preço.
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))

    # Erro percentual absoluto médio: |erro| / valor real, em %.
    # Atenção: estoura se algum y_true for zero. Para preço de ação isso
    # não acontece na prática.
    mape = float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100)

    return {
        "MAE": round(float(mae), 4),
        "RMSE": round(rmse, 4),
        "MAPE_%": round(mape, 4),
    }


def naive_baseline(y_true: ArrayLike) -> dict[str, float]:
    """
    Calcula as mesmas métricas, mas para a regra "amanhã = hoje".

    Como funciona: pegamos `y_true` e simulamos uma "previsão" que é
    simplesmente a série deslocada um dia. Ou seja:
        previsão do dia 2 = valor real do dia 1
        previsão do dia 3 = valor real do dia 2
        ...

    Comparamos `y_true[1:]` (valores reais, sem o primeiro) com `y_true[:-1]`
    (mesma série, sem o último). A diferença é exatamente o "erro do palpite
    preguiçoso".

    Isso vira a régua que o modelo precisa BATER. Se a LSTM treinada erra
    menos que o baseline, ela está agregando algo. Se erra mais, está atrás
    do palpite mais bobo possível.
    """
    y_true = np.asarray(y_true).ravel()
    return regression_metrics(y_true[1:], y_true[:-1])
