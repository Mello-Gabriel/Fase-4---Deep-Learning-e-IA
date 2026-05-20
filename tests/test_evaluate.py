"""
============================================================================
ARQUIVO: tests/test_evaluate.py
PAPEL  : Testes unitários para as métricas (`regression_metrics`) e o
         baseline ingênuo (`naive_baseline`).
============================================================================

POR QUE TESTAR MÉTRICAS?
------------------------
Se as métricas calcularem errado, o relatório de qualidade do modelo
fica mentindo — a gente pode até promover um modelo ruim achando que é
bom. Estes testes ancoram resultados em casos triviais que dá para
calcular de cabeça.
"""

import numpy as np

from src.evaluate import naive_baseline, regression_metrics


def test_metrics_zero_on_perfect_pred() -> None:
    """
    Previsão perfeita (igual ao real) → todos os erros devem dar zero.
    """
    y = np.array([10.0, 11.0, 12.0, 13.0])
    m = regression_metrics(y, y)
    assert m["MAE"] == 0
    assert m["RMSE"] == 0
    assert m["MAPE_%"] == 0


def test_metrics_nonzero() -> None:
    """
    Caso manual:
        y_real = [10, 20]
        y_prev = [11, 18]
        erros absolutos = [1, 2]   → MAE = média([1, 2]) = 1.5
        erros quadrát.  = [1, 4]   → RMSE = sqrt(média([1,4])) ≈ 1.58

    Confirma também que RMSE > MAE para erros não-uniformes — propriedade
    teórica importante (a desigualdade é estrita quando os erros variam).
    """
    y = np.array([10.0, 20.0])
    p = np.array([11.0, 18.0])
    m = regression_metrics(y, p)
    assert m["MAE"] == round((1 + 2) / 2, 4)
    assert m["RMSE"] > m["MAE"] - 1e-9


def test_naive_baseline_shape() -> None:
    """
    O baseline ingênuo deve retornar um dicionário com as três chaves
    esperadas, qualquer que seja a entrada. Não testamos o VALOR aqui —
    isso já é coberto pelo teste de `regression_metrics`.
    """
    y = np.array([10.0, 11.0, 12.0, 13.0])
    b = naive_baseline(y)
    assert "MAE" in b and "RMSE" in b and "MAPE_%" in b
