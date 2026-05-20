"""
============================================================================
ARQUIVO: tests/test_data.py
PAPEL  : Testes unitários para a função `make_windows` do módulo de dados.
============================================================================

O QUE É UM "TESTE UNITÁRIO"?
----------------------------
É um programinha que chama uma função do código real, com entradas
pré-determinadas, e verifica se a saída bate com o esperado. Se algo no
código mudar e quebrar essa lógica, o teste falha e a gente descobre
ANTES de o usuário sentir.

POR QUE TESTAR JANELAS?
-----------------------
A função `make_windows` é a peça mais sensível do pré-processamento: se
ela errar uma posição, o gabarito sai deslocado e o modelo treina sobre
respostas erradas, sem reclamar. Métricas pareceriam até razoáveis, mas
em produção a previsão seria de outro dia. Por isso vale testar.

ESTRUTURA DE UM ARQUIVO PYTEST
------------------------------
- Cada função que começa com `test_` é executada pelo pytest.
- `assert <expressão>` faz a verificação. Se a expressão for falsa,
  o teste falha.
- Não há `if __name__ == "__main__"`: o pytest descobre os arquivos sozinho.
"""

import numpy as np

from src.data import make_windows


def test_make_windows_shapes() -> None:
    """
    Confere que as FORMAS dos arrays de saída estão corretas.

    Entrada: série com 100 pontos, janela de 60.
    Esperado: 40 amostras (100 - 60), cada uma com 60 passos e 1 feature.
    """
    series = np.arange(100, dtype=np.float32).reshape(-1, 1)
    X, y = make_windows(series, window=60)
    assert X.shape == (40, 60, 1)
    assert y.shape == (40,)


def test_make_windows_continuity() -> None:
    """
    Confere que os VALORES dentro das janelas seguem a ordem certa.

    Com uma série [0, 1, 2, ..., 69] e janela 60, o esquema esperado é:

        X[0]  = [0, 1, ..., 59]  → y[0]  = 60
        X[-1] = [9, 10, ..., 68] → y[-1] = 69

    Se esse alinhamento estiver errado, o modelo aprenderia gabaritos
    deslocados, então este teste é uma rede de segurança importante.
    """
    series = np.arange(70, dtype=np.float32).reshape(-1, 1)
    X, y = make_windows(series, window=60)

    # Primeira janela: vai de 0 a 59 (inclusive); a resposta é o índice 60.
    np.testing.assert_array_equal(X[0, :, 0], np.arange(60))
    assert y[0] == 60

    # Última janela: vai de 9 a 68 (inclusive); a resposta é o índice 69.
    np.testing.assert_array_equal(X[-1, :, 0], np.arange(9, 69))
    assert y[-1] == 69


def test_make_windows_rejects_short() -> None:
    """
    Garante que a função RECUSA séries menores que a janela, em vez de
    produzir um array vazio silenciosamente.

    "Falhar barulhento" é melhor que "passar errado" — assim a pessoa que
    chamou a função descobre o problema na hora.
    """
    series = np.arange(30, dtype=np.float32).reshape(-1, 1)
    try:
        make_windows(series, window=60)
    except ValueError:
        # Esperado: a função levantou ValueError. Teste passa.
        return
    # Se chegou aqui, NÃO foi levantado o erro esperado. Falha proposital.
    raise AssertionError("Expected ValueError for series shorter than window")
