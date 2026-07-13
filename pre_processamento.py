"""Compatibilidade de importação para notebooks e scripts.

Este módulo reexporta as classes definidas em src.pre_processamento para que
possam ser importadas por nomes curtos como `from pre_processamento import ...`.
"""

from src.pre_processamento import Engenharia_Features


class EngenhariaFeatureNLTK(Engenharia_Features):
    """Alias de compatibilidade para importações antigas."""

    pass


# Alias para compatibilidade com importações antigas.
Engenharia_Fetures = Engenharia_Features

__all__ = [
    "Engenharia_Features"
]

    