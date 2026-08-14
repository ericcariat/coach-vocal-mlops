"""Tous les tests tournent sur CPU (ADR-002), configuré AVANT tout import TF.

Sans ça, le premier test qui importe TensorFlow initialise le backend avec le
GPU Metal visible, et plus personne ne peut le masquer ensuite (la liste des
devices est figée à l'initialisation).
"""

from coachvocal import runtime

runtime.configure(use_gpu=False)
