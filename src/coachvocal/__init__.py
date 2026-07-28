"""coachvocal — détection de wake word, du dataset au service.

Organisation (cf. docs/ARCHITECTURE.md) :
    config    schémas pydantic, composition des YAML
    audio     front-end log-mel (implémentation UNIQUE, train == inférence)
    data      sources → manifest → tf.data, audit qualité, corpus de vérité terrain
    models    registre d'architectures
    training  protocole multi-candidats + sélection par la validation
    evaluation  métriques par clip (contrôle) + banc streaming (décision)
    registry  champion courant et historique des promotions
    inference machine à états partagée live/banc
    serving   API FastAPI
"""

__version__ = "0.1.0"
