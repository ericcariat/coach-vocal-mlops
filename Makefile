.DEFAULT_GOAL := help
EXP ?= v03_replica
WORD ?= eloquence
MINUTES ?= 4

help:  ## Liste les cibles disponibles
	@grep -hE '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | \
		awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

install:  ## Installe les dépendances (uv)
	uv sync --all-groups

migrate:  ## Récupère les données de coach-vocal_etape1 (liens physiques)
	uv run python scripts/migrate_from_etape1.py

split:  ## Fige splits.csv (refuse d'écraser un split existant)
	uv run coachvocal data split $(WORD)

data:  ## Construit le manifest du dataset de $(EXP)
	uv run coachvocal data build $(EXP)

audit:  ## Audit qualité du dataset de $(EXP)
	uv run coachvocal data audit $(EXP)

tts:  ## Génère le pool de positifs synthétiques Piper
	uv run coachvocal data tts-pool $(WORD)

smoke:  ## Contrôle de bout en bout (~2 min)
	uv run coachvocal train smoke --no-track

train:  ## Entraîne $(EXP) (multi-candidats, élection par la validation)
	uv run coachvocal train $(EXP)

bench:  ## Banc streaming sur $(MINUTES) minutes d'audio continu
	uv run coachvocal bench --minutes $(MINUTES)

dashboard:  ## Génère le dashboard HTML comparatif
	uv run coachvocal dashboard

runs:  ## Tableau de tous les runs
	uv run coachvocal registry list

live:  ## Détection always-on au micro (champion)
	uv run coachvocal live listen

api:  ## API FastAPI (Swagger sur http://127.0.0.1:8000/docs)
	uv run coachvocal serve

ui:  ## Interface Streamlit
	uv run coachvocal ui

mlflow:  ## Interface MLflow (http://127.0.0.1:5000)
	uv run mlflow ui --backend-store-uri file://$(PWD)/artifacts/mlruns --port 5000

test:  ## Tests unitaires
	uv run pytest

lint:  ## Analyse statique
	uv run ruff check src tests app scripts

fix:  ## Corrige ce qui est corrigible automatiquement
	uv run ruff check --fix src tests app scripts

dvc-init:  ## Initialise DVC + remote local (une seule fois)
	uv run dvc init
	uv run dvc remote add -d local ../.coachvocal-dvc-store
	uv run dvc add data/wakewords/$(WORD)/clean data/wakewords/$(WORD)/raw
	@echo "→ git add data/wakewords/$(WORD)/*.dvc .dvc/config && git commit"

docker:  ## Construit l'image de service
	docker build -t coachvocal:latest .

.PHONY: help install migrate split data audit tts smoke train bench dashboard runs \
        live api ui mlflow test lint fix dvc-init docker
