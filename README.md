# MyPaper2Code

**MyPaper2Code** est un assistant local de compréhension et de reproduction d'articles scientifiques en intelligence artificielle.

L'objectif du projet est d'aider un utilisateur à passer d'un papier de recherche à une implémentation structurée, explicable et partiellement vérifiable, tout en conservant une traçabilité claire entre les décisions de génération de code, les contraintes utilisateur et le contenu du papier.

Le projet ne vise pas à transformer automatiquement n'importe quel article en une reproduction parfaite. Il cherche plutôt à fournir un environnement assisté permettant de comprendre un papier, d'interroger son contenu, de construire un plan d'implémentation, d'imposer des contraintes techniques, puis de générer progressivement une base de code cohérente dans un workspace dédié.

## Objectif

La reproduction d'un article scientifique est souvent complexe : les papiers omettent certains détails d'implémentation, les hyperparamètres ne sont pas toujours complets, les architectures sont parfois décrites de manière ambiguë et les choix expérimentaux peuvent être dispersés entre le corps principal, les tableaux, les figures et les annexes.

**MyPaper2Code** a pour objectif de réduire cette friction en proposant un pipeline complet :

1. ingestion d'un article scientifique ;
2. extraction et structuration de ses sections importantes ;
3. indexation du contenu pour permettre des questions/réponses sourcées ;
4. analyse de la méthode proposée ;
5. génération d'un plan d'implémentation ;
6. prise en compte de contraintes utilisateur ;
7. création d'un workspace dédié ;
8. génération progressive d'un squelette de projet ;
9. validation minimale du code généré ;
10. production d'un rapport d'hypothèses et de fidélité au papier.

L'idée centrale est de fournir un assistant capable d'accompagner l'utilisateur dans la reproduction d'un papier, tout en signalant clairement les zones d'incertitude ou les choix arbitraires nécessaires.

## Cas d'usage

Un utilisateur peut fournir un article sous forme de PDF ou de lien arXiv, puis interagir avec le système pour :

- poser des questions sur la méthode ;
- retrouver la fonction de perte utilisée ;
- identifier les datasets et métriques d'évaluation ;
- comprendre l'architecture du modèle ;
- extraire les hyperparamètres disponibles ;
- demander un plan d'implémentation ;
- imposer un framework ou une structure de projet ;
- choisir le modèle LLM utilisé pour l'analyse et la génération ;
- générer un squelette de code PyTorch ;
- créer un workspace isolé pour l'implémentation ;
- obtenir un rapport sur les hypothèses prises par l'agent.

Exemples de requêtes utilisateur :

```text
Quelle est l'architecture du modèle proposé ?

Quels sont les datasets utilisés dans les expériences ?

Génère un plan d'implémentation en PyTorch.

Implémente une version minimale compatible avec CIFAR-10.

Utilise une configuration YAML et sépare le code en modules data, model, training et evaluation.

Utilise le modèle local llama3 avec Ollama.

Utilise un modèle NVIDIA Build pour la génération de code.

Quelles parties du papier ne sont pas assez précises pour une reproduction fidèle ?
```

## Fonctionnalités principales

### 1. Ingestion de papier

Le système permet d'importer un papier scientifique à partir d'un fichier PDF ou d'une URL.

L'étape d'ingestion extrait le texte et tente de structurer le document en sections logiques :

- abstract ;
- introduction ;
- related work ;
- method ;
- experiments ;
- results ;
- appendix ;
- références ;
- tableaux et légendes de figures lorsque disponibles.

Cette structuration permet ensuite au système de différencier les informations générales des informations réellement utiles à l'implémentation.

### 2. Recherche et questions/réponses sourcées

Une fois le papier indexé, l'utilisateur peut poser des questions en langage naturel.

Le système s'appuie sur un mécanisme de retrieval-augmented generation afin de répondre en citant les passages pertinents du papier.

L'objectif est d'éviter les réponses non vérifiables. Chaque réponse doit idéalement être accompagnée de références vers :

- la section du papier ;
- la page ;
- le passage utilisé ;
- éventuellement une figure ou un tableau.

Cette étape transforme le papier en base de connaissances exploitable pour l'utilisateur et pour les agents de génération de code.

### 3. Analyse de la méthode

Le système tente d'identifier les composants nécessaires à l'implémentation :

- architecture du modèle ;
- modules principaux ;
- fonction de perte ;
- stratégie d'entraînement ;
- datasets utilisés ;
- métriques d'évaluation ;
- hyperparamètres ;
- prétraitements ;
- détails d'optimisation ;
- protocole expérimental.

Lorsque certaines informations sont absentes ou ambiguës, elles sont marquées comme incertaines.

Le système ne doit pas masquer ces incertitudes : elles sont conservées pour être affichées dans le rapport final.

### 4. Plan d'implémentation

À partir de l'analyse du papier, MyPaper2Code génère un plan d'implémentation structuré.

Ce plan peut inclure :

```text
project/
├── configs/
│   └── default.yaml
├── src/
│   ├── data/
│   ├── models/
│   ├── losses/
│   ├── training/
│   ├── evaluation/
│   └── utils/
├── tests/
├── scripts/
│   ├── train.py
│   └── evaluate.py
├── README.md
└── requirements.txt
```

Le plan décrit le rôle de chaque fichier, les composants à implémenter et les dépendances entre modules.

### 5. Contraintes utilisateur

L'utilisateur peut imposer des contraintes afin d'adapter l'implémentation à ses besoins.

Exemples de contraintes :

```yaml
framework: pytorch
style: research
dataset: cifar10
config_format: yaml
target_gpu_memory: 8GB
implementation_level: minimal
include_tests: true
include_training_script: true
include_evaluation_script: true
provider: ollama
model: llama3
```

Ces contraintes permettent de générer une implémentation plus adaptée au contexte réel de l'utilisateur.

Par exemple, un utilisateur peut demander :

- une version minimale pour comprendre l'idée ;
- une version plus proche du papier original ;
- une version compatible avec un dataset différent ;
- une structure orientée recherche ;
- une structure plus production-ready ;
- une implémentation avec tests et configuration reproductible ;
- l'utilisation d'un modèle local via Ollama ;
- l'utilisation d'un modèle disponible via NVIDIA Build.

### 6. Choix du provider LLM

MyPaper2Code est conçu pour être compatible avec plusieurs fournisseurs de modèles.

L'utilisateur peut choisir le provider et le modèle utilisés pour :

- l'analyse du papier ;
- les questions/réponses sourcées ;
- la génération du plan d'implémentation ;
- la génération du code ;
- la revue automatique ;
- la génération du rapport d'hypothèses.

Deux modes principaux sont prévus :

#### Ollama

Le mode Ollama permet d'utiliser des modèles exécutés localement.

Ce mode est utile pour :

- travailler hors ligne ;
- éviter d'envoyer le contenu du papier à un service externe ;
- tester différents modèles open source ;
- garder un environnement de développement entièrement local.

Exemple :

```bash
mypaper2code config set provider ollama
mypaper2code config set model llama3
```

#### NVIDIA Build

Le mode NVIDIA Build permet d'utiliser des modèles accessibles via la plateforme `https://build.nvidia.com/`.

Ce mode est utile pour :

- tester des modèles performants hébergés ;
- comparer différents modèles sur les mêmes tâches ;
- utiliser des modèles spécialisés pour le raisonnement, le code ou la compréhension de documents ;
- accélérer certaines étapes comme l'analyse, la planification ou la génération de code.

Exemple :

```bash
mypaper2code config set provider nvidia
mypaper2code config set model <model-name>
```

Le système doit permettre de changer de modèle facilement selon la tâche. Par exemple, l'utilisateur peut choisir un modèle pour la compréhension du papier et un autre pour la génération de code.

### 7. Création d'un workspace

Lorsqu'un papier est ingéré et qu'un plan d'implémentation est validé, l'agent crée un **workspace dédié**.

Ce workspace sert d'environnement de travail isolé dans lequel l'agent peut progressivement implémenter le papier.

Exemple de structure :

```text
workspaces/
└── paper_name_timestamp/
    ├── paper/
    │   ├── original.pdf
    │   ├── extracted_sections.json
    │   └── chunks.json
    ├── analysis/
    │   ├── method_summary.md
    │   ├── implementation_plan.md
    │   ├── assumptions.md
    │   └── requirements.yaml
    ├── generated_code/
    │   ├── configs/
    │   ├── src/
    │   ├── tests/
    │   ├── scripts/
    │   └── README.md
    ├── runs/
    │   ├── lint.log
    │   ├── tests.log
    │   └── dry_run.log
    └── metadata.json
```

Le workspace permet de conserver :

- le papier original ;
- les chunks indexés ;
- les réponses aux questions importantes ;
- le plan d'implémentation ;
- les contraintes utilisateur ;
- le code généré ;
- les logs de validation ;
- les hypothèses prises par l'agent.

L'objectif est de rendre la génération traçable et itérative. L'utilisateur peut inspecter, modifier ou relancer certaines étapes sans perdre l'historique du projet.

### 8. Génération de code

Le système génère progressivement les fichiers du projet.

La génération ne se fait pas comme une réponse unique et monolithique, mais comme une construction incrémentale :

1. génération de la structure du workspace ;
2. génération des fichiers de configuration ;
3. génération des modules de données ;
4. génération du modèle ;
5. génération de la fonction de perte ;
6. génération de la boucle d'entraînement ;
7. génération de l'évaluation ;
8. génération des tests ;
9. génération du README d'implémentation.

Chaque fichier généré doit être lié, lorsque c'est possible, aux éléments du papier qui justifient son contenu.

### 9. Validation du code généré

MyPaper2Code intègre une étape de validation minimale afin de détecter les erreurs évidentes.

Cette étape peut inclure :

- vérification des imports ;
- linting avec Ruff ;
- exécution de tests unitaires ;
- dry-run sur des données factices ;
- vérification de la cohérence entre configuration et code ;
- détection de fichiers incomplets ou de fonctions non implémentées.

L'objectif n'est pas de garantir une reproduction scientifique complète, mais de produire un squelette de projet exécutable et améliorable.

### 10. Rapport d'hypothèses et de fidélité

Un des éléments centraux du projet est la génération d'un rapport final.

Ce rapport documente :

- les sections du papier utilisées ;
- les composants implémentés ;
- les détails explicitement présents dans le papier ;
- les hypothèses prises par l'agent ;
- les éléments manquants ou ambigus ;
- les écarts avec la méthode originale ;
- les suggestions pour améliorer la reproduction ;
- le niveau de confiance global.

Exemple :

```text
Assumption Report

Implemented:
- Model backbone
- Training loop
- Loss function
- Evaluation metrics

Unclear in paper:
- Exact learning rate schedule
- Batch size for ablation experiments
- Data augmentation details

Assumptions made:
- AdamW optimizer used by default
- Batch size set to 64
- Cosine scheduler selected for training stability

Fidelity level:
- Medium
```

Ce rapport rend le système plus transparent et plus crédible. Il évite de présenter le code généré comme une reproduction exacte lorsque le papier ne fournit pas tous les détails nécessaires.

## Architecture proposée

Le système peut être organisé autour de plusieurs agents spécialisés.

### PaperParser

Responsable de l'extraction du contenu du papier.

Il extrait :

- texte brut ;
- sections ;
- titres ;
- tableaux ;
- légendes de figures ;
- références utiles.

### PaperRetriever

Responsable de la recherche d'information dans le papier.

Il permet de répondre aux questions utilisateur avec des sources précises.

### MethodAnalyzer

Responsable de l'analyse technique du papier.

Il identifie :

- architecture ;
- loss ;
- datasets ;
- métriques ;
- protocole expérimental ;
- détails d'entraînement ;
- éléments nécessaires à l'implémentation.

### ImplementationPlanner

Responsable de la transformation de l'analyse en plan de projet.

Il propose :

- une arborescence ;
- les fichiers à créer ;
- les modules à implémenter ;
- les dépendances entre composants.

### RequirementResolver

Responsable de l'application des contraintes utilisateur.

Il adapte le plan selon :

- le framework ;
- le dataset ;
- le style de code ;
- les limites matérielles ;
- le niveau de fidélité souhaité ;
- le provider LLM choisi ;
- le modèle sélectionné.

### WorkspaceManager

Responsable de la création et de la gestion du workspace.

Il conserve :

- les fichiers du papier ;
- les analyses intermédiaires ;
- les requirements ;
- les fichiers générés ;
- les logs ;
- les métadonnées du projet.

### CodeWriter

Responsable de la génération de code.

Il génère les fichiers un par un, en respectant le plan d'implémentation et les contraintes utilisateur.

### CodeReviewer

Responsable de la revue automatique.

Il vérifie :

- la cohérence globale ;
- les imports ;
- les signatures de fonctions ;
- les TODOs ;
- les erreurs probables ;
- l'alignement avec les requirements.

### ExperimentRunner

Responsable de l'exécution contrôlée.

Il peut lancer :

- tests unitaires ;
- lint ;
- dry-run ;
- scripts d'entraînement minimaux.

## Stack technique envisagée

### Environnement de développement

Le projet utilise **uv** pour gérer l'environnement Python, les dépendances et l'exécution des commandes.

L'utilisation de `uv` permet de disposer d'un environnement reproductible, rapide à installer et simple à manipuler.

Exemples :

```bash
uv sync
uv run mypaper2code --help
uv run pytest
uv run ruff check .
```

### Backend

- Python
- FastAPI
- Pydantic
- SQLAlchemy ou stockage JSON pour le MVP

### Parsing PDF

- PyMuPDF
- éventuellement `unstructured` ou `marker` pour une extraction plus avancée

### RAG

- Chroma, Qdrant ou FAISS
- SentenceTransformers ou embeddings locaux
- BM25 pour recherche lexicale
- reranker optionnel

### Agents et LLM

- orchestration maison ou LangGraph
- support Ollama local
- support NVIDIA Build
- sélection configurable du modèle
- prompts versionnés

### Génération et validation de code

- génération de fichiers multi-modules
- Ruff
- pytest
- exécution contrôlée via subprocess
- sandbox Docker dans une version avancée

### Interface

- CLI pour le MVP
- API FastAPI
- interface web légère dans une version ultérieure

## Exemple de workflow CLI

```bash
uv run mypaper2code ingest paper.pdf

uv run mypaper2code ask "Quelle est la fonction de perte utilisée ?"

uv run mypaper2code plan \
  --framework pytorch \
  --dataset cifar10 \
  --style research \
  --config yaml \
  --provider ollama \
  --model llama3

uv run mypaper2code generate \
  --workspace ./workspaces/my_paper

uv run mypaper2code validate ./workspaces/my_paper
```

## Exemple de workflow API

```text
POST /papers
POST /papers/{paper_id}/ask
POST /papers/{paper_id}/analyze
POST /papers/{paper_id}/plan
POST /papers/{paper_id}/workspace
POST /papers/{paper_id}/generate
POST /papers/{paper_id}/validate
```

## Limites assumées

MyPaper2Code ne prétend pas résoudre automatiquement tous les problèmes liés à la reproduction scientifique.

Certaines limites sont assumées :

- les papiers peuvent être incomplets ;
- certaines équations peuvent être difficiles à extraire proprement ;
- les figures et tableaux peuvent nécessiter une interprétation approximative ;
- les résultats expérimentaux peuvent dépendre de détails non publiés ;
- le code généré peut nécessiter une revue humaine ;
- les choix par défaut doivent être explicitement documentés ;
- les performances obtenues peuvent différer de celles du papier original.

Le système est donc conçu comme un assistant de reproduction, et non comme un générateur parfait.

## Vision du projet

À terme, MyPaper2Code pourrait devenir un environnement complet pour explorer, comprendre et prototyper rapidement des articles de recherche en IA.

L'objectif est de permettre à un utilisateur de passer plus rapidement de la lecture d'un papier à une première implémentation expérimentale, tout en conservant une traçabilité claire entre le papier original, les choix techniques, les contraintes utilisateur et le code généré.

Le projet met l'accent sur quatre principes :

1. **Transparence** : chaque décision importante doit être expliquée ou reliée au papier.
2. **Contrôle utilisateur** : l'utilisateur peut imposer ses contraintes et orienter la génération.
3. **Modularité** : le provider LLM, le modèle, le framework et la structure du projet doivent rester configurables.
4. **Validation progressive** : le code généré doit être vérifiable, testable et améliorable.

MyPaper2Code se positionne ainsi comme un pont entre lecture scientifique, compréhension assistée par LLM et prototypage logiciel reproductible.
