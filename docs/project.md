# Data Science Projekt Template - Dokumentation

## Überblick

Dieses Template bietet eine strukturierte Grundlage für Data Science und Data Analytics Projekte. Es folgt Best Practices für Organisation, Reproduzierbarkeit und kollaboratives Arbeiten.

## Projektstruktur

```
template/
├── .git/                       # Git Repository
├── .venv/                      # Virtuelle Python-Umgebung
├── .vscode/                    # VS Code Konfiguration
│   ├── extensions.json         # Empfohlene VS Code Extensions
│   └── settings.json           # Workspace-spezifische Einstellungen
├── data/                       # Datenverzeichnis (nicht versioniert)
│   ├── raw/                    # Rohdaten
│   └── processed/              # Verarbeitete Daten
├── docs/                       # Dokumentation
│   └── project.md              # Projektbeschreibung und Anleitungen
├── notebooks/                  # Jupyter Notebooks
│   └── 01_exploration.ipynb    # Beispiel-Notebook
├── src/                        # Python Source Code
│   └── core/
│       ├── __init__.py         # Macht core zu einem Package
│       └── data.py             # Datenlade-Funktionen
├── .gitignore                  # Git Ignore Regeln
├── .python-version             # Python Version für uv
├── pyproject.toml              # Projekt-Konfiguration
├── README.md                   # Projekt-Beschreibung
└── uv.lock                     # Abhängigkeiten Lock-File
```

## Verzeichnisse und Dateien

### `.vscode/`

Enthält VS Code spezifische Konfigurationen:

- **`extensions.json`**: Empfohlene VS Code Extensions für das Projekt (z.B. Python, Jupyter)
- **`settings.json`**: Workspace-spezifische Einstellungen

### `data/`

Das Datenverzeichnis ist durch `.gitignore` vom Versionskontrollsystem ausgeschlossen und sollte **nicht** ins Repository committed werden.

- **`raw/`**: Ursprüngliche, unveränderte Rohdaten
- **`processed/`**: Bereinigte, transformierte oder aggregierte Daten

**Wichtig**: Rohdaten sollten niemals überschrieben werden. Alle Transformationen werden in verarbeitete Daten gespeichert.

### `notebooks/`

Jupyter Notebooks für explorative Datenanalyse und Experimente.

- **`01_exploration.ipynb`**: Beispiel-Notebook zum Testen des Setups
  - Lädt einen Datensatz von Kaggle
  - Zeigt erste explorative Analysen
  - Dient als Vorlage für weitere Notebooks

**Naming Convention**: Nummerierung mit Präfix (01*, 02*, ...) für chronologische Reihenfolge.

### `src/core/`

Python-Module mit wiederverwendbarem Code, der aus Notebooks ausgelagert wird.

- **`__init__.py`**: Macht `core` zu einem Python-Package
- **`data.py`**: Funktionen zum Laden und Verarbeiten von Daten (z.B. Kaggle Downloads)

**Best Practice**: Code, der in mehreren Notebooks verwendet wird, sollte in Module ausgelagert werden.

### `pyproject.toml`

Zentrale Konfigurationsdatei für das Python-Projekt. Sie definiert Metadaten und Abhängigkeiten.

#### Projekt-Metadaten anpassen

```toml
[project]
name = "dpp-template"                       # Ändern Sie dies auf Ihren Projektnamen
version = "0.1.0"                           # Versionsnummer
description = "Add your description here"   # Kurze Projektbeschreibung
readme = "README.md"                        # Pfad zur README-Datei
requires-python = ">=3.14"                  # Unterstützte Python-Version
```

*Passe diese Felder an dein Projekt an!*

#### Abhängigkeiten

Die `dependencies` Liste enthält alle benötigten Python-Pakete:

```toml
dependencies = [
    "ipykernel>=7.0.1",      # Jupyter Kernel
    "kagglehub>=0.3.13",     # Kaggle Integration
    "matplotlib>=3.10.7",    # Plotting
    "pandas>=2.3.3",         # Datenanalyse
    "seaborn>=0.13.2",       # Visualisierung
    "duckdb>=1.4.1",         # In-Memory Datenbank für Analysen mit SQL
    "scikit-learn>=1.7.2",   # Machine Learning
]
```

> [**ipykernel**](https://ipykernel.readthedocs.io/)  
  Stellt die Verbindung zwischen Jupyter Notebooks und dem Python-Kernel her. Ermöglicht die interaktive Ausführung von Python-Code in Jupyter-Umgebungen.

> [**kagglehub**](https://github.com/Kaggle/kagglehub)  
  Offizielles Python-Package für die Integration mit Kaggle, ermöglicht den einfachen Zugriff auf Kaggle-Datasets und -Modelle. Vereinfacht das Herunterladen und Verwalten von Kaggle-Ressourcen direkt aus Python-Code.

> [**matplotlib**](https://matplotlib.org/stable/index.html)  
  Die grundlegende Plotting-Bibliothek für Python, bietet umfangreiche Möglichkeiten zur Erstellung statischer, animierter und interaktiver Visualisierungen. Dient als Basis für viele andere Visualisierungsbibliotheken.

> [**pandas**](https://pandas.pydata.org/docs/)  
  Die zentrale Bibliothek für Datenanalyse in Python mit leistungsstarken Datenstrukturen wie DataFrame und Series. Bietet intuitive Werkzeuge für Datenmanipulation, -bereinigung und -analyse.

> [**seaborn**](https://seaborn.pydata.org/)  
  High-Level-Visualisierungsbibliothek basierend auf matplotlib, spezialisiert auf statistische Grafiken. Ermöglicht die Erstellung ästhetisch ansprechender und informativer Visualisierungen mit weniger Code.

> [**duckdb**](https://duckdb.org/docs/)  
  Hochperformante In-Memory SQL-Datenbank, optimiert für analytische Workloads (OLAP). Ermöglicht SQL-Abfragen direkt auf DataFrames und CSV-Dateien ohne externe Datenbankserver.

> [**scikit-learn**](https://scikit-learn.org/stable/documentation.html)  
  Die umfassendste Machine-Learning-Bibliothek für Python mit Algorithmen für Klassifikation, Regression, Clustering und mehr. Bietet eine einheitliche API und umfangreiche Tools für Modelltraining, -evaluation und -präprozessierung.

#### Build System

Das Build System ermöglicht es, den `src/core` Code als importierbares Python-Package zu verwenden:

```toml
[build-system]
requires = ["uv_build>=0.8.9,<0.9.0"]
build-backend = "uv_build"

[tool.uv.build-backend]
module-name = "core"
```

Dadurch kannst du in Notebooks `from core.data import ...` verwenden.

### `.gitignore`

Definiert Dateien und Ordner, die nicht ins Git Repository committed werden sollen:

- Virtuelle Umgebungen (`.venv/`)
- Datenverzeichnisse (`data/`)
- Cache-Dateien (`__pycache__/`, `*.pyc`)
- IDE-spezifische Dateien
- Wurde mit [gitignore.io](https://www.toptal.com/developers/gitignore) generiert

### `uv.lock`

Lock-File, das exakte Versionen aller Abhängigkeiten festhält. Gewährleistet Reproduzierbarkeit.

**Wichtig**: Diese Datei sollte ins Repository committed werden!

## Dependency Management mit UV

UV ist ein moderner, schneller Python Package Manager. Hier sind die wichtigsten Befehle:

### Pakete hinzufügen

```bash
uv add <paketname>
```

Beispiel:

```bash
uv add numpy
uv add plotly>=5.0.0
```

Dies fügt das Paket zu `pyproject.toml` hinzu und aktualisiert `uv.lock`.

### Pakete entfernen

```bash
uv remove <paketname>
```

Beispiel:

```bash
uv remove numpy
```

### Abhängigkeiten synchronisieren

```bash
uv sync
```

Installiert alle in `pyproject.toml` definierten Pakete und bringt die virtuelle Umgebung auf den aktuellen Stand.

**Wichtig**: Führe `uv sync` aus nach:

- Clone eines Repositories
- Änderungen an `pyproject.toml`
- Pull von Updates

## Workflow

### Projekt-Setup

1. Repository klonen oder Template kopieren
2. Abhängigkeiten installieren: `uv sync`
3. Öffne `01_exploration.ipynb` in VS Code und Wähle oben rechts dein Environment als Kernel
4. `01_exploration.ipynb` ausführen, um Setup zu testen.
5. Bei Problemen an Mentoring wenden

### Entwicklung

1. Analysen in Notebooks durchführen
2. Wiederverwendbaren Code in `src/core/` Module auslagern
3. Daten in `data/raw/` ablegen (nicht committen!)
4. Verarbeitete Daten in `data/processed/` speichern
5. Regelmäßig committen mit aussagekräftigen Commit-Messages
6. Mindestens einmal pro Tag Änderungen auf GitHub pushen

### Kollaboration

1. Regelmäßig pushen und pullen um Konflikte zu vermeiden
2. Nach Änderungen an `pyproject.toml` bzw. `uv.lock` immer `uv sync` ausführen

## Best Practices

- **Daten nicht versionieren**: Große Datensätze gehören nicht ins Git Repository
- **Reproduzierbarkeit**: `uv.lock` committen für identische Umgebungen
- **Code-Organisation**: Wiederholter Code → Module, Einmalige Analysen → Notebooks
- **Dokumentation**: README und Docstrings aktuell halten
- **Naming**: Aussagekräftige Namen für Notebooks und Module
