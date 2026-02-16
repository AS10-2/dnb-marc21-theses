# [Datengetriebene Analyse deutschsprachiger Dissertationen] 🚀

Analyse von DNB-Hochschulschriften anhand der MARC21-Metadaten, um den Wandel fachlicher Zuordnungen im zeitlichen Verlauf zu erkennen und ältere Arbeiten prototypisch zu klassifizieren.

**Ziel:**  

## 📊 Projektübersicht

**Problemstellung:** 
Die Deutsche Nationalbibliothek (DNB) sammelt seit 1913 alle Hochschulschriften aus Deutschland. Viele ältere Datensätze sind nicht vollständig nach Fachgebieten klassifiziert, und die Zuordnungen haben sich über die Jahrzehnte verändert. Als Testfall wird das Fach Mineralogie untersucht.

**Ziel:** 
Analyse des Wandels der fachlichen Zuordnungen von Mineralogie-Hochschulschriften, Visualisierung von Trends und prototypische Retroklassifizierung älterer Datensätze auf moderne Fachgruppen.

**Methoden:**  

**Methoden:** 
- Parsing der MARC21-XML-Daten der DNB  
- Extraktion relevanter Metadaten in einen Pandas DataFrame  
- Explorative Datenanalyse (EDA) zur Identifikation von Trends und Mustern  
- Visualisierung der Ergebnisse (Zeitreihen, Balkendiagramme, Heatmaps)  
- TODO: Prototypische Retroklassifizierung älterer Dissertationen auf moderne Fachgruppen


**Links**
[Hochschulschriften auf der Website der DNB](https://www.dnb.de/dnblabdatendiss)  

[Übersicht zu den MARC21-Feldern (PDF)](https://www.dnb.de/SharedDocs/Downloads/DE/Professionell/Services/handoutInhalteInMarc.pdf?__blob=publicationFile&v=2)


## Setup

Klone das Repository
```bash
# Repository klonen
git clone https://github.com/AS10-2/dnb-marc21-theses.git
cd dnb-marc21-theses
```

Installiere [uv](https://uv.dev) (ein Tool zur Projektverwaltung und Dependency-Synchronisation) und synchronisiere die Abhängigkeiten

```bash
# Dependencies installieren
uv sync
```

### Ausführung

Führe die Notebooks in der folgenden Reihenfolge aus:

1. **`notebooks/01_exploration.ipynb`**  
   - Import der benötigten Module: `Marc21Parser` & `DataFrameAnalyzer`  
   - Einlesen der MARC21-Datei (über `parse_dnb_theses` oder direkt `Marc21Parser`)  
   - Übersicht über den DataFrame erstellen (`DataFrameAnalyzer.overview()`)  
   - Erste Visualisierungen, z. B. Jahresverteilung, Klassifikationen  

2. **`notebooks/02_preprocessing.ipynb`** *(optional)*  
   - Bereinigung fehlender Werte und Normalisierung der Klassifikationen  
   - Vorbereitung für zukünftige Klassifizierungen  
   - TODO-Zelle für die prototypische Retroklassifizierung älterer Dissertationen  

3. **`notebooks/03_modeling.ipynb`** *(optional)*  
   - Platzhalter: Modellierung oder Analyseerweiterungen  

4. **`notebooks/04_results.ipynb`** *(in progress)*  
   - Platzhalter: Ergebnisse, Visualisierungen, Zusammenfassungen

