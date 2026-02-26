# Datengetriebene Analyse deutschsprachiger Dissertationen 🚀

Analyse von DNB-Hochschulschriften anhand der MARC21-Metadaten, um den Wandel fachlicher Zuordnungen im zeitlichen Verlauf zu erkennen und ältere Arbeiten prototypisch neu zu klassifizieren.

**Ziel:**  

## 📊 Projektübersicht

**Problemstellung:** 
Die Deutsche Nationalbibliothek (DNB) sammelt seit 1913 alle Dissertationen aus Deutschland. Die DNB hat die Klassifikationssysteme zweimal gewechselt — von keiner Klassifikation (vor 1970) über SDNB (1970–2003) zu DDC (ab 2003). Diese Systemwechsel sind im Korpus empirisch nachweisbar. Durch die verschiedenen Klassifikationssystem und den fehlenden Klassifikationen lassen sich Trends im wissenschaftlichen Diskurs nur schwer nachvollziehen. Als Testfall wwerden das Fach Mineralogie, die Lagerstättekunde und die Geowissenschaften untersucht.

**Ziel:** 
Analyse des Wandels der fachlichen Zuordnungen von Mineralogie-Hochschulschriften, Visualisierung von Trends und prototypische Retroklassifizierung für die von 1945 bis 1970 publizierten Dissertationen.

**Ergebnisse**
Analysierter Korpus: 1,6 Mio. DNB-Hochschulschriften
Unklassifizierte Records: 149.000 (9,2 %)
Retro-Kandidaten (Geowiss.) 896
F1-Score (5-fold CV)0,882 ± 0,004ROC-AUC0,952 ± 0,002
Retroklassifiziert (prob ≥ 0,5)692
Hoch-konfident (prob ≥ 0,8)307  

**Methoden:** 
- Parsing der MARC21-XML-Daten der DNB  
- Extraktion relevanter Metadaten in einen Pandas DataFrame  
- Explorative Datenanalyse (EDA) zur Identifikation von Trends und Mustern  
- Visualisierung der Ergebnisse mit Plotly
- Prototypische Retroklassifizierung mit TF-IDF + LogisticRegression (zukünftige Erweiterung)


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
   - Import der benötigten Module: `Marc21Parser` & `DataExplorer`  
   - Einlesen der MARC21-Datei  
   - Übersicht über den DataFrame
   - Erste Visualisierungen: Jahresverteilung und Klassifikationen  

2. **`notebooks/02_preprocessing.ipynb`** 
   - Bereinigung fehlender Werte und Normalisierung der Klassifikationen  
   - Vorbereitung für zukünftige Klassifizierungen  
   - Marc21-Felder 082, 083, 084
   -  

3. **`notebooks/03_modeling.ipynb`**  
   - Modellierung mit TF-IDF und Logistischer Regression
   - Evaluierung u.a. mit Cross-Validation und Lernkurve 

4. **`notebooks/04_results.ipynb`** *(in progress)*  
   - Ergebnisse, Visualisierungen, Zusammenfassungen

### Lizenz: 
GNU General Public License v3.0 

### Kontakt: 
angelastrauss10[at]gmail.com

