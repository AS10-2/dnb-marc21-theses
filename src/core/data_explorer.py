import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import seaborn as sns


class DataExplorer:
    """
    Schlanke Utility-Klasse für strukturierte DataFrame-EDA.
    Erwartet bereits korrekt typisierte Daten
    (z.B. MARC-Listen als list[dict], nicht als JSON-Strings).
    """

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()

    # --------------------------------------------------
    # Übersicht über Spalten
    # --------------------------------------------------
    def overview(self, max_elements_preview: int = 5):
        result = pd.DataFrame(
            {
                "dtype": self.df.dtypes,
                "non_null": self.df.notna().sum(),
                "missing_n": self.df.isna().sum(),
                "missing_%": (self.df.isna().mean() * 100).round(2),
            }
        )

        # Unique-Werte + Preview
        uniques_n = []
        uniques_preview = []

        for col in self.df.columns:
            series = self.df[col]

            # Listen/Dicts serialisieren für eindeutige Zählung
            if series.apply(lambda x: isinstance(x, (list, dict))).any():
                series = series.apply(str)

            uniques_n.append(series.nunique(dropna=True))

            vals = series.dropna().unique()
            if len(vals) > max_elements_preview:
                preview = list(vals[:max_elements_preview])
                preview.append(f"... (+{len(vals) - max_elements_preview})")
            else:
                preview = list(vals)

            uniques_preview.append(preview)

        result["uniques_n"] = uniques_n
        result["uniques_preview"] = uniques_preview

        return result

    # --------------------------------------------------
    # Value Counts (funktioniert auch für Listenfelder)
    # --------------------------------------------------
    def value_counts(self, column):
        if column not in self.df.columns:
            raise ValueError(f"Spalte '{column}' existiert nicht")

        series = self.df[column]

        # Listen explodieren
        if series.apply(lambda x: isinstance(x, list)).any():
            series = series.explode()

        return series.value_counts(dropna=False)

    # --------------------------------------------------
    # Missing Report (optional: leere Listen zählen)
    # --------------------------------------------------
    def missing_report(self, treat_empty_lists_as_missing=True):
        """
        Erstellt einen Bericht über fehlende Werte im DataFrame.

        Ein Wert gilt als "fehlend" (missing), wenn:
        - er NaN oder None ist
        - er eine leere Liste, Tupel oder Dict ist (optional)
        - er ein leeres numpy Array ist
        - er ein numerisches numpy Array ist, in dem alle Werte NaN sind

        Parameters
        ----------
        treat_empty_lists_as_missing : bool, default True
            Ob leere Listen, Tupel, Dicts als fehlend gezählt werden sollen.

        Returns
        -------
        pandas.DataFrame
            DataFrame mit den Spalten:
            - "missing_n": absolute Anzahl fehlender Werte pro Spalte
            - "missing_%": prozentualer Anteil fehlender Werte pro Spalte
            sortiert nach "missing_%" absteigend.
        """

        def is_missing(x):
            # numpy array zuerst abfangen
            if isinstance(x, np.ndarray):
                if x.size == 0:
                    return True
                if np.issubdtype(x.dtype, np.number):
                    return np.isnan(x).all()
                return False

            # NaN / None für normale Objekte
            if x is None:
                return True
            try:
                if pd.isna(x):
                    return True
            except Exception:
                pass  # manche Objekte lösen pd.isna aus

            if treat_empty_lists_as_missing:
                if isinstance(x, (list, tuple)):
                    return len(x) == 0
                if isinstance(x, dict):
                    return len(x) == 0

            return False

        # apply pro Spalte
        missing_n = self.df.apply(lambda col: col.apply(is_missing)).sum()
        missing_pct = (missing_n / len(self.df) * 100).round(2)

        return pd.DataFrame(
            {
                "missing_n": missing_n,
                "missing_%": missing_pct,
            }
        ).sort_values("missing_%", ascending=False)




# ======================================================
# MARC21-Spezifische Erweiterung
# ======================================================


class Marc21Explorer(DataExplorer):
    """
    Spezialisierte Analysefunktionen für MARC21-Listenfelder.
    """

    # --------------------------------------------------
    # Feldbelegung & Wiederholbarkeit
    # --------------------------------------------------
    def field_stats(self, column):
        if column not in self.df.columns:
            raise ValueError(f"Spalte '{column}' existiert nicht")

        lengths = self.df[column].apply(lambda x: len(x) if isinstance(x, list) else 0)

        total = len(lengths)
        non_empty = (lengths > 0).sum()

        return {
            "total_records": total,
            "records_with_field": non_empty,
            "presence_%": round(non_empty / total * 100, 2),
            "mean_occurrences": round(lengths.mean(), 2),
            "max_occurrences": lengths.max(),
        }

    # --------------------------------------------------
    # Subfield-Häufigkeiten (z.B. 650 $a)
    # --------------------------------------------------
    def subfield_counts(self, column, subfield):
        if column not in self.df.columns:
            raise ValueError(f"Spalte '{column}' existiert nicht")

        series = self.df[column].explode()

        values = series.apply(
            lambda x: x.get(subfield) if isinstance(x, dict) else None
        )

        return values.value_counts(dropna=False)

    # --------------------------------------------------
    # Welche Subfields existieren?
    # --------------------------------------------------
    def subfield_overview(self, column):
        if column not in self.df.columns:
            raise ValueError(f"Spalte '{column}' existiert nicht")

        series = self.df[column].explode()

        keys = series.apply(
            lambda x: list(x.keys()) if isinstance(x, dict) else []
        ).explode()

        return keys.value_counts()

    # --------------------------------------------------
    # Kontrollfeld 008 Positionsanalyse
    # --------------------------------------------------
    def analyze_008(self, start, end):
        if "control_008" not in self.df.columns:
            raise ValueError("Spalte 'control_008' existiert nicht")

        return self.df["control_008"].str[start:end].value_counts()

    # --------------------------------------------------
    # SDNB
    # --------------------------------------------------
    def analyze_sdnb_ddc_plotly(self, top_n: int = 10):
        """
        Analysiert SDNB-Codes vs DDC-Hauptklasse.

        - Anteil Dokumente mit SDNB
        - Cross-Tab der Überschneidungen
        - Heatmap mit Plotly
        - Top-N Mappings pro DDC-Hauptklasse

        Returns
        -------
        pd.DataFrame : Top-N SDNB-DDC Mappings
        """

        if (
            "has_sdnb" not in self.df.columns
            or "sdnb_ddc_mapped" not in self.df.columns
        ):
            raise ValueError("Benötigt Spalten 'has_sdnb' und 'sdnb_ddc_mapped'")

        # Anteil SDNB
        sdnb_coverage = self.df["has_sdnb"].mean()
        print(f"Anteil Dokumente mit SDNB: {sdnb_coverage:.2%}")

        # Crosstab
        ct = pd.crosstab(
            self.df["ddc_main_class"], self.df["sdnb_ddc_mapped"].explode()
        )

        if ct.empty or ct.values.sum() == 0:
            print("Keine Daten für Heatmap verfügbar.")
            return ct

        # Heatmap mit Plotly
        fig = px.imshow(
            ct,
            labels=dict(x="SDNB-DDC", y="DDC Main Class", color="Anzahl"),
            text_auto=True,
            aspect="auto",
        )

        fig.update_layout(title="DDC Main Class vs SDNB-DDC Mapping")
        fig.show()

        # Top-N Mappings
        top_mappings = {}
        for ddc_class in ct.index:
            top_mappings[ddc_class] = ct.loc[ddc_class].nlargest(top_n)

        top_df = pd.DataFrame(top_mappings).T.fillna(0).astype(int)
        print(f"Top {top_n} Mappings pro DDC Class:\n")
        print(top_df)

        return top_df

    def analyze_sdnb_ddc_diff(self, top_n: int = 10, plot: bool = True):
        """
        Zeigt SDNB-DDC-Mappings, die von der DDC Primary Class abweichen.

        - Nur Datensätze mit SDNB
        - Cross-Tab der abweichenden Kombinationen
        - Optional Heatmap
        """
        if (
            "has_sdnb" not in self.df.columns
            or "sdnb_ddc_mapped" not in self.df.columns
        ):
            raise ValueError(
                "Benötigt Spalten 'has_sdnb' und 'sdnb_ddc_mapped' im DataFrame"
            )

        # Filter: nur Datensätze mit SDNB
        df_sdnb = self.df[self.df["has_sdnb"]].copy()

        # Explodiere SDNB-Listen
        exploded = df_sdnb.explode("sdnb_ddc_mapped")

        # Filter: SDNB-DDC != DDC Primary
        mask = exploded["sdnb_ddc_mapped"] != exploded["ddc_primary_3digit"]
        exploded_diff = exploded[mask]

        # Cross-Tab
        ct = pd.crosstab(
            exploded_diff["ddc_primary_3digit"], exploded_diff["sdnb_ddc_mapped"]
        )

        if plot:
            plt.figure(figsize=(10, 6))
            sns.heatmap(ct, annot=True, fmt="d", cmap="Reds")
            plt.title("Abweichende SDNB-DDC vs DDC Primary Class")
            plt.ylabel("DDC Primary 3-digit")
            plt.xlabel("SDNB-DDC Mapped")
            plt.show()

        # Top-N SDNB pro DDC Primary
        top_mappings = {}
        for ddc_class in ct.index:
            sorted_cols = ct.loc[ddc_class].sort_values(ascending=False)
            top_mappings[ddc_class] = sorted_cols.head(top_n)

        top_df = pd.DataFrame(top_mappings).T.fillna(0).astype(int)
        print(f"Top {top_n} abweichende SDNB-DDC Mappings pro DDC Primary Class:\n")
        print(top_df)

        return top_df

# Ergänzen für DDC-Verteilung nach Zeitraum, Binary Target, Textfeld-Kombination, Top-Wörter pro Klasse
    # --------------------------------------------------
    # DDC-Verteilung nach Zeitraum
    # --------------------------------------------------
    def ddc_distribution_by_period(self, bins):
        """
        Zeigt Verteilung der ddc_primary_3digit pro Zeitsegment.
        bins: Liste von Jahresgrenzen, z.B. [1913, 1945, 1980, 2000, 2025]
        """
        if "publication_year" not in self.df.columns:
            raise ValueError("Spalte 'publication_year' fehlt")

        if "ddc_primary_3digit" not in self.df.columns:
            raise ValueError("Spalte 'ddc_primary_3digit' fehlt")

        df = self.df.copy()
        df["period"] = pd.cut(df["publication_year"], bins=bins)

        return pd.crosstab(df["period"], df["ddc_primary_3digit"])


    # --------------------------------------------------
    # Binary Classification Target erstellen
    # --------------------------------------------------
    def build_binary_target(self, positive_class, negative_class):
        """
        Filtert Datensatz auf zwei DDC-Klassen
        und erzeugt binäres Label.
        """
        df_subset = self.df[
            self.df["ddc_primary_3digit"].isin([positive_class, negative_class])
        ].copy()

        df_subset["target"] = (
            df_subset["ddc_primary_3digit"] == positive_class
        ).astype(int)

        print("Klassenverteilung:")
        print(df_subset["target"].value_counts())

        return df_subset


    # --------------------------------------------------
    # Textfeld kombinieren (Titel + Subjects)
    # --------------------------------------------------
    def build_text_column(self, columns, new_column="text_combined"):
        """
        Kombiniert mehrere Textspalten zu einem Feature-Feld.
        Listen werden automatisch zusammengeführt.
        """
        def combine(row):
            texts = []
            for col in columns:
                val = row.get(col)

                if isinstance(val, list):
                    texts.append(" ".join(map(str, val)))
                elif pd.notna(val):
                    texts.append(str(val))

            return " ".join(texts)

        self.df[new_column] = self.df.apply(combine, axis=1)
        return self.df

    # --------------------------------------------------
    # DDC-Verteilung nach Zeitraum
    # --------------------------------------------------
    def ddc_distribution_by_period(self, bins):
        """
        Zeigt Verteilung der ddc_primary_3digit pro Zeitsegment.
        bins: Liste von Jahresgrenzen, z.B. [1913, 1945, 1980, 2000, 2025]
        """
        if "publication_year" not in self.df.columns:
            raise ValueError("Spalte 'publication_year' fehlt")

        if "ddc_primary_3digit" not in self.df.columns:
            raise ValueError("Spalte 'ddc_primary_3digit' fehlt")

        df = self.df.copy()
        df["period"] = pd.cut(df["publication_year"], bins=bins)

        return pd.crosstab(df["period"], df["ddc_primary_3digit"])

    # --------------------------------------------------
    # Binary Classification Target erstellen
    # --------------------------------------------------
    def build_binary_target(self, positive_class, negative_class):
        """
        Filtert Datensatz auf zwei DDC-Klassen
        und erzeugt binäres Label.
        """
        df_subset = self.df[
            self.df["ddc_primary_3digit"].isin([positive_class, negative_class])
        ].copy()

        df_subset["target"] = (
            df_subset["ddc_primary_3digit"] == positive_class
        ).astype(int)

        print("Klassenverteilung:")
        print(df_subset["target"].value_counts())

        return df_subset

    # --------------------------------------------------
    # Textfeld kombinieren (Titel + Subjects)
    # --------------------------------------------------
    def build_text_column(self, columns, new_column="text_combined"):
        """
        Kombiniert mehrere Textspalten zu einem Feature-Feld.
        Listen werden automatisch zusammengeführt.
        """
        def combine(row):
            texts = []
            for col in columns:
                val = row.get(col)

                if isinstance(val, list):
                    texts.append(" ".join(map(str, val)))
                elif pd.notna(val):
                    texts.append(str(val))

            return " ".join(texts)

        self.df[new_column] = self.df.apply(combine, axis=1)
        return self.df

    
    # --------------------------------------------------
    # Top-Wörter pro Klasse
    # --------------------------------------------------
    def top_terms_by_class(self, text_column, target_column, top_n=20):
        from sklearn.feature_extraction.text import TfidfVectorizer

        df = self.df.dropna(subset=[text_column, target_column])

        vectorizer = TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 2),
            min_df=5
        )

        X = vectorizer.fit_transform(df[text_column])
        y = df[target_column]

        feature_names = np.array(vectorizer.get_feature_names_out())

        for label in sorted(y.unique()):
            mean_tfidf = X[y == label].mean(axis=0).A1
            top_idx = mean_tfidf.argsort()[-top_n:][::-1]

            print(f"\nTop Terms für Klasse {label}:")
            print(feature_names[top_idx])
