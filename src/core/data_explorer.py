import numpy as np
import pandas as pd


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
