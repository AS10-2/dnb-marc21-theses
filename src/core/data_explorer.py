"""
explorer.py

EDA und Visualisierung für DNB-Hochschulschriften.

Funktioniert defensiv auf allen Pipeline-Stufen:
    df_raw        → overview, missing_report, field_stats
    df_clean      → zusätzlich value_counts, subfield_counts
    df_transformed → zusätzlich alle Klassifikations-Analysen

Pipeline:
    Marc21Parser → Cleaner → ClassificationTransformer → Explorer
"""

from typing import List

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.feature_extraction.text import TfidfVectorizer


# ── DDC-Hauptklassen ──────────────────────────────────────────
DDC_MAIN = {
    "0": "Allgemeines",
    "1": "Philosophie",
    "2": "Religion",
    "3": "Sozialwissenschaften",
    "4": "Sprache",
    "5": "Naturwissenschaften",
    "6": "Technik",
    "7": "Kunst",
    "8": "Literatur",
    "9": "Geschichte",
}


# ======================================================
# Basis-Explorer — funktioniert auf df_raw
# ======================================================


class DataExplorer:
    """Generische EDA-Utilities für beliebige DataFrames."""

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()

    # --------------------------------------------------
    # Spaltenübersicht
    # --------------------------------------------------
    def overview(self, max_preview: int = 5) -> pd.DataFrame:
        result = pd.DataFrame(
            {
                "dtype": self.df.dtypes,
                "non_null": self.df.notna().sum(),
                "missing_n": self.df.isna().sum(),
                "missing_%": (self.df.isna().mean() * 100).round(2),
            }
        )
        uniques_n, uniques_preview = [], []
        for col in self.df.columns:
            s = self.df[col]
            if s.apply(lambda x: isinstance(x, (list, dict))).any():
                s = s.apply(str)
            n = s.nunique(dropna=True)
            uniques_n.append(n)
            vals = s.dropna().unique()
            preview = list(vals[:max_preview])
            if len(vals) > max_preview:
                preview.append(f"... (+{len(vals) - max_preview})")
            uniques_preview.append(preview)
        result["uniques_n"] = uniques_n
        result["uniques_preview"] = uniques_preview
        return result

    # --------------------------------------------------
    # Missing Report
    # --------------------------------------------------
    def missing_report(self, treat_empty_lists_as_missing: bool = True) -> pd.DataFrame:
        def is_missing(x):
            if isinstance(x, np.ndarray):
                return x.size == 0 or (
                    np.issubdtype(x.dtype, np.number) and np.isnan(x).all()
                )
            if x is None:
                return True
            if treat_empty_lists_as_missing and isinstance(x, (list, tuple, dict)):
                return len(x) == 0
            try:
                return bool(pd.isna(x))
            except (ValueError, TypeError):
                return False

        missing_n = self.df.apply(lambda col: col.apply(is_missing)).sum()
        missing_pct = (missing_n / len(self.df) * 100).round(2)
        return pd.DataFrame(
            {
                "missing_n": missing_n,
                "missing_%": missing_pct,
            }
        ).sort_values("missing_%", ascending=False)

    # --------------------------------------------------
    # Value Counts (auch für Listenfelder)
    # --------------------------------------------------
    def value_counts(self, column: str) -> pd.Series:
        self._require(column)
        s = self.df[column]
        if s.apply(lambda x: isinstance(x, list)).any():
            s = s.explode()
        return s.value_counts(dropna=False)

    # --------------------------------------------------
    # Memory Report
    # --------------------------------------------------
    def memory_report(self) -> pd.DataFrame:
        return (
            self.df.dtypes.to_frame("dtype")
            .join(self.df.memory_usage(deep=True).rename("bytes"))
            .assign(MB=lambda d: (d["bytes"] / 1024**2).round(2))
            .sort_values("MB", ascending=False)
        )

    def _require(self, *cols):
        for col in cols:
            if col not in self.df.columns:
                raise ValueError(f"Spalte '{col}' nicht vorhanden")


# ======================================================
# MARC21-Explorer — df_clean und df_transformed
# ======================================================


class Marc21Explorer(DataExplorer):
    """
    Spezialisierte Analyse für MARC21-Daten.
    Methoden prüfen defensiv ob benötigte Spalten vorhanden sind.
    """

    # --------------------------------------------------
    # Feldbelegung für Listenfelder
    # --------------------------------------------------
    def field_stats(self, column: str) -> dict:
        self._require(column)
        lengths = self.df[column].apply(lambda x: len(x) if isinstance(x, list) else 0)
        total = len(lengths)
        non_empty = (lengths > 0).sum()
        return {
            "total_records": total,
            "records_with_field": int(non_empty),
            "presence_%": round(non_empty / total * 100, 2),
            "mean_occurrences": round(lengths.mean(), 2),
            "max_occurrences": int(lengths.max()),
        }

    # --------------------------------------------------
    # Subfield-Analyse (für _list-Felder aus df_raw)
    # --------------------------------------------------
    def subfield_counts(self, column: str, subfield: str) -> pd.Series:
        self._require(column)
        series = self.df[column].explode()
        values = series.apply(
            lambda x: x.get(subfield) if isinstance(x, dict) else None
        )
        return values.value_counts(dropna=False)

    def subfield_overview(self, column: str) -> pd.Series:
        self._require(column)
        keys = (
            self.df[column]
            .explode()
            .apply(lambda x: list(x.keys()) if isinstance(x, dict) else [])
            .explode()
        )
        return keys.value_counts()

    # --------------------------------------------------
    # Kontrollfeld 008
    # --------------------------------------------------
    def analyze_008(self, start: int, end: int) -> pd.Series:
        self._require("control_008")
        return self.df["control_008"].str[start:end].value_counts()

    # --------------------------------------------------
    # Klassifikationsabdeckung nach Jahr — Kern-Chart
    # --------------------------------------------------
    def plot_coverage_by_year(self, year_min: int = 1913, year_max: int = 2024):
        self._require("publication_year", "ddc_primary_3digit", "has_sdnb")

        df = self.df.copy()
        df["_year"] = pd.to_numeric(df["publication_year"], errors="coerce")
        df = df[df["_year"].between(year_min, year_max)]
        df["_year"] = df["_year"].astype(int)

        has_ddc = df["ddc_primary_3digit"].fillna("").str.len() > 0
        has_sdnb = df["has_sdnb"].fillna(False)

        df["_kat"] = np.select(
        [has_ddc, has_sdnb],
        ["DDC (082/083)", "SDNB (084)"],
        default="Keine Klassifikation"
    )

        by_year = (
            df.groupby(["_year", "_kat"])
            .size()
            .reset_index(name="count")
        )

        fig = px.bar(
        by_year,
        x="_year",
        y="count",
        color="_kat",
        color_discrete_map={
            "DDC (082/083)": "#1E88E5",
            "SDNB (084)": "#FF9800",
            "Keine Klassifikation": "#CFD8DC",
        },
        barmode="stack",
        title="Klassifikationsabdeckung DNB-Hochschulschriften",
        labels={"_year": "Erscheinungsjahr", "count": "Anteil (%)", "_kat": ""},
        category_orders={
            "_kat": ["Keine Klassifikation", "SDNB (084)", "DDC (082/083)"]
        },
    )

        # Prozentuale Stappelung
        fig.update_layout(
            plot_bgcolor="white",
            height=500,
            barnorm="percent",
            yaxis=dict(ticksuffix="%", range=[0, 100]),
            legend=dict(orientation="h", y=1.08, x=0.5, xanchor="center"),
        )

        fig.update_xaxes(showgrid=True, gridcolor="#EEEEEE", dtick=5, tickangle=-45)
        fig.update_yaxes(showgrid=True, gridcolor="#EEEEEE")

        return fig

    # --------------------------------------------------
    # Mineralogie nach Jahrzehnt
    # --------------------------------------------------
    def plot_mineralogie_by_decade(self) -> go.Figure:
        """Mineralogie-Records nach Jahrzehnt, gefärbt nach Klassifikationsstatus."""
        self._require(
        "publication_year",
        "is_mineralogie",
        "ddc_primary_3digit",
        "has_sdnb",
        )
        df = self.df[self.df["is_mineralogie"]].copy()
        df["_year"] = pd.to_numeric(df["publication_year"], errors="coerce")
        df = df[df["_year"].between(1940, 2024)]
        df["decade"] = (df["_year"] // 10 * 10).astype(int)

        def klass_status(row):
            has_ddc = (
                isinstance(row.get("ddc_primary_3digit"), str)
                and len(row["ddc_primary_3digit"]) > 0
            )
            has_sdnb = row.get("has_sdnb", False)
            if has_ddc:
                return "DDC (082/083)"
            if has_sdnb:
                return "SDNB (084)"
            return "Unklassifiziert"

        df["status"] = df.apply(klass_status, axis=1)

        agg = df.groupby(["decade", "status"]).size().reset_index(name="count")
        fig = px.bar(
            agg,
            x="decade",
            y="count",
            color="status",
            color_discrete_map={
                "DDC (082/083)": "#1E88E5",
                "SDNB (084)": "#FF9800",
                "Unklassifiziert": "#CFD8DC",
            },
            barmode="stack",
            title="Mineralogie-Hochschulschriften nach Jahrzehnt",
            labels={
                "decade": "Jahrzehnt",
                "count": "Anzahl",
                "status": "Klassifikation",
            },
        )
        fig.update_layout(
            plot_bgcolor="white",
            height=450,
            legend=dict(orientation="h", y=1.08, x=0.5, xanchor="center"),
        )
        return fig

    # --------------------------------------------------
    # Retro-Bedarf nach Jahrzehnt
    # --------------------------------------------------
    def plot_retro_bedarf(self) -> go.Figure:
        """Unklassifizierte Mineralogie-Records — das Retro-Argument."""
        self._require(
        "publication_year",
        "is_mineralogie",
        "ddc_primary_3digit",
        "has_sdnb",
        "record_id",
        )
        df = self.df[self.df["is_mineralogie"]].copy()
        df["_year"] = pd.to_numeric(df["publication_year"], errors="coerce")
        df = df[df["_year"].between(1940, 2024)]
        df["decade"] = (df["_year"] // 10 * 10).astype(int)
        df["_unclass"] = (df["ddc_primary_3digit"].fillna("").str.len() == 0) & ~df[
            "has_sdnb"
        ].fillna(False)

        agg = (
            df.groupby("decade")
            .agg(total=("record_id", "count"), unclass=("_unclass", "sum"))
            .reset_index()
        )
        agg["pct"] = (agg["unclass"] / agg["total"] * 100).round(1)

        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(
            go.Bar(
                x=agg["decade"],
                y=agg["unclass"],
                name="Unklassifiziert (absolut)",
                marker_color="#CFD8DC",
            ),
            secondary_y=False,
        )
        fig.add_trace(
            go.Scatter(
                x=agg["decade"],
                y=agg["pct"],
                name="Anteil (%)",
                mode="lines+markers",
                line=dict(color="#E53935", width=2.5),
            ),
            secondary_y=True,
        )
        fig.add_vrect(
            x0=1935,
            x1=1975,
            fillcolor="#E53935",
            opacity=0.07,
            annotation_text="Retro-Bedarf",
            annotation_position="top left",
            annotation_font_color="#E53935",
        )
        fig.update_layout(
            title="Retroklassifizierungs-Bedarf: Unklassifizierte Mineralogie-Records",
            plot_bgcolor="white",
            height=450,
            legend=dict(orientation="h", y=1.08, x=0.5, xanchor="center"),
        )
        fig.update_yaxes(
            title_text="Anzahl", secondary_y=False, showgrid=True, gridcolor="#EEEEEE"
        )
        fig.update_yaxes(
            title_text="Anteil (%)", secondary_y=True, ticksuffix="%", showgrid=False
        )
        return fig

    # --------------------------------------------------
    # DDC-Hierarchie für Sunburst
    # --------------------------------------------------
    def build_ddc_hierarchy(self) -> pd.DataFrame:
        """
        Aggregierter DDC-Hierarchie-DataFrame für Sunburst-Chart.
        Benötigt ddc_primary_3digit aus ClassificationTransformer.
        """
        self._require("ddc_primary_3digit")
        df = self.df.copy()
        df["ddc3"] = df["ddc_primary_3digit"].astype(str).str.strip()
        df = df[df["ddc3"].str.match(r"^\d{3}$")].copy()

        df["DDC_1_key"] = df["ddc3"].str[0]
        df["DDC_1_label"] = df["DDC_1_key"].map(DDC_MAIN).fillna("Unbekannt")
        df["DDC_1"] = df["DDC_1_label"] + " (" + df["DDC_1_key"] + "00)"
        df["DDC_2"] = df["ddc3"].str[:2] + "0"
        df["DDC_3"] = df["ddc3"]

        return df.groupby(["DDC_1", "DDC_2", "DDC_3"]).size().reset_index(name="count")

    def plot_sunburst(self, max_depth: int = 2) -> go.Figure:
        """Sunburst der DDC-Hierarchie."""
        df_hier = self.build_ddc_hierarchy()
        fig = px.sunburst(
            df_hier,
            path=["DDC_1", "DDC_2", "DDC_3"],
            values="count",
            title="DDC-Hierarchie DNB-Hochschulschriften",
            color="count",
            color_continuous_scale="Blues",
            maxdepth=max_depth,
        )
        fig.update_traces(
            hovertemplate="<b>%{label}</b><br>Records: %{value:,}<extra></extra>",
        )
        fig.update_layout(height=600, coloraxis_showscale=False)
        return fig

    # --------------------------------------------------
    # Text-Feature aufbauen
    # --------------------------------------------------
    def build_text_column(
        self,
        columns: List[str],
        new_column: str = "text_combined",
    ) -> pd.DataFrame:
        """
        Kombiniert Textspalten zu einem Feature-Feld.
        Subjects (List[str]) werden automatisch gejoined.
        """

        def combine(row):
            texts = []
            for col in columns:
                val = row.get(col)
                if isinstance(val, list):
                    texts.append(
                        " ".join(
                            s.split(";")[0].strip()  # ersten Term vor ; nehmen
                            for s in map(str, val)
                            if s.strip()
                        )
                    )
                elif pd.notna(val) and str(val).strip():
                    texts.append(str(val).strip())
            return " ".join(texts).lower()

        self.df[new_column] = self.df.apply(combine, axis=1)
        return self.df

    # --------------------------------------------------
    # Klassifikations-Label für Retroklassifizierer
    # --------------------------------------------------
    def build_classification_target(
        self,
        positive_codes: dict,
        label_col: str = "target_label",
        verbose: bool = False,
    ) -> pd.DataFrame:
        """
        Erstellt Mehrklassen-Label.

        Parameters
        ----------
        positive_codes : dict
            z.B. {"Mineralogie": ["549"], "Petrologie": ["552"]}
        """
        self._require("ddc_primary_3digit")
        df = self.df.copy()
        df[label_col] = "Sonstige"
        for label, codes in positive_codes.items():
            mask = df["ddc_primary_3digit"].isin(codes)
            df.loc[mask, label_col] = label

        if verbose:
            print("Klassifikationsverteilung:")
            print(df[label_col].value_counts())

        return df

    # --------------------------------------------------
    # Top-Terme pro Klasse
    # --------------------------------------------------
    def top_terms_by_class(
        self,
        text_column: str,
        target_column: str,
        top_n: int = 20,
    ) -> pd.DataFrame:
        """
        Ermittelt Top-TF-IDF-Terme pro Klasse (ML/Feature-Store-Style).

        Returns
        -------
        DataFrame mit Spalten:
            - class: Klassenlabel
            - term: Term
            - score: mittlerer TF-IDF-Score

        Notes
        -----
        Geeignet für:
            - Feature-Analyse
            - Feature-Store-Export
            - ML-Interpretation
        """

        self._require(text_column, target_column)
        df = self.df.dropna(subset=[text_column, target_column])

        if df.shape[0] < 5:
            return pd.DataFrame(columns=["class", "term", "score"])

        vec = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), min_df=5)
        X = vec.fit_transform(df[text_column])
        feature_names = np.array(vec.get_feature_names_out())

        rows = []

        for label in sorted(df[target_column].unique()):
            mask = df[target_column] == label
            if mask.sum() < 5:
                continue

            X_subset = X[mask.values]
            mean_tfidf = X_subset.mean(axis=0).A1
            top_idx = mean_tfidf.argsort()[-top_n:][::-1]

            for idx in top_idx:
                rows.append({
                    "class": label,
                    "term": feature_names[idx],
                    "score": float(mean_tfidf[idx]),
                })

        return pd.DataFrame(rows)
