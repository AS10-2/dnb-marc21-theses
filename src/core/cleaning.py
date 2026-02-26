"""
cleaning.py

Datenhygiene- und Typisierungsmodul für aus MARC21 extrahierte
DNB-Hochschulschriften-Daten.

Verantwortung:
- Listen-Spalten absichern
- Datentypen säubern
- 082 / 083 / 084 Rohfelder in flache Listen extrahieren
- SDNB-Codes aus 084 isolieren
- keine Klassifikationslogik (gehört in ClassificationTransformer)

Pipeline:
    Marc21Parser → Cleaner → ClassificationTransformer → Explorer
"""

from typing import List

import pandas as pd


class Cleaner:
    # Textspalten die zu StringDtype konvertiert werden
    _STRING_COLS = [
        "record_id",
        "control_008",
        "author_name",
        "author_dates",
        "title",
        "title_remainder",
        "dissertation_note",
    ]

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df = self._ensure_list_columns(df)
        df = self._clean_publication_year(df)
        df = self._clean_subjects(df)
        df = self._extract_ddc_field(df, "082")
        df = self._extract_ddc_field(df, "083")
        df = self._extract_sdnb_from_084(df)
        df = self._drop_raw_list_columns(df)  # ← Speicher freigeben
        df = self._optimize_dtypes(df)
        return df

    # --------------------------------------------------
    # Listen absichern
    # --------------------------------------------------
    def _ensure_list_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        for col in ["082_list", "083_list", "084_list", "subjects"]:
            if col in df.columns:
                df[col] = df[col].apply(lambda x: x if isinstance(x, list) else [])
        return df

    # --------------------------------------------------
    # publication_year
    # --------------------------------------------------
    def _clean_publication_year(self, df: pd.DataFrame) -> pd.DataFrame:
        if "publication_year" not in df.columns:
            return df
        df["publication_year"] = (
            pd.to_numeric(df["publication_year"], errors="coerce")
            .round()
            .astype("Int64")
        )
        valid = df["publication_year"].between(1800, 2030)
        df.loc[~valid, "publication_year"] = pd.NA
        # Int16 reicht: 1800–2030 liegt im Bereich -32768..32767
        df["publication_year"] = df["publication_year"].astype("Int16")
        return df

    # --------------------------------------------------
    # Subjects splitten
    # --------------------------------------------------
    def _clean_subjects(self, df: pd.DataFrame) -> pd.DataFrame:
        if "subjects" not in df.columns:
            return df
        df["subjects"] = df["subjects"].apply(
            lambda lst: [
                s.strip() for item in lst for s in str(item).split(";") if s.strip()
            ]
        )
        return df

    # --------------------------------------------------
    # 082 / 083 extrahieren
    # --------------------------------------------------
    def _extract_ddc_field(self, df: pd.DataFrame, tag: str) -> pd.DataFrame:
        col = f"{tag}_list"
        col_a = f"{tag}_a"
        col_2 = f"{tag}_2"

        if col not in df.columns:
            df[col_a] = [[] for _ in range(len(df))]
            df[col_2] = [[] for _ in range(len(df))]
            return df

        def _extract(lst: list, subfield: str) -> List[str]:
            result = []
            for entry in lst:
                if not isinstance(entry, dict):
                    continue
                val = entry.get(subfield)
                if val is None:
                    continue
                if isinstance(val, list):
                    result.extend(v for v in val if v)
                else:
                    result.append(str(val))
            return result

        df[col_a] = df[col].apply(lambda x: _extract(x, "a"))
        df[col_2] = df[col].apply(lambda x: _extract(x, "2"))
        return df

    # --------------------------------------------------
    # SDNB-Codes aus 084 isolieren
    # --------------------------------------------------
    def _extract_sdnb_from_084(self, df: pd.DataFrame) -> pd.DataFrame:
        if "084_list" not in df.columns:
            df["sdnb_codes"] = [[] for _ in range(len(df))]
            return df

        def extract_sdnb(entries: list) -> List[str]:
            result = []
            if not isinstance(entries, list):
                return result
            for entry in entries:
                if not isinstance(entry, dict):
                    continue
                source = entry.get("2", "")
                if isinstance(source, list):
                    source = source[0] if source else ""
                if "dnb" not in source.lower():
                    continue
                codes = entry.get("a", [])
                if isinstance(codes, list):
                    result.extend(c.strip() for c in codes if c.strip())
                elif codes:
                    result.append(str(codes).strip())
            return result

        df["sdnb_codes"] = df["084_list"].apply(extract_sdnb)
        return df

    # --------------------------------------------------
    # Datentypen optimieren
    # --------------------------------------------------
    def _optimize_dtypes(self, df: pd.DataFrame) -> pd.DataFrame:

        # Textspalten: object → StringDtype (nullable, memory-effizienter)
        for col in self._STRING_COLS:
            if col in df.columns:
                df[col] = df[col].astype(pd.StringDtype())

        # Kategorie-Spalten (wenige Unique Values)
        for col in ["language", "publication_place", "publisher"]:
            if col in df.columns:
                df[col] = df[col].astype("category")

        # Numerisch
        if "subject_count" in df.columns:
            df["subject_count"] = pd.to_numeric(
                df["subject_count"], errors="coerce"
            ).astype("Int16")

        return df

    # --------------------------------------------------
    # Rohe List-Spalten droppen
    # --------------------------------------------------
    def _drop_raw_list_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        drop_cols = [
            "082_list",
            "083_list",
            "084_list",
            "author_gnd",  # vorerst nicht benötigt
            "dissertation_note",  # vorerst nicht benötigt
        ]
        df = df.drop(columns=[c for c in drop_cols if c in df.columns])
        return df

    # --------------------------------------------------
    # Convenience + Diagnose
    # --------------------------------------------------
    @staticmethod
    def clean_library_df(df: pd.DataFrame) -> pd.DataFrame:
        return Cleaner().apply(df)

    @staticmethod
    def memory_report(df: pd.DataFrame) -> pd.DataFrame:
        """Zeigt Speicherverbrauch pro Spalte."""
        return (
            df.dtypes.to_frame("dtype")
            .join(df.memory_usage(deep=True).rename("bytes"))
            .assign(MB=lambda d: (d["bytes"] / 1024**2).round(2))
            .sort_values("MB", ascending=False)
        )
