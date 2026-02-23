"""
cleaning.py

Datenhygiene- und Typisierungsmodul für aus MARC21 extrahierte
DNB-Hochschulschriften-Daten.

Verantwortung:
- Listen-Spalten absichern
- primitive Datentypen säubern
- 082 / 083 / 084 Rohfelder extrahieren
- keine Klassifikationslogik

Pipeline:
    marc21_parser_full → Cleaner → ClassificationTransformer → Analyse
"""

import pandas as pd


class Cleaner:

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df = self._ensure_list_columns(df)
        df = self._clean_publication_year(df)
        df = self._extract_082(df)
        df = self._extract_083(df)
        df = self._extract_sdnb_from_084(df)
        df = self._optimize_dtypes(df)
        return df

    # --------------------------------------------------
    # 1) Listen absichern
    # --------------------------------------------------
    def _ensure_list_columns(self, df):
        for col in ["082_list", "083_list", "084_list", "subjects"]:
            if col in df.columns:
                df[col] = df[col].apply(lambda x: x if isinstance(x, list) else [])
        return df

    # --------------------------------------------------
    # 2) publication_year säubern
    # --------------------------------------------------
    def _clean_publication_year(self, df):
        if "publication_year" in df.columns:
            df["publication_year"] = (
                pd.to_numeric(df["publication_year"], errors="coerce")
                .astype("float")
            )
            df["publication_year"] = (
                df["publication_year"]
                .round()
                .astype("Int64")
            )
        return df

    # --------------------------------------------------
    # 3) 082 extrahieren (DDC Hauptfeld)
    # --------------------------------------------------
    def _extract_082(self, df):
        if "082_list" not in df.columns:
            df["082_a"] = [[] for _ in range(len(df))]
            df["082_2"] = [[] for _ in range(len(df))]
            return df

        def extract_a(lst):
            return [
                e.get("a")
                for e in lst
                if isinstance(e, dict) and "a" in e
            ]

        def extract_2(lst):
            return [
                e.get("2")
                for e in lst
                if isinstance(e, dict) and "2" in e
            ]

        df["082_a"] = df["082_list"].apply(extract_a)
        df["082_2"] = df["082_list"].apply(extract_2)
        return df

    # --------------------------------------------------
    # 4) 083 extrahieren (zusätzliche DDC)
    # --------------------------------------------------
    def _extract_083(self, df):
        if "083_list" not in df.columns:
            df["083_a"] = [[] for _ in range(len(df))]
            df["083_2"] = [[] for _ in range(len(df))]
            return df

        def extract_a(lst):
            return [
                e.get("a")
                for e in lst
                if isinstance(e, dict) and "a" in e
            ]

        def extract_2(lst):
            return [
                e.get("2")
                for e in lst
                if isinstance(e, dict) and "2" in e
            ]

        df["083_a"] = df["083_list"].apply(extract_a)
        df["083_2"] = df["083_list"].apply(extract_2)
        return df

    # --------------------------------------------------
    # 5) 084 → SDNB extrahieren
    # --------------------------------------------------
    def _extract_sdnb_from_084(self, df):
        if "084_list" not in df.columns:
            df["sdnb_codes"] = [[] for _ in range(len(df))]
            df["has_sdnb"] = False
            df["sdnb_prefix"] = [[] for _ in range(len(df))]
            return df

        def extract_sdnb(entries):
            result = []
            if not isinstance(entries, list):
                return result

            for entry in entries:
                if not isinstance(entry, dict):
                    continue

                source = entry.get("2", "")
                code = entry.get("a", "")

                if source and "dnb" in source.lower() and code:
                    result.append(code)

            return result

        df["sdnb_codes"] = df["084_list"].apply(extract_sdnb)
        df["has_sdnb"] = df["sdnb_codes"].str.len().gt(0)
        df["sdnb_prefix"] = df["sdnb_codes"].apply(
            lambda lst: [code[:2] for code in lst]
        )

        return df

    # --------------------------------------------------
    # 6) Datentypen optimieren
    # --------------------------------------------------
    def _optimize_dtypes(self, df):
        for col in ["language", "publication_place", "publisher"]:
            if col in df.columns:
                df[col] = df[col].astype("category")

        if "record_id" in df.columns:
            df["record_id"] = df["record_id"].astype("string")

        if "subject_count" in df.columns:
            df["subject_count"] = (
                pd.to_numeric(df["subject_count"], errors="coerce")
                .astype("Int32")
            )

        return df

    @staticmethod
    def clean_library_df(df: pd.DataFrame) -> pd.DataFrame:
        return Cleaner().apply(df)
