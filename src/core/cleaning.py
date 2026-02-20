"""
cleaning.py

Datenhygiene- und Typisierungsmodul für aus MARC21 extrahierte
DNB-Hochschulschriften-Daten.

Verantwortung:
- Listen-Spalten absichern
- Datentypen optimieren
- publication_year und ddc_main sauber als Int64

Pipeline:
    marc21_parser_full → Cleaner → ClassificationTransformer → Analyse
"""

from typing import List
import pandas as pd


class Cleaner:
    """
    Führt grundlegende Datenbereinigung und Typoptimierung durch.

    Erwartet ein DataFrame aus Marc21Parser.

    Cleaning-Schritte:
        1. Sicherstellen, dass definierte Spalten Listen enthalten
        2. Konvertierung von 'publication_year' zu Int64
        3. Konvertierung von 'ddc_main' zu Int64
        4. Optimierung weiterer Datentypen (Kategorie, Int16, String)
    """

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Führt alle Cleaning-Schritte aus.

        Parameter
        ----------
        df : pandas.DataFrame
            Output des Marc21Parser

        Returns
        -------
        pandas.DataFrame
            Bereinigtes DataFrame
        """
        df = df.copy()
        df = self._ensure_list_columns(df)
        df = self._clean_publication_year(df)
        df = self._clean_ddc_main(df)
        df = self._extract_sdnb_from_084(df)
        df = self._optimize_dtypes(df)
        return df

    # ---------------------------------------------------------
    # Cleaning Steps
    # ---------------------------------------------------------

    def _ensure_list_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Stellt sicher, dass definierte Spalten immer Listen enthalten.
        """
        list_cols: List[str] = ["082_list", "083_list", "084_list", "subjects"]
        for col in list_cols:
            if col in df.columns:
                df[col] = df[col].apply(lambda x: x if isinstance(x, list) else [])
        return df
        
    def _clean_publication_year(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Konvertiert 'publication_year' sauber zu Int64,
        ungültige Werte werden NaN.
        """
        if "publication_year" in df.columns:
            df["publication_year"] = pd.to_numeric(df["publication_year"], errors="coerce").astype("Int64")
        return df

    def _clean_ddc_main(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Bereinigt 'ddc_main':
        - Konvertiert zu numerisch (Int64)
        - Ungültige Werte werden NaN
        - Extrahiert Hauptklasse (erste 3 Ziffern) als Kategorie
        """
        if "ddc_main" in df.columns:
            # Rohwerte zu numerisch, Fehler -> NaN
            df["ddc_main_raw"] = pd.to_numeric(df["ddc_main"], errors="coerce").astype("Int64")

            # Hauptklasse: erste 3 Ziffern (z.B. 512.3 -> 512)
            def extract_main_class(val):
                if pd.isna(val):
                    return pd.NA
                s = str(val)
                return int(s[:3]) if len(s) >= 3 else int(s)

            df["ddc_main"] = df["ddc_main_raw"].apply(extract_main_class).astype("category")

        return df    

    def _extract_sdnb_from_084(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extrahiert alte DNB-Sachgruppen (SDNB) aus 084.

        Bedingung:
            - Subfield $2 enthält 'dnb'
            - Subfield $a enthält den Sachgruppen-Code

        Ergebnis:
            Neue Spalten:
                - sdnb_codes (List[str])
                - has_sdnb (bool)
        """

        if "084_list" not in df.columns:
            df["sdnb_codes"] = [[] for _ in range(len(df))]
            df["has_sdnb"] = False
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

        # Vektorisierte Bool-Spalte (performanter als lambda)
        df["has_sdnb"] = df["sdnb_codes"].str.len().gt(0)

        df["sdnb_prefix"] = df["sdnb_codes"].apply(
        lambda lst: [code[:2] for code in lst] 
        )

        return df
            
    def _optimize_dtypes(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Optimiert Datentypen für Speicher und Analyse.
        """
        # --- Kategorische Variablen ---
        cat_cols = ["language", "publication_place", "publisher"]
        for col in cat_cols:
            if col in df.columns:
                df[col] = df[col].astype("category")

        # --- record_id als String ---
        if "record_id" in df.columns:
            df["record_id"] = df["record_id"].astype("string")

        # --- subject_count ---
        if "subject_count" in df.columns:
            df["subject_count"] = pd.to_numeric(df["subject_count"], errors="coerce").astype("Int16")

        return df

    # ---------------------------------------------------------
    # Statische Convenience-Methode
    # ---------------------------------------------------------
    @staticmethod
    def clean_library_df(df: pd.DataFrame) -> pd.DataFrame:
        """
        Führt das komplette Cleaning direkt aus, ohne Cleaner instanziieren zu müssen.
        """
        return Cleaner().apply(df)
