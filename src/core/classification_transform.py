"""
classification_transform.py
Feature Engineering auf Basis bereinigter MARC21-Klassifikationsdaten.

Pipeline:
    Marc21Parser → Cleaner → ClassificationTransformer → Explorer
"""

import re
from typing import List, Callable

import pandas as pd


class ClassificationTransformer:
    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        # Zwischenspalte — wird nach Verwendung gedroppt
        df["_ddc_all"] = df.apply(self._collect_ddc_codes, axis=1)

        df["ddc_primary_3digit"] = df["_ddc_all"].apply(
            lambda lst: self._normalize_ddc(lst[0]) if lst else ""
        )

        df["has_sdnb"] = df["sdnb_codes"].apply(
            lambda x: isinstance(x, list) and len(x) > 0
        )

        # Fachcluster
        is_geowiss = self.make_topic_flag(
            ddc_prefixes=["549", "55"],
            sdnb_prefixes=["31"]
        )
        df["is_geowiss"] = df.apply(is_geowiss, axis=1)

        df = df.drop(columns=["_ddc_all"])
        return df

    @staticmethod
    def transform_dataset(df: pd.DataFrame) -> pd.DataFrame:
        return ClassificationTransformer().apply(df)

    # =====================================================
    # DDC Handling
    # =====================================================

    def _collect_ddc_codes(self, row: pd.Series) -> List[str]:
        codes = []
        codes += self._extract_valid_ddc(
            list(row.get("082_a") or []),
            list(row.get("082_2") or [])
        )
        codes += self._extract_valid_ddc(
            list(row.get("083_a") or []),
            list(row.get("083_2") or [])
        )
        return codes

    def _extract_valid_ddc(
        self,
        codes: List[str],
        editions: List[str]
    ) -> List[str]:
        """
        Permissiv: Edition 22/23 oder leer werden akzeptiert.
        Leer deckt ältere Records ab — wichtig für Retroklassifizierung.
        """
        result: List[str] = []

        if not isinstance(codes, list):
            return result

        for i, code in enumerate(codes):
            if not isinstance(code, str) or not code:
                continue

            edition = editions[i] if i < len(editions) else ""
            if not isinstance(edition, str):
                edition = ""

            # ablehnen nur wenn Edition explizit gesetzt aber nicht 22/23
            if edition and not edition.startswith(("22", "23")):
                continue

            result.append(code)

        return result

    # =====================================================
    # Normalisierung auf 3 Stellen
    # =====================================================

    @staticmethod
    def _normalize_ddc(ddc: str) -> str:
        if not isinstance(ddc, str) or not ddc:
            return ""

        ddc = ddc.split(";")[0].split("/")[0].strip()
        match = re.match(r"^(\d+)", ddc)
        if not match:
            return ""

        digits = match.group(1)
        if len(digits) == 1:
            return digits + "00"
        if len(digits) == 2:
            return digits + "0"
        return digits[:3]

    # =====================================================
    # Topic Factory
    # =====================================================

    @staticmethod
    def make_topic_flag(
        ddc_prefixes: List[str],
        sdnb_prefixes: List[str] | None = None
    ) -> Callable[[pd.Series], bool]:
        """
        Erzeugt eine Klassifikationsfunktion für fachliche Zuordnung
        auf Basis von DDC- und SDNB-Präfixen.

        Beispiel:
            is_geowiss = make_topic_flag(["549", "55"], ["31"])
        """
        sdnb_prefixes = sdnb_prefixes or []

        def flag(row: pd.Series) -> bool:
            # DDC prüfen (082 + 083)
            for code in list(row.get("082_a") or []) + list(row.get("083_a") or []):
                norm = ClassificationTransformer._normalize_ddc(code)
                if any(norm.startswith(p) for p in ddc_prefixes):
                    return True

            # SDNB prüfen
            for code in list(row.get("sdnb_codes") or []):
                if isinstance(code, str) and any(code.startswith(p) for p in sdnb_prefixes):
                    return True

            return False

        return flag
