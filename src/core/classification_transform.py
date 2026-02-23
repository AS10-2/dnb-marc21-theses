"""
classification_transform.py

Transformationsmodul zur Weiterverarbeitung von aus MARC21 extrahierten
Klassifikationsdaten (insbesondere DDC).

Dieses Modul übernimmt ausschließlich fachliche Logik:
- Sammlung gültiger DDC-Codes (082 + 083)
- Priorisierung (082 vor 083)
- Normalisierung auf 3-Stellen-Ebene
- Ableitung der Hauptklasse
- Optionale SDNB→DDC-Mappings

Pipeline:
    MARC21 → Cleaner → ClassificationTransformer → Analyse
"""

from typing import Dict, List, Optional
import pandas as pd
import re


class ClassificationTransformer:

    def __init__(self, sdnb_to_ddc_mapping: Optional[Dict[str, str]] = None):
        self.mapping = sdnb_to_ddc_mapping or {}

    # =====================================================
    # Hauptfunktion
    # =====================================================
    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        # Alle gültigen DDC-Codes sammeln
        df["ddc_all_raw"] = df.apply(self._collect_ddc_codes, axis=1)

        # Primary = erster valider Code (082 priorisiert)
        df["ddc_primary"] = df["ddc_all_raw"].apply(
            lambda lst: lst[0] if lst else ""
        )

        # Normalisierung
        df["ddc_primary_3digit"] = df["ddc_primary"].apply(
            self._normalize_ddc
        )

        # Hauptklasse
        df["ddc_main_class"] = df["ddc_primary_3digit"].apply(
            self._derive_main_class
        )

        # Geoscience-Flag
        df["is_geoscience"] = df["ddc_all_raw"].apply(
            self._is_geoscience
        )

        # Optional: SDNB-Mapping
        if "sdnb_prefix" in df.columns and self.mapping:
            df["sdnb_ddc_mapped"] = df["sdnb_prefix"].apply(
                self._map_sdnb_to_ddc
            )

        return df

    # =====================================================
    # 1) DDC sammeln (082 priorisiert vor 083)
    # =====================================================
    def _collect_ddc_codes(self, row: pd.Series) -> List[str]:
        codes = []

        # 082 zuerst
        codes += self._extract_valid_ddc(
            row.get("082_a", []),
            row.get("082_2", [])
        )

        # dann 083
        codes += self._extract_valid_ddc(
            row.get("083_a", []),
            row.get("083_2", [])
        )

        return codes

    def _extract_valid_ddc(self, codes: List[str], editions: List[str]) -> List[str]:
        result = []

        if not isinstance(codes, list):
            return result

        for i, code in enumerate(codes):
            edition = editions[i] if i < len(editions) else ""

            if not isinstance(code, str):
                continue

            # Nur DDC-Edition 22 oder 23 zulassen
            if isinstance(edition, str) and edition.startswith(("22", "23")):
                result.append(code)

        return result

    # =====================================================
    # 2) Normalisierung
    # =====================================================
    def _normalize_ddc(self, ddc: str) -> str:
        if not isinstance(ddc, str) or not ddc:
            return ""

        # nur erster Block (vor ; oder /)
        ddc = ddc.split(";")[0].split("/")[0].strip()

        # führende Ziffern extrahieren
        match = re.match(r"^(\d+)", ddc)
        if not match:
            return ""

        digits = match.group(1)

        # 1-stellig → 500
        if len(digits) == 1:
            return digits + "00"

        # 2-stellig → 330
        if len(digits) == 2:
            return digits + "0"

        # >=3 → erste 3
        return digits[:3]

    # =====================================================
    # 3) Hauptklasse (z.B. 550 → 500)
    # =====================================================
    def _derive_main_class(self, ddc3: str) -> str:
        if not isinstance(ddc3, str) or not ddc3.isdigit():
            return ""

        return ddc3[0] + "00"

    # =====================================================
    # 4) Geoscience-Erkennung
    # =====================================================
    def _is_geoscience(self, codes: List[str]) -> bool:
        if not isinstance(codes, list):
            return False

        for code in codes:
            norm = self._normalize_ddc(code)
            if norm.startswith("550"):
                return True

        return False

    # =====================================================
    # 5) SDNB-Mapping (optional)
    # =====================================================
    def _map_sdnb_to_ddc(self, prefixes: List[str]) -> List[str]:
        if not isinstance(prefixes, list):
            return []

        return [
            self.mapping[p]
            for p in prefixes
            if p in self.mapping
        ]
