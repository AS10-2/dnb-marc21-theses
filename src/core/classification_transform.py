"""
classification_transform.py

Feature Engineering auf Basis bereinigter MARC21-Klassifikationsdaten.

Verantwortung:
- DDC-Codes aus 082/083 validieren und priorisieren
- Normalisierung auf 3-Stellen-Ebene
- SDNB-Präsenz feststellen
- Mineralogie-Label ableiten (DDC 549 oder SDNB 38)

Bewusst nicht enthalten:
- 084-DDC (Fremddaten, unzuverlässig)
- RVK (nicht im DNB-Datensatz vorhanden)
- SDNB→DDC Mapping (nicht nötig, direkte Flags reichen)
- ddc_main_class (Explorer-Logik, on-the-fly ableitbar)

Erwartet df_clean aus Cleaner:
    082_a, 082_2, 083_a, 083_2, sdnb_codes

Pipeline:
    Marc21Parser → Cleaner → ClassificationTransformer → Explorer
"""

import re
from typing import List

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
        df["is_mineralogie"] = df.apply(self._is_mineralogie, axis=1)

        df = df.drop(columns=["_ddc_all"])
        return df

    # =====================================================
    # 1) DDC sammeln — 082 priorisiert vor 083
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

    def _extract_valid_ddc(self, codes: List[str], editions: List[str]) -> List[str]:
        """
        Permissiv: Edition 22/23 oder leer werden akzeptiert.
        Leer deckt ältere Records ab — wichtig für Retroklassifizierung.
        """
        result = []
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
    # 2) Normalisierung auf 3 Stellen
    # =====================================================
    def _normalize_ddc(self, ddc: str) -> str:
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
    # 3) Mineralogie-Label
    # =====================================================
    def _is_mineralogie(self, row: pd.Series) -> bool:
        """
        Prüft ALLE DDC- und SDNB-Codes — nicht nur den primären.
        Mineralogie kann als Zweit- oder Drittnotation auftreten.

        DDC-Weg:  irgendein Code normalisiert auf 549  (aus 082/083)
        SDNB-Weg: irgendein Code beginnt mit 38        (aus 084)
        """
        # list() absichern — nach Parquet-Roundtrip können Arrays vorliegen
        codes_082 = list(row.get("082_a") or [])
        codes_083 = list(row.get("083_a") or [])

        # DDC: alle Codes aus 082 + 083 prüfen
        for code in codes_082 + codes_083:
            if self._normalize_ddc(code).startswith("549"):
                return True


        # SDNB: alle Codes aus 084 prüfen
        for code in list(row.get("sdnb_codes") or []):
            if isinstance(code, str) and code.startswith("38"):
                return True

        return False

    # =====================================================
    # Convenience
    # =====================================================
    @staticmethod
    def transform(df: pd.DataFrame) -> pd.DataFrame:
        return ClassificationTransformer().apply(df)
