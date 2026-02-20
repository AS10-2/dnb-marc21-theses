"""
classification_transform.py

Transformationsmodul zur Weiterverarbeitung von aus MARC21 extrahierten
Klassifikationsdaten (insbesondere DDC und alte DNB-Sachgruppen).

Dieses Modul trennt bewusst die reine Datenextraktion (Parser)
von der fachlichen Logik (Priorisierung, Mapping, Normalisierung).

Funktionalität:
- Priorisierung konkurrierender DDC-Felder (082 > 083 > 084)
- Normalisierung von DDC-Werten (z. B. auf 3-Stellen-Ebene)
- Mapping alter DNB-Sachgruppen auf DDC-Klassen
- Transparente Beibehaltung aller Ursprungswerte

Empfohlene Pipeline:
    1. MARC21-Parsing → Roh-DataFrame
    2. Transformation mit ClassificationTransformer
    3. Analyse / Visualisierung
"""

from typing import Dict, List, Optional
import pandas as pd


class ClassificationTransformer:
    """
    Transformiert ein DataFrame mit extrahierten MARC21-Klassifikationsdaten.

    Erwartete Spalten im Input-DataFrame:
        - ddc_082_all : List[str]
        - ddc_083_all : List[str]
        - ddc_084_all : List[str]
        - sachgruppe  : List[str]

    Hinzugefügte Spalten:
        - ddc_primary
        - ddc_primary_3digit
        - sachgruppe_ddc_mapped

    Parameter
    ----------
    sachgruppe_mapping : dict, optional
        Mapping-Tabelle von DNB-Sachgruppen (Prefix) zu DDC-Klassen.
        Beispiel:
            {
                "01": "000",
                "02": "100",
                ...
            }
    """

    def __init__(self, sdnb_to_ddc_mapping: Optional[Dict[str, str]] = None):
        self.mapping = sdnb_to_ddc_mapping or {}

    # ---------------------------------------------------------
    # Öffentliche API
    # ---------------------------------------------------------

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Führt alle Transformationsschritte auf dem DataFrame aus.

        Schritte:
            1. Priorisierung konkurrierender DDC-Felder
            2. Normalisierung der Haupt-DDC auf 3-Stellen-Ebene
            3. Mapping alter DNB-Sachgruppen auf DDC

        Parameter
        ----------
        df : pandas.DataFrame
            DataFrame mit extrahierten MARC21-Daten.

        Returns
        -------
        pandas.DataFrame
            Transformiertes DataFrame mit zusätzlichen Analyse-Spalten.
        """
        df = df.copy()

        df["ddc_primary"] = df.apply(self._prioritize_ddc, axis=1)
        df["ddc_primary_3digit"] = df["ddc_primary"].apply(
            self._normalize_ddc
        )

        df["sachgruppe_ddc_mapped"] = df["sachgruppe"].apply(
            self._map_sachgruppe
        )

        return df

    # ---------------------------------------------------------
    # Interne Methoden
    # ---------------------------------------------------------

    def _prioritize_ddc(self, row: pd.Series) -> str:
        """
        Priorisiert konkurrierende DDC-Felder.

        Reihenfolge:
            1. 082
            2. 083
            3. 084

        Gibt die erste verfügbare Klasse zurück.
        """
        for col in ["ddc_082_all", "ddc_083_all", "ddc_084_all"]:
            values = row.get(col)
            if isinstance(values, list) and values:
                return values[0]
        return ""

    def _normalize_ddc(self, ddc: str) -> str:
        """
        Normalisiert eine DDC-Notation.

        - Entfernt Zusätze (z. B. nach "/")
        - Kürzt auf die ersten drei Stellen
        - Entfernt führende/trailing Spaces

        Beispiel:
            "530.12/045" → "530"
        """
        if not isinstance(ddc, str) or not ddc:
            return ""

        base = ddc.split("/")[0].strip()
        return base[:3]

    def _map_sachgruppe(self, sachgruppen: List[str]) -> List[str]:
        """
        Mappt alte DNB-Sachgruppen auf DDC-Klassen anhand
        eines Prefix-Mappings.

        Beispiel:
            "05.12" → Prefix "05" → Mapping → "500"
        """
        if not isinstance(sachgruppen, list):
            return []

        mapped = []
        for sg in sachgruppen:
            if not isinstance(sg, str):
                continue

            prefix = sg[:2]
            if prefix in self.mapping:
                mapped.append(self.mapping[prefix])

        return mapped
