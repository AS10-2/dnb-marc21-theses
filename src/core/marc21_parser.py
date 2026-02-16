import gzip
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd


class Marc21Parser:
    """
    Parser für MARC21-XML-Dateien mit Fokus auf bibliographische Kerndaten.

    Extrahiert Kontrollfelder, Autor, Titel, Publikation, Dissertation,
    Klassifikationen (084, 082, 490, 830, 648, 650) und Schlagwörter als JSON-Listen.
    """

    MARC_NS = {"marc": "http://www.loc.gov/MARC21/slim"}

    def parse_file(
        self, filepath: str, limit: Optional[int] = None, verbose: bool = True
    ) -> pd.DataFrame:
        """
        Parsen einer MARC21-XML-Datei (.xml oder .xml.gz) und Rückgabe als DataFrame.

        Parameters
        ----------
        filepath : str
            Pfad zur MARC21-XML-Datei.
        limit : int, optional
            Maximale Anzahl der zu parsenden Datensätze. None = alle Datensätze.
        verbose : bool, default True
            Fortschrittsinformationen anzeigen.

        Returns
        -------
        pd.DataFrame
            DataFrame mit extrahierten Feldern, JSON-kompatible Listen für Klassifikationen und Schlagwörter.
        """
        filepath_obj = Path(filepath)
        if not filepath_obj.exists():
            raise FileNotFoundError(f"File not found: {filepath}")

        if filepath_obj.suffix == ".gz":
            file_obj = gzip.open(filepath_obj, "rb")
        else:
            file_obj = open(filepath_obj, "rb")

        try:
            records = []
            count = 0
            context = ET.iterparse(file_obj, events=("end",))
            for event, elem in context:
                if elem.tag.endswith("record"):
                    rec = self._parse_record(elem)
                    if rec:
                        records.append(rec)
                        count += 1
                        if verbose and count % 1000 == 0:
                            print(f"Parsed {count} records...")
                        if limit is not None and count >= limit:
                            if verbose:
                                print(f"Reached limit of {limit} records.")
                            break
                    elem.clear()
            if verbose:
                print(f"Total records parsed: {count}")
            df = pd.DataFrame(records)
            return df
        finally:
            file_obj.close()

    def _parse_record(self, elem: ET.Element) -> Optional[Dict[str, Any]]:
        try:
            data: Dict[str, Any] = {}

            # Controlfields
            data["record_id"] = self._get_controlfield(elem, "001")
            data["control_008"] = self._get_controlfield(elem, "008")

            # Autor
            author = self._get_datafield(elem, "100")
            data["author_name"] = author.get("a", "")
            data["author_dates"] = author.get("d", "")
            data["author_gnd"] = self._extract_gnd(author.get("0", ""))

            # Titel
            title = self._get_datafield(elem, "245")
            data["title"] = title.get("a", "")
            data["title_remainder"] = title.get("b", "")

            # Publikation (264 komplett als Dict-Liste)
            pub_fields = self._get_datafield_list_dicts(elem, "264")

            publication_year = ""
            publication_place = ""
            publisher = ""

            if pub_fields:
                first_pub = pub_fields[0]
                publication_place = first_pub.get("a", "")
                publisher = first_pub.get("b", "")
                publication_year = first_pub.get("c", "")

            data["publication_place"] = publication_place
            data["publisher"] = publisher
            data["publication_year"] = publication_year

            # Dissertation
            diss = self._get_datafield(elem, "502")
            data["dissertation_note"] = diss.get("a", "")

            # Mehrfachfelder
            for tag in ["084", "083", "082", "490", "648", "650", "830"]:
                values = self._get_datafield_list_dicts(elem, tag)
                data[f"{tag}_list"] = (
                    values if values else []
                )  # JSON-kompatible Liste von Dicts

            return data

        except Exception as e:
            print(f"Error parsing record: {e}")
            return None

    # --- Hilfsfunktionen ---
    def _get_controlfield(self, elem: ET.Element, tag: str) -> str:
        """
        Extrahiert den Wert eines Controlfield-Tags.

        Parameters
        ----------
        elem : ET.Element
            MARC21 record Element
        tag : str
            Controlfield-Tag (z.B. '001', '005', '008')

        Returns
        -------
        str
            Textwert des Controlfields, oder '' falls nicht vorhanden.
        """
        cf = elem.find(f".//marc:controlfield[@tag='{tag}']", self.MARC_NS)
        if cf is None:
            cf = elem.find(f".//controlfield[@tag='{tag}']")
        if cf is not None and cf.text:
            return cf.text.strip()
        return ""

    def _get_datafield(self, elem: ET.Element, tag: str) -> Dict[str, str]:
        """
        Extrahiert die Subfields eines Datafield-Tags als Dictionary.

        Parameters
        ----------
        elem : ET.Element
            MARC21 record Element
        tag : str
            Datafield-Tag (z.B. '100', '245')

        Returns
        -------
        dict
            Schlüssel = Subfield-Code, Wert = Text
        """
        res: Dict[str, str] = {}
        df = elem.find(f".//marc:datafield[@tag='{tag}']", self.MARC_NS)
        if df is None:
            df = elem.find(f".//datafield[@tag='{tag}']")
        if df is not None:
            subfields = df.findall("marc:subfield", self.MARC_NS)
            if not subfields:
                subfields = df.findall("subfield")
            for sf in subfields:
                code = sf.get("code", "")
                text = sf.text or ""
                if code:
                    if code in res:
                        res[code] += f" ; {text.strip()}"
                    else:
                        res[code] = text.strip()
        return res

    def _get_datafield_list(self, elem: ET.Element, tag: str, code: str) -> List[str]:
        """
        Extrahiert alle Werte eines bestimmten Subfields als Liste.

        Parameters
        ----------
        elem : ET.Element
            MARC21 record Element
        tag : str
            Datafield-Tag (z.B. '084', '650')
        code : str
            Subfield-Code (z.B. 'a')

        Returns
        -------
        list
            Liste der Textwerte (JSON-kompatibel)
        """
        vals: List[str] = []
        dfs = elem.findall(f".//marc:datafield[@tag='{tag}']", self.MARC_NS)
        if not dfs:
            dfs = elem.findall(f".//datafield[@tag='{tag}']")
        for df in dfs:
            sfs = df.findall(f"marc:subfield[@code='{code}']", self.MARC_NS)
            if not sfs:
                sfs = df.findall(f"subfield[@code='{code}']")
            for sf in sfs:
                if sf.text:
                    vals.append(sf.text.strip())
        return vals

    def _get_datafield_list_dicts(
        self, elem: ET.Element, tag: str
    ) -> List[Dict[str, str]]:
        """
        Gibt alle Datafields eines Tags als Liste von Dictionaries zurück.
        Jedes Datafield wird vollständig mit allen Subfields gespeichert.
        """
        results: List[Dict[str, str]] = []

        dfs = elem.findall(f".//marc:datafield[@tag='{tag}']", self.MARC_NS)
        if not dfs:
            dfs = elem.findall(f".//datafield[@tag='{tag}']")

        for df in dfs:
            entry: Dict[str, str] = {}

            subfields = df.findall("marc:subfield", self.MARC_NS)
            if not subfields:
                subfields = df.findall("subfield")

            for sf in subfields:
                code = sf.get("code", "")
                text = sf.text.strip() if sf.text else ""

                if code:
                    if code in entry:
                        entry[code] += f" ; {text}"
                    else:
                        entry[code] = text

            if entry:
                results.append(entry)

        return results

    def _extract_gnd(self, gnd_str: str) -> str:
        """
        Extrahiert die GND-Kennung aus verschiedenen Formaten.

        Parameters
        ----------
        gnd_str : str
            GND-Feld (z.B. "(DE-588)10675131X" oder "https://d-nb.info/gnd/10675131X")

        Returns
        -------
        str
            Saubere GND-Kennung oder ''.
        """
        if not gnd_str:
            return ""
        vals = [v.strip() for v in gnd_str.split(" ; ")]
        for fmt in ["(DE-588)", "gnd/", "(DE-101)"]:
            for v in vals:
                if fmt in v:
                    gnd = v.split(fmt)[-1].strip().split()[0]
                    return gnd
        return vals[0] if vals else ""


def parse_dnb_theses(
    filepath: str, limit: Optional[int] = None, verbose: bool = True
) -> pd.DataFrame:
    """
    Einfache Funktion zum Parsen von DNB-Hochschulschriften MARC21-XML.

    Parameters
    ----------
    filepath : str
        Pfad zur MARC21-XML-Datei (.xml oder .xml.gz)
    limit : int, optional
        Maximale Anzahl der Datensätze
    verbose : bool, default True
        Fortschrittsinformationen anzeigen

    Returns
    -------
    pd.DataFrame
        DataFrame mit allen extrahierten Feldern, Listen als JSON-kompatible Arrays.
    """
    parser = Marc21Parser()
    return parser.parse_file(filepath, limit=limit, verbose=verbose)


if __name__ == "__main__":
    # Example usage
    print("MARC21 Parser for Deutsche Nationalbibliothek")
    print("=" * 50)
    print()
    print("Usage example:")
    print(">>> from marc21_parser import parse_dnb_theses")
    print(">>> df = parse_dnb_theses('your_file.xml.gz', limit=100)")
    print(">>> df.head()")
