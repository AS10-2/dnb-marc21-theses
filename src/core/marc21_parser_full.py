import gzip
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd


class Marc21Parser:
    """
    Robuster MARC21-XML Parser (z.B. DNB-Hochschulschriften).

    Extrahiert:
    - ID & Controlfields
    - Autor inkl. GND
    - Titel
    - Publikationsangaben
    - Dissertation-Vermerk
    - DDC/Sachgruppen (082, 083, 084)
    - Sprache
    - Schlagwörter
    - Externe Identifikatoren (024)

    Speicher- und performance-optimiert für große XML-Dateien.
    """

    MARC_NS = {"marc": "http://www.loc.gov/MARC21/slim"}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def parse_file(
        self,
        filepath: str,
        limit: Optional[int] = None,
        verbose: bool = True,
    ) -> pd.DataFrame:
        """
        Parst eine MARC21 XML oder XML.GZ Datei in ein Pandas DataFrame.
        """

        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {filepath}")

        opener = gzip.open if path.suffix == ".gz" else open

        records: List[Dict[str, Any]] = []
        count = 0

        with opener(path, "rb") as file_obj:
            context = ET.iterparse(file_obj, events=("end",))

            for _, elem in context:
                if self._is_record(elem):
                    record = self._parse_record(elem)
                    if record:
                        records.append(record)
                        count += 1

                        if verbose and count % 1000 == 0:
                            print(f"Parsed {count} records")

                        if limit and count >= limit:
                            break

                    elem.clear()  # Memory cleanup

        if verbose:
            print(f"Finished. Total records parsed: {count}")

        return pd.DataFrame(records)

    # ------------------------------------------------------------------
    # Core Record Parsing
    # ------------------------------------------------------------------

    def _parse_record(self, elem: ET.Element) -> Optional[Dict[str, Any]]:
        try:
            data: Dict[str, Any] = {}

            # --- Controlfields ---
            data["record_id"] = self._get_controlfield(elem, "001")
            control_008 = self._get_controlfield(elem, "008")
            data["control_008"] = control_008

            # --- Sprache ---
            lang_list = self._get_datafield_list(elem, "041", "a")
            data["language"] = lang_list[0] if lang_list else ""

            # --- Autor ---
            author = self._get_first_datafield(elem, ["100", "110"])
            data["author_name"] = author.get("a", "")
            data["author_dates"] = author.get("d", "")
            data["author_gnd"] = self._extract_gnd(author.get("0", ""))

            # --- Titel ---
            title = self._get_datafield(elem, "245")
            data["title"] = title.get("a", "")
            data["title_remainder"] = title.get("b", "")

            # --- Publikation ---
            publication = self._extract_publication(elem, control_008)
            data.update(publication)

            # --- Dissertation ---
            diss = self._get_datafield(elem, "502")
            data["dissertation_note"] = self._join_subfields(
                diss, ["a", "b", "c", "d"]
            )

            # --- DDC / Sachgruppen ---
            for tag in ["082", "083", "084"]:
                data[f"{tag}_list"] = self._get_datafield_list_dicts(elem, tag)

            # --- Schlagwörter ---
            subjects: List[str] = []
            for tag in ["600", "610", "650", "651", "655"]:
                subjects.extend(self._get_datafield_list(elem, tag, "a"))
            data["subjects"] = subjects
            data["subject_count"] = len(subjects)

            # --- Identifikatoren (024) ---
            data["idn_list"] = self._get_datafield_list(elem, "024", "a")

            return data

        except Exception as e:
            print(f"Error parsing record: {e}")
            return None

    # ------------------------------------------------------------------
    # Extraction Helpers
    # ------------------------------------------------------------------

    def _is_record(self, elem: ET.Element) -> bool:
        return elem.tag.endswith("record")

    def _get_controlfield(self, elem: ET.Element, tag: str) -> str:
        cf = elem.find(f".//marc:controlfield[@tag='{tag}']", self.MARC_NS)
        if cf is None:
            cf = elem.find(f".//controlfield[@tag='{tag}']")
        return cf.text.strip() if cf is not None and cf.text else ""

    def _get_first_datafield(
        self, elem: ET.Element, tags: List[str]
    ) -> Dict[str, str]:
        for tag in tags:
            df = self._get_datafield(elem, tag)
            if df:
                return df
        return {}

    def _get_datafield(self, elem: ET.Element, tag: str) -> Dict[str, str]:
        dfs = self._get_datafield_list_dicts(elem, tag)
        return dfs[0] if dfs else {}

    def _get_datafield_list(self, elem: ET.Element, tag: str, code: str) -> List[str]:
        dfs = self._get_datafield_list_dicts(elem, tag)
        return [d[code] for d in dfs if code in d]

    def _get_datafield_list_dicts(
        self, elem: ET.Element, tag: str
    ) -> List[Dict[str, str]]:
        dfs = elem.findall(f".//marc:datafield[@tag='{tag}']", self.MARC_NS)
        if not dfs:
            dfs = elem.findall(f".//datafield[@tag='{tag}']")

        results: List[Dict[str, str]] = []

        for df in dfs:
            entry: Dict[str, str] = {}
            for sf in df.findall("marc:subfield", self.MARC_NS) or df.findall(
                "subfield"
            ):
                code = sf.get("code")
                if code and sf.text:
                    text = sf.text.strip()
                    entry[code] = (
                        f"{entry[code]} ; {text}" if code in entry else text
                    )
            if entry:
                results.append(entry)

        return results

    # ------------------------------------------------------------------
    # Publication Extraction
    # ------------------------------------------------------------------

    def _extract_publication(
        self, elem: ET.Element, control_008: str
    ) -> Dict[str, str]:

        publication_place = ""
        publisher = ""
        publication_year = ""

        pub_fields = self._get_datafield_list_dicts(elem, "264")

        if pub_fields:
            first = pub_fields[0]
            publication_place = first.get("a", "")
            publisher = first.get("b", "")
            publication_year = first.get("c", "")

        # Fallback 502
        if not publication_year:
            year_502 = self._get_datafield(elem, "502").get("c", "")
            publication_year = year_502

        # Fallback 008
        if not publication_year and len(control_008) >= 11:
            publication_year = control_008[7:11]

        return {
            "publication_place": publication_place,
            "publisher": publisher,
            "publication_year": publication_year,
        }

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def _join_subfields(
        self, data: Dict[str, str], keys: List[str]
    ) -> str:
        return " ; ".join([data[k] for k in keys if k in data])

    def _extract_gnd(self, raw: str) -> str:
        if not raw:
            return ""

        parts = [p.strip() for p in raw.split(";")]

        for part in parts:
            if "gnd/" in part:
                return part.split("gnd/")[-1].strip()
            if "(DE-588)" in part:
                return part.split("(DE-588)")[-1].strip()

        return parts[0] if parts else ""


# ----------------------------------------------------------------------
# Convenience Function
# ----------------------------------------------------------------------

def parse_dnb_theses(
    filepath: str,
    limit: Optional[int] = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Convenience-Wrapper für den Marc21Parser.
    """
    return Marc21Parser().parse_file(filepath, limit=limit, verbose=verbose)


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------

if __name__ == "__main__":
    print("Marc21Parser – DNB Hochschulschriften")
    df = parse_dnb_theses("your_file.xml.gz", limit=100)
    print(df.head())
