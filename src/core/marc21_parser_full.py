import pandas as pd
import gzip
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Dict, List, Optional, Union #, TypedDict

SubfieldValue = Union[str, List[str]]
DatafieldDict = Dict[str, SubfieldValue]
DatafieldList = List[DatafieldDict]
RecordDict = Dict[str, Any]

class Marc21Parser:
    """
    Robuster MARC21-XML Parser (DNB-Hochschulschriften).

    Extrahiert:
    - ID & Controlfields
    - Autor inkl. GND-Nummer
    - Titel
    - Publikationsangaben
    - Dissertation-Vermerk
    - DDC/Sachgruppen (082, 083, 084)
    - Sprache
    - Schlagwörter
    
    Wichtig:
    - 082/083/084 behalten wiederholbare Subfelder (z. B. $a)
    als echte Listen pro Datafield.
    - Andere Felder aggregieren Mehrfach-Subfelder mit " ; ".

    Output:
    - Ein Record = Dict[str, Any]
    - Klassifikationen: List[Dict[str, Union[str, List[str]]]]

    Speicher- und performance-optimiert für große XML-Dateien.
    """

    MARC_NS = {"marc": "http://www.loc.gov/MARC21/slim"}

    def parse_file(
        self, filepath: str, limit: Optional[int] = None, verbose=True
    ) -> pd.DataFrame:
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {filepath}")
        opener = gzip.open if path.suffix == ".gz" else open
        records: List[Dict[str, Any]] = []
        count = 0
        with opener(path, "rb") as file_obj:
            context = ET.iterparse(file_obj, events=("end",))
            for _, elem in context:
                if elem.tag.endswith("record"):
                    record = self._parse_record(elem)
                    if record:
                        records.append(record)
                        count += 1
                        if verbose and count % 1000 == 0:
                            print(f"Parsed {count} records")
                        if limit and count >= limit:
                            break
                    elem.clear()
        if verbose:
            print(f"Finished parsing {count} records")
        return pd.DataFrame(records)

    # ------------- Convenience-Methode ----------------
    @staticmethod
    def parse_dnb_theses(
        filepath: str, limit: Optional[int] = None, verbose=True
    ) -> pd.DataFrame:
        return Marc21Parser().parse_file(filepath, limit=limit, verbose=verbose)

    # ------------- Record Parsing ----------------
    def _parse_record(self, elem: ET.Element) -> Optional[Dict[str, Any]]:
        try:
            data: Dict[str, Any] = {}
            data["record_id"] = self._get_controlfield(elem, "001")
            control_008 = self._get_controlfield(elem, "008")
            data["control_008"] = control_008

            # Language
            lang_list = self._get_datafield_list(elem, "041", "a")
            data["language"] = lang_list[0] if lang_list else ""

            # Author
            author = self._get_first_datafield(elem, ["100", "110"])
            data["author_name"] = author.get("a", "")
            data["author_dates"] = author.get("d", "")
            data["author_gnd"] = self._extract_gnd(author.get("0", ""))
        

            # Title
            title = self._get_datafield(elem, "245")
            data["title"] = title.get("a", "")
            data["title_remainder"] = title.get("b", "")

            # Publication
            pub = self._extract_publication(elem, control_008)
            data.update(pub)

            # Dissertation
            diss = self._get_datafield(elem, "502")
            data["dissertation_note"] = " ; ".join(
                [diss.get(k, "") for k in ["a", "b", "c", "d"] if k in diss]
            )

            # DDC/SDNB
            for tag in ["082", "083", "084"]:
                data[f"{tag}_list"] = self._get_datafield_list_dicts(
                        elem,
                        tag,
                        repeatable_subfields=True
                    )

            # Subjects
            subjects: List[str] = []
            for tag in ["600", "610", "650", "651", "655"]:
                subjects.extend(self._get_datafield_list(elem, tag, "a"))
            data["subjects"] = subjects
            data["subject_count"] = len(subjects)
            return data
        except Exception as e:
            print(f"Error parsing record: {e}")
            return None

    # ---------------- Helper Methods ----------------
    def _get_controlfield(self, elem: ET.Element, tag: str) -> str:
        cf = elem.find(f".//marc:controlfield[@tag='{tag}']", self.MARC_NS)
        if cf is None:
            cf = elem.find(f".//controlfield[@tag='{tag}']")
        return cf.text.strip() if cf is not None and cf.text else ""

    def _get_first_datafield(self, elem: ET.Element, tags)-> DatafieldDict:
        for tag in tags:
            df = self._get_datafield(elem, tag)
            if df:
                return df
        return {}

    def _get_datafield(self, elem, tag) -> DatafieldDict:
        """Gibt erstes Datafield zurück. Nur für nicht-wiederholbare Felder."""
        dfs = self._get_datafield_list_dicts(elem, tag, repeatable_subfields=False)
        return dfs[0] if dfs else {}

    def _get_datafield_list(self, elem, tag, code) -> List[str]:
        """Nur für nicht-wiederholbare Felder (Subjects, Language)."""
        dfs = self._get_datafield_list_dicts(elem, tag, repeatable_subfields=False)
        values = []
        for d in dfs:
            val = d.get(code)
            if val is None:
                continue
            if isinstance(val, list):
                values.extend(val)
            else:
                values.append(val)
        return values

    def _get_datafield_list_dicts(self, elem, tag, repeatable_subfields=False) -> List[DatafieldDict]:
        """
        Extrahiert alle Datafields eines Tags als Liste von Dicts.

        Parameter
        ---------
        repeatable_subfields : bool
            Wenn True, werden mehrfach vorkommende Subfelder
            als List[str] gespeichert (z.B. 082/083/084).
            Wenn False (Default), werden sie mit " ; " aggregiert.

        Returns
        -------
        List[Dict[str, Union[str, List[str]]]]
        """     
        dfs = elem.findall(
            f".//marc:datafield[@tag='{tag}']", self.MARC_NS
        ) or elem.findall(f".//datafield[@tag='{tag}']")

        results = []

        for df in dfs:
            entry = {}

            for sf in df.findall("marc:subfield", self.MARC_NS) or df.findall("subfield"):
                code = sf.get("code")
                if code and sf.text:
                    text = sf.text.strip()

                    if repeatable_subfields:
                        entry.setdefault(code, []).append(text)
                    else:
                        entry[code] = (
                            f"{entry[code]} ; {text}"
                            if code in entry
                            else text
                        )

            if entry:
                results.append(entry)

        return results

    def _extract_publication(self, elem, control_008) -> Dict[str, str]:
        place = publisher = year = ""
        pub_fields = self._get_datafield_list_dicts(elem, "264")
        
        if pub_fields:
            first = pub_fields[0]
            place = first.get("a", "")
            if isinstance(place, list):
                place = place[0] if place else ""
            publisher = first.get("b", "")
            if isinstance(publisher, list):
                publisher = publisher[0] if publisher else ""
            year = first.get("c", "")
            if isinstance(year, list):
                year = year[0] if year else ""
                year = str(year).strip()
        if not year:
            year = self._get_datafield(elem, "502").get("c", "")
        if not year and len(control_008) >= 11:
            year = control_008[7:11]
        return {
            "publication_place": place,
            "publisher": publisher,
            "publication_year": year,
        }

    def _extract_gnd(self, raw: str) -> str:
        if not raw:
            return ""
        for part in [p.strip() for p in raw.split(";")]:
            if "gnd/" in part:
                return part.split("gnd/")[-1].strip()
            if "(DE-588)" in part:
                return part.split("(DE-588)")[-1].strip()
        return ""
