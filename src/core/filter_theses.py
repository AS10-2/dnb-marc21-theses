"""
filter_theses.py

Filtert DNB-Hochschulschriften-DataFrames nach Jahr und Schlagwörtern.

Funktioniert auf allen Pipeline-Stufen:
    df_raw        → subjects als String (noch nicht gesplittet)
    df_clean      → subjects als List[str]
    df_transformed → wie df_clean, zusätzlich is_mineralogie, has_sdnb

Pipeline:
    Marc21Parser → Cleaner → ClassificationTransformer → Filter → Explorer
"""

import pandas as pd
from typing import List, Optional, Union


class Filter:
    """
    Filtert Dissertationen nach Jahr, Schlagwörtern,
    Sprache, Klassifikation und Vollständigkeit.

    Erkennt automatisch ob subjects als String oder List[str] vorliegt.
    """

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.df["publication_year"] = pd.to_numeric(
            self.df["publication_year"], errors="coerce"
        )
        # Merken ob subjects Liste oder String ist
        self._subjects_are_lists = (
            "subjects" in self.df.columns
            and self.df["subjects"].apply(lambda x: isinstance(x, list)).any()
        )

    # --------------------------------------------------
    # Jahr-Filter
    # --------------------------------------------------
    def filter_by_year(
        self,
        year: Optional[int] = None,
        year_range: Optional[tuple] = None,
        year_list: Optional[List[int]] = None,
    ) -> "Filter":
        df = self.df.copy()
        if year is not None:
            df = df[df["publication_year"] == year]
        elif year_range is not None:
            start, end = year_range
            df = df[df["publication_year"].between(start, end)]
        elif year_list is not None:
            df = df[df["publication_year"].isin(year_list)]
        return Filter(df)

    # --------------------------------------------------
    # Schlagwort-Filter — List[str] und String
    # --------------------------------------------------
    def filter_by_keywords(
        self,
        keywords: Union[str, List[str]],
        exact_match: bool = False,
        require_all: bool = False,
    ) -> "Filter":
        """
        Filtert nach Schlagwörtern in subjects.

        Parameters
        ----------
        keywords    : str oder List[str]
        exact_match : Ganzes Wort matchen (\\b)
        require_all : True → alle Keywords müssen vorkommen (AND)
                      False → mindestens eines (OR)
        """
        if isinstance(keywords, str):
            keywords = [keywords]
        keywords_lower = [k.lower() for k in keywords]

        if self._subjects_are_lists:
            # subjects ist List[str] — nach Cleaner
            def matches(subject_list):
                if not isinstance(subject_list, list):
                    return False
                text = " ".join(s.lower() for s in subject_list)
                if exact_match:
                    hits = [f" {k} " in f" {text} " for k in keywords_lower]
                else:
                    hits = [k in text for k in keywords_lower]
                return all(hits) if require_all else any(hits)

            mask = self.df["subjects"].apply(matches)
        else:
            # subjects ist String — df_raw
            if exact_match:
                pattern = "|".join(rf"\b{k}\b" for k in keywords_lower)
            else:
                pattern = "|".join(keywords_lower)

            if require_all:
                mask = pd.Series(True, index=self.df.index)
                for k in keywords_lower:
                    p = rf"\b{k}\b" if exact_match else k
                    mask &= self.df["subjects"].str.contains(
                        p, case=False, na=False, regex=True
                    )
            else:
                mask = self.df["subjects"].str.contains(
                    pattern, case=False, na=False, regex=True
                )

        return Filter(self.df[mask])

    # --------------------------------------------------
    # Titel-Filter
    # --------------------------------------------------
    def filter_by_title(
        self,
        keywords: Union[str, List[str]],
        require_all: bool = False,
    ) -> "Filter":
        """Sucht in title + title_remainder."""
        if isinstance(keywords, str):
            keywords = [keywords]

        text_col = (
            self.df.get("title", pd.Series("", index=self.df.index)).fillna("")
            + " "
            + self.df.get("title_remainder", pd.Series("", index=self.df.index)).fillna("")
        ).str.lower()

        if require_all:
            mask = pd.Series(True, index=self.df.index)
            for k in keywords:
                mask &= text_col.str.contains(k.lower(), na=False)
        else:
            pattern = "|".join(k.lower() for k in keywords)
            mask = text_col.str.contains(pattern, na=False)

        return Filter(self.df[mask])

    # --------------------------------------------------
    # Sprach-Filter
    # --------------------------------------------------
    def filter_by_language(
        self, languages: Union[str, List[str]]
    ) -> "Filter":
        """z.B. filter_by_language('ger') oder ['ger', 'eng']"""
        if isinstance(languages, str):
            languages = [languages]
        mask = self.df["language"].isin(languages)
        return Filter(self.df[mask])

    # --------------------------------------------------
    # Klassifikations-Filter (df_transformed)
    # --------------------------------------------------
    def filter_mineralogie(self) -> "Filter":
        """Nur Records mit is_mineralogie == True."""
        if "is_mineralogie" not in self.df.columns:
            raise ValueError("Benötigt df_transformed (is_mineralogie fehlt)")
        return Filter(self.df[self.df["is_mineralogie"]])

    def filter_by_ddc(
        self, prefixes: Union[str, List[str]]
    ) -> "Filter":
        """
        Filtert auf DDC-Präfix.
        filter_by_ddc("549") → alle Mineralogie-Records via DDC
        filter_by_ddc(["549", "550"]) → Mineralogie + Geowissenschaften
        """
        if "ddc_primary_3digit" not in self.df.columns:
            raise ValueError("Benötigt df_transformed (ddc_primary_3digit fehlt)")
        if isinstance(prefixes, str):
            prefixes = [prefixes]
        mask = self.df["ddc_primary_3digit"].apply(
            lambda x: any(str(x).startswith(p) for p in prefixes)
        )
        return Filter(self.df[mask])

    def filter_unclassified(self) -> "Filter":
        """Records ohne jede Klassifikation — Retro-Kandidaten."""
        conditions = []
        if "ddc_primary_3digit" in self.df.columns:
            conditions.append(self.df["ddc_primary_3digit"].str.len() == 0)
        if "has_sdnb" in self.df.columns:
            conditions.append(~self.df["has_sdnb"])
        if "sdnb_codes" in self.df.columns and "has_sdnb" not in self.df.columns:
            conditions.append(self.df["sdnb_codes"].apply(len) == 0)
        if not conditions:
            raise ValueError("Keine Klassifikationsspalten gefunden")
        mask = conditions[0]
        for c in conditions[1:]:
            mask = mask & c
        return Filter(self.df[mask])

    # --------------------------------------------------
    # Kombinierter Filter
    # --------------------------------------------------
    def filter(
        self,
        year: Optional[int] = None,
        year_range: Optional[tuple] = None,
        year_list: Optional[List[int]] = None,
        keywords: Optional[Union[str, List[str]]] = None,
        title_keywords: Optional[Union[str, List[str]]] = None,
        language: Optional[Union[str, List[str]]] = None,
        ddc_prefix: Optional[Union[str, List[str]]] = None,
        only_mineralogie: bool = False,
        only_unclassified: bool = False,
        require_all_keywords: bool = False,
    ) -> "Filter":
        """Kombiniert alle Filter in einem Aufruf."""
        result = self
        if year or year_range or year_list:
            result = result.filter_by_year(year, year_range, year_list)
        if keywords:
            result = result.filter_by_keywords(
                keywords, require_all=require_all_keywords
            )
        if title_keywords:
            result = result.filter_by_title(title_keywords)
        if language:
            result = result.filter_by_language(language)
        if ddc_prefix:
            result = result.filter_by_ddc(ddc_prefix)
        if only_mineralogie:
            result = result.filter_mineralogie()
        if only_unclassified:
            result = result.filter_unclassified()
        return result

    # --------------------------------------------------
    # Ergebnis abrufen
    # --------------------------------------------------
    def result(self) -> pd.DataFrame:
        return self.df.copy()

    def __len__(self) -> int:
        return len(self.df)

    def summary(self) -> None:
        print(f"Records:          {len(self.df):,}")
        if "publication_year" in self.df.columns:
            yr = self.df["publication_year"].dropna()
            if len(yr):
                print(f"Jahr:             {int(yr.min())}–{int(yr.max())}")
        if "language" in self.df.columns:
            print(f"Sprachen:         {dict(self.df['language'].value_counts().head(3))}")
        if "is_mineralogie" in self.df.columns:
            n = self.df["is_mineralogie"].sum()
            print(f"Mineralogie:      {n:,} ({n/len(self.df)*100:.1f}%)")
        if "has_sdnb" in self.df.columns:
            n = self.df["has_sdnb"].sum()
            print(f"Mit SDNB:         {n:,} ({n/len(self.df)*100:.1f}%)")
