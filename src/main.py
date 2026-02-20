from core.marc21_parser_full import Marc21Parser
from core.cleaning import Cleaner
from core.classification_transform import ClassificationTransformer
from core.data_explorer import DataExplorer

# Pfad zum Datensatz
parquet_file = "/Users/AS/Desktop/Portfolio/dnb-marc21-theses/data/raw/dnb_all_theses_100.parquet"

# -----------------------------
# 1️⃣ Parsing
# -----------------------------
df_parsed = Marc21Parser.parse_dnb_theses(parquet_file, verbose=True)
print(f"Nach Parsing: {df_parsed.shape[0]} Zeilen, {df_parsed.shape[1]} Spalten")

# -----------------------------
# 2️⃣ Cleaning
# -----------------------------
cleaner = Cleaner()
df_clean = cleaner.apply(df_parsed)
print(f"Nach Cleaning: {df_clean.shape[0]} Zeilen, {df_clean.shape[1]} Spalten")

# -----------------------------
# 3️⃣ Transformation
# -----------------------------
transformer = ClassificationTransformer()
df_transformed = transformer.apply(df_clean)
print(f"Nach Transformation: {df_transformed.shape[0]} Zeilen, {df_transformed.shape[1]} Spalten")

# -----------------------------
# 4️⃣ Data Exploration
# -----------------------------
explorer = DataExplorer(df_transformed)
print(explorer.overview(max_elements_preview=3).head())
print(explorer.missing_report().head())
