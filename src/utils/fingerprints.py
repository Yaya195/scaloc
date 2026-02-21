import pandas as pd


def ap_columns(df: pd.DataFrame) -> list[str]:
    """Infer AP columns by naming convention WAP###."""
    return [col for col in df.columns if col.upper().startswith("WAP")]
