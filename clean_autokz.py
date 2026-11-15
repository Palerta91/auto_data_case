#!/usr/bin/env python3
"""Clean Kazakhstan sales; preserve observations or remove exact output repeats.

Requires Python 3.10+ and pandas 2.2+.
Examples:
  python clean_autokz.py autokz2019.csv autokz2019_analysis_base.csv
  python clean_autokz.py autokz2019.csv autokz2019_clean.csv --drop-exact-duplicates

The second mode satisfies the uniqueness requirement for the delivered CSV.
It cannot establish transaction-level uniqueness without a VIN or transaction ID.
"""

import argparse
import calendar
import re
from pathlib import Path

import pandas as pd


MONTHS_RU = {
    "январь": 1, "февраль": 2, "март": 3, "апрель": 4, "май": 5,
    "июнь": 6, "июль": 7, "август": 8, "сентябрь": 9,
    "октябрь": 10, "ноябрь": 11, "декабрь": 12,
}

COUNTRY_ALPHA3 = {
    "германия": "DEU", "сша": "USA", "австрия": "AUT",
    "республика казахстан": "KAZ", "российская федерация": "RUS",
    "корея": "KOR", "япония": "JPN", "таиланд": "THA", "китай": "CHN",
    "uk": "GBR", "узбекистан": "UZB", "венгрия": "HUN", "турция": "TUR",
    "испания": "ESP", "нидерланды": "NLD", "польша": "POL",
    "швеция": "SWE", "белоруссия": "BLR", "бельгия": "BEL",
}

COMPANY_ALIASES = {
    "astana motors": "Astana Motors",
    "mercur auto": "Mercur Auto",
    "mercur autos": "Mercur Auto",
    "меркур авто": "Mercur Auto",
    "равон моторс казахстан": "Ravon Motors Kazakhstan",
    "ravon motors kazakstan": "Ravon Motors Kazakhstan",
    "автокапитал": "Autokapital",
    "ммс рус": "MMC RUS",
    "caspian motors": "Каспиан Моторс",
    "hino motors": "Хино Моторс Казахстан",
}

DRIVE_ALIASES = {
    "4wd": "4WD", "4 wd": "4WD", "полный": "4WD", "полный привод": "4WD", "awd": "4WD",
    "quattro": "4WD", "4motion": "4WD", "4x4": "4WD", "4х4": "4WD",
    "2wd": "2WD", "2 wd": "2WD", "4x2": "2WD", "4х2": "2WD",
    "4х2.2": "2WD", "fwd": "FWD", "ff": "FWD", "передний": "FWD",
    "передний (ff)": "FWD", "rwd": "RWD", "задний": "RWD",
}


def number(value):
    """Parse a comma- or dot-decimal numeric field."""
    if pd.isna(value) or not str(value).strip():
        return pd.NA
    return pd.to_numeric(str(value).replace("\xa0", "").replace(" ", "").replace(",", "."), errors="coerce")


def engine_volume(value):
    """Extract the engine displacement and ignore trims or power units."""
    if pd.isna(value):
        return pd.NA
    text = str(value).strip().replace(",", ".")
    if not text or "kwh" in text.casefold() or "л.с" in text.casefold():
        return pd.NA
    match = re.search(r"\d+(?:\.\d+)?", text)
    return float(match.group()) if match else pd.NA


def transmission(value):
    """Reduce gearbox labels to the two requested groups."""
    if pd.isna(value):
        return pd.NA
    text = str(value).strip().casefold().replace(" ", "")
    if text.startswith(("мех", "механ")):
        return "Механика"
    text = text.translate(str.maketrans({"а": "a", "т": "t", "м": "m", "к": "k", "п": "p"}))
    if not text or text in {"0", "#н/д"}:
        return pd.NA
    # Automated manual gearboxes belong to the automatic group in this binary taxonomy.
    if "amt" in text:
        return "Автомат"
    if any(token in text for token in ("mt", "mkp", "mex", "m/t", "5m")):
        return "Механика"
    if "редуктор" in str(value).casefold() or any(token in text for token in (
        "at", "a/t", "akp", "cvt", "dsg", "dct", "tronic", "tiptronic",
        "steptronic", "powershift", "pdk",
    )):
        return "Автомат"
    if re.fullmatch(r"\d*a", text):
        return "Автомат"
    return pd.NA


def sale_date(year, month):
    month_number = MONTHS_RU.get(str(month).strip().casefold())
    year_number = number(str(year).replace(" ", ""))
    if pd.isna(year_number) or month_number is None:
        return pd.NaT
    year_number = int(year_number)
    last_day = calendar.monthrange(year_number, month_number)[1]
    return pd.Timestamp(year_number, month_number, last_day)


def clean(input_path, output_path, drop_exact_duplicates=False):
    # Keep the source as text so mixed decimal and label errors stay visible.
    df = pd.read_csv(
        input_path, sep=";", decimal=",", dtype=str,
        keep_default_na=False, encoding="utf-8-sig", low_memory=False,
    )
    df = df.apply(lambda col: col.str.replace("\xa0", " ", regex=False).str.strip())
    df = df.replace("", pd.NA)

    rows_read = len(df)
    df = df.dropna(how="all").copy()
    blank_rows = rows_read - len(df)
    # Count repeats on all source fields before removing informational columns.
    duplicate_rows = int(df.duplicated().sum())
    duplicates_dropped = 0
    if drop_exact_duplicates:
        df = df.drop_duplicates().copy()
        duplicates_dropped = duplicate_rows

    rename = {
        "Год": "sale_year", "Месяц": "sale_month", "Компания": "company",
        "Бренд": "brand", "Модель": "model", "Год выпуска": "production_year",
        "Страна-производитель": "country_of_origin_alpha3", "Вид топлива": "fuel_type",
        "Объём двиг, л,": "engine_volume_l", "Коробка передач": "transmission_type",
        "Тип привода": "drive_type", "Регион": "region", "Количество": "quantity",
        "Цена, USD": "unit_price_usd", "Продажа, USD": "sales_usd", "Область": "oblast",
        "Сегментация 2013": "segment_2013", "Класс 2013": "class_2013",
    }
    # Remove source fields that do not support the requested analysis.
    df = df.drop(columns=[
        "Модификация", "Наименование дилерского центра", "Форма расчета", "Сегмент",
        "Сегментация Eng", "Локализация производства", "Тип клиента",
    ])
    df = df.rename(columns=rename)

    # Renault Duster/Sandero rows have displacement and gearbox shifted one column left.
    shifted = (
        df["brand"].eq("Renault")
        & df["model"].isin(["Duster", "Sandero"])
        & df["engine_volume_l"].str.casefold().isin(["at", "mt", "0"])
        & df["fuel_type"].isin(["2", "1,6"])
    )
    df.loc[shifted, "drive_type"] = df.loc[shifted, "transmission_type"]
    df.loc[shifted, "transmission_type"] = df.loc[shifted, "engine_volume_l"]
    df.loc[shifted, "engine_volume_l"] = df.loc[shifted, "fuel_type"]
    df.loc[shifted, "fuel_type"] = pd.NA

    df["sale_date"] = [sale_date(y, m) for y, m in zip(df["sale_year"], df["sale_month"])]
    df["production_year"] = pd.to_numeric(
        df["production_year"].str.replace(" ", "", regex=False), errors="coerce"
    ).astype("Int64")
    df["quantity"] = df["quantity"].map(number).astype("Int64")
    for column in ("unit_price_usd", "sales_usd"):
        df[column] = df[column].map(number).astype("Float64")

    # Six records have no quantity and zero sales. Preserve the missing quantity;
    # neither a sale nor a cancelled transaction can be established from this alone.
    if df[["unit_price_usd", "sales_usd"]].isna().any().any():
        raise ValueError("Unparseable monetary fields; inspect the source before export")
    known_quantity = df["quantity"].notna()
    mismatch = (df["sales_usd"] - df["quantity"] * df["unit_price_usd"]).abs().gt(0.02)
    if mismatch.loc[known_quantity].any():
        raise ValueError("Sales differ from quantity × unit price by more than 0.02 USD")
    if (df["quantity"].isna() & df["sales_usd"].ne(0)).any():
        raise ValueError("Nonzero sales with unknown quantity require manual review")

    df["country_of_origin_alpha3"] = df["country_of_origin_alpha3"].str.casefold().map(COUNTRY_ALPHA3)
    fuel_map = {
        "бензин": "F", "дизель": "D", "электро": "E",
        "электричество": "E", "гибрид": "HYB",
    }
    df["fuel_type"] = df["fuel_type"].str.casefold().map(fuel_map)
    df["company"] = df["company"].map(
        lambda value: COMPANY_ALIASES.get(str(value).casefold(), value)
    )
    df["drive_type"] = df["drive_type"].str.casefold().map(DRIVE_ALIASES)
    df["transmission_type"] = df["transmission_type"].map(transmission)

    df["engine_volume_l"] = df["engine_volume_l"].map(engine_volume).astype("Float64")
    df.loc[df["engine_volume_l"].eq(0), "engine_volume_l"] = pd.NA
    # Chevrolet Niva engines are 1.7 L; the 2.7–26.7 sequence is an export error.
    niva_fix = (
        df["brand"].eq("Chevrolet") & df["model"].eq("Niva")
        & df["engine_volume_l"].between(2.7, 26.7, inclusive="both")
    )
    df.loc[niva_fix, "engine_volume_l"] = 1.7

    df["region"] = df["region"].str.title()
    df["oblast"] = df["oblast"].str.title().str.replace("Г.", "Г. ", regex=False)
    df["sale_date"] = pd.to_datetime(df["sale_date"])
    if df["sale_date"].isna().any():
        raise ValueError("Unparseable sale year/month")
    for column in ("fuel_type", "transmission_type", "drive_type", "segment_2013", "class_2013"):
        df[column] = df[column].astype("category")

    columns = [
        "sale_date", "company", "brand", "model", "production_year",
        "country_of_origin_alpha3", "fuel_type", "engine_volume_l", "transmission_type",
        "drive_type", "region", "quantity", "unit_price_usd", "sales_usd", "oblast",
        "segment_2013", "class_2013",
    ]
    df = df[columns]
    # Column removal and normalization create additional exact output matches.
    # Strict mode guarantees uniqueness of all 17 delivered fields as requested.
    output_duplicates_dropped = 0
    if drop_exact_duplicates:
        output_duplicates_dropped = int(df.duplicated().sum())
        df = df.drop_duplicates().copy()
    df = df.reset_index(drop=True)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False, encoding="utf-8-sig", date_format="%Y-%m-%d")
    print(
        f"Read {rows_read:,} rows; removed {blank_rows:,} blank rows; found "
        f"{duplicate_rows:,} exact source repeats; dropped {duplicates_dropped:,} "
        f"source repeats and {output_duplicates_dropped:,} additional output repeats."
    )
    print(f"Saved {len(df):,} rows and {len(df.columns)} columns to {output_path}")
    print("Missing fields:", {c: int(n) for c, n in df.isna().sum().items() if n})
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_csv", help="Source semicolon-delimited CSV")
    parser.add_argument("output_csv", help="Path for the cleaned CSV")
    parser.add_argument(
        "--drop-exact-duplicates", action="store_true",
        help="Remove exact source repeats and all exact repeats in the final 17 fields",
    )
    args = parser.parse_args()
    clean(args.input_csv, args.output_csv, args.drop_exact_duplicates)
