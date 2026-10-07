"""Prepare a reproducible product catalog from the local fashion dataset."""

import argparse
import csv
import random
import sys
from pathlib import Path

import pandas as pd
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = Path("data/raw/fashion-product-images-small")
OUTPUT_PATH = Path("data/processed/catalog.csv")
CATEGORIES = ("Apparel", "Footwear", "Accessories")
OUTPUT_COLUMNS = (
    "image_id",
    "product_id",
    "image_path",
    "master_category",
    "sub_category",
    "article_type",
    "base_colour",
    "product_display_name",
)
REQUIRED_METADATA_COLUMNS = {
    "id",
    "masterCategory",
    "subCategory",
    "articleType",
    "baseColour",
    "productDisplayName",
}


def read_metadata(metadata_path: Path) -> tuple[pd.DataFrame, int]:
    """Read metadata and recover unquoted commas in the final name column."""
    with metadata_path.open(encoding="utf-8-sig", newline="") as source:
        header = next(csv.reader(source), [])

    missing_columns = REQUIRED_METADATA_COLUMNS.difference(header)
    if missing_columns:
        raise ValueError(f"Missing metadata columns: {', '.join(sorted(missing_columns))}")
    if header[-1] != "productDisplayName":
        raise ValueError("Expected productDisplayName to be the last metadata column.")

    repaired_rows = 0

    def repair_row(fields: list[str]) -> list[str]:
        nonlocal repaired_rows
        repaired_rows += 1
        name_index = len(header) - 1
        return fields[:name_index] + [",".join(fields[name_index:])]

    metadata = pd.read_csv(
        metadata_path,
        dtype=str,
        keep_default_na=False,
        encoding="utf-8-sig",
        engine="python",
        on_bad_lines=repair_row,
    )
    return metadata, repaired_rows


def image_is_readable(image_path: Path) -> bool:
    """Check the image file structure and fully decode its pixels."""
    try:
        if not image_path.is_file():
            return False
        with Image.open(image_path) as image:
            image.verify()
        with Image.open(image_path) as image:
            image.load()
    except (OSError, ValueError, SyntaxError, Image.DecompressionBombError):
        return False
    return True


def prepare_catalog(limit: int) -> None:
    metadata, repaired_rows = read_metadata(REPO_ROOT / DATASET_PATH / "styles.csv")
    filtered = metadata[metadata["masterCategory"].isin(CATEGORIES)]

    records = []
    seen_ids = set()
    invalid_images = 0
    invalid_ids = 0
    duplicate_ids = 0

    for row in filtered.itertuples(index=False):
        product_id = row.id
        if not isinstance(product_id, str) or not product_id.isascii() or not product_id.isdecimal():
            invalid_ids += 1
            continue

        relative_image_path = DATASET_PATH / "images" / f"{product_id}.jpg"
        if not image_is_readable(REPO_ROOT / relative_image_path):
            invalid_images += 1
            continue
        if product_id in seen_ids:
            duplicate_ids += 1
            continue

        seen_ids.add(product_id)
        records.append(
            {
                "image_id": product_id,
                "product_id": product_id,
                "image_path": relative_image_path.as_posix(),
                "master_category": row.masterCategory,
                "sub_category": row.subCategory,
                "article_type": row.articleType,
                "base_colour": row.baseColour,
                "product_display_name": row.productDisplayName,
            }
        )

    # Validate the complete candidate set before shuffling exactly once.
    random.Random(42).shuffle(records)
    catalog = pd.DataFrame(records[:limit], columns=OUTPUT_COLUMNS)
    if not catalog["image_id"].is_unique or not catalog["product_id"].is_unique:
        raise ValueError("Catalog IDs must be unique.")

    output_path = REPO_ROOT / OUTPUT_PATH
    output_path.parent.mkdir(parents=True, exist_ok=True)
    catalog.to_csv(output_path, index=False, encoding="utf-8", lineterminator="\n")

    print(f"Metadata rows: {len(metadata)}")
    print(f"After category filter: {len(filtered)}")
    print(f"Invalid/missing images: {invalid_images}")
    print(f"Final catalog size: {len(catalog)}")
    print("Category counts:")
    counts = catalog["master_category"].value_counts()
    for category in CATEGORIES:
        print(f"{category}: {counts.get(category, 0)}")

    if repaired_rows:
        print(f"Metadata rows with repaired name commas: {repaired_rows}")
    if invalid_ids:
        print(f"Invalid product IDs skipped: {invalid_ids}")
    if duplicate_ids:
        print(f"Duplicate product IDs skipped: {duplicate_ids}")
    if len(catalog) < limit:
        print(
            f"Warning: requested {limit} rows, but only {len(catalog)} valid unique products are available.",
            file=sys.stderr,
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=5000, help="Maximum catalog size (default: 5000).")
    args = parser.parse_args()
    if args.limit <= 0:
        parser.error("--limit must be a positive integer.")

    try:
        prepare_catalog(args.limit)
    except (OSError, ValueError, csv.Error, pd.errors.ParserError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
