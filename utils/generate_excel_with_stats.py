import sys
import os
import pandas as pd
import numpy as np
from openpyxl import Workbook
from openpyxl.styles import PatternFill, Font
from openpyxl.utils import get_column_letter

def round_or_div0(value):
    if pd.isna(value):
        return '#DIV/0!'
    return round(value, 3)

def generate_excel_from_csv(csv_path):
    # Load CSV
    df = pd.read_csv(csv_path)

    # Extract base prefix (e.g., "deadlock", "sleeping") from filename
    base_name = os.path.basename(csv_path)
    parts = base_name.split('_')
    base_prefix = parts[1] if len(parts) > 1 else parts[0]

    # Metadata
    methods = ['Ans', 'Classifier', 'MLP', 'BF', 'SA', 'GA']
    variants = ['best', '5th', '10th']
    blocks = [f"{i}k" for i in range(1, 21)]
    filtered_cols = [f"{m}_{v}" for m in methods for v in variants if not (m in ['SA', 'GA'] and v != 'best')]
    header = ['', '', *filtered_cols]

    # Create Excel workbook
    wb = Workbook()
    ws = wb.active
    ws.title = "Results"
    ws.append(header)

    # Style header
    gray_fill = PatternFill(start_color='D9D9D9', end_color='D9D9D9', fill_type='solid')
    bold_font = Font(bold=True)
    for cell in ws[1]:
        cell.font = bold_font
        cell.fill = gray_fill

    # Output rows
    for idx, block in enumerate(blocks):
        prefix = f"{base_prefix}_{block}_"
        ordered_cols = [prefix + col for col in filtered_cols]
        available_cols = [col for col in ordered_cols if col in df.columns]

        # Create a sub-DataFrame with fallback if some columns are missing
        block_df = df[available_cols].apply(pd.to_numeric, errors='coerce')

        # Reindex to match expected structure (missing values will be NaN)
        block_df = block_df.reindex(columns=ordered_cols)

        avg = block_df.mean(skipna=True)
        std = block_df.std(skipna=True, ddof=1)

        row_label = 100 + idx * 200
        ws.append([row_label, 'AVRG'] + list(map(round_or_div0, avg)))
        ws.append(['', 'STD'] + list(map(round_or_div0, std)))

    # Merge row labels
    for i in range(2, 42, 2):
        ws.merge_cells(start_row=i, start_column=1, end_row=i+1, end_column=1)

    # Set column widths
    for i in range(1, len(header) + 1):
        ws.column_dimensions[get_column_letter(i)].width = 15

    # Save Excel file
    base, _ = os.path.splitext(csv_path)
    out_path = base + "_data.xlsx"
    wb.save(out_path)
    print(f"✅ Saved: {out_path}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python generate_excel_from_csv.py <input.csv>")
        sys.exit(1)
    generate_excel_from_csv(sys.argv[1])
