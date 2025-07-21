import pandas as pd
import openpyxl
from openpyxl.styles import Font, PatternFill, Border, Side, Alignment
from openpyxl.utils import get_column_letter
import os
import math # For math.isnan

def convert_csv_to_excel_with_stats(csv_filepath, num_to_check=20, num_of_tests=50):
    """
    Converts a CSV file containing raw test results into an Excel file
    formatted with Average (AVRG) and Standard Deviation (STD) for
    specified increment steps, applying comprehensive formatting.

    Args:
        csv_filepath (str): The path to the input CSV file.
        num_to_check (int): The number of increment steps to display in the Excel output.
                            (Default: 20, as per the benchmark parameters).
        num_of_tests (int): The conceptual number of test runs per increment step.
                            (Default: 50, as per the benchmark parameters).
                            Note: For the provided sample CSVs with 50 rows, this means
                            the entire CSV is treated as data for one block of 50 tests.
                            The calculated AVR/STD will be repeated for all 'num_to_check' steps.
    """
    try:
        # Load the CSV into a pandas DataFrame
        df = pd.read_csv(csv_filepath)

        # --- BUG FIX START ---
        # Convert all columns to numeric, coercing errors to NaN.
        # This ensures that all columns that should contain numbers are correctly interpreted,
        # and any non-numeric entries that caused the bug are converted to NaN,
        # allowing mean() and std() to process them.
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        # --- BUG FIX END ---

        # Get column headers from the CSV
        csv_headers = df.columns.tolist()

        # Generate the list of increment steps (e.g., 100, 300, ..., 3900)
        increment_steps = [100 + i * 200 for i in range(num_to_check)]

        # Create a new Excel workbook and select the active sheet
        workbook = openpyxl.Workbook()
        sheet = workbook.active
        sheet.title = "Sheet1"

        # --- Define Styles ---
        header_font = Font(bold=True)
        header_fill = PatternFill(start_color="D3D3D3", end_color="D3D3D3", fill_type="solid") # Light gray
        thin_border = Border(left=Side(style='thin'),
                             right=Side(style='thin'),
                             top=Side(style='thin'),
                             bottom=Side(style='thin'))
        center_aligned_text = Alignment(horizontal="center", vertical="center")
        # Number format for 3 decimal places
        number_format = '0.000'

        # --- Prepare Combined Headers for Row 1 ---
        # Write 'Step' and 'Type' labels in header row to define the columns clearly
        sheet.cell(row=1, column=1, value='Step').font = header_font
        sheet.cell(row=1, column=1).fill = header_fill
        sheet.cell(row=1, column=1).alignment = center_aligned_text
        sheet.cell(row=1, column=1).border = thin_border

        sheet.cell(row=1, column=2, value='Type').font = header_font
        sheet.cell(row=1, column=2).fill = header_fill
        sheet.cell(row=1, column=2).alignment = center_aligned_text
        sheet.cell(row=1, column=2).border = thin_border

        # Write data-related headers (original CSV headers) from column 3 onwards
        for col_idx, header_value in enumerate(csv_headers):
            cell = sheet.cell(row=1, column=col_idx + 3, value=header_value) # Data headers start from column 3 (C)
            cell.font = header_font
            cell.fill = header_fill
            cell.alignment = center_aligned_text
            cell.border = thin_border

        # --- Calculate overall average and standard deviation for each column ---
        # Now, overall_averages and overall_stds will correctly include values for
        # deadlock_1k_BF_5th and deadlock_1k_BF_10th if they were successfully converted to numeric.
        overall_averages = df.mean(numeric_only=True)
        overall_stds = df.std(numeric_only=True)

        # Initialize the current row for writing data in Excel (starts after header row)
        excel_row = 2 # Data starts from row 2 now

        # Populate the Excel sheet with AVRG and STD rows for each increment step
        for step in increment_steps:
            # Write AVRG row
            cell_step = sheet.cell(row=excel_row, column=1, value=step)
            cell_step.border = thin_border
            cell_avrg_label = sheet.cell(row=excel_row, column=2, value='AVRG')
            cell_avrg_label.font = header_font # AVRG label is bold
            cell_avrg_label.border = thin_border

            for col_idx, header in enumerate(csv_headers):
                value = overall_averages.get(header)
                cell = sheet.cell(row=excel_row, column=col_idx + 3) # Data starts from column 3
                if value is not None and not math.isnan(value):
                    cell.value = round(value, 3) # Round to 3 decimal places
                    cell.number_format = number_format
                else:
                    cell.value = '' # Empty for NaN
                cell.border = thin_border
            excel_row += 1

            # Write STD row
            cell_empty = sheet.cell(row=excel_row, column=1, value='') # Empty as per example
            cell_empty.border = thin_border
            cell_std_label = sheet.cell(row=excel_row, column=2, value='STD')
            cell_std_label.font = header_font # STD label is bold
            cell_std_label.border = thin_border

            for col_idx, header in enumerate(csv_headers):
                value = overall_stds.get(header)
                cell = sheet.cell(row=excel_row, column=col_idx + 3) # Data starts from column 3
                if value is not None and not math.isnan(value):
                    cell.value = round(value, 3) # Round to 3 decimal places
                    cell.number_format = number_format
                else:
                    cell.value = '' # Empty for NaN
                cell.border = thin_border
            excel_row += 1

        # --- Adjust Column Widths ---
        # Estimate column widths based on header content or a reasonable default
        for col_idx in range(1, sheet.max_column + 1):
            max_length = 0
            column = get_column_letter(col_idx)
            for cell in sheet[column]:
                try:
                    if cell.value is not None and len(str(cell.value)) > max_length:
                        max_length = len(str(cell.value))
                except:
                    pass
            adjusted_width = (max_length + 2) # Add a little padding
            sheet.column_dimensions[column].width = adjusted_width if adjusted_width > 5 else 5 # Min width 5

        # Determine the output Excel filename
        output_dir = os.path.dirname(csv_filepath)
        csv_filename_without_ext = os.path.splitext(os.path.basename(csv_filepath))[0]
        # Remove the "_data.xlsx - גיליון1" suffix if present from the original name
        if csv_filename_without_ext.endswith("_data.xlsx - גיליון1"):
            csv_filename_without_ext = csv_filename_without_ext.replace("_data.xlsx - גיליון1", "")
        excel_filename = os.path.join(output_dir, f"{csv_filename_without_ext}_processed.xlsx")

        # Save the workbook
        workbook.save(excel_filename)
        print(f"Successfully converted '{csv_filepath}' to '{excel_filename}'")

    except FileNotFoundError:
        print(f"Error: CSV file not found at '{csv_filepath}'")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        csv_file_path = sys.argv[1]
        convert_csv_to_excel_with_stats(csv_file_path)
    else:
        print("Usage: python your_script_name.py <path_to_csv_file>")
        print("Example: python your_script_name.py results_deadlock_4071245.csv")
        print("\nAttempting to process the provided example CSVs if they exist in the current directory:")
        # Attempt to process the provided example CSVs for testing purposes
        example_csvs = [
            'results_deadlock_4071245.csv',
            'results_RaceToWait_4527458.csv',
            'results_sleeping_4268692.csv',
            # Also try with the "data.xlsx - גיליון1.csv" names if they were provided as input
            'results_deadlock_4071245_data.xlsx - גיליון1.csv',
            'results_RaceToWait_4527458_data.xlsx - גיליון1.csv',
            'results_sleeping_4268692_data.xlsx - גיליון1.csv'
        ]
        for csv_file in example_csvs:
            if os.path.exists(csv_file):
                print(f"Processing {csv_file}...")
                convert_csv_to_excel_with_stats(csv_file)
            else:
                print(f"Skipping {csv_file} as it was not found.")