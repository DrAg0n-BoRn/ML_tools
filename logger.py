import os
import pandas as pd
from datetime import datetime
from typing import Union, List, Dict, Any
from openpyxl.styles import Font, PatternFill


def custom_logger(
    data: Union[List[Any], Dict[Any, List[Any]], pd.DataFrame],
    save_directory: str,
    log_name: str,
) -> None:
    """
    Logs data to:
    - .txt (if input is a list),
    - .csv (if input is a dict),
    - .xlsx (if input is a DataFrame, with bold headers and background color).

    Args:
        data: List (-> .txt), Dict (-> .csv), or DataFrame (-> .xlsx).
        save_directory: Target directory to save the log.
        log_name: Base name for the log file (timestamp appended).
    """
    try:
        os.makedirs(save_directory, exist_ok=True)
        timestamp = datetime.now().strftime(r"%Y%m%d_%H%M")

        if isinstance(data, list):
            # List -> .txt
            log_lines = []
            for item in data:
                try:
                    log_lines.append(str(item).strip())
                except Exception:
                    log_lines.append(f"(type={type(item)})")

            log_path = os.path.join(save_directory, f"{log_name}_{timestamp}.txt")
            with open(log_path, 'w') as f:
                f.write('\n'.join(log_lines))
            print(f"Log saved to: {log_path}")

        elif isinstance(data, dict):
            # Dict -> .csv
            sanitized_dict = {}
            max_length = max(len(v) for v in data.values()) if data else 0

            for key, value in data.items():
                sanitized_key = str(key).strip().replace('\n', '_').replace('\r', '_')
                if not isinstance(value, list):
                    raise ValueError(f"Dictionary value for key '{sanitized_key}' must be a list.")
                padded_value = value + [None] * (max_length - len(value))
                sanitized_dict[sanitized_key] = padded_value

            log_path = os.path.join(save_directory, f"{log_name}_{timestamp}.csv")
            pd.DataFrame(sanitized_dict).to_csv(log_path, index=False)
            print(f"Log saved to: {log_path}")

        elif isinstance(data, pd.DataFrame):
            # DataFrame -> .xlsx (with styled headers)
            log_path = os.path.join(save_directory, f"{log_name}_{timestamp}.xlsx")
            writer = pd.ExcelWriter(log_path, engine='openpyxl')
            data.to_excel(writer, index=False, sheet_name='Data')

            # Access the workbook and worksheet
            workbook = writer.book
            worksheet = writer.sheets['Data']

            # Define header style (bold + light blue background)
            header_font = Font(bold=True)
            header_fill = PatternFill(
                start_color="ADD8E6",  # Light blue
                end_color="ADD8E6",
                fill_type="solid"
            )

            # Apply style to headers (first row)
            for cell in worksheet[1]:
                cell.font = header_font
                cell.fill = header_fill

            writer.close()
            print(f"Log saved to: {log_path}")

        else:
            raise ValueError("Input data must be a list, dict, or pandas DataFrame.")

    except Exception as e:
        print(f"Error in logger: {e}")

