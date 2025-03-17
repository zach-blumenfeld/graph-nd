import csv
from typing import List

from PyPDF2 import PdfReader

def format_table_as_markdown_preview(headers, rows, truncate_length=30, max_rows=5) -> str:
    """
    Formats the provided headers and rows as a well-aligned Markdown table.

    Args:
        headers (list of str): List of column names (header row).
        rows (list of list of str): List of rows, where each row is a list of strings.
        truncate_length (int): Maximum length of a cell value before truncation.
        max_rows (int): Maximum number of rows to read from the file.

    Returns:
        str: A Markdown formatted table.
    """
    def truncate(inp):
        text = str(inp)
        """Utility to truncate text if it exceeds the specified length."""
        return (text[:truncate_length-1] + "...") if len(text) > truncate_length else text
    clean_rows = []
    i = 0
    for row in rows:
        if i >= max_rows:
            break
        i+=1
        clean_rows.append([truncate(cell) for cell in row])
    # Combine headers and rows to calculate column widths
    all_rows = [headers] + clean_rows
    col_widths = [max(len(row[i]) for row in all_rows) for i in range(len(headers))]

    # Generate aligned Markdown table
    def format_row(r):
        """Formats a single row with padding for alignment."""
        return "| " + " | ".join(r[i].ljust(col_widths[i]) for i in range(len(r))) + " |"

    # Build the table
    table = format_row(headers) + "\n"
    table += "|" + "|".join("-" * (col_widths[i]+2) for i in range(len(headers))) + "|\n"
    for row in clean_rows:
        table += format_row(row) + "\n"

    return table.strip()



def read_csv_preview(file_path, truncate_length=30, max_rows=5):
    """
    Reads the header and first few rows of a CSV file, truncating cell contents as needed.
    Formats as a well-aligned Markdown table.

    Args:
        file_path (str): Path to the CSV file.
        truncate_length (int): Maximum length of a cell value before truncation.
        max_rows (int): Maximum number of rows to read from the file.

    Returns:
        str: A Markdown formatted table.
    """
    try:
        with open(file_path, mode='r', encoding='utf-8') as csv_file:
            reader = csv.reader(csv_file)

            # Read the header
            headers = next(reader)

            # Read the first rows
            rows = []
            for i, row in enumerate(reader):
                if i >= max_rows:
                    break
                rows.append(row)

            return format_table_as_markdown_preview(headers, rows, truncate_length, max_rows)
    except FileNotFoundError:
        raise FileNotFoundError("Error: File not found.")
    except Exception as e:
        raise Exception(f"Error while reading CSV: {str(e)}")

def read_csv(file_path):
    """
    Reads a CSV file and returns its content as a list of dictionaries.

    :param file_path: The path to the CSV file.
    :return: A list of dictionaries where keys are column headers and values are row values.
    """
    with open(file_path, mode='r', newline='', encoding='utf-8') as csv_file:
        reader = csv.DictReader(csv_file)
        records = [row for row in reader]
    return records


def load_pdf(file_path: str, chunk_strategy="BY_PAGE", chunk_size=20) -> List[str]:
    """
    Reads a PDF file and splits its content into text chunks.

    Currently, the only supported chunking strategy is 'BY_PAGE'.
    If a different strategy is provided, an error will be raised.
    The chunking will group 'chunk_size' number of pages into one chunk.

    Args:
        file_path (str): The path to the PDF file.
        chunk_strategy (str): The chunking strategy. Defaults to 'BY_PAGE'.
        chunk_size (int): Number of pages per chunk. Defaults to 1.

    Returns:
        List[str]: A list of text chunks (each containing 'chunk_size' pages).

    Raises:
        ValueError: If the chunk strategy is not implemented.
    """
    if chunk_strategy != "BY_PAGE":
        raise ValueError(
            f"Chunking strategy '{chunk_strategy}' is not implemented yet. The only supported strategy is 'BY_PAGE'.")

    if chunk_size < 1:
        raise ValueError(f"Chunk size must be at least 1. Provided: {chunk_size}")

    try:
        # Read the PDF file
        reader = PdfReader(file_path)
        chunks = []
        current_chunk = []

        # Extract text and group by chunk_size
        for i, page in enumerate(reader.pages):
            page_text = page.extract_text()

            # Add page content to the current chunk
            current_chunk.append(page_text)

            # If the current chunk has reached the desired chunk_size or it's the last page
            if (i + 1) % chunk_size == 0 or (i + 1) == len(reader.pages):
                # Combine pages in the current chunk with the delimiter
                chunks.append("\n\n__NEXT_PAGE__\n\n".join(current_chunk))
                current_chunk = []  # Reset for the next chunk

        return chunks
    except Exception as e:
        raise RuntimeError(f"An error occurred while reading the PDF: {e}")
