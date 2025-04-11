import unittest
from graph_nd.graphrag.utils import format_table_as_markdown_preview


class TestFormatAsMarkdownTable(unittest.TestCase):
    maxDiff = None

    def test_truncation_and_row_limit(self):
        """
        Test that the function respects max_rows and truncates overly long text correctly.
        """
        headers = ["id", "name", "age"]
        # Create rows with more than 5 rows and a very long name for Alice
        rows = [
            [1, "Alice " + "verylongtext " * 5, "30"],  # Long name for Alice, should be truncated
            [2, "Bob", "25"],
            [3, "Charlie", "35"],
            [4, "Diana", "29"],
            [5, "Evan", "40"],
            [6, "Frank", "50"],  # This row should not appear as max_rows=5
        ]
        expected_output = (
            "| id | name                             | age |\n"
            "|----|----------------------------------|-----|\n"
            "| 1  | Alice verylongtext verylongte... | 30  |\n"
            "| 2  | Bob                              | 25  |\n"
            "| 3  | Charlie                          | 35  |\n"
            "| 4  | Diana                            | 29  |\n"
            "| 5  | Evan                             | 40  |"
        )

        # Pass max_rows=5 and truncate_length=30 to the function
        formatted_table = format_table_as_markdown_preview(headers, rows, truncate_length=30, max_rows=5)
        print(formatted_table)
        # Validate that the output matches the expected Markdown table
        self.assertEqual(formatted_table, expected_output)


if __name__ == "__main__":
    unittest.main()
