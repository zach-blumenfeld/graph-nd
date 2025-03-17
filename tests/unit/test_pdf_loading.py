import os
import unittest
from utils import load_pdf


class TestPDFLoader(unittest.TestCase):

    def test_load_pdf_chunking(self):
        """
        Test the load_pdf function to ensure it correctly chunks a PDF.
        Specifically, we check that a 79-page PDF is divided into 5 chunks
        when chunk_size is 20.
        """
        # Path to the PDF file in the data folder
        file_path = 'data/component-catalog.pdf'

        # Expecting 5 chunks for 79 pages with chunking by 20 pages
        expected_chunk_count = 4
        chunk_size = 20

        # Call the load_pdf function
        chunks = load_pdf(file_path=file_path, chunk_strategy="BY_PAGE", chunk_size=chunk_size)
        print(chunks[0])
        # Assert the number of chunks is as expected
        self.assertEqual(len(chunks), expected_chunk_count,
                         f"Expected {expected_chunk_count} chunks but got {len(chunks)}")


if __name__ == '__main__':
    unittest.main()
