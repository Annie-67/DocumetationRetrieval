import pandas as pd
import json
from typing import List

class CSVProcessor:
    def __init__(self, max_rows_per_chunk: int = 100):
        """
        Initialize the CSV processor with chunk settings
        
        Args:
            max_rows_per_chunk (int): Maximum number of rows per chunk
        """
        self.max_rows_per_chunk = max_rows_per_chunk
    
    def process(self, file_path: str) -> List[str]:
        """
        Process a CSV file and convert it to text chunks
        
        Args:
            file_path (str): Path to the CSV file
            
        Returns:
            List[str]: List of text chunks representing the CSV data
        """
        try:
            # Read CSV file
            df = pd.read_csv(file_path)
            
            # Basic statistics and information about the CSV
            info_chunk = self._generate_info_chunk(df)
            
            # Generate chunks from the data
            data_chunks = self._chunk_data(df)
            
            # Combine info chunk with data chunks
            all_chunks = [info_chunk] + data_chunks
            
            return all_chunks
        
        except Exception as e:
            raise Exception(f"Error processing CSV: {str(e)}")
    
    def _generate_info_chunk(self, df: pd.DataFrame) -> str:
        """
        Generate an informational chunk about the CSV data
        
        Args:
            df (pd.DataFrame): DataFrame from the CSV
            
        Returns:
            str: Text containing information about the CSV
        """
        # Basic statistics
        row_count = len(df)
        col_count = len(df.columns)
        
        # Column information
        columns_info = []
        for column in df.columns:
            col_type = str(df[column].dtype)
            unique_values = df[column].nunique()
            null_count = df[column].isnull().sum()
            
            col_info = f"Column '{column}': Type={col_type}, Unique Values={unique_values}, Null Values={null_count}"
            columns_info.append(col_info)
        
        # Combine information
        info_text = f"CSV Dataset Summary:\n"
        info_text += f"Total Rows: {row_count}\n"
        info_text += f"Total Columns: {col_count}\n\n"
        info_text += "Column Information:\n"
        info_text += "\n".join(columns_info)
        
        return info_text
    
    def _chunk_data(self, df: pd.DataFrame) -> List[str]:
        """
        Split DataFrame into text chunks
        
        Args:
            df (pd.DataFrame): DataFrame to chunk
            
        Returns:
            List[str]: List of text chunks
        """
        chunks = []
        
        # If the DataFrame is small, process it as a single chunk
        if len(df) <= self.max_rows_per_chunk:
            chunk_text = self._dataframe_to_text(df)
            chunks.append(chunk_text)
            return chunks
        
        # Split into chunks based on max_rows_per_chunk
        for i in range(0, len(df), self.max_rows_per_chunk):
            chunk_df = df.iloc[i:i + self.max_rows_per_chunk]
            chunk_text = self._dataframe_to_text(chunk_df, start_row=i)
            chunks.append(chunk_text)
        
        return chunks
    
    def _dataframe_to_text(self, df: pd.DataFrame, start_row: int = 0) -> str:
        """
        Convert a DataFrame to a text representation
        
        Args:
            df (pd.DataFrame): DataFrame to convert
            start_row (int): Starting row index for reference
            
        Returns:
            str: Text representation of the DataFrame
        """
        # Get column headers
        columns = df.columns.tolist()
        
        # Convert each row to a dictionary and then to text
        rows_text = []
        for i, (_, row) in enumerate(df.iterrows()):
            row_dict = {}
            for col in columns:
                value = row[col]
                # Handle different data types
                if pd.isna(value):
                    row_dict[col] = "NULL"
                elif isinstance(value, (int, float, bool)):
                    row_dict[col] = value
                else:
                    row_dict[col] = str(value)
            
            # Add row index information and convert to JSON string
            row_info = f"Row {start_row + i}: {json.dumps(row_dict, default=str)}"
            rows_text.append(row_info)
        
        # Combine all rows
        result = f"Data Rows (Columns: {', '.join(columns)}):\n"
        result += "\n".join(rows_text)
        
        return result
