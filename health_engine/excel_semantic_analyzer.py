import pandas as pd
import numpy as np
import json
import os
from typing import Dict, List, Any, Optional, Union, IO
from io import BytesIO
from openai import AzureOpenAI
from dotenv import load_dotenv
import logging
from .json_utils import safe_json_dump

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ExcelSemanticAnalyzer:
    def __init__(self, excel_source: Union[str, bytes, IO[bytes]]):
        """
        Initialize the Excel Semantic Analyzer
        
        Args:
            excel_source: Path to the Excel file, raw bytes, or a file-like object of bytes
        """
        self.excel_source = excel_source
        self.df = None
        self.column_analysis = {}
        
        # Load environment variables
        load_dotenv()
        
        # Initialize Azure OpenAI client
        self.client = AzureOpenAI(
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview")
        )
        self.deployment_name = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
        
        if not all([os.getenv("AZURE_OPENAI_ENDPOINT"), 
                   os.getenv("AZURE_OPENAI_API_KEY"), 
                   self.deployment_name]):
            raise ValueError("Azure OpenAI configuration not found in environment variables")
    
    def load_excel_file(self) -> None:
        """Load the Excel file into a pandas DataFrame"""
        try:
            if isinstance(self.excel_source, (bytes, bytearray)):
                logger.info("Loading Excel from in-memory bytes")
                self.df = pd.read_excel(BytesIO(self.excel_source))
            elif hasattr(self.excel_source, 'read'):
                logger.info("Loading Excel from file-like object")
                data = self.excel_source.read()
                self.df = pd.read_excel(BytesIO(data))
            else:
                logger.info(f"Loading Excel file: {self.excel_source}")
                self.df = pd.read_excel(self.excel_source)
            logger.info(f"Successfully loaded Excel file with {len(self.df)} rows and {len(self.df.columns)} columns")
        except Exception as e:
            logger.error(f"Error loading Excel file: {e}")
            raise
    
    def analyze_column(self, column_name: str) -> Dict[str, Any]:
        """
        Analyze a single column and return its metrics
        
        Args:
            column_name (str): Name of the column to analyze
            
        Returns:
            Dict containing column analysis metrics
        """
        column_data = self.df[column_name]
        
        # Calculate metrics
        unique_count = column_data.nunique()
        null_count = column_data.isnull().sum()
        data_type = str(column_data.dtype)
        
        # Get 5 unique values (or all if less than 5)
        unique_values = column_data.dropna().unique()
        sample_values = unique_values[:5].tolist()
        
        # Convert numpy types and pandas timestamps to Python types for JSON serialization
        if len(sample_values) > 0:
            sample_values = []
            for val in unique_values[:5]:
                if isinstance(val, (np.integer, np.floating)):
                    sample_values.append(str(val))
                elif pd.isna(val):
                    sample_values.append(None)
                elif hasattr(val, 'strftime'):  # Handle datetime/timestamp objects
                    sample_values.append(val.strftime('%Y-%m-%d %H:%M:%S'))
                else:
                    sample_values.append(str(val))
        
        return {
            "unique_count": int(unique_count),
            "null_count": int(null_count),
            "data_type": data_type,
            "sample_values": sample_values
        }
    
    def analyze_all_columns(self) -> None:
        """Analyze all columns in the DataFrame"""
        logger.info("Starting column analysis...")
        
        for column_name in self.df.columns:
            logger.info(f"Analyzing column: {column_name}")
            self.column_analysis[column_name] = self.analyze_column(column_name)
        
        logger.info("Column analysis completed")
    
    def create_prompt_for_column(self, target_column: str) -> str:
        """
        Create a comprehensive prompt for generating semantic description of a column
        
        Args:
            target_column (str): The column for which to generate description
            
        Returns:
            str: Formatted prompt for Azure OpenAI
        """
        target_analysis = self.column_analysis[target_column]
        
        prompt = f"""You are a data analyst tasked with generating semantic descriptions for columns in a dataset. 
        
        Please analyze the following column and provide a comprehensive description that considers its relationship with other columns in the dataset.
        

        
        TARGET COLUMN: {target_column}
        
        Target Column Analysis:
        - Unique value count: {target_analysis['unique_count']}
        - Null value count: {target_analysis['null_count']}
        - Data type: {target_analysis['data_type']}
        - Sample values: {target_analysis['sample_values']}
        
        ALL COLUMNS IN DATASET:
        """
        
        # Add analysis of all other columns for context
        for col_name, analysis in self.column_analysis.items():
            if col_name != target_column:
                prompt += f"""
        Column: {col_name}
        - Unique value count: {analysis['unique_count']}
        - Null value count: {analysis['null_count']}
        - Data type: {analysis['data_type']}
        - Sample values: {analysis['sample_values']}
        """
        
        prompt += f"""
        
        Based on the above analysis, provide a comprehensive semantic description for the column "{target_column}". 
        
        Consider:
        1. What the column likely represents based on its name and data
        2. How it relates to other columns in the dataset
        3. The business context and domain knowledge
        4. Data quality indicators (null values, unique values)
        5. The data type and what it suggests about the column's purpose
        
        Provide a clear, concise description that would be useful for data scientists, analysts, and business users.
        Focus on the semantic meaning and business context rather than technical details.
        
        Return only the description text, no additional formatting or explanations.
        """
        
        return prompt
    
    def generate_semantic_description(self, column_name: str) -> str:
        """
        Generate semantic description for a column using Azure OpenAI
        
        Args:
            column_name (str): Name of the column
            
        Returns:
            str: Generated semantic description
        """
        try:
            prompt = self.create_prompt_for_column(column_name)
            
            response = self.client.chat.completions.create(
                model=self.deployment_name,
                messages=[
                    {"role": "system", "content": "You are a data analyst expert specializing in semantic data analysis and column description generation."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=500
            )
            
            description = response.choices[0].message.content.strip()
            logger.info(f"Generated description for column '{column_name}': {description[:100]}...")
            return description
            
        except Exception as e:
            logger.error(f"Error generating description for column '{column_name}': {e}")
            return f"Error generating description: {str(e)}"
    
    def generate_semantic_json(self) -> Dict[str, Any]:
        """
        Generate the complete semantic JSON structure
        
        Returns:
            Dict containing the semantic analysis in the specified JSON format
        """
        logger.info("Generating semantic descriptions for all columns...")
        
        file_name = os.path.splitext(os.path.basename(self.excel_file_path))[0]
        
        result = {
            file_name: {}
        }
        
        for column_name in self.df.columns:
            logger.info(f"Generating semantic description for column: {column_name}")
            description = self.generate_semantic_description(column_name)
            
            result[file_name][column_name] = {
                "description": description
            }
        
        logger.info("Semantic JSON generation completed")
        return result
    
    def save_semantic_json(self, output_file: str = "semantic_analysis.json") -> None:
        """
        Save the semantic analysis to a JSON file
        
        Args:
            output_file (str): Output file path
        """
        try:
            semantic_json = self.generate_semantic_json()
            
            safe_json_dump(semantic_json, output_file)
            
            logger.info(f"Semantic analysis saved to: {output_file}")
            
        except Exception as e:
            logger.error(f"Error saving semantic analysis: {e}")
            raise
    
    def print_column_analysis_summary(self) -> None:
        """Print a summary of the column analysis"""
        print("\n" + "="*80)
        print("COLUMN ANALYSIS SUMMARY")
        print("="*80)
        
        for column_name, analysis in self.column_analysis.items():
            print(f"\nColumn: {column_name}")
            print(f"  - Unique values: {analysis['unique_count']}")
            print(f"  - Null values: {analysis['null_count']}")
            print(f"  - Data type: {analysis['data_type']}")
            print(f"  - Sample values: {analysis['sample_values']}")
        
        print("\n" + "="*80)

def main():
    """Main function to run the Excel semantic analysis"""
    excel_file = "Unsafe Event - EI Tech App.xlsx"
    
    try:
        # Initialize analyzer
        analyzer = ExcelSemanticAnalyzer(excel_file)
        
        # Load and analyze the Excel file
        analyzer.load_excel_file()
        analyzer.analyze_all_columns()
        
        # Print summary
        analyzer.print_column_analysis_summary()
        
        # Generate and save semantic analysis
        analyzer.save_semantic_json()
        
        print(f"\n✅ Analysis completed successfully!")
        print(f"📄 Results saved to: semantic_analysis.json")
        
    except Exception as e:
        logger.error(f"Error in main execution: {e}")
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()
