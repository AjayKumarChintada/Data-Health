import pandas as pd
import numpy as np
import json
import os
from typing import Dict, List, Any
from openai import AzureOpenAI
from dotenv import load_dotenv
import logging
from .json_utils import safe_json_dump

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EnhancedDataQualityAnalyzer:
    def __init__(self, excel_source, semantic_analysis_file: str = "semantic_analysis.json", semantic_analysis_data: dict | None = None):
        """
        Initialize the Enhanced Data Quality Analyzer
        
        Args:
            excel_source: Path to the Excel file, raw bytes, or a file-like object
            semantic_analysis_file (str): Path to the semantic analysis JSON file
            semantic_analysis_data (dict): Optional in-memory semantic analysis data
        """
        self.excel_file_path = excel_source
        self.semantic_analysis_file = semantic_analysis_file
        self.semantic_descriptions = {}
        # If semantic data is provided directly, prefer it over a file path
        if semantic_analysis_data is not None:
            self.semantic_descriptions = semantic_analysis_data
        self.df = None
        self.column_analysis = {}
        self.semantic_descriptions = {}
        
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
    
    def load_semantic_analysis(self) -> None:
        """Load existing semantic analysis from direct data or JSON file"""
        try:
            if self.semantic_descriptions:
                logger.info(f"Using in-memory semantic analysis for {len(self.semantic_descriptions)} columns")
                return
            if os.path.exists(self.semantic_analysis_file):
                with open(self.semantic_analysis_file, 'r', encoding='utf-8') as f:
                    semantic_data = json.load(f)
                # If the JSON is stored as {filename: {col: desc, ...}}, extract the first entry
                if isinstance(semantic_data, dict) and semantic_data:
                    first_key = list(semantic_data.keys())[0]
                    value = semantic_data[first_key]
                    self.semantic_descriptions = value if isinstance(value, dict) else semantic_data
                else:
                    self.semantic_descriptions = {}
                logger.info(f"Loaded semantic descriptions for {len(self.semantic_descriptions)} columns")
            else:
                logger.warning(f"Semantic analysis file {self.semantic_analysis_file} not found. Proceeding without semantic context.")
                self.semantic_descriptions = {}
        except Exception as e:
            logger.error(f"Error loading semantic analysis: {e}")
            self.semantic_descriptions = {}
    
    def load_excel_file(self) -> None:
        """Load the Excel file into a pandas DataFrame"""
        try:
            from io import BytesIO
            if isinstance(self.excel_file_path, (bytes, bytearray)):
                logger.info("Loading Excel from in-memory bytes")
                self.df = pd.read_excel(BytesIO(self.excel_file_path))
            elif hasattr(self.excel_file_path, 'read'):
                logger.info("Loading Excel from file-like object")
                data = self.excel_file_path.read()
                self.df = pd.read_excel(BytesIO(data))
            else:
                logger.info(f"Loading Excel file: {self.excel_file_path}")
                self.df = pd.read_excel(self.excel_file_path)
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
        total_count = len(column_data)
        
        # Get 5 unique values (or all if less than 5)
        unique_values = column_data.dropna().unique()
        
        # Convert numpy types and pandas timestamps to Python types for JSON serialization
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
        
        # Additional analysis for data quality
        duplicate_count = total_count - unique_count
        unique_percentage = (unique_count / total_count) * 100 if total_count > 0 else 0
        null_percentage = (null_count / total_count) * 100 if total_count > 0 else 0
        
        # Check for zero-length strings
        zero_length_count = 0
        if data_type == 'object':
            zero_length_count = (column_data.astype(str).str.len() == 0).sum()
        
        # Check for future dates (if datetime)
        future_date_count = 0
        if 'datetime' in data_type.lower():
            from datetime import datetime
            current_date = pd.Timestamp.now()
            future_date_count = (column_data > current_date).sum()
        
        return {
            "unique_count": int(unique_count),
            "null_count": int(null_count),
            "data_type": data_type,
            "sample_values": sample_values,
            "total_count": int(total_count),
            "duplicate_count": int(duplicate_count),
            "unique_percentage": float(unique_percentage),
            "null_percentage": float(null_percentage),
            "zero_length_count": int(zero_length_count),
            "future_date_count": int(future_date_count)
        }
    
    def analyze_all_columns(self) -> None:
        """Analyze all columns in the DataFrame"""
        logger.info("Starting column analysis...")
        
        for column_name in self.df.columns:
            logger.info(f"Analyzing column: {column_name}")
            self.column_analysis[column_name] = self.analyze_column(column_name)
        
        logger.info("Column analysis completed")
    
    def create_enhanced_quality_prompt_for_column(self, target_column: str) -> str:
        """
        Create a comprehensive prompt for determining data quality dimensions and checks
        using existing semantic descriptions for better context
        
        Args:
            target_column (str): The column for which to determine quality checks
            
        Returns:
            str: Formatted prompt for Azure OpenAI
        """
        target_analysis = self.column_analysis[target_column]
        
        # Get semantic description if available
        semantic_description = self.semantic_descriptions.get(target_column, {}).get('description', 'No semantic description available')
        
        prompt = f"""You are a data quality expert tasked with determining which data quality dimensions and checks are applicable for a specific column in a dataset.

Please analyze the following column and determine which data quality checks should be applied based on the column's characteristics, semantic meaning, and the overall dataset context.

TARGET COLUMN: {target_column}

SEMANTIC DESCRIPTION:
{semantic_description}

Target Column Analysis:
- Unique value count: {target_analysis['unique_count']}
- Null value count: {target_analysis['null_count']} ({target_analysis['null_percentage']:.2f}%)
- Data type: {target_analysis['data_type']}
- Sample values: {target_analysis['sample_values']}
- Total count: {target_analysis['total_count']}
- Duplicate count: {target_analysis['duplicate_count']}
- Unique percentage: {target_analysis['unique_percentage']:.2f}%
- Zero-length strings: {target_analysis['zero_length_count']}
- Future dates: {target_analysis['future_date_count']}

ALL COLUMNS IN DATASET:
"""
        
        # Add analysis of all other columns for context
        for col_name, analysis in self.column_analysis.items():
            if col_name != target_column:
                other_semantic = self.semantic_descriptions.get(col_name, {}).get('description', 'No description available')
                prompt += f"""
Column: {col_name}
- Unique value count: {analysis['unique_count']}
- Null value count: {analysis['null_count']} ({analysis['null_percentage']:.2f}%)
- Data type: {analysis['data_type']}
- Sample values: {analysis['sample_values']}
- Semantic description: {other_semantic[:200]}...
"""
        
        prompt += f"""

Based on the above analysis, including the semantic description of the column, determine which data quality dimensions and checks are applicable for the column "{target_column}".

Available Data Quality Dimensions and Checks:

1. COMPLETENESS:
   - Missing Value Count and Percentage: Check for null/missing values
   - Required Field Validation: Check if this field should be mandatory
   - Zero-Length String Detection: Check for empty strings

2. UNIQUENESS:
   - Duplicate Value Count: Check for duplicate values
   - Unique Value Percentage: Analyze uniqueness ratio
   - Primary Key Validation: Check if this should be a unique identifier
   - Check uniqueness across Rows: Verify row-level uniqueness

3. VALIDITY:
   - Data Type: Validate data type consistency
   - Format Pattern Validation: Check format patterns (dates, IDs, etc.)
   - Length Validation: Check string/numeric length constraints
   - Category Validation: Check if values are from allowed categories

4. CONSISTENCY:
   - Format Consistency: Check if similar data follows same format
   - Case Consistency: Check for consistent capitalization

5. TIMELINESS:
   - Data Freshness: Check how recent the data is
   - Future Date Detection: Identify illogical future dates

For each check, provide:
1. "yes" if the check is applicable and meaningful for this column, or "no" if it's not applicable
2. A brief explanation (1-2 sentences) of why this check is applicable or not applicable

Consider:
1. The semantic description and business purpose of the column
2. The column name and its likely function
3. The data type and format
4. The sample values and patterns
5. The relationship with other columns
6. Business context and domain knowledge
7. Data quality best practices

The semantic description provides crucial context about what this column represents and how it's used in the business context. Use this information to make informed decisions about which quality checks are most relevant.

Return your response as a JSON object with the exact structure shown below:

{{
  "Completeness": {{
    "Missing Value Count and Percentage": {{
      "applicable": "yes/no",
      "explanation": "Brief explanation of why this check is applicable or not"
    }},
    "Required Field Validation": {{
      "applicable": "yes/no",
      "explanation": "Brief explanation of why this check is applicable or not"
    }},
    "Zero-Length String Detection": {{
      "applicable": "yes/no",
      "explanation": "Brief explanation of why this check is applicable or not"
    }}
  }},
  "Uniqueness": {{
    "Duplicate Value Count": {{
      "applicable": "yes/no",
      "explanation": "Brief explanation of why this check is applicable or not"
    }},
    "Unique Value Percentage": {{
      "applicable": "yes/no",
      "explanation": "Brief explanation of why this check is applicable or not"
    }},
    "Primary Key Validation": {{
      "applicable": "yes/no",
      "explanation": "Brief explanation of why this check is applicable or not"
    }},
    "Check uniqueness across Rows": {{
      "applicable": "yes/no",
      "explanation": "Brief explanation of why this check is applicable or not"
    }}
  }},
  "Validity": {{
    "Data Type": {{
      "applicable": "yes/no",
      "explanation": "Brief explanation of why this check is applicable or not"
    }},
    "Format Pattern Validation": {{
      "applicable": "yes/no",
      "explanation": "Brief explanation of why this check is applicable or not"
    }},
    "Length Validation": {{
      "applicable": "yes/no",
      "explanation": "Brief explanation of why this check is applicable or not"
    }},
    "Category Validation": {{
      "applicable": "yes/no",
      "explanation": "Brief explanation of why this check is applicable or not"
    }}
  }},
  "Consistency": {{
    "Format Consistency": {{
      "applicable": "yes/no",
      "explanation": "Brief explanation of why this check is applicable or not"
    }},
    "Case Consistency": {{
      "applicable": "yes/no",
      "explanation": "Brief explanation of why this check is applicable or not"
    }}
  }},
  "Timeliness": {{
    "Data Freshness": {{
      "applicable": "yes/no",
      "explanation": "Brief explanation of why this check is applicable or not"
    }},
    "Future Date Detection": {{
      "applicable": "yes/no",
      "explanation": "Brief explanation of why this check is applicable or not"
    }}
  }}
}}

Only return the JSON object, no additional text or explanations. don't include ```json or ``` in your response.
"""
        
        return prompt
    
    def generate_quality_assessment(self, column_name: str) -> Dict[str, Any]:
        """
        Generate data quality assessment for a column using Azure OpenAI
        
        Args:
            column_name (str): Name of the column
            
        Returns:
            Dict: Generated quality assessment
        """
        try:
            prompt = self.create_enhanced_quality_prompt_for_column(column_name)
            
            response = self.client.chat.completions.create(
                model=self.deployment_name,
                messages=[
                    {"role": "system", "content": "You are a data quality expert specializing in determining applicable data quality checks and dimensions for database columns. Always respond with valid JSON only."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,
                max_tokens=1000
            )
            
            response_text = response.choices[0].message.content.strip()
            
            # Try to parse the JSON response
            try:
                quality_assessment = json.loads(response_text)
                logger.info(f"Generated quality assessment for column '{column_name}'")
                return quality_assessment
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse JSON response for column '{column_name}': {e}")
                logger.error(f"Response: {response_text}")
                # Return a default structure if JSON parsing fails
                return self.get_default_quality_assessment()
            
        except Exception as e:
            logger.error(f"Error generating quality assessment for column '{column_name}': {e}")
            return self.get_default_quality_assessment()
    
    def get_default_quality_assessment(self) -> Dict[str, Any]:
        """Return a default quality assessment structure"""
        return {
            "Completeness": {
                "Missing Value Count and Percentage": {
                    "applicable": "yes",
                    "explanation": "Default explanation for missing value count"
                },
                "Required Field Validation": {
                    "applicable": "yes",
                    "explanation": "Default explanation for required field validation"
                },
                "Zero-Length String Detection": {
                    "applicable": "yes",
                    "explanation": "Default explanation for zero-length string detection"
                }
            },
            "Uniqueness": {
                "Duplicate Value Count": {
                    "applicable": "yes",
                    "explanation": "Default explanation for duplicate value count"
                },
                "Unique Value Percentage": {
                    "applicable": "yes",
                    "explanation": "Default explanation for unique value percentage"
                },
                "Primary Key Validation": {
                    "applicable": "no",
                    "explanation": "Default explanation for primary key validation"
                },
                "Check uniqueness across Rows": {
                    "applicable": "yes",
                    "explanation": "Default explanation for check uniqueness across rows"
                }
            },
            "Validity": {
                "Data Type": {
                    "applicable": "yes",
                    "explanation": "Default explanation for data type validity"
                },
                "Format Pattern Validation": {
                    "applicable": "no",
                    "explanation": "Default explanation for format pattern validation"
                },
                "Length Validation": {
                    "applicable": "no",
                    "explanation": "Default explanation for length validation"
                },
                "Category Validation": {
                    "applicable": "no",
                    "explanation": "Default explanation for category validation"
                }
            },
            "Consistency": {
                "Format Consistency": {
                    "applicable": "yes",
                    "explanation": "Default explanation for format consistency"
                },
                "Case Consistency": {
                    "applicable": "no",
                    "explanation": "Default explanation for case consistency"
                }
            },
            "Timeliness": {
                "Data Freshness": {
                    "applicable": "no",
                    "explanation": "Default explanation for data freshness"
                },
                "Future Date Detection": {
                    "applicable": "no",
                    "explanation": "Default explanation for future date detection"
                }
            }
        }
    
    def generate_quality_json(self) -> Dict[str, Any]:
        """
        Generate the complete data quality JSON structure
        
        Returns:
            Dict containing the quality analysis in the specified JSON format
        """
        logger.info("Generating enhanced data quality assessments for all columns...")
        
        file_name = os.path.splitext(os.path.basename(self.excel_file_path))[0]
        
        result = {
            file_name: {}
        }
        
        for column_name in self.df.columns:
            logger.info(f"Generating enhanced quality assessment for column: {column_name}")
            quality_assessment = self.generate_quality_assessment(column_name)
            
            result[file_name][column_name] = {
                "dimensions": quality_assessment
            }
        
        logger.info("Enhanced data quality JSON generation completed")
        return result
    
    def save_quality_json(self, output_file: str = "enhanced_data_quality_analysis.json") -> None:
        """
        Save the enhanced data quality analysis to a JSON file
        
        Args:
            output_file (str): Output file path
        """
        try:
            quality_json = self.generate_quality_json()
            
            safe_json_dump(quality_json, output_file)
            
            logger.info(f"Enhanced data quality analysis saved to: {output_file}")
            
        except Exception as e:
            logger.error(f"Error saving enhanced data quality analysis: {e}")
            raise
    
    def print_column_analysis_summary(self) -> None:
        """Print a summary of the column analysis"""
        print("\n" + "="*80)
        print("ENHANCED COLUMN ANALYSIS SUMMARY")
        print("="*80)
        
        for column_name, analysis in self.column_analysis.items():
            print(f"\nColumn: {column_name}")
            print(f"  - Unique values: {analysis['unique_count']} ({analysis['unique_percentage']:.2f}%)")
            print(f"  - Null values: {analysis['null_count']} ({analysis['null_percentage']:.2f}%)")
            print(f"  - Data type: {analysis['data_type']}")
            print(f"  - Sample values: {analysis['sample_values']}")
            print(f"  - Duplicates: {analysis['duplicate_count']}")
            print(f"  - Zero-length strings: {analysis['zero_length_count']}")
            print(f"  - Future dates: {analysis['future_date_count']}")
            
            # Show if semantic description is available
            if column_name in self.semantic_descriptions:
                semantic_desc = self.semantic_descriptions[column_name].get('description', 'No description')
                print(f"  - Semantic description: {semantic_desc[:100]}...")
            else:
                print(f"  - Semantic description: Not available")
        
        print("\n" + "="*80)

def main():
    """Main function to run the enhanced data quality analysis"""
    excel_file = "Unsafe Event - EI Tech App.xlsx"
    semantic_file = "semantic_analysis.json"
    
    try:
        # Initialize analyzer
        analyzer = EnhancedDataQualityAnalyzer(excel_file, semantic_file)
        
        # Load semantic analysis first
        analyzer.load_semantic_analysis()
        
        # Load and analyze the Excel file
        analyzer.load_excel_file()
        analyzer.analyze_all_columns()
        
        # Print summary
        analyzer.print_column_analysis_summary()
        
        # Generate and save enhanced quality analysis
        analyzer.save_quality_json()
        
        print(f"\n✅ Enhanced data quality analysis completed successfully!")
        print(f"📄 Results saved to: enhanced_data_quality_analysis.json")
        print(f"🔍 Analysis used semantic descriptions for better context")
        
    except Exception as e:
        logger.error(f"Error in main execution: {e}")
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()
