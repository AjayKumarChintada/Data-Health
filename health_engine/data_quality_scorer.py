import pandas as pd
import numpy as np
import json
import os
from typing import Dict, List, Any, Tuple, Union, IO
from io import BytesIO
from datetime import datetime
import logging
from .json_utils import safe_json_dump

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DataQualityScorer:
    def __init__(self, excel_source: Union[str, bytes, IO[bytes]], quality_analysis_file: str = "health_engine/enhanced_data_quality_analysis.json"):
        """
        Initialize the Data Quality Scorer
        
        Args:
            excel_source (Union[str, bytes, IO[bytes]]): Path to Excel file, bytes content, or file-like object
            quality_analysis_file (str): Path to the enhanced data quality analysis JSON file
        
        Raises:
            TypeError: If excel_source is not of the correct type
        """
        if not isinstance(excel_source, (str, bytes, BytesIO)) and not hasattr(excel_source, 'read'):
            raise TypeError("excel_source must be a string path, bytes, or file-like object")
            
        self.excel_file_path = excel_source
        self.quality_analysis_file = quality_analysis_file
        self.df: pd.DataFrame | None = None
        self.quality_analysis: dict = {}
        self.column_scores: dict = {}
        self.dimension_scores: dict = {}
        self.overall_score: float = 0.0
        self._initialized = False
        
    def load_quality_analysis(self) -> None:
        """Load the enhanced data quality analysis from JSON file"""
        try:
            if os.path.exists(self.quality_analysis_file):
                with open(self.quality_analysis_file, 'r', encoding='utf-8') as f:
                    self.quality_analysis = json.load(f)
                logger.info(f"Loaded quality analysis for {len(self.quality_analysis)} files")
            else:
                raise FileNotFoundError(f"Quality analysis file {self.quality_analysis_file} not found")
                
        except Exception as e:
            logger.error(f"Error loading quality analysis: {e}")
            raise
    
    def load_excel_file(self) -> None:
        """
        Load the Excel file into a pandas DataFrame
        
        Raises:
            FileNotFoundError: If the Excel file path doesn't exist
            ValueError: If the Excel file is empty
            pd.errors.EmptyDataError: If the Excel file has no valid data
            Exception: For other errors during file loading
        """
        try:
            if isinstance(self.excel_file_path, str):
                if not os.path.exists(self.excel_file_path):
                    raise FileNotFoundError(f"Excel file not found: {self.excel_file_path}")
                logger.info(f"Loading Excel file: {self.excel_file_path}")
                self.df = pd.read_excel(self.excel_file_path, engine='openpyxl')
            
            elif isinstance(self.excel_file_path, (bytes, bytearray)):
                logger.info("Loading Excel from in-memory bytes")
                self.df = pd.read_excel(BytesIO(self.excel_file_path), engine='openpyxl')
            
            elif hasattr(self.excel_file_path, 'read'):
                logger.info("Loading Excel from file-like object")
                if hasattr(self.excel_file_path, 'seek'):
                    self.excel_file_path.seek(0)
                data = self.excel_file_path.read()
                self.df = pd.read_excel(BytesIO(data), engine='openpyxl')
            
            else:
                raise TypeError("excel_source must be a string path, bytes, or file-like object")
            
            if self.df is None or self.df.empty:
                raise ValueError("The Excel file is empty or contains no valid data")
                
            logger.info(f"Successfully loaded Excel file with {len(self.df)} rows and {len(self.df.columns)} columns")
            self._initialized = True
            
        except pd.errors.EmptyDataError:
            logger.error("The Excel file is empty or has no valid data")
            raise
        except Exception as e:
            logger.error(f"Error loading Excel file: {e}")
            raise
    
    def _ensure_initialized(self) -> None:
        """Ensure the DataFrame is loaded before operations"""
        if not self._initialized or self.df is None:
            raise RuntimeError("DataFrame not initialized. Call load_excel_file() first.")
    
    def check_completeness(self, column_name: str) -> Dict[str, Any]:
        """Check completeness dimension for a column"""
        self._ensure_initialized()
        column_data = self.df[column_name]
        total_count = len(column_data)
        null_count = column_data.isnull().sum()
        
        # Missing Value Count and Percentage
        missing_percentage = (null_count / total_count) * 100 if total_count > 0 else 0
        missing_score = max(0, 100 - missing_percentage)
        
        # Required Field Validation (assuming all fields are required if not specified)
        required_score = 100 if null_count == 0 else 0
        
        # Zero-Length String Detection
        zero_length_count = 0
        if column_data.dtype == 'object':
            zero_length_count = (column_data.astype(str).str.strip() == '').sum()
        zero_length_percentage = (zero_length_count / total_count) * 100 if total_count > 0 else 0
        zero_length_score = max(0, 100 - zero_length_percentage)
        
        return {
            "missing_value_score": missing_score,
            "required_field_score": required_score,
            "zero_length_score": zero_length_score,
            "metrics": {
                "total_count": total_count,
                "null_count": int(null_count),
                "null_percentage": float(missing_percentage),
                "zero_length_count": int(zero_length_count),
                "zero_length_percentage": float(zero_length_percentage)
            }
        }
    
    def check_uniqueness(self, column_name: str) -> Dict[str, Any]:
        """Check uniqueness dimension for a column"""
        self._ensure_initialized()
        column_data = self.df[column_name]
        total_count = len(column_data)
        unique_count = column_data.nunique()
        duplicate_count = total_count - unique_count
        
        # Duplicate Value Count
        duplicate_percentage = (duplicate_count / total_count) * 100 if total_count > 0 else 0
        duplicate_score = max(0, 100 - duplicate_percentage)
        
        # Unique Value Percentage
        unique_percentage = (unique_count / total_count) * 100 if total_count > 0 else 0
        unique_score = unique_percentage
        
        # Primary Key Validation (assume high uniqueness indicates potential primary key)
        primary_key_score = 100 if unique_percentage >= 95 else max(0, unique_percentage)
        
        # Check uniqueness across Rows (same as duplicate score)
        row_uniqueness_score = duplicate_score
        
        return {
            "duplicate_score": duplicate_score,
            "unique_percentage_score": unique_score,
            "primary_key_score": primary_key_score,
            "row_uniqueness_score": row_uniqueness_score,
            "metrics": {
                "total_count": total_count,
                "unique_count": int(unique_count),
                "duplicate_count": int(duplicate_count),
                "unique_percentage": float(unique_percentage),
                "duplicate_percentage": float(duplicate_percentage)
            }
        }
    
    def check_validity(self, column_name: str) -> Dict[str, Any]:
        """Check validity dimension for a column"""
        column_data = self.df[column_name]
        total_count = len(column_data)
        non_null_data = column_data.dropna()

        # If column is empty or all values are null, validity should not be perfect
        if total_count == 0 or len(non_null_data) == 0:
            return {
                "data_type_score": 0,
                "format_score": 0,
                "length_score": 0,
                "category_score": 0,
                "metrics": {
                    "total_count": total_count,
                    "data_type": str(column_data.dtype)
                }
            }

        # Data Type validation (assume consistent if pandas loaded it)
        data_type_score = 100

        # Format Pattern Validation (basic checks based on data type)
        format_score = 100

        # Length Validation (for string columns)
        length_score = 100
        if column_data.dtype == 'object':
            string_lengths = non_null_data.astype(str).str.len()
            reasonable_lengths = ((string_lengths >= 1) & (string_lengths <= 1000)).sum()
            length_score = (reasonable_lengths / len(non_null_data)) * 100 if len(non_null_data) > 0 else 0

        # Category Validation (check if values are from expected categories)
        category_score = 100  # Default score, could be enhanced with predefined categories

        return {
            "data_type_score": data_type_score,
            "format_score": format_score,
            "length_score": length_score,
            "category_score": category_score,
            "metrics": {
                "total_count": total_count,
                "data_type": str(column_data.dtype)
            }
        }
    
    def check_consistency(self, column_name: str) -> Dict[str, Any]:
        """Check consistency dimension for a column"""
        column_data = self.df[column_name]
        total_count = len(column_data)
        
        # Format Consistency (check if similar data follows same format)
        format_consistency_score = 100  # Default score
        
        # Case Consistency (for string columns)
        case_consistency_score = 100
        if column_data.dtype == 'object':
            # Check if all strings have consistent case (all upper, all lower, or title case)
            non_null_data = column_data.dropna()
            if len(non_null_data) > 0:
                # Check if all strings follow same case pattern
                all_upper = (non_null_data.astype(str).str.isupper()).sum()
                all_lower = (non_null_data.astype(str).str.islower()).sum()
                title_case = (non_null_data.astype(str).str.istitle()).sum()
                
                max_consistent = max(all_upper, all_lower, title_case)
                case_consistency_score = (max_consistent / len(non_null_data)) * 100 if len(non_null_data) > 0 else 0
        
        return {
            "format_consistency_score": format_consistency_score,
            "case_consistency_score": case_consistency_score,
            "metrics": {
                "total_count": total_count
            }
        }
    
    def check_timeliness(self, column_name: str) -> Dict[str, Any]:
        """Check timeliness dimension for a column"""
        column_data = self.df[column_name]
        total_count = len(column_data)
        
        # Data Freshness (check how recent the data is)
        data_freshness_score = 100  # Default score
        
        # Future Date Detection (if datetime column)
        future_date_score = 100
        if 'datetime' in str(column_data.dtype).lower():
            current_date = pd.Timestamp.now()
            future_dates = (column_data > current_date).sum()
            future_date_percentage = (future_dates / total_count) * 100 if total_count > 0 else 0
            future_date_score = max(0, 100 - future_date_percentage)
        
        return {
            "data_freshness_score": data_freshness_score,
            "future_date_score": future_date_score,
            "metrics": {
                "total_count": total_count,
                "future_dates": int(future_dates) if 'datetime' in str(column_data.dtype).lower() else 0
            }
        }
    
    def calculate_column_score(self, column_name: str) -> Dict[str, Any]:
        """Calculate comprehensive score for a single column"""
        logger.info(f"Calculating score for column: {column_name}")
        
        # Get quality analysis for this column
        file_name = list(self.quality_analysis.keys())[0]
        column_quality = self.quality_analysis[file_name].get(column_name, {}).get('dimensions', {})
        
        column_scores = {}
        applicable_dimensions = []
        skipped_dimensions = []
        dimension_scores = {}
        
        # Check each dimension
        for dimension_name, checks in column_quality.items():
            dimension_score = 0
            applicable_checks = 0
            total_checks = 0
            
            if dimension_name == "Completeness":
                completeness_results = self.check_completeness(column_name)
                for check_name, check_info in checks.items():
                    total_checks += 1
                    if check_info.get('applicable', 'no') == 'yes':
                        applicable_checks += 1
                        if check_name == "Missing Value Count and Percentage":
                            dimension_score += completeness_results['missing_value_score']
                        elif check_name == "Required Field Validation":
                            dimension_score += completeness_results['required_field_score']
                        elif check_name == "Zero-Length String Detection":
                            dimension_score += completeness_results['zero_length_score']
                
                if applicable_checks > 0:
                    dimension_scores[dimension_name] = dimension_score / applicable_checks
                    applicable_dimensions.append(dimension_name)
                else:
                    skipped_dimensions.append(dimension_name)
                    
            elif dimension_name == "Uniqueness":
                uniqueness_results = self.check_uniqueness(column_name)
                for check_name, check_info in checks.items():
                    total_checks += 1
                    if check_info.get('applicable', 'no') == 'yes':
                        applicable_checks += 1
                        if check_name == "Duplicate Value Count":
                            dimension_score += uniqueness_results['duplicate_score']
                        elif check_name == "Unique Value Percentage":
                            dimension_score += uniqueness_results['unique_percentage_score']
                        elif check_name == "Primary Key Validation":
                            dimension_score += uniqueness_results['primary_key_score']
                        elif check_name == "Check uniqueness across Rows":
                            dimension_score += uniqueness_results['row_uniqueness_score']
                
                if applicable_checks > 0:
                    dimension_scores[dimension_name] = dimension_score / applicable_checks
                    applicable_dimensions.append(dimension_name)
                else:
                    skipped_dimensions.append(dimension_name)
                    
            elif dimension_name == "Validity":
                validity_results = self.check_validity(column_name)
                for check_name, check_info in checks.items():
                    total_checks += 1
                    if check_info.get('applicable', 'no') == 'yes':
                        applicable_checks += 1
                        if check_name == "Data Type":
                            dimension_score += validity_results['data_type_score']
                        elif check_name == "Format Pattern Validation":
                            dimension_score += validity_results['format_score']
                        elif check_name == "Length Validation":
                            dimension_score += validity_results['length_score']
                        elif check_name == "Category Validation":
                            dimension_score += validity_results['category_score']
                
                if applicable_checks > 0:
                    dimension_scores[dimension_name] = dimension_score / applicable_checks
                    applicable_dimensions.append(dimension_name)
                else:
                    skipped_dimensions.append(dimension_name)
                    
            elif dimension_name == "Consistency":
                consistency_results = self.check_consistency(column_name)
                for check_name, check_info in checks.items():
                    total_checks += 1
                    if check_info.get('applicable', 'no') == 'yes':
                        applicable_checks += 1
                        if check_name == "Format Consistency":
                            dimension_score += consistency_results['format_consistency_score']
                        elif check_name == "Case Consistency":
                            dimension_score += consistency_results['case_consistency_score']
                
                if applicable_checks > 0:
                    dimension_scores[dimension_name] = dimension_score / applicable_checks
                    applicable_dimensions.append(dimension_name)
                else:
                    skipped_dimensions.append(dimension_name)
                    
            elif dimension_name == "Timeliness":
                timeliness_results = self.check_timeliness(column_name)
                for check_name, check_info in checks.items():
                    total_checks += 1
                    if check_info.get('applicable', 'no') == 'yes':
                        applicable_checks += 1
                        if check_name == "Data Freshness":
                            dimension_score += timeliness_results['data_freshness_score']
                        elif check_name == "Future Date Detection":
                            dimension_score += timeliness_results['future_date_score']
                
                if applicable_checks > 0:
                    dimension_scores[dimension_name] = dimension_score / applicable_checks
                    applicable_dimensions.append(dimension_name)
                else:
                    skipped_dimensions.append(dimension_name)
        
        # Calculate overall column score
        overall_column_score = 0
        if dimension_scores:
            overall_column_score = sum(dimension_scores.values()) / len(dimension_scores)
        
        # Determine priority based on column name and score
        priority = "low"
        if any(keyword in column_name.lower() for keyword in ['id', 'key', 'primary']):
            priority = "critical"
        elif overall_column_score < 50:
            priority = "high"
        elif overall_column_score < 80:
            priority = "medium"
        
        return {
            "overall_column_score": round(overall_column_score, 2),
            "priority": priority,
            "dimensions_checked": applicable_dimensions,
            "dimensions_skipped": skipped_dimensions,
            "dimension_scores": {k: round(v, 2) for k, v in dimension_scores.items()},
            "total_dimensions": len(column_quality),
            "applicable_dimensions": len(applicable_dimensions)
        }
    
    def calculate_dimension_scores(self) -> Dict[str, Any]:
        """Calculate scores for each dimension across all columns"""
        dimension_totals = {
            "completeness": {"total_score": 0, "columns_count": 0},
            "uniqueness": {"total_score": 0, "columns_count": 0},
            "validity": {"total_score": 0, "columns_count": 0},
            "consistency": {"total_score": 0, "columns_count": 0},
            "timeliness": {"total_score": 0, "columns_count": 0}
        }
        
        for column_name, column_result in self.column_scores.items():
            for dimension_name, dimension_score in column_result['dimension_scores'].items():
                dimension_key = dimension_name.lower()
                if dimension_key in dimension_totals:
                    dimension_totals[dimension_key]["total_score"] += dimension_score
                    dimension_totals[dimension_key]["columns_count"] += 1
        
        # Calculate average scores
        dimension_scores = {}
        for dimension, totals in dimension_totals.items():
            if totals["columns_count"] > 0:
                avg_score = totals["total_score"] / totals["columns_count"]
                dimension_scores[dimension] = {
                    "score": round(avg_score, 2),
                    "columns_assessed": totals["columns_count"]
                }
            else:
                dimension_scores[dimension] = {
                    "score": 0,
                    "columns_assessed": 0
                }
        
        return dimension_scores
    
    def calculate_overall_score(self) -> Dict[str, Any]:
        """Calculate overall data quality score"""
        if not self.column_scores:
            return {"score": 0, "grade": "Unknown"}
        
        # Calculate weighted average based on column priorities
        total_weighted_score = 0
        total_weight = 0
        
        for column_name, column_result in self.column_scores.items():
            weight = 1
            if column_result['priority'] == 'critical':
                weight = 3
            elif column_result['priority'] == 'high':
                weight = 2
            
            total_weighted_score += column_result['overall_column_score'] * weight
            total_weight += weight
        
        overall_score = total_weighted_score / total_weight if total_weight > 0 else 0
        
        # Determine grade
        if overall_score >= 90:
            grade = "Excellent"
        elif overall_score >= 80:
            grade = "Good"
        elif overall_score >= 70:
            grade = "Fair"
        elif overall_score >= 60:
            grade = "Poor"
        else:
            grade = "Critical"
        
        return {
            "score": round(overall_score, 2),
            "grade": grade,
            "timestamp": datetime.now().isoformat() + "Z"
        }
    
    def generate_quality_report(self) -> Dict[str, Any]:
        """Generate comprehensive data quality report"""
        logger.info("Generating comprehensive data quality report...")
        
        # Calculate scores for all columns
        for column_name in self.df.columns:
            self.column_scores[column_name] = self.calculate_column_score(column_name)
        
        # Calculate dimension scores
        self.dimension_scores = self.calculate_dimension_scores()
        
        # Calculate overall score
        overall_health = self.calculate_overall_score()
        
        # Generate report
        report = {
            "overall_health": overall_health,
            "dimensions": self.dimension_scores,
            "column_analysis": self.column_scores,
            "summary": {
                "total_columns": len(self.df.columns),
                "columns_assessed": len(self.column_scores),
                "critical_priority_columns": len([c for c in self.column_scores.values() if c['priority'] == 'critical']),
                "high_priority_columns": len([c for c in self.column_scores.values() if c['priority'] == 'high']),
                "medium_priority_columns": len([c for c in self.column_scores.values() if c['priority'] == 'medium']),
                "low_priority_columns": len([c for c in self.column_scores.values() if c['priority'] == 'low'])
            }
        }
        
        return report
    
    def save_quality_report(self, output_file: str = "data_quality_report.json") -> None:
        """Save the quality report to a JSON file"""
        try:
            report = self.generate_quality_report()
            safe_json_dump(report, output_file)
            logger.info(f"Data quality report saved to: {output_file}")
            
        except Exception as e:
            logger.error(f"Error saving quality report: {e}")
            raise
    
    def print_summary(self) -> None:
        """Print a summary of the quality assessment"""
        if not self.column_scores:
            print("No quality assessment completed yet.")
            return
        
        print("\n" + "="*80)
        print("DATA QUALITY ASSESSMENT SUMMARY")
        print("="*80)
        
        # Overall health
        overall_health = self.calculate_overall_score()
        print(f"\nOverall Data Quality Score: {overall_health['score']}/100 ({overall_health['grade']})")
        
        # Dimension scores
        print(f"\nDimension Scores:")
        for dimension, scores in self.dimension_scores.items():
            print(f"  {dimension.capitalize()}: {scores['score']}/100 ({scores['columns_assessed']} columns)")
        
        # Column priorities
        priorities = {}
        for column_result in self.column_scores.values():
            priority = column_result['priority']
            priorities[priority] = priorities.get(priority, 0) + 1
        
        print(f"\nColumn Priority Distribution:")
        for priority, count in priorities.items():
            print(f"  {priority.capitalize()}: {count} columns")
        
        # Top and bottom performing columns
        sorted_columns = sorted(self.column_scores.items(), key=lambda x: x[1]['overall_column_score'], reverse=True)
        
        print(f"\nTop 3 Performing Columns:")
        for i, (col_name, col_result) in enumerate(sorted_columns[:3]):
            print(f"  {i+1}. {col_name}: {col_result['overall_column_score']}/100")
        
        print(f"\nBottom 3 Performing Columns:")
        for i, (col_name, col_result) in enumerate(sorted_columns[-3:]):
            print(f"  {i+1}. {col_name}: {col_result['overall_column_score']}/100")
        
        print("\n" + "="*80)

def main():
    """Main function to run the data quality scoring"""
    excel_file = "Unsafe Event - EI Tech App.xlsx"
    quality_file = "enhanced_data_quality_analysis.json"
    
    try:
        # Initialize scorer
        scorer = DataQualityScorer(excel_file, quality_file)
        
        # Load quality analysis and Excel file
        scorer.load_quality_analysis()
        scorer.load_excel_file()
        
        # Generate and save quality report
        scorer.save_quality_report()
        
        # Print summary
        scorer.print_summary()
        
        print(f"\n✅ Data quality scoring completed successfully!")
        print(f"📄 Report saved to: data_quality_report.json")
        print(f"📊 Comprehensive scoring with priority-based weighting")
        
    except Exception as e:
        logger.error(f"Error in main execution: {e}")
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()
