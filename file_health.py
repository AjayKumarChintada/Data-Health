"""
File Health Service
Handles file data health operations including database queries, S3 file retrieval, and data quality analysis
"""

import os
import sys
import pandas as pd
import sqlalchemy as sa
from sqlalchemy import text
from datetime import datetime
import logging
import json
import tempfile
from io import BytesIO
from typing import Dict, Any, Optional, Tuple
from dotenv import load_dotenv
import uuid

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Load environment variables
load_dotenv()

# Import database configuration
from config.database_config import db_manager

# Import S3 service
from document_processing.s3_service import s3_client, s3_bucket, S3Service

# Import data quality modules
from health_engine.data_quality_scorer import DataQualityScorer
from health_engine.data_quality_insights_generator import DataQualityInsightsGenerator

logger = logging.getLogger(__name__)

class FileHealthService:
    """Service for handling file data health operations"""
    
    def __init__(self):
        self.db_manager = db_manager
    
    def get_file_info(self, file_id: str) -> Optional[Dict[str, Any]]:
        """
        Get file information from uploaded_files table
        
        Args:
            file_id: The file ID to query
            
        Returns:
            Dictionary with file information or None if not found
        """
        try:
            query = """
            SELECT file_id, original_filename, folder_type, s3_bucket, s3_key, s3_url, upload_status, tab_id,
                   is_processed, processing_status, processing_error, processed_at,
                   health_status, health_error, health_created_at, health_updated_at
            FROM uploaded_files 
            WHERE file_id = :file_id
            """
            
            engine = self.db_manager.postgres_engine
            with engine.connect() as conn:
                result = conn.execute(text(query), {"file_id": file_id})
                row = result.fetchone()
                
                if row:
                    return {
                        "file_id": row.file_id,
                        "file_name": row.original_filename,
                        "folder_type": row.folder_type,
                        "s3_bucket": row.s3_bucket,
                        "s3_key": row.s3_key,
                        "s3_url": row.s3_url,
                        "upload_status": row.upload_status,
                        "tab_id": row.tab_id,
                        "is_processed": row.is_processed,
                        "processing_status": row.processing_status,
                        "processing_error": row.processing_error,
                        "processed_at": row.processed_at,
                        "health_status": row.health_status,
                        "health_error": row.health_error,
                        "health_created_at": row.health_created_at,
                        "health_updated_at": row.health_updated_at
                    }
                else:
                    logger.warning(f"File with ID {file_id} not found in uploaded_files table")
                    return None
                    
        except Exception as e:
            logger.error(f"Error getting file info for {file_id}: {e}")
            raise
    
    def fetch_excel_bytes_from_s3(self, s3_key: str, bucket: str) -> bytes:
        """Fetch Excel bytes directly from S3 without writing to disk."""
        try:
            logger.info(f"Fetching Excel bytes from S3: {bucket}/{s3_key}")
            svc = S3Service()
            success, data = svc.download_file(s3_key, bucket)
            if not success:
                raise ValueError(data.get('error', 'Unknown S3 download error'))
            return data['content']
        except Exception as e:
            logger.error(f"Error fetching Excel bytes from S3: {e}")
            raise
    
    def validate_dimension_checks_columns(self, excel_source, dimension_checks: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
        """
        Validate that the columns in dimension checks match the actual file columns
        
        Args:
            file_path: Path to the Excel file
            dimension_checks: Dimension checks data from database
            
        Returns:
            Tuple of (is_valid, validation_details)
        """
        try:
            logger.info("Validating dimension checks columns against file columns")
            
            # Load Excel file to get actual columns
            if isinstance(excel_source, (bytes, bytearray)):
                df = pd.read_excel(BytesIO(excel_source))
            elif hasattr(excel_source, 'read'):
                df = pd.read_excel(BytesIO(excel_source.read()))
            else:
                df = pd.read_excel(excel_source)
            file_columns = set(df.columns.tolist())
            
            # Get expected columns from dimension checks
            # The structure is: {filename: {column_name: {dimensions: {...}}}}
            expected_columns = set()
            for filename, file_data in dimension_checks.items():
                if isinstance(file_data, dict):
                    expected_columns.update(file_data.keys())
            
            # Find missing and extra columns
            missing_columns = expected_columns - file_columns
            extra_columns = file_columns - expected_columns
            
            # Validation result
            is_valid = len(missing_columns) == 0 and len(extra_columns) == 0
            
            validation_details = {
                "is_valid": is_valid,
                "file_columns": sorted(list(file_columns)),
                "expected_columns": sorted(list(expected_columns)),
                "missing_columns": sorted(list(missing_columns)),
                "extra_columns": sorted(list(extra_columns)),
                "total_file_columns": len(file_columns),
                "total_expected_columns": len(expected_columns),
                "missing_count": len(missing_columns),
                "extra_count": len(extra_columns)
            }
            
            if not is_valid:
                logger.warning(f"Column validation failed: {len(missing_columns)} missing columns, {len(extra_columns)} extra columns")
                logger.warning(f"Missing columns: {missing_columns}")
                logger.warning(f"Extra columns: {extra_columns}")
            else:
                logger.info("Column validation passed: all expected columns found in file")
            
            return is_valid, validation_details
            
        except Exception as e:
            logger.error(f"Error validating dimension checks columns: {e}")
            return False, {
                "is_valid": False,
                "error": str(e),
                "file_columns": [],
                "expected_columns": [],
                "missing_columns": [],
                "extra_columns": [],
                "total_file_columns": 0,
                "total_expected_columns": 0,
                "missing_count": 0,
                "extra_count": 0
            }

    def perform_data_quality_analysis(self, excel_source, tab_id: str) -> Dict[str, Any]:
        """
        Perform data quality analysis on the Excel file
        
        Args:
            file_path: Path to the Excel file
            tab_id: The tab ID to get dimension checks from database
            
        Returns:
            Dictionary with data quality analysis results
        """
        try:
            logger.info("Starting data quality analysis")
            
            # Get dimension checks from database
            dimension_checks = self.get_dimension_checks(tab_id)
            if not dimension_checks:
                raise ValueError(f"Dimension checks not found for tab_id: {tab_id}")
            
            # Validate that columns in dimension checks match file columns
            is_valid, validation_details = self.validate_dimension_checks_columns(excel_source, dimension_checks)
            
            if not is_valid:
                # Return validation error instead of proceeding with analysis
                error_response = {
                    "error": "Column validation failed",
                    "message": "File columns do not match the expected columns in dimension checks. Cannot perform data quality analysis.",
                    "validation_details": validation_details,
                    "recommendation": "Please ensure the uploaded file contains all expected columns or update the dimension checks configuration."
                }
                logger.error("Column validation failed for provided Excel source")
                return error_response
            
            # Initialize data quality scorer with database data
            scorer = DataQualityScorer(excel_source)
            
            # Set the quality analysis data from database
            scorer.quality_analysis = dimension_checks
            
            # Load Excel file
            scorer.load_excel_file()
            
            # Generate quality report
            quality_report = scorer.generate_quality_report()
            
            # Add validation details to the report
            quality_report["column_validation"] = validation_details
            
            logger.info("Data quality analysis completed successfully")
            return quality_report
            
        except Exception as e:
            logger.error(f"Error performing data quality analysis: {e}")
            raise
    
    def generate_data_quality_insights(self, excel_source, quality_report: Dict[str, Any], tab_id: str) -> Dict[str, Any]:
        """
        Generate data quality insights using the insights generator
        
        Args:
            file_path: Path to the Excel file
            quality_report: Data quality report from scorer
            tab_id: The tab ID to get semantic analysis from database
            
        Returns:
            Dictionary with data quality insights
        """
        try:
            logger.info("Starting data quality insights generation")
            
            # Get semantic analysis from database
            semantic_analysis = self.get_semantic_analysis(tab_id)
            if not semantic_analysis:
                raise ValueError(f"Semantic analysis not found for tab_id: {tab_id}")
            
            # Initialize insights generator with in-memory data
            generator = DataQualityInsightsGenerator(
                excel_source=excel_source,
                semantic_analysis_data=semantic_analysis,
                quality_report_data=quality_report
            )
            
            # Generate insights report
            insights_report = generator.generate_insights_report()
            
            logger.info("Data quality insights generation completed successfully")
            return insights_report
            
        except Exception as e:
            logger.error(f"Error generating data quality insights: {e}")
            raise
    
    def save_quality_score(self, file_id: str, quality_report: Dict[str, Any]) -> bool:
        """
        Save data quality score to database
        
        Args:
            file_id: The file ID
            quality_report: Data quality report
            
        Returns:
            True if saved successfully, False otherwise
        """
        try:
            overall_health = quality_report.get("overall_health", {})
            overall_score = float(overall_health.get("score", 0))
            overall_grade = overall_health.get("grade", "Unknown")
            
            # First try to delete any existing record
            delete_sql = "DELETE FROM public.data_quality_scores WHERE file_id = :file_id"
            
            insert_sql = """
            INSERT INTO public.data_quality_scores (id, file_id, quality_score_data, overall_score, overall_grade, created_at)
            VALUES (:id, :file_id, :quality_score_data, :overall_score, :overall_grade, CURRENT_TIMESTAMP)
            """
            
            # Clean the quality report data to remove local file paths
            cleaned_quality_report = self._clean_report_data(quality_report)
            
            engine = self.db_manager.postgres_engine
            with engine.connect() as conn:
                # First delete any existing record
                delete_result = conn.execute(text(delete_sql), {'file_id': file_id})
                logger.info(f"Deleted existing data_quality_scores rows for file_id={file_id}: {delete_result.rowcount}")
                
                # Then insert new record
                insert_result = conn.execute(text(insert_sql), {
                    'id': str(uuid.uuid4()),
                    'file_id': file_id,
                    'quality_score_data': json.dumps(cleaned_quality_report),
                    'overall_score': overall_score,
                    'overall_grade': overall_grade
                })
                logger.info(f"Inserted data_quality_scores row for file_id={file_id}: {insert_result.rowcount}")
                conn.commit()
                
            logger.info(f"Quality score saved successfully for file {file_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving quality score for file {file_id}: {e}")
            return False
    
    def save_quality_insights(self, file_id: str, insights_report: Dict[str, Any]) -> bool:
        """
        Save data quality insights to database
        
        Args:
            file_id: The file ID
            insights_report: Data quality insights report
            
        Returns:
            True if saved successfully, False otherwise
        """
        try:
            overall_insights = insights_report.get("overall_insights", {})
            business_impact_level = overall_insights.get("business_impact_level", "low")
            implementation_ease = overall_insights.get("implementation_ease", "medium")
            
            # First try to delete any existing record
            delete_sql = "DELETE FROM public.data_quality_insights WHERE file_id = :file_id"
            
            insert_sql = """
            INSERT INTO public.data_quality_insights (id, file_id, insights_data, business_impact_level, implementation_ease, created_at)
            VALUES (:id, :file_id, :insights_data, :business_impact_level, :implementation_ease, CURRENT_TIMESTAMP)
            """
            
            # Clean the insights report data to remove local file paths
            cleaned_insights_report = self._clean_report_data(insights_report)
            
            engine = self.db_manager.postgres_engine
            with engine.connect() as conn:
                # First delete any existing record
                delete_result = conn.execute(text(delete_sql), {'file_id': file_id})
                logger.info(f"Deleted existing data_quality_insights rows for file_id={file_id}: {delete_result.rowcount}")
                
                # Then insert new record
                insert_result = conn.execute(text(insert_sql), {
                    'id': str(uuid.uuid4()),
                    'file_id': file_id,
                    'insights_data': json.dumps(cleaned_insights_report),
                    'business_impact_level': business_impact_level,
                    'implementation_ease': implementation_ease
                })
                logger.info(f"Inserted data_quality_insights row for file_id={file_id}: {insert_result.rowcount}")
                conn.commit()
                
            logger.info(f"Quality insights saved successfully for file {file_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving quality insights for file {file_id}: {e}")
            return False
    
    def _clean_report_data(self, data: Any) -> Any:
        """
        Clean report data to remove local file paths, binary data, and sensitive information
        
        Args:
            data: The data to clean
            
        Returns:
            Cleaned data without local file paths or binary data
        """
        if isinstance(data, dict):
            cleaned_data = {}
            for key, value in data.items():
                # Skip keys that contain file paths or binary data
                if any(path_key in key.lower() for path_key in ['file', 'path', 'excel_file', 'quality_report_file', 'excel_source']):
                    continue
                # Skip binary data
                if isinstance(value, (bytes, bytearray)):
                    cleaned_data[key] = f"<binary_data_{len(value)}_bytes>"
                    continue
                # Recursively clean nested data
                cleaned_data[key] = self._clean_report_data(value)
            return cleaned_data
        elif isinstance(data, list):
            return [self._clean_report_data(item) for item in data]
        elif isinstance(data, (bytes, bytearray)):
            return f"<binary_data_{len(data)}_bytes>"
        else:
            return data
    
    def update_processing_status(self, file_id: str, processing_status: str, processing_error: str = None) -> bool:
        """
        Update the processing status of a file in the uploaded_files table
        
        Args:
            file_id: The file ID
            processing_status: The new processing status (e.g., 'completed', 'failed', 'in_progress')
            processing_error: Error message if processing failed
            
        Returns:
            True if updated successfully, False otherwise
        """
        try:
            if processing_status == "completed":
                update_sql = """
                UPDATE uploaded_files 
                SET is_processed = true,
                    processing_status = :processing_status,
                    processed_at = CURRENT_TIMESTAMP,
                    processing_error = :processing_error,
                    updated_at = CURRENT_TIMESTAMP
                WHERE file_id = :file_id
                """
            else:
                update_sql = """
                UPDATE uploaded_files 
                SET is_processed = false,
                    processing_status = :processing_status,
                    processing_error = :processing_error,
                    updated_at = CURRENT_TIMESTAMP
                WHERE file_id = :file_id
                """
            
            engine = self.db_manager.postgres_engine
            with engine.connect() as conn:
                result = conn.execute(text(update_sql), {
                    'file_id': file_id,
                    'processing_status': processing_status,
                    'processing_error': processing_error
                })
                conn.commit()
                
                if result.rowcount > 0:
                    logger.info(f"Updated processing status to '{processing_status}' for file {file_id}")
                    return True
                else:
                    logger.warning(f"No rows updated for file {file_id}")
                    return False
                    
        except Exception as e:
            logger.error(f"Error updating processing status for file {file_id}: {e}")
            return False
    
    def update_health_status(self, file_id: str, health_status: str, health_error: str = None) -> bool:
        """
        Update the health status of a file in the uploaded_files table
        
        Args:
            file_id: The file ID
            health_status: The new health status ('processing', 'completed', 'error')
            health_error: Error message if health analysis failed
            
        Returns:
            True if updated successfully, False otherwise
        """
        try:
            update_sql = """
            UPDATE uploaded_files 
            SET health_status = :health_status,
                health_error = :health_error,
                health_updated_at = CURRENT_TIMESTAMP
            WHERE file_id = :file_id
            """
            
            engine = self.db_manager.postgres_engine
            with engine.connect() as conn:
                result = conn.execute(text(update_sql), {
                    'file_id': file_id,
                    'health_status': health_status,
                    'health_error': health_error
                })
                conn.commit()
                
                if result.rowcount > 0:
                    logger.info(f"Updated health status to '{health_status}' for file {file_id}")
                    return True
                else:
                    logger.warning(f"No rows updated for file {file_id}")
                    return False
                    
        except Exception as e:
            logger.error(f"Error updating health status for file {file_id}: {e}")
            return False
    
    def get_semantic_analysis(self, tab_id: str) -> Optional[Dict[str, Any]]:
        """
        Get semantic analysis data from schindler_semantics table
        
        Args:
            tab_id: The tab ID to look up
            
        Returns:
            Dictionary with semantic analysis data or None if not found
        """
        try:
            query = """
            SELECT semantics
            FROM schindler_semantics 
            WHERE tab_id = :tab_id
            """
            
            engine = self.db_manager.postgres_engine
            with engine.connect() as conn:
                result = conn.execute(text(query), {"tab_id": tab_id})
                row = result.fetchone()
                
                if row:
                    return row.semantics
                else:
                    logger.warning(f"Semantic analysis not found for tab_id: {tab_id}")
                    return None
                    
        except Exception as e:
            logger.error(f"Error getting semantic analysis for tab_id {tab_id}: {e}")
            return None
    
    def get_dimension_checks(self, tab_id: str) -> Optional[Dict[str, Any]]:
        """
        Get dimension checks data from schindler_dimension_checks table
        
        Args:
            tab_id: The tab ID to look up
            
        Returns:
            Dictionary with dimension checks data or None if not found
        """
        try:
            query = """
            SELECT dimension_checks
            FROM schindler_dimension_checks 
            WHERE tab_id = :tab_id
            """
            
            engine = self.db_manager.postgres_engine
            with engine.connect() as conn:
                result = conn.execute(text(query), {"tab_id": tab_id})
                row = result.fetchone()
                
                if row:
                    return row.dimension_checks
                else:
                    logger.warning(f"Dimension checks not found for tab_id: {tab_id}")
                    return None
                    
        except Exception as e:
            logger.error(f"Error getting dimension checks for tab_id {tab_id}: {e}")
            return None
    
    def process_file_health(self, file_id: str) -> Dict[str, Any]:
        """
        Main method to process file health analysis
        
        Args:
            file_id: The file ID to analyze
            
        Returns:
            Dictionary with complete file health analysis results
        """
        try:
            logger.info(f"Starting file health analysis for file ID: {file_id}")
            
            # Step 1: Get file information from database
            file_info = self.get_file_info(file_id)
            if not file_info:
                raise ValueError(f"File with ID {file_id} not found in database")
            
            # Check if upload status is completed
            upload_status = file_info.get("upload_status", "pending")
            if upload_status != "completed":
                raise ValueError(f"File upload status is {upload_status}. Only files with 'completed' status can be processed.")
            
            # Step 2: Set health status to processing
            self.update_health_status(file_id, "processing")
            
            # Step 3: Fetch Excel bytes from S3
            if not file_info.get("s3_key") or not file_info.get("s3_bucket"):
                raise ValueError("S3 information not available for file")
            
            excel_bytes = self.fetch_excel_bytes_from_s3(file_info["s3_key"], file_info["s3_bucket"])
            
            try:
                # Step 4: Perform data quality analysis
                quality_report = self.perform_data_quality_analysis(excel_bytes, file_info["tab_id"])
                
                # Check if quality report contains validation error
                if "error" in quality_report and quality_report["error"] == "Column validation failed":
                    # Update health status to error due to validation error
                    self.update_health_status(file_id, "error", quality_report["message"])
                    
                    # Return validation error response
                    response = {
                        "file_id": file_id,
                        "file_name": file_info["file_name"],
                        "folder_type": file_info["folder_type"],
                        "upload_status": file_info["upload_status"],  # Keep original upload status
                        "health_status": "error",
                        "health_error": quality_report["message"],
                        "analysis_status": "failed",
                        "error": quality_report["error"],
                        "message": quality_report["message"],
                        "validation_details": quality_report["validation_details"],
                        "recommendation": quality_report["recommendation"],
                        "timestamp": datetime.now().isoformat()
                    }
                    
                    logger.error(f"File health analysis failed due to column validation error for file {file_id}")
                    return response
                
                # Step 5: Generate data quality insights
                insights_report = self.generate_data_quality_insights(excel_bytes, quality_report, file_info["tab_id"])
                
                # Step 6: Save results to database
                quality_saved = self.save_quality_score(file_id, quality_report)
                insights_saved = self.save_quality_insights(file_id, insights_report)
                
                # Step 7: Update health status to completed
                self.update_health_status(file_id, "completed")
                
                # Step 8: Prepare response
                response = {
                    "file_id": file_id,
                    "file_name": file_info["file_name"],
                    "folder_type": file_info["folder_type"],
                    "upload_status": file_info["upload_status"],  # Keep original upload status
                    "health_status": "completed",
                    "analysis_status": "completed",
                    "data_quality_score": quality_report.get("overall_health", {}).get("score", 0),
                    "data_quality_grade": quality_report.get("overall_health", {}).get("grade", "Unknown"),
                    "business_impact_level": insights_report.get("overall_insights", {}).get("business_impact_level", "low"),
                    "implementation_ease": insights_report.get("overall_insights", {}).get("implementation_ease", "medium"),
                    "quality_report": quality_report,
                    "insights_report": insights_report,
                    "database_saved": {
                        "quality_score": quality_saved,
                        "quality_insights": insights_saved
                    },
                    "timestamp": datetime.now().isoformat()
                }
                
                logger.info(f"File health analysis completed successfully for file {file_id}")
                return response
                
            finally:
                pass
                    
        except Exception as e:
            logger.error(f"Error processing file health for {file_id}: {e}")
            # Update health status to error
            self.update_health_status(file_id, "error", str(e))
            raise
