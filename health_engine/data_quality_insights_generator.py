#!/usr/bin/env python3
"""
Data Quality Insights Generator

This script generates actionable insights and recommendations for improving data quality
by analyzing semantic descriptions and data quality reports using Azure OpenAI.
"""

import json
import logging
import os
from typing import Dict, Any, List, Union, IO, Optional
from io import BytesIO
from datetime import datetime
import pandas as pd
from openai import AzureOpenAI
from dotenv import load_dotenv
from .json_utils import safe_json_dump

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DataQualityInsightsGenerator:
    """
    Generates data quality improvement insights using Azure OpenAI.
    Analyzes semantic descriptions and data quality reports to provide actionable recommendations.
    """
    
    def __init__(self, 
                 excel_source: Union[str, bytes, IO[bytes]],
                 semantic_file: str = None,
                 quality_report_file: str = None,
                 semantic_analysis_data: Optional[Dict[str, Any]] = None,
                 quality_report_data: Optional[Dict[str, Any]] = None):
        """
        Initialize the insights generator.
        
        Args:
            excel_file_path: Path to the Excel file
            semantic_file: Path to semantic analysis JSON
            quality_report_file: Path to data quality report JSON
        """
        self.excel_file_path = excel_source
        self.semantic_file = semantic_file
        self.quality_report_file = quality_report_file
        
        # Initialize data containers
        self.df = None
        self.semantic_analysis = semantic_analysis_data or {}
        self.quality_report = quality_report_data or {}
        self.insights_report = {}
        
        # Initialize Azure OpenAI client
        self.client = AzureOpenAI(
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview")
        )
        self.deployment_name = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
        
        logger.info("Data Quality Insights Generator initialized")
    
    def load_excel_file(self) -> None:
        """Load Excel file into pandas DataFrame from path or in-memory bytes."""
        try:
            if isinstance(self.excel_file_path, (bytes, bytearray)):
                logger.info("Loading Excel from in-memory bytes (insights generator)")
                self.df = pd.read_excel(BytesIO(self.excel_file_path))
            elif hasattr(self.excel_file_path, 'read'):
                logger.info("Loading Excel from file-like object (insights generator)")
                data = self.excel_file_path.read()
                self.df = pd.read_excel(BytesIO(data))
            else:
                logger.info(f"Loading Excel file: {self.excel_file_path}")
                self.df = pd.read_excel(self.excel_file_path)
            logger.info(f"Successfully loaded Excel file with {len(self.df)} rows and {len(self.df.columns)} columns")
        except Exception as e:
            logger.error(f"Error loading Excel file: {e}")
            raise
    
    def load_semantic_analysis(self) -> None:
        """Load semantic analysis from provided data or JSON file."""
        try:
            if self.semantic_analysis:
                logger.info("Using in-memory semantic analysis data")
                return
            if self.semantic_file and os.path.exists(self.semantic_file):
                logger.info(f"Loading semantic analysis from: {self.semantic_file}")
                with open(self.semantic_file, 'r', encoding='utf-8') as f:
                    self.semantic_analysis = json.load(f)
                logger.info("Successfully loaded semantic analysis")
            else:
                logger.warning("No semantic analysis data provided and no file path specified")
                self.semantic_analysis = {}
        except Exception as e:
            logger.error(f"Error loading semantic analysis: {e}")
            raise
    
    def load_quality_report(self) -> None:
        """Load data quality report from provided data or JSON file."""
        try:
            if self.quality_report:
                logger.info("Using in-memory quality report data")
                return
            if self.quality_report_file and os.path.exists(self.quality_report_file):
                logger.info(f"Loading quality report from: {self.quality_report_file}")
                with open(self.quality_report_file, 'r', encoding='utf-8') as f:
                    self.quality_report = json.load(f)
                logger.info("Successfully loaded quality report")
            else:
                logger.warning("No quality report data provided and no file path specified")
                self.quality_report = {}
        except Exception as e:
            logger.error(f"Error loading quality report: {e}")
            raise
    
    def get_column_quality_summary(self, column_name: str) -> Dict[str, Any]:
        """
        Get comprehensive quality summary for a column.
        
        Args:
            column_name: Name of the column
            
        Returns:
            Dictionary with quality summary
        """
        if column_name not in self.quality_report.get("column_analysis", {}):
            return {}
        
        column_analysis = self.quality_report["column_analysis"][column_name]
        
        # Get semantic description
        semantic_desc = ""
        for file_name, file_data in self.semantic_analysis.items():
            if column_name in file_data:
                semantic_desc = file_data[column_name].get("description", "")
                break
        
        # Get basic metrics
        total_rows = len(self.df)
        null_count = self.df[column_name].isnull().sum()
        unique_count = self.df[column_name].nunique()
        duplicate_count = total_rows - unique_count
        
        return {
            "column_name": column_name,
            "semantic_description": semantic_desc,
            "overall_score": column_analysis.get("overall_column_score", 0),
            "priority": column_analysis.get("priority", "low"),
            "dimensions_checked": column_analysis.get("dimensions_checked", []),
            "dimension_scores": column_analysis.get("dimension_scores", {}),
            "total_rows": total_rows,
            "null_count": null_count,
            "null_percentage": (null_count / total_rows) * 100 if total_rows > 0 else 0,
            "unique_count": unique_count,
            "duplicate_count": duplicate_count,
            "duplicate_percentage": (duplicate_count / total_rows) * 100 if total_rows > 0 else 0,
            "data_type": str(self.df[column_name].dtype),
            "sample_values": self.df[column_name].dropna().unique()[:5].tolist()
        }
    
    def create_insights_prompt(self, column_summary: Dict[str, Any]) -> str:
        """
        Create detailed prompt for generating business-focused insights.
        
        Args:
            column_summary: Quality summary for the column
            
        Returns:
            Formatted prompt string
        """
        prompt = f"""
You are a business data analyst helping non-technical Excel users improve their data quality. Analyze the following column data and provide business-focused, actionable insights that can be easily understood and implemented by someone who only knows Excel.

COLUMN INFORMATION:
- Column Name: {column_summary['column_name']}
- Business Purpose: {column_summary['semantic_description']}
- Data Quality Score: {column_summary['overall_score']}/100
- Business Priority: {column_summary['priority']}
- Data Type: {column_summary['data_type']}

BUSINESS METRICS:
- Total Records: {column_summary['total_rows']}
- Missing Information: {column_summary['null_count']} records ({column_summary['null_percentage']:.2f}%)
- Unique Values: {column_summary['unique_count']}
- Duplicate Records: {column_summary['duplicate_count']} ({column_summary['duplicate_percentage']:.2f}%)
- Sample Data: {column_summary['sample_values']}

QUALITY SCORES BY AREA:
{json.dumps(column_summary['dimension_scores'], indent=2)}

AREAS CHECKED:
{', '.join(column_summary['dimensions_checked'])}

TASK:
Provide business-focused insights and recommendations for improving this column's data quality. Focus on practical Excel-based solutions that non-technical users can implement. Structure your response as a JSON object with the following format:

{{
  "column_name": "{column_summary['column_name']}",
  "business_assessment": {{
    "overall_score": {column_summary['overall_score']},
    "business_strengths": ["List 2-3 business benefits of this column's current state"],
    "business_concerns": ["List 2-3 business problems caused by data quality issues"],
    "urgent_issues": ["List any critical business issues that need immediate attention"]
  }},
  "business_recommendations": {{
    "immediate_actions": [
      {{
        "action": "Specific business action using Excel features",
        "business_impact": "How this affects business operations and decision-making",
        "expected_benefit": "Business benefit of implementing this action",
        "excel_effort": "easy/medium/hard (for Excel users)"
      }}
    ],
    "short_term_improvements": [
      {{
        "action": "Business process improvement using Excel",
        "business_impact": "How this affects business operations and decision-making",
        "expected_benefit": "Business benefit of implementing this action",
        "excel_effort": "easy/medium/hard (for Excel users)"
      }}
    ],
    "long_term_improvements": [
      {{
        "action": "Strategic business process change",
        "business_impact": "How this affects business operations and decision-making",
        "expected_benefit": "Business benefit of implementing this action",
        "excel_effort": "easy/medium/hard (for Excel users)"
      }}
    ]
  }},
  "excel_best_practices": [
    "Specific Excel tips and techniques for this column"
  ],
  "business_process_suggestions": [
    "Suggestions for improving business processes around this data"
  ],
  "business_impact_level": "low/medium/high",
  "implementation_ease": "easy/medium/hard"
}}

Focus on:
1. Business impact and benefits, not technical implementation
2. Excel-based solutions that non-technical users can implement
3. Practical business process improvements
4. Clear, simple language avoiding technical jargon
5. Specific Excel features and techniques users can apply
6. How data quality affects business decision-making and operations

Remember: The user only knows Excel and wants to improve their business data quality.
"""
        return prompt
    
    def generate_column_insights(self, column_name: str) -> Dict[str, Any]:
        """
        Generate insights for a single column using Azure OpenAI.
        
        Args:
            column_name: Name of the column
            
        Returns:
            Dictionary with insights and recommendations
        """
        try:
            logger.info(f"Generating insights for column: {column_name}")
            
            # Get column quality summary
            column_summary = self.get_column_quality_summary(column_name)
            if not column_summary:
                logger.warning(f"No quality data found for column: {column_name}")
                return {}
            
            # Create prompt
            prompt = self.create_insights_prompt(column_summary)
            
            # Call Azure OpenAI
            response = self.client.chat.completions.create(
                model=self.deployment_name,
                messages=[
                    {"role": "system", "content": "You are a business data analyst helping non-technical Excel users improve their data quality. Focus on business impact, Excel-based solutions, and practical recommendations."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=2000
            )
            
            # Parse response
            insights_text = response.choices[0].message.content.strip()
            
            # Try to parse as JSON
            try:
                insights = json.loads(insights_text)
                logger.info(f"Successfully generated insights for column: {column_name}")
                return insights
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse insights JSON for column {column_name}: {e}")
                logger.error(f"Raw response: {insights_text}")
                return {
                    "column_name": column_name,
                    "error": "Failed to parse insights response",
                    "raw_response": insights_text
                }
                
        except Exception as e:
            logger.error(f"Error generating insights for column {column_name}: {e}")
            return {
                "column_name": column_name,
                "error": str(e)
            }
    
    def create_overall_insights_prompt(self) -> str:
        """
        Create prompt for generating high-level business insights.
        
        Returns:
            Formatted prompt string
        """
        overall_health = self.quality_report.get("overall_health", {})
        dimensions = self.quality_report.get("dimensions", {})
        summary = self.quality_report.get("summary", {})
        
        # Get top and bottom performing columns
        column_analysis = self.quality_report.get("column_analysis", {})
        sorted_columns = sorted(
            column_analysis.items(),
            key=lambda x: x[1].get("overall_column_score", 0),
            reverse=True
        )
        
        top_columns = sorted_columns[:3]
        bottom_columns = sorted_columns[-3:]
        
        prompt = f"""
You are a business data analyst providing high-level insights for improving data quality. Analyze the overall data quality assessment and provide 5-10 strategic business recommendations that focus on WHAT needs to be improved, not HOW to do it.

OVERALL BUSINESS DATA ASSESSMENT:
- Overall Data Quality Score: {overall_health.get('score', 0)}/100
- Business Grade: {overall_health.get('grade', 'Unknown')}
- Total Data Fields: {summary.get('total_columns', 0)}
- Fields Analyzed: {summary.get('columns_assessed', 0)}

QUALITY PERFORMANCE BY AREA:
{json.dumps(dimensions, indent=2)}

BUSINESS PRIORITY DISTRIBUTION:
- Critical Business Fields: {summary.get('critical_priority_columns', 0)}
- High Priority Fields: {summary.get('high_priority_columns', 0)}
- Medium Priority Fields: {summary.get('medium_priority_columns', 0)}
- Low Priority Fields: {summary.get('low_priority_columns', 0)}

BEST PERFORMING DATA FIELDS:
{json.dumps([{'field': col, 'score': data.get('overall_column_score', 0)} for col, data in top_columns], indent=2)}

FIELDS NEEDING ATTENTION:
{json.dumps([{'field': col, 'score': data.get('overall_column_score', 0)} for col, data in bottom_columns], indent=2)}

TASK:
Provide 5-10 high-level business insights and recommendations for improving data quality. Focus on WHAT needs to be improved for better data health, not HOW to implement it. Structure your response as a JSON object with the following format:

{{
  "business_overview": {{
    "current_business_state": "Brief assessment of how data quality affects business operations",
    "business_strengths": ["List 2-3 business benefits of current data quality"],
    "business_risks": ["List 2-3 business risks caused by poor data quality"]
  }},
  "data_quality_insights": [
    {{
      "insight": "Specific data quality issue that needs attention",
      "business_impact": "How this affects business operations and decision-making",
      "priority": "high/medium/low",
      "expected_benefit": "Business benefit of addressing this issue"
    }}
  ],
  "business_recommendations": [
    "High-level business recommendations for improving data quality"
  ],
  "data_health_priorities": [
    "List of data health priorities that need immediate attention"
  ]
}}

Focus on:
1. Business impact and benefits, not technical implementation
2. High-level data quality issues that affect business operations
3. Strategic recommendations for improving data health
4. Clear, simple language avoiding technical jargon
5. How data quality affects business decision-making and operations
6. What business problems are caused by poor data quality

Remember: Focus on WHAT needs to be improved for better data health. The HOW (implementation) is up to the employee.
"""
        return prompt
    
    def generate_overall_insights(self) -> Dict[str, Any]:
        """
        Generate overall insights for the entire dataset.
        
        Returns:
            Dictionary with overall insights and recommendations
        """
        try:
            logger.info("Generating overall insights for the dataset")
            
            # Create prompt
            prompt = self.create_overall_insights_prompt()
            
            # Call Azure OpenAI
            response = self.client.chat.completions.create(
                model=self.deployment_name,
                messages=[
                    {"role": "system", "content": "You are a business data analyst helping non-technical Excel users improve their overall data quality. Focus on business impact, Excel-based solutions, and practical strategic recommendations."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=2500
            )
            
            # Parse response
            insights_text = response.choices[0].message.content.strip()
            
            # Try to parse as JSON
            try:
                insights = json.loads(insights_text)
                logger.info("Successfully generated overall insights")
                return insights
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse overall insights JSON: {e}")
                logger.error(f"Raw response: {insights_text}")
                return {
                    "error": "Failed to parse overall insights response",
                    "raw_response": insights_text
                }
                
        except Exception as e:
            logger.error(f"Error generating overall insights: {e}")
            return {
                "error": str(e)
            }
    
    def generate_insights_report(self) -> Dict[str, Any]:
        """
        Generate overall insights report focusing on business recommendations.
        
        Returns:
            Complete insights report
        """
        logger.info("Starting insights report generation")
        
        # Load all required data
        self.load_excel_file()
        self.load_semantic_analysis()
        self.load_quality_report()
        
        # Generate overall insights only
        overall_insights = self.generate_overall_insights()
        
        # Compile complete report
        self.insights_report = {
            "metadata": {
                "generated_at": datetime.now().isoformat(),
                "excel_source_type": "in_memory_bytes" if isinstance(self.excel_file_path, (bytes, bytearray)) else "file_path",
                "semantic_file": self.semantic_file,
                "quality_report_file": self.quality_report_file,
                "total_columns_analyzed": len(self.quality_report.get("column_analysis", {}))
            },
            "overall_insights": overall_insights,
            "summary": {
                "overall_quality_score": self.quality_report.get("overall_health", {}).get("score", 0),
                "overall_grade": self.quality_report.get("overall_health", {}).get("grade", "Unknown")
            }
        }
        
        logger.info("Insights report generation completed")
        return self.insights_report
    
    def save_insights_report(self, output_file: str = "data_quality_insights_report.json") -> None:
        """
        Save insights report to JSON file.
        
        Args:
            output_file: Path to output file
        """
        try:
            if not self.insights_report:
                logger.warning("No insights report generated. Call generate_insights_report() first.")
                return
            
            logger.info(f"Saving insights report to: {output_file}")
            safe_json_dump(self.insights_report, output_file)
            logger.info(f"Successfully saved insights report to: {output_file}")
            
        except Exception as e:
            logger.error(f"Error saving insights report: {e}")
            raise
    
    def print_summary(self) -> None:
        """Print human-readable summary of insights."""
        if not self.insights_report:
            logger.warning("No insights report available. Generate insights first.")
            return
        
        print("\n" + "="*80)
        print("DATA QUALITY INSIGHTS SUMMARY")
        print("="*80)
        
        # Overall summary
        overall = self.insights_report.get("overall_insights", {})
        if "error" not in overall:
            print(f"\n📊 BUSINESS OVERVIEW:")
            print(f"   Business Impact: {overall.get('business_overview', {}).get('current_business_state', 'N/A')}")
            print(f"   Business Strengths: {len(overall.get('business_overview', {}).get('business_strengths', []))} identified")
            print(f"   Business Risks: {len(overall.get('business_overview', {}).get('business_risks', []))} identified")
        
        # Data quality insights
        insights = overall.get("data_quality_insights", [])
        if insights:
            print(f"\n🎯 DATA QUALITY INSIGHTS ({len(insights)} recommendations):")
            for i, insight in enumerate(insights, 1):
                print(f"   {i}. {insight.get('insight', 'N/A')}")
                print(f"      Priority: {insight.get('priority', 'N/A').upper()}")
                print(f"      Business Impact: {insight.get('business_impact', 'N/A')}")
                print()
        
        # Business recommendations
        recommendations = overall.get("business_recommendations", [])
        if recommendations:
            print(f"\n💡 BUSINESS RECOMMENDATIONS:")
            for i, rec in enumerate(recommendations, 1):
                print(f"   {i}. {rec}")
        
        # Data health priorities
        priorities = overall.get("data_health_priorities", [])
        if priorities:
            print(f"\n🚨 DATA HEALTH PRIORITIES:")
            for i, priority in enumerate(priorities, 1):
                print(f"   {i}. {priority}")
        
        # Summary
        summary = self.insights_report.get("summary", {})
        print(f"\n📈 DATA QUALITY METRICS:")
        print(f"   Overall Data Quality: {summary.get('overall_quality_score', 0)}/100")
        print(f"   Business Grade: {summary.get('overall_grade', 'Unknown')}")
        
        print("\n" + "="*80)


def main():
    """Main function to run the insights generator."""
    try:
        logger.info("Starting Data Quality Insights Generator")
        
        # Initialize generator
        generator = DataQualityInsightsGenerator()
        
        # Generate insights report
        insights_report = generator.generate_insights_report()
        
        # Save report
        generator.save_insights_report()
        
        # Print summary
        generator.print_summary()
        
        logger.info("Data Quality Insights Generator completed successfully")
        
    except Exception as e:
        logger.error(f"Error in main execution: {e}")
        raise


if __name__ == "__main__":
    main()
