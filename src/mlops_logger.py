# src/mlops_logger.py
"""
MLOps Logging Module
Comprehensive logging system for production ML monitoring and error tracking.
"""

import pandas as pd
import numpy as np
import json
import os
import sqlite3
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union
import logging
from pathlib import Path
import hashlib
import pickle

class MLOpsLogger:
    """Production-ready MLOps logging system for error monitoring and model tracking."""
    
    def __init__(self, log_dir: str = "logs", db_path: str = "logs/mlops.db"):
        """
        Initialize MLOps Logger.
        
        Args:
            log_dir: Directory for log files
            db_path: Path to SQLite database for structured logging
        """
        self.log_dir = Path(log_dir)
        self.db_path = Path(db_path)
        self.log_dir.mkdir(exist_ok=True)
        
        # Initialize database
        self._init_database()
        
        # Setup logging
        self._setup_logging()
        
        # Logging counters
        self.prediction_count = 0
        self.error_count = 0
        
        print(f"MLOps Logger initialized. Database: {self.db_path}")
    
    def _init_database(self):
        """Initialize SQLite database for structured logging."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create predictions table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                model_name TEXT NOT NULL,
                dataset_name TEXT,
                sample_id TEXT,
                true_label INTEGER,
                predicted_label INTEGER,
                confidence REAL,
                is_error BOOLEAN,
                error_type TEXT,
                features_hash TEXT,
                metadata TEXT
            )
        ''')
        
        # Create errors table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS errors (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                model_name TEXT NOT NULL,
                dataset_name TEXT,
                sample_id TEXT,
                true_label INTEGER,
                predicted_label INTEGER,
                confidence REAL,
                error_type TEXT,
                error_category TEXT,
                mitigation_strategy TEXT,
                mitigation_result TEXT,
                features_hash TEXT,
                explanation TEXT,
                metadata TEXT
            )
        ''')
        
        # Create model_performance table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS model_performance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                model_name TEXT NOT NULL,
                dataset_name TEXT,
                accuracy REAL,
                precision_score REAL,
                recall_score REAL,
                f1_score REAL,
                error_rate REAL,
                total_samples INTEGER,
                error_samples INTEGER,
                metadata TEXT
            )
        ''')
        
        # Create mitigation_strategies table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS mitigation_strategies (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                model_name TEXT NOT NULL,
                dataset_name TEXT,
                strategy_name TEXT NOT NULL,
                strategy_params TEXT,
                baseline_accuracy REAL,
                improved_accuracy REAL,
                improvement REAL,
                execution_time REAL,
                success BOOLEAN,
                error_message TEXT,
                metadata TEXT
            )
        ''')
        
        # Create indexes for better performance
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_predictions_timestamp ON predictions(timestamp)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_predictions_model ON predictions(model_name)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_predictions_error ON predictions(is_error)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_errors_timestamp ON errors(timestamp)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_errors_model ON errors(model_name)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_errors_type ON errors(error_type)')
        
        conn.commit()
        conn.close()
    
    def _setup_logging(self):
        """Setup file logging for MLOps."""
        log_file = self.log_dir / "mlops.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        
        self.logger = logging.getLogger('MLOpsLogger')
    
    def log_prediction(self, 
                      model_name: str,
                      sample_id: str,
                      true_label: Optional[int],
                      predicted_label: int,
                      confidence: float,
                      dataset_name: Optional[str] = None,
                      features: Optional[np.ndarray] = None,
                      metadata: Optional[Dict] = None) -> str:
        """
        Log a single prediction.
        
        Args:
            model_name: Name of the model
            sample_id: Unique identifier for the sample
            true_label: True label (if available)
            predicted_label: Predicted label
            confidence: Prediction confidence
            dataset_name: Name of the dataset
            features: Feature vector
            metadata: Additional metadata
            
        Returns:
            Prediction ID
        """
        # Determine if this is an error
        is_error = true_label is not None and true_label != predicted_label
        error_type = self._classify_error_type(confidence, is_error)
        
        # Create features hash for deduplication
        features_hash = self._hash_features(features) if features is not None else None
        
        # Prepare metadata
        metadata_json = json.dumps(metadata) if metadata else None
        
        # Insert into database
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO predictions 
            (model_name, dataset_name, sample_id, true_label, predicted_label, 
             confidence, is_error, error_type, features_hash, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (model_name, dataset_name, sample_id, true_label, predicted_label,
              confidence, is_error, error_type, features_hash, metadata_json))
        
        prediction_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        # Update counters
        self.prediction_count += 1
        if is_error:
            self.error_count += 1
        
        # Log to file
        self.logger.info(f"Prediction logged: {model_name} - {sample_id} - "
                        f"Predicted: {predicted_label}, Confidence: {confidence:.3f}, "
                        f"Error: {is_error}")
        
        return str(prediction_id)
    
    def log_error_analysis(self,
                          model_name: str,
                          sample_id: str,
                          true_label: int,
                          predicted_label: int,
                          confidence: float,
                          error_category: str,
                          mitigation_strategy: Optional[str] = None,
                          mitigation_result: Optional[Dict] = None,
                          explanation: Optional[str] = None,
                          dataset_name: Optional[str] = None,
                          features: Optional[np.ndarray] = None,
                          metadata: Optional[Dict] = None) -> str:
        """
        Log detailed error analysis.
        
        Args:
            model_name: Name of the model
            sample_id: Unique identifier for the sample
            true_label: True label
            predicted_label: Predicted label
            confidence: Prediction confidence
            error_category: Category of the error
            mitigation_strategy: Strategy used to mitigate the error
            mitigation_result: Result of mitigation
            explanation: Human-readable explanation
            dataset_name: Name of the dataset
            features: Feature vector
            metadata: Additional metadata
            
        Returns:
            Error ID
        """
        error_type = self._classify_error_type(confidence, True)
        features_hash = self._hash_features(features) if features is not None else None
        
        # Prepare JSON fields
        mitigation_result_json = json.dumps(mitigation_result) if mitigation_result else None
        metadata_json = json.dumps(metadata) if metadata else None
        
        # Insert into database
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO errors 
            (model_name, dataset_name, sample_id, true_label, predicted_label,
             confidence, error_type, error_category, mitigation_strategy,
             mitigation_result, features_hash, explanation, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (model_name, dataset_name, sample_id, true_label, predicted_label,
              confidence, error_type, error_category, mitigation_strategy,
              mitigation_result_json, features_hash, explanation, metadata_json))
        
        error_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        # Log to file
        self.logger.warning(f"Error logged: {model_name} - {sample_id} - "
                           f"True: {true_label}, Predicted: {predicted_label}, "
                           f"Category: {error_category}")
        
        return str(error_id)
    
    def log_model_performance(self,
                            model_name: str,
                            dataset_name: str,
                            accuracy: float,
                            precision_score: float,
                            recall_score: float,
                            f1_score: float,
                            error_rate: float,
                            total_samples: int,
                            error_samples: int,
                            metadata: Optional[Dict] = None) -> str:
        """
        Log model performance metrics.
        
        Args:
            model_name: Name of the model
            dataset_name: Name of the dataset
            accuracy: Model accuracy
            precision_score: Precision score
            recall_score: Recall score
            f1_score: F1 score
            error_rate: Error rate
            total_samples: Total number of samples
            error_samples: Number of error samples
            metadata: Additional metadata
            
        Returns:
            Performance log ID
        """
        metadata_json = json.dumps(metadata) if metadata else None
        
        # Insert into database
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO model_performance 
            (model_name, dataset_name, accuracy, precision_score, recall_score,
             f1_score, error_rate, total_samples, error_samples, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (model_name, dataset_name, accuracy, precision_score, recall_score,
              f1_score, error_rate, total_samples, error_samples, metadata_json))
        
        performance_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        # Log to file
        self.logger.info(f"Performance logged: {model_name} - {dataset_name} - "
                        f"Accuracy: {accuracy:.4f}, Error Rate: {error_rate:.4f}")
        
        return str(performance_id)
    
    def log_mitigation_strategy(self,
                              model_name: str,
                              dataset_name: str,
                              strategy_name: str,
                              strategy_params: Dict,
                              baseline_accuracy: float,
                              improved_accuracy: float,
                              improvement: float,
                              execution_time: float,
                              success: bool,
                              error_message: Optional[str] = None,
                              metadata: Optional[Dict] = None) -> str:
        """
        Log mitigation strategy results.
        
        Args:
            model_name: Name of the model
            dataset_name: Name of the dataset
            strategy_name: Name of the mitigation strategy
            strategy_params: Parameters used for the strategy
            baseline_accuracy: Accuracy before mitigation
            improved_accuracy: Accuracy after mitigation
            improvement: Improvement in accuracy
            execution_time: Time taken to execute the strategy
            success: Whether the strategy was successful
            error_message: Error message if strategy failed
            metadata: Additional metadata
            
        Returns:
            Strategy log ID
        """
        strategy_params_json = json.dumps(strategy_params)
        metadata_json = json.dumps(metadata) if metadata else None
        
        # Insert into database
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO mitigation_strategies 
            (model_name, dataset_name, strategy_name, strategy_params,
             baseline_accuracy, improved_accuracy, improvement, execution_time,
             success, error_message, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (model_name, dataset_name, strategy_name, strategy_params_json,
              baseline_accuracy, improved_accuracy, improvement, execution_time,
              success, error_message, metadata_json))
        
        strategy_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        # Log to file
        status = "SUCCESS" if success else "FAILED"
        self.logger.info(f"Strategy logged: {model_name} - {strategy_name} - "
                        f"{status} - Improvement: {improvement:+.4f}")
        
        return str(strategy_id)
    
    def get_error_summary(self, 
                         model_name: Optional[str] = None,
                         dataset_name: Optional[str] = None,
                         days: int = 7) -> Dict[str, Any]:
        """
        Get error summary for the specified period.
        
        Args:
            model_name: Filter by model name
            dataset_name: Filter by dataset name
            days: Number of days to look back
            
        Returns:
            Error summary dictionary
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Build query
        where_conditions = ["timestamp >= datetime('now', '-{} days')".format(days)]
        params = []
        
        if model_name:
            where_conditions.append("model_name = ?")
            params.append(model_name)
        
        if dataset_name:
            where_conditions.append("dataset_name = ?")
            params.append(dataset_name)
        
        where_clause = " AND ".join(where_conditions)
        
        # Get error counts by type
        cursor.execute(f'''
            SELECT error_type, COUNT(*) as count
            FROM errors
            WHERE {where_clause}
            GROUP BY error_type
        ''', params)
        
        error_types = dict(cursor.fetchall())
        
        # Get error counts by category
        cursor.execute(f'''
            SELECT error_category, COUNT(*) as count
            FROM errors
            WHERE {where_clause}
            GROUP BY error_category
        ''', params)
        
        error_categories = dict(cursor.fetchall())
        
        # Get total error count
        cursor.execute(f'''
            SELECT COUNT(*) as total_errors
            FROM errors
            WHERE {where_clause}
        ''', params)
        
        total_errors = cursor.fetchone()[0]
        
        # Get error rate over time
        cursor.execute(f'''
            SELECT DATE(timestamp) as date, COUNT(*) as error_count
            FROM errors
            WHERE {where_clause}
            GROUP BY DATE(timestamp)
            ORDER BY date
        ''', params)
        
        error_timeline = dict(cursor.fetchall())
        
        conn.close()
        
        return {
            'total_errors': total_errors,
            'error_types': error_types,
            'error_categories': error_categories,
            'error_timeline': error_timeline,
            'period_days': days,
            'model_name': model_name,
            'dataset_name': dataset_name
        }
    
    def get_model_performance_history(self,
                                    model_name: str,
                                    dataset_name: Optional[str] = None,
                                    days: int = 30) -> pd.DataFrame:
        """
        Get model performance history.
        
        Args:
            model_name: Name of the model
            dataset_name: Filter by dataset name
            days: Number of days to look back
            
        Returns:
            DataFrame with performance history
        """
        conn = sqlite3.connect(self.db_path)
        
        where_conditions = [
            "model_name = ?",
            "timestamp >= datetime('now', '-{} days')".format(days)
        ]
        params = [model_name]
        
        if dataset_name:
            where_conditions.append("dataset_name = ?")
            params.append(dataset_name)
        
        where_clause = " AND ".join(where_conditions)
        
        query = f'''
            SELECT timestamp, dataset_name, accuracy, precision_score, 
                   recall_score, f1_score, error_rate, total_samples, error_samples
            FROM model_performance
            WHERE {where_clause}
            ORDER BY timestamp
        '''
        
        df = pd.read_sql_query(query, conn, params=params)
        conn.close()
        
        return df
    
    def get_mitigation_strategy_effectiveness(self,
                                            model_name: Optional[str] = None,
                                            days: int = 30) -> pd.DataFrame:
        """
        Get mitigation strategy effectiveness analysis.
        
        Args:
            model_name: Filter by model name
            days: Number of days to look back
            
        Returns:
            DataFrame with strategy effectiveness
        """
        conn = sqlite3.connect(self.db_path)
        
        where_conditions = ["timestamp >= datetime('now', '-{} days')".format(days)]
        params = []
        
        if model_name:
            where_conditions.append("model_name = ?")
            params.append(model_name)
        
        where_clause = " AND ".join(where_conditions)
        
        query = f'''
            SELECT strategy_name, dataset_name, baseline_accuracy, 
                   improved_accuracy, improvement, execution_time, success
            FROM mitigation_strategies
            WHERE {where_clause}
            ORDER BY improvement DESC
        '''
        
        df = pd.read_sql_query(query, conn, params=params)
        conn.close()
        
        return df
    
    def export_logs_to_csv(self, 
                          output_dir: str = "logs/exports",
                          days: int = 30) -> Dict[str, str]:
        """
        Export logs to CSV files for analysis.
        
        Args:
            output_dir: Directory to save CSV files
            days: Number of days to export
            
        Returns:
            Dictionary with file paths
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_paths = {}
        
        conn = sqlite3.connect(self.db_path)
        
        # Export predictions
        predictions_df = pd.read_sql_query(f'''
            SELECT * FROM predictions 
            WHERE timestamp >= datetime('now', '-{days} days')
        ''', conn)
        
        predictions_file = output_dir / f"predictions_{timestamp}.csv"
        predictions_df.to_csv(predictions_file, index=False)
        file_paths['predictions'] = str(predictions_file)
        
        # Export errors
        errors_df = pd.read_sql_query(f'''
            SELECT * FROM errors 
            WHERE timestamp >= datetime('now', '-{days} days')
        ''', conn)
        
        errors_file = output_dir / f"errors_{timestamp}.csv"
        errors_df.to_csv(errors_file, index=False)
        file_paths['errors'] = str(errors_file)
        
        # Export model performance
        performance_df = pd.read_sql_query(f'''
            SELECT * FROM model_performance 
            WHERE timestamp >= datetime('now', '-{days} days')
        ''', conn)
        
        performance_file = output_dir / f"model_performance_{timestamp}.csv"
        performance_df.to_csv(performance_file, index=False)
        file_paths['model_performance'] = str(performance_file)
        
        # Export mitigation strategies
        strategies_df = pd.read_sql_query(f'''
            SELECT * FROM mitigation_strategies 
            WHERE timestamp >= datetime('now', '-{days} days')
        ''', conn)
        
        strategies_file = output_dir / f"mitigation_strategies_{timestamp}.csv"
        strategies_df.to_csv(strategies_file, index=False)
        file_paths['mitigation_strategies'] = str(strategies_file)
        
        conn.close()
        
        self.logger.info(f"Logs exported to {output_dir}")
        return file_paths
    
    def generate_mlops_report(self, 
                            output_file: str = "logs/mlops_report.txt",
                            days: int = 7) -> str:
        """
        Generate comprehensive MLOps report.
        
        Args:
            output_file: Path to save the report
            days: Number of days to include in report
            
        Returns:
            Path to the generated report
        """
        output_file = Path(output_file)
        output_file.parent.mkdir(exist_ok=True)
        
        # Get summary data
        error_summary = self.get_error_summary(days=days)
        performance_history = self.get_model_performance_history("all", days=days)
        strategy_effectiveness = self.get_mitigation_strategy_effectiveness(days=days)
        
        with open(output_file, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("MLOPS MONITORING REPORT\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Report period: Last {days} days\n")
            f.write(f"Total predictions logged: {self.prediction_count}\n")
            f.write(f"Total errors logged: {self.error_count}\n\n")
            
            # Error summary
            f.write("ERROR SUMMARY:\n")
            f.write("-" * 20 + "\n")
            f.write(f"Total errors: {error_summary['total_errors']}\n")
            
            if error_summary['error_types']:
                f.write("\nErrors by type:\n")
                for error_type, count in error_summary['error_types'].items():
                    f.write(f"  {error_type}: {count}\n")
            
            if error_summary['error_categories']:
                f.write("\nErrors by category:\n")
                for category, count in error_summary['error_categories'].items():
                    f.write(f"  {category}: {count}\n")
            
            # Performance trends
            if not performance_history.empty:
                f.write("\n\nPERFORMANCE TRENDS:\n")
                f.write("-" * 25 + "\n")
                latest_performance = performance_history.iloc[-1]
                f.write(f"Latest accuracy: {latest_performance['accuracy']:.4f}\n")
                f.write(f"Latest error rate: {latest_performance['error_rate']:.4f}\n")
                
                if len(performance_history) > 1:
                    first_performance = performance_history.iloc[0]
                    accuracy_change = latest_performance['accuracy'] - first_performance['accuracy']
                    f.write(f"Accuracy change: {accuracy_change:+.4f}\n")
            
            # Strategy effectiveness
            if not strategy_effectiveness.empty:
                f.write("\n\nMITIGATION STRATEGY EFFECTIVENESS:\n")
                f.write("-" * 40 + "\n")
                successful_strategies = strategy_effectiveness[strategy_effectiveness['success'] == True]
                
                if not successful_strategies.empty:
                    f.write("Most effective strategies:\n")
                    for _, row in successful_strategies.head(5).iterrows():
                        f.write(f"  {row['strategy_name']}: {row['improvement']:+.4f} improvement\n")
                else:
                    f.write("No successful mitigation strategies found.\n")
            
            # Recommendations
            f.write("\n\nRECOMMENDATIONS:\n")
            f.write("-" * 20 + "\n")
            
            if error_summary['total_errors'] > 0:
                error_rate = error_summary['total_errors'] / self.prediction_count if self.prediction_count > 0 else 0
                if error_rate > 0.1:
                    f.write("• High error rate detected. Consider model retraining or data quality review.\n")
                elif error_rate > 0.05:
                    f.write("• Moderate error rate. Monitor closely and consider mitigation strategies.\n")
                else:
                    f.write("• Error rate is within acceptable range.\n")
            
            if not strategy_effectiveness.empty:
                best_strategy = strategy_effectiveness.iloc[0]
                if best_strategy['improvement'] > 0.01:
                    f.write(f"• Consider implementing {best_strategy['strategy_name']} as it shows significant improvement.\n")
            
            f.write("• Continue monitoring model performance and error patterns.\n")
            f.write("• Set up automated alerts for error rate increases.\n")
        
        self.logger.info(f"MLOps report generated: {output_file}")
        return str(output_file)
    
    def _classify_error_type(self, confidence: float, is_error: bool) -> str:
        """Classify error type based on confidence and error status."""
        if not is_error:
            return "Correct Prediction"
        elif confidence < 0.4:
            return "Low Confidence Error"
        elif confidence < 0.7:
            return "Medium Confidence Error"
        else:
            return "High Confidence Error"
    
    def _hash_features(self, features: np.ndarray) -> str:
        """Create hash of features for deduplication."""
        if features is None:
            return None
        return hashlib.md5(features.tobytes()).hexdigest()
    
    def cleanup_old_logs(self, days_to_keep: int = 90):
        """Clean up old log entries to manage database size."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cutoff_date = datetime.now() - timedelta(days=days_to_keep)
        
        # Delete old predictions
        cursor.execute('''
            DELETE FROM predictions 
            WHERE timestamp < ?
        ''', (cutoff_date,))
        
        # Delete old errors
        cursor.execute('''
            DELETE FROM errors 
            WHERE timestamp < ?
        ''', (cutoff_date,))
        
        # Delete old performance logs
        cursor.execute('''
            DELETE FROM model_performance 
            WHERE timestamp < ?
        ''', (cutoff_date,))
        
        # Delete old strategy logs
        cursor.execute('''
            DELETE FROM mitigation_strategies 
            WHERE timestamp < ?
        ''', (cutoff_date,))
        
        conn.commit()
        conn.close()
        
        self.logger.info(f"Cleaned up logs older than {days_to_keep} days")

# Example usage
if __name__ == "__main__":
    # Initialize MLOps logger
    mlops_logger = MLOpsLogger()
    
    # Log some sample predictions
    mlops_logger.log_prediction(
        model_name="test_model",
        sample_id="sample_001",
        true_label=5,
        predicted_label=3,
        confidence=0.85,
        dataset_name="digits"
    )
    
    # Log an error
    mlops_logger.log_error_analysis(
        model_name="test_model",
        sample_id="sample_001",
        true_label=5,
        predicted_label=3,
        confidence=0.85,
        error_category="High Confidence Error",
        explanation="Model confused similar digit shapes"
    )
    
    # Generate report
    report_path = mlops_logger.generate_mlops_report()
    print(f"MLOps report generated: {report_path}")
