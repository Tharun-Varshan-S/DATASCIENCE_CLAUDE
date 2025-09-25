# demo_enhanced_features.py
"""
Demo script showcasing all the enhanced features of the Intelligent Error Mitigation System.
This script demonstrates XAI, error storytelling, strategy comparison, and MLOps logging.
"""

import numpy as np
import os
import sys
from datetime import datetime

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def main():
    """Demonstrate all enhanced features."""
    
    print("=" * 80)
    print("🚀 INTELLIGENT ERROR MITIGATION SYSTEM - ENHANCED FEATURES DEMO")
    print("=" * 80)
    print()
    
    try:
        # Import all enhanced modules
        from data_loader import DataLoader
        from baseline_model import BaselineModel
        from error_monitor import ErrorMonitor
        from error_analyzer import ErrorAnalyzer
        from mitigation_strategies import MitigationStrategies
        from xai_visualizer import XAIVisualizer
        from error_storyteller import ErrorStoryteller
        from strategy_comparator import StrategyComparator
        from mlops_logger import MLOpsLogger
        
        print("✅ All enhanced modules imported successfully!")
        print()
        
        # Step 1: Load data and train model
        print("STEP 1: LOADING DATA AND TRAINING MODEL")
        print("-" * 50)
        
        loader = DataLoader('digits')
        data = loader.load_dataset()
        
        model = BaselineModel('random_forest')
        model.train(data['X_train'], data['y_train'])
        baseline_eval = model.evaluate(data['X_test'], data['y_test'])
        
        print(f"✅ Model trained successfully!")
        print(f"   - Dataset: digits ({data['X_test'].shape[0]} test samples)")
        print(f"   - Model: random_forest")
        print(f"   - Baseline accuracy: {baseline_eval['metrics']['accuracy']:.4f}")
        print()
        
        # Step 2: Initialize MLOps Logger
        print("STEP 2: INITIALIZING MLOPS LOGGING")
        print("-" * 50)
        
        mlops_logger = MLOpsLogger()
        print("✅ MLOps logger initialized with SQLite database")
        print()
        
        # Step 3: Monitor errors
        print("STEP 3: MONITORING PREDICTION ERRORS")
        print("-" * 50)
        
        error_monitor = ErrorMonitor()
        monitoring_results = error_monitor.monitor_predictions(
            data['X_test'], data['y_test'],
            baseline_eval['predictions'], baseline_eval['probabilities'],
            "demo_baseline"
        )
        
        print(f"✅ Error monitoring completed!")
        print(f"   - Total errors: {monitoring_results['total_errors']}")
        print(f"   - Error rate: {monitoring_results['error_rate']:.4f}")
        print()
        
        # Step 4: XAI Explanations
        print("STEP 4: GENERATING XAI EXPLANATIONS")
        print("-" * 50)
        
        xai_visualizer = XAIVisualizer(
            model.model, data['X_train'], data['y_train'],
            data.get('feature_names'), data['target_names'], data['data_type']
        )
        
        # Get first error case for explanation
        if monitoring_results['error_details']:
            error_case = monitoring_results['error_details'][0]
            sample_idx = error_case['sample_index']
            sample = data['X_test'][sample_idx:sample_idx+1]
            
            # Generate SHAP explanation
            shap_explanation = xai_visualizer.explain_prediction_shap(sample, sample_idx)
            
            if not shap_explanation.get('error'):
                print("✅ SHAP explanation generated!")
                print(f"   - Predicted class: {shap_explanation['predicted_class']}")
                print(f"   - Confidence: {shap_explanation['confidence']:.3f}")
                print(f"   - Top feature: {shap_explanation['feature_importance'][0]['feature']}")
            else:
                print("⚠️ SHAP explanation not available")
            
            # Generate LIME explanation
            lime_explanation = xai_visualizer.explain_prediction_lime(sample, sample_idx)
            
            if not lime_explanation.get('error'):
                print("✅ LIME explanation generated!")
                print(f"   - Top LIME feature: {lime_explanation['feature_importance'][0]['feature']}")
            else:
                print("⚠️ LIME explanation not available")
        else:
            print("ℹ️ No errors found for XAI demonstration")
        
        print()
        
        # Step 5: Error Storytelling
        print("STEP 5: GENERATING ERROR STORIES")
        print("-" * 50)
        
        storyteller = ErrorStoryteller(
            data['target_names'], data.get('feature_names'), data['data_type']
        )
        
        if monitoring_results['error_details']:
            # Generate story for first error case
            error_case = monitoring_results['error_details'][0]
            story = storyteller.generate_error_story(
                {
                    'true_class': data['target_names'][error_case['true_label']],
                    'predicted_class': data['target_names'][error_case['predicted_label']],
                    'confidence': error_case['confidence'],
                    'error_type': 'High Confidence Error' if error_case['confidence'] > 0.8 else 'Low Confidence Error',
                    'sample_idx': error_case['sample_index']
                },
                shap_explanation if not shap_explanation.get('error') else None
            )
            
            print("✅ Error story generated!")
            print(f"   - Title: {story['story_components']['title']}")
            print(f"   - Summary: {story['story_components']['summary'][:100]}...")
        else:
            print("ℹ️ No errors found for storytelling demonstration")
        
        print()
        
        # Step 6: Strategy Comparison
        print("STEP 6: COMPARING MITIGATION STRATEGIES")
        print("-" * 50)
        
        mitigation_strategies = MitigationStrategies()
        strategy_comparator = StrategyComparator(data['target_names'])
        
        # Test a few strategies
        strategies_to_test = [
            ('ensemble_learning', {'n_estimators': 3}),
            ('confidence_thresholding', {'confidence_threshold': 0.7})
        ]
        
        print("Testing mitigation strategies...")
        for strategy_name, params in strategies_to_test:
            try:
                print(f"   Testing {strategy_name}...")
                result = mitigation_strategies.apply_mitigation(
                    strategy_name, model.model,
                    data['X_train'], data['y_train'],
                    data['X_test'], data['y_test'],
                    {}, **params
                )
                strategy_comparator.add_strategy_result(strategy_name, result)
                print(f"   ✅ {strategy_name} completed")
            except Exception as e:
                print(f"   ❌ {strategy_name} failed: {e}")
        
        # Create comparison visualization
        try:
            comparison_path = strategy_comparator.create_performance_comparison(
                baseline_eval['metrics']['accuracy']
            )
            print(f"✅ Strategy comparison chart created: {comparison_path}")
        except Exception as e:
            print(f"⚠️ Could not create comparison chart: {e}")
        
        print()
        
        # Step 7: MLOps Logging
        print("STEP 7: MLOPS LOGGING DEMONSTRATION")
        print("-" * 50)
        
        # Log some predictions
        for i in range(min(5, len(data['X_test']))):
            sample = data['X_test'][i]
            true_label = data['y_test'][i]
            predicted_label = baseline_eval['predictions'][i]
            confidence = baseline_eval['max_probabilities'][i]
            
            prediction_id = mlops_logger.log_prediction(
                model_name="demo_random_forest",
                sample_id=f"demo_sample_{i}",
                true_label=true_label,
                predicted_label=predicted_label,
                confidence=confidence,
                dataset_name="digits",
                features=sample,
                metadata={"demo": True}
            )
        
        print("✅ Logged 5 predictions to MLOps database")
        
        # Log model performance
        performance_id = mlops_logger.log_model_performance(
            model_name="demo_random_forest",
            dataset_name="digits",
            accuracy=baseline_eval['metrics']['accuracy'],
            precision_score=baseline_eval['metrics']['precision'],
            recall_score=baseline_eval['metrics']['recall'],
            f1_score=baseline_eval['metrics']['f1'],
            error_rate=monitoring_results['error_rate'],
            total_samples=len(data['X_test']),
            error_samples=monitoring_results['total_errors'],
            metadata={"demo": True}
        )
        
        print("✅ Logged model performance metrics")
        
        # Generate MLOps report
        report_path = mlops_logger.generate_mlops_report(days=1)
        print(f"✅ MLOps report generated: {report_path}")
        
        print()
        
        # Step 8: Summary
        print("STEP 8: DEMO SUMMARY")
        print("-" * 50)
        
        print("🎉 Enhanced features demonstration completed successfully!")
        print()
        print("✅ Features demonstrated:")
        print("   - XAI Visualizations (SHAP & LIME)")
        print("   - Error Storytelling & Analysis")
        print("   - Strategy Comparison Dashboard")
        print("   - MLOps Logging System")
        print("   - Comprehensive Error Monitoring")
        print()
        print("📁 Generated files:")
        print("   - MLOps database: logs/mlops.db")
        print("   - MLOps report: logs/mlops_report.txt")
        print("   - Strategy comparison: results/strategy_comparison_*.png")
        print("   - Error logs: logs/error_monitor.log")
        print()
        print("🚀 Next steps:")
        print("   1. Launch the enhanced dashboard: streamlit run dashboard/streamlit_app.py")
        print("   2. Start the FastAPI server: python api/app.py")
        print("   3. Explore the interactive features in the dashboard")
        print("   4. Test the API endpoints at http://localhost:8000/docs")
        print()
        print("📚 Documentation:")
        print("   - Read ENHANCED_FEATURES_GUIDE.md for detailed instructions")
        print("   - Check the API documentation at /docs when running the server")
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Please ensure all dependencies are installed:")
        print("pip install -r requirements.txt")
        return 1
    
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
