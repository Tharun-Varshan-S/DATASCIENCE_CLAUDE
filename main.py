# main.py
"""
Main execution script for Intelligent Error Mitigation System
Demonstrates the complete pipeline from baseline model to error mitigation.
"""

import numpy as np
import os
import sys
import warnings
warnings.filterwarnings('ignore')

# Import our modules
from src.data_loader import DataLoader
from src.baseline_model import BaselineModel
from src.error_monitor import ErrorMonitor
from src.error_analyzer import ErrorAnalyzer
from src.mitigation_strategies import MitigationStrategies
from src.self_correction import SelfCorrectionSystem
from src.evaluator import ModelEvaluator
from src.explainer import ModelExplainer

def create_directories():
    """Create necessary directories for the project."""
    directories = ['data', 'models', 'logs', 'results']
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"Created directory: {directory}")

def main():
    """Main execution function demonstrating the complete error mitigation pipeline."""
    
    print("=" * 60)
    print("INTELLIGENT ERROR MITIGATION SYSTEM FOR ML MODELS")
    print("=" * 60)
    print()
    
    # Create necessary directories
    create_directories()
    
    # Configuration
    DATASET_NAME = 'digits'  # Change to 'breast_cancer', 'wine', 'mnist', etc.
    MODEL_TYPE = 'random_forest'  # Change to 'gradient_boosting', 'svm', etc.
    
    print(f"Configuration:")
    print(f"  - Dataset: {DATASET_NAME}")
    print(f"  - Model: {MODEL_TYPE}")
    print()
    
    try:
        # Step 1: Load and prepare data
        print("STEP 1: LOADING DATA")
        print("-" * 30)
        
        loader = DataLoader(DATASET_NAME)
        data = loader.load_dataset()
        
        print(f"Dataset loaded successfully:")
        print(f"  - Training samples: {data['X_train'].shape[0]}")
        print(f"  - Test samples: {data['X_test'].shape[0]}")
        print(f"  - Features: {data['X_train'].shape[1]}")
        print(f"  - Classes: {data['n_classes']}")
        print(f"  - Target names: {data['target_names']}")
        print()
        
        # Save processed data
        data_path = f"data/{DATASET_NAME}_processed.pkl"
        loader.save_processed_data(data, data_path)
        
        # Step 2: Train baseline model
        print("STEP 2: TRAINING BASELINE MODEL")
        print("-" * 30)
        
        baseline_model = BaselineModel(MODEL_TYPE)
        training_info = baseline_model.train(data['X_train'], data['y_train'])
        
        # Save baseline model
        model_path = f"models/baseline_{MODEL_TYPE}_{DATASET_NAME}.joblib"
        baseline_model.save_model(model_path)
        print()
        
        # Step 3: Evaluate baseline model
        print("STEP 3: EVALUATING BASELINE MODEL")
        print("-" * 30)
        
        baseline_evaluation = baseline_model.evaluate(data['X_test'], data['y_test'])
        baseline_accuracy = baseline_evaluation['metrics']['accuracy']
        print(f"Baseline model accuracy: {baseline_accuracy:.4f}")
        print()
        
        # Step 4: Monitor errors
        print("STEP 4: MONITORING PREDICTION ERRORS")
        print("-" * 30)
        
        error_monitor = ErrorMonitor()
        monitoring_results = error_monitor.monitor_predictions(
            data['X_test'],
            data['y_test'],
            baseline_evaluation['predictions'],
            baseline_evaluation['probabilities'],
            f"baseline_{MODEL_TYPE}"
        )
        
        print(f"Error monitoring completed:")
        print(f"  - Total errors: {monitoring_results['total_errors']}")
        print(f"  - Error rate: {monitoring_results['error_rate']:.4f}")
        print(f"  - Low confidence errors: {monitoring_results['low_confidence_errors']}")
        print(f"  - High confidence errors: {monitoring_results['high_confidence_errors']}")
        print()
        
        # Step 5: Analyze errors
        print("STEP 5: ANALYZING ERRORS")
        print("-" * 30)
        
        error_analyzer = ErrorAnalyzer()
        error_analysis = error_analyzer.analyze_errors(
            monitoring_results['error_details'],
            data['X_test'],
            data['y_test'],
            dict(zip(*np.unique(data['y_train'], return_counts=True)))
        )
        
        print(f"Error analysis completed:")
        print(f"  - Total errors analyzed: {error_analysis['total_errors']}")
        print(f"  - Low confidence errors: {error_analysis['confidence_analysis']['low_confidence_errors']}")
        print(f"  - Class imbalance errors: {error_analysis['class_imbalance_analysis']['imbalance_errors']}")
        print(f"  - Boundary case errors: {error_analysis['boundary_analysis']['boundary_errors']}")
        print(f"  - Systematic biases: {len(error_analysis['bias_analysis']['systematic_biases'])}")
        print()
        
        # Save error analysis visualizations
        error_analyzer.visualize_error_analysis('results')
        
        # Step 6: Apply mitigation strategies individually
        print("STEP 6: APPLYING INDIVIDUAL MITIGATION STRATEGIES")
        print("-" * 30)
        
        mitigation_strategies = MitigationStrategies()
        individual_results = {}
        
        # List of strategies to try
        strategies_to_test = [
            ('confidence_thresholding', {'confidence_threshold': 0.7}),
            ('ensemble_learning', {'ensemble_type': 'bagging', 'n_estimators': 5}),
            ('data_augmentation', {'augmentation_method': 'smote'}),
            ('class_balancing', {'balancing_method': 'class_weight'}),
        ]
        
        print("Testing individual mitigation strategies:")
        for strategy_name, params in strategies_to_test:
            try:
                print(f"  Testing {strategy_name}...")
                result = mitigation_strategies.apply_mitigation(
                    strategy_name,
                    baseline_model.model,
                    data['X_train'],
                    data['y_train'],
                    data['X_test'],
                    data['y_test'],
                    error_analysis,
                    **params
                )
                individual_results[strategy_name] = result
                
                # Get accuracy improvement
                if 'improvement' in result:
                    improvement = result['improvement']
                elif 'improved_accuracy' in result:
                    improvement = result['improved_accuracy'] - baseline_accuracy
                elif 'ensemble_accuracy' in result:
                    improvement = result['ensemble_accuracy'] - baseline_accuracy
                elif 'augmented_accuracy' in result:
                    improvement = result['augmented_accuracy'] - baseline_accuracy
                elif 'balanced_accuracy' in result:
                    improvement = result['balanced_accuracy'] - baseline_accuracy
                else:
                    improvement = 0
                
                print(f"    Improvement: {improvement:+.4f}")
                
            except Exception as e:
                print(f"    Failed: {str(e)}")
                individual_results[strategy_name] = {'error': str(e)}
        
        print()
        
        # Step 7: Run self-correction system
        print("STEP 7: RUNNING SELF-CORRECTION SYSTEM")
        print("-" * 30)
        
        self_correction = SelfCorrectionSystem(
            correction_threshold=0.05,  # Trigger correction if error rate > 5%
            min_errors_for_correction=5,
            max_iterations=3,
            improvement_threshold=0.01
        )
        
        correction_results = self_correction.run_self_correction(
            baseline_model.model,
            data['X_train'],
            data['y_train'],
            data['X_test'],
            data['y_test'],
            error_monitor,
            error_analyzer,
            mitigation_strategies
        )
        
        print(f"Self-correction completed:")
        print(f"  - Initial accuracy: {correction_results['initial_accuracy']:.4f}")
        print(f"  - Final accuracy: {correction_results['final_accuracy']:.4f}")
        print(f"  - Total improvement: {correction_results['total_improvement']:+.4f}")
        print(f"  - Iterations completed: {correction_results['iterations_completed']}")
        print()
        
        # Step 8: Comprehensive evaluation
        print("STEP 8: COMPREHENSIVE MODEL EVALUATION")
        print("-" * 30)
        
        evaluator = ModelEvaluator()
        
        # Get the corrected model (use best model from self-correction)
        corrected_model = self_correction.current_model
        
        evaluation_results = evaluator.comprehensive_evaluation(
            baseline_model.model,
            corrected_model,
            data['X_test'],
            data['y_test'],
            data['X_train'],
            data['y_train'],
            data['target_names'],
            f"{MODEL_TYPE}_{DATASET_NAME}_evaluation"
        )
        
        print(f"Comprehensive evaluation completed:")
        print(f"  - Baseline accuracy: {evaluation_results['metrics_comparison']['before']['accuracy']:.4f}")
        print(f"  - Corrected accuracy: {evaluation_results['metrics_comparison']['after']['accuracy']:.4f}")
        print(f"  - Accuracy improvement: {evaluation_results['metrics_comparison']['improvements']['accuracy']:+.4f}")
        print(f"  - Errors fixed: {evaluation_results['error_analysis']['errors_fixed']}")
        print(f"  - New errors: {evaluation_results['error_analysis']['new_errors']}")
        print(f"  - Net error reduction: {evaluation_results['error_analysis']['net_error_reduction']}")
        print()
        
        # Step 9: Model explanations
        print("STEP 9: GENERATING MODEL EXPLANATIONS")
        print("-" * 30)
        
        explainer = ModelExplainer(
            corrected_model,
            data['X_train'],
            data['y_train'],
            data.get('feature_names', None)
        )
        
        # Explain a few error cases
        error_indices = monitoring_results['error_details'][:5]  # First 5 errors
        error_explanations = []
        
        for error_detail in error_indices:
            sample_idx = error_detail['sample_index']
            sample = data['X_test'][sample_idx:sample_idx+1]
            
            explanation = explainer.explain_prediction(
                sample, 0, ['feature_importance', 'local_surrogate']
            )
            error_explanations.append(explanation)
        
        print(f"Generated explanations for {len(error_explanations)} error cases")
        
        # Generate explanation report
        if error_explanations:
            report_path = 'results/explanation_report.txt'
            explainer.generate_explanation_report(error_explanations, report_path)
            print(f"Explanation report saved to {report_path}")
        
        print()
        
        # Step 10: Summary and recommendations
        print("STEP 10: SUMMARY AND RECOMMENDATIONS")
        print("-" * 30)
        
        print("FINAL RESULTS SUMMARY:")
        print(f"Dataset: {DATASET_NAME} ({data['X_test'].shape[0]} test samples)")
        print(f"Model: {MODEL_TYPE}")
        print()
        
        print("PERFORMANCE METRICS:")
        final_metrics = evaluation_results['metrics_comparison']
        for metric in ['accuracy', 'precision', 'recall', 'f1']:
            before = final_metrics['before'][metric]
            after = final_metrics['after'][metric]
            improvement = final_metrics['improvements'][metric]
            rel_improvement = final_metrics['relative_improvements'][metric]
            
            print(f"  {metric.capitalize():<12}: {before:.4f} → {after:.4f} ({improvement:+.4f}, {rel_improvement:+.1f}%)")
        
        print()
        
        print("ERROR ANALYSIS:")
        error_stats = evaluation_results['error_analysis']
        print(f"  Errors Fixed: {error_stats['errors_fixed']}")
        print(f"  New Errors: {error_stats['new_errors']}")
        print(f"  Net Reduction: {error_stats['net_error_reduction']}")
        print(f"  Error Reduction Rate: {error_stats['error_reduction_rate']:.2%}")
        
        print()
        
        print("MITIGATION STRATEGY EFFECTIVENESS:")
        best_strategies = []
        for strategy, result in individual_results.items():
            if 'error' not in result:
                if 'improvement' in result:
                    improvement = result['improvement']
                else:
                    improvement = 0
                best_strategies.append((strategy, improvement))
        
        best_strategies.sort(key=lambda x: x[1], reverse=True)
        
        for i, (strategy, improvement) in enumerate(best_strategies[:3], 1):
            print(f"  {i}. {strategy.replace('_', ' ').title()}: {improvement:+.4f}")
        
        print()
        
        print("RECOMMENDATIONS:")
        recommendations = error_analysis.get('mitigation_recommendations', [])
        for i, rec in enumerate(recommendations[:3], 1):
            print(f"  {i}. {rec['strategy']} ({rec['priority']} priority)")
            print(f"     Target: {rec['category']}")
            print(f"     Action: {rec['description']}")
        
        print()
        
        # Step 11: Save all results
        print("STEP 11: SAVING RESULTS")
        print("-" * 30)
        
        # Save final corrected model
        final_model_path = f"models/corrected_{MODEL_TYPE}_{DATASET_NAME}.joblib"
        if hasattr(corrected_model, 'save_model'):
            corrected_model.save_model(final_model_path)
        else:
            import joblib
            joblib.dump(corrected_model, final_model_path)
        print(f"Final corrected model saved to {final_model_path}")
        
        # Save error monitoring log
        error_monitor.save_error_log(f"error_log_{MODEL_TYPE}_{DATASET_NAME}.pkl")
        
        print("All results saved successfully!")
        print()
        
        # Final message
        print("=" * 60)
        print("INTELLIGENT ERROR MITIGATION SYSTEM COMPLETED!")
        print("=" * 60)
        print()
        print("Next steps:")
        print("1. Review the generated visualizations in the 'results/' directory")
        print("2. Check the detailed logs in the 'logs/' directory")
        print("3. Launch the Streamlit dashboard to explore results interactively:")
        print("   streamlit run dashboard/streamlit_app.py")
        print()
        print("The system successfully:")
        final_improvement = correction_results['total_improvement']
        if final_improvement > 0:
            print(f"✅ Improved model accuracy by {final_improvement:+.4f}")
            print(f"✅ Fixed {error_stats['errors_fixed']} prediction errors")
            print(f"✅ Applied effective mitigation strategies")
        else:
            print("ℹ️  Analyzed errors and provided recommendations")
            print("ℹ️  Demonstrated the complete error mitigation pipeline")
        
        print("✅ Generated comprehensive evaluation reports")
        print("✅ Created model explanations for error cases")
        
    except Exception as e:
        print(f"❌ Error occurred during execution: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

def run_quick_demo():
    """Run a quick demo with a small dataset for testing purposes."""
    print("RUNNING QUICK DEMO...")
    print("-" * 30)
    
    # Use a smaller, faster configuration for demo
    global DATASET_NAME, MODEL_TYPE
    DATASET_NAME = 'digits'  # Fastest dataset
    MODEL_TYPE = 'random_forest'  # Fast and reliable
    
    return main()

if __name__ == "__main__":
    # Check command line arguments
    if len(sys.argv) > 1 and sys.argv[1] == '--demo':
        exit_code = run_quick_demo()
    else:
        exit_code = main()
    
    sys.exit(exit_code)