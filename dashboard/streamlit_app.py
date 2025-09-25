# dashboard/streamlit_app.py
"""
Enhanced Streamlit Dashboard for Intelligent Error Mitigation System
Interactive dashboard with advanced features: file upload, confidence sliders, error heatmaps, and XAI visualizations.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import os
import sys
from datetime import datetime
import io
import base64
from PIL import Image
import cv2

# Add src directory to path to import our modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

try:
    from data_loader import DataLoader
    from baseline_model import BaselineModel
    from error_monitor import ErrorMonitor
    from error_analyzer import ErrorAnalyzer
    from mitigation_strategies import MitigationStrategies
    from self_correction import SelfCorrectionSystem
    from evaluator import ModelEvaluator
    from explainer import ModelExplainer
    from xai_visualizer import XAIVisualizer
    from error_storyteller import ErrorStoryteller
    from strategy_comparator import StrategyComparator
except ImportError as e:
    st.error(f"Error importing modules: {e}")
    st.stop()

# Page configuration
st.set_page_config(
    page_title="Intelligent Error Mitigation System",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
.main > div {
    padding-top: 2rem;
}
.stMetric > div > div > div > div {
    font-size: 1rem;
}
</style>
""", unsafe_allow_html=True)

def load_results_data():
    """Load results from the logs and results directories."""
    results_data = {
        'monitoring_results': [],
        'evaluation_results': [],
        'correction_results': []
    }
    
    # Resolve absolute base directory (project root)
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    logs_dir = os.path.join(base_dir, 'logs')
    results_dir = os.path.join(base_dir, 'results')
    
    # Load monitoring results
    if os.path.exists(logs_dir):
        for filename in sorted(os.listdir(logs_dir)):
            if filename.startswith('monitoring_results_') and filename.endswith('.json'):
                try:
                    with open(os.path.join(logs_dir, filename), 'r') as f:
                        data = json.load(f)
                        results_data['monitoring_results'].append(data)
                except:
                    pass
    
    # Load evaluation results
    if os.path.exists(results_dir):
        for filename in sorted(os.listdir(results_dir)):
            if filename.endswith('_results.json'):
                try:
                    with open(os.path.join(results_dir, filename), 'r') as f:
                        data = json.load(f)
                        results_data['evaluation_results'].append(data)
                except:
                    pass
    
    # Load correction results
    if os.path.exists(logs_dir):
        for filename in sorted(os.listdir(logs_dir)):
            if filename.startswith('self_correction_') and filename.endswith('.json'):
                try:
                    with open(os.path.join(logs_dir, filename), 'r') as f:
                        data = json.load(f)
                        results_data['correction_results'].append(data)
                except:
                    pass
    
    return results_data

def main():
    """Main dashboard application."""
    
    # Title and description
    st.title("🤖 Intelligent Error Mitigation System")
    st.markdown("""
    This dashboard provides comprehensive monitoring, analysis, and visualization of ML model errors 
    and their mitigation strategies. Use the sidebar to navigate between different views.
    """)
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Select Page",
        ["Overview", "Live Demo", "Custom Upload", "Error Analysis", "Mitigation Results", 
         "Model Comparison", "XAI Explanations", "Strategy Comparison", "Error Stories"]
    )
    
    # Load data
    with st.spinner("Loading results data..."):
        results_data = load_results_data()
    
    # Route to selected page
    if page == "Overview":
        show_overview(results_data)
    elif page == "Live Demo":
        show_live_demo()
    elif page == "Custom Upload":
        show_custom_upload()
    elif page == "Error Analysis":
        show_error_analysis(results_data)
    elif page == "Mitigation Results":
        show_mitigation_results(results_data)
    elif page == "Model Comparison":
        show_model_comparison(results_data)
    elif page == "XAI Explanations":
        show_xai_explanations()
    elif page == "Strategy Comparison":
        show_strategy_comparison()
    elif page == "Error Stories":
        show_error_stories()

def show_overview(results_data):
    """Show system overview and key metrics."""
    
    st.header("📊 System Overview")
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        n_monitoring = len(results_data['monitoring_results'])
        st.metric("Monitoring Sessions", n_monitoring)
    
    with col2:
        n_evaluations = len(results_data['evaluation_results'])
        st.metric("Evaluations", n_evaluations)
    
    with col3:
        n_corrections = len(results_data['correction_results'])
        st.metric("Correction Attempts", n_corrections)
    
    with col4:
        # Calculate average improvement
        if results_data['evaluation_results']:
            improvements = []
            for eval_result in results_data['evaluation_results']:
                if 'metrics_comparison' in eval_result:
                    acc_improvement = eval_result['metrics_comparison']['improvements']['accuracy']
                    improvements.append(acc_improvement)
            avg_improvement = np.mean(improvements) if improvements else 0
            st.metric("Avg. Accuracy Improvement", f"{avg_improvement:.3f}")
        else:
            st.metric("Avg. Accuracy Improvement", "N/A")
    
    # Performance trends
    if results_data['evaluation_results']:
        st.subheader("📈 Performance Trends")
        
        # Create performance timeline
        performance_data = []
        for i, eval_result in enumerate(results_data['evaluation_results']):
            if 'metrics_comparison' in eval_result:
                metrics = eval_result['metrics_comparison']
                performance_data.append({
                    'Evaluation': i + 1,
                    'Before Accuracy': metrics['before']['accuracy'],
                    'After Accuracy': metrics['after']['accuracy'],
                    'Improvement': metrics['improvements']['accuracy']
                })
        
        if performance_data:
            df = pd.DataFrame(performance_data)
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=df['Evaluation'], y=df['Before Accuracy'],
                mode='lines+markers', name='Before Mitigation',
                line=dict(color='red', width=2)
            ))
            fig.add_trace(go.Scatter(
                x=df['Evaluation'], y=df['After Accuracy'],
                mode='lines+markers', name='After Mitigation',
                line=dict(color='green', width=2)
            ))
            
            fig.update_layout(
                title="Model Performance Over Time",
                xaxis_title="Evaluation Session",
                yaxis_title="Accuracy",
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
    
    # Recent activities
    st.subheader("🕐 Recent Activities")
    
    activities = []
    
    # Add monitoring activities
    for result in results_data['monitoring_results'][-5:]:
        activities.append({
            'Time': result.get('timestamp', 'Unknown'),
            'Type': 'Error Monitoring',
            'Details': f"Found {result.get('total_errors', 0)} errors in {result.get('total_samples', 0)} samples"
        })
    
    # Add evaluation activities
    for result in results_data['evaluation_results'][-5:]:
        activities.append({
            'Time': result.get('timestamp', 'Unknown'),
            'Type': 'Model Evaluation',
            'Details': f"Evaluated {result.get('evaluation_name', 'model')}"
        })
    
    # Add correction activities
    for result in results_data['correction_results'][-5:]:
        activities.append({
            'Time': result.get('timestamp', 'Unknown') if isinstance(result, dict) else 'Unknown',
            'Type': 'Self-Correction',
            'Details': f"Completed {result.get('iterations_completed', 0)} iterations"
        })
    
    if activities:
        activities.sort(key=lambda x: x['Time'], reverse=True)
        df_activities = pd.DataFrame(activities[:10])  # Show last 10 activities
        st.dataframe(df_activities, use_container_width=True)
    else:
        st.info("No recent activities found. Run the system to see activities here.")

def show_live_demo():
    """Show live demo of the error mitigation system."""
    
    st.header("🚀 Live Demo")
    st.markdown("Run the error mitigation system interactively with different datasets and parameters.")
    
    # Dataset selection
    col1, col2 = st.columns(2)
    
    with col1:
        dataset_name = st.selectbox(
            "Select Dataset",
            ["digits", "breast_cancer", "wine", "mnist"]
        )
    
    with col2:
        model_type = st.selectbox(
            "Select Model Type",
            ["random_forest", "gradient_boosting", "svm", "logistic_regression"]
        )
    
    # Mitigation parameters
    st.subheader("Mitigation Parameters")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        confidence_threshold = st.slider("Confidence Threshold", 0.5, 0.9, 0.7, 0.05)
    
    with col2:
        max_iterations = st.slider("Max Correction Iterations", 1, 5, 3)
    
    with col3:
        improvement_threshold = st.slider("Improvement Threshold", 0.001, 0.05, 0.01, 0.001)
    
    # Run demo button
    if st.button("🎯 Run Error Mitigation Demo", type="primary"):
        with st.spinner("Running error mitigation system..."):
            try:
                # Load data
                loader = DataLoader(dataset_name)
                data = loader.load_dataset()
                
                # Train baseline model
                model = BaselineModel(model_type)
                training_info = model.train(data['X_train'], data['y_train'])
                
                # Evaluate baseline
                baseline_eval = model.evaluate(data['X_test'], data['y_test'])
                
                # Initialize components
                error_monitor = ErrorMonitor()
                error_analyzer = ErrorAnalyzer()
                mitigation_strategies = MitigationStrategies()
                
                # Monitor errors
                monitoring_results = error_monitor.monitor_predictions(
                    data['X_test'], 
                    data['y_test'], 
                    baseline_eval['predictions'], 
                    baseline_eval['probabilities'],
                    "demo_baseline"
                )
                
                # Analyze errors
                error_analysis = error_analyzer.analyze_errors(
                    monitoring_results['error_details'],
                    data['X_test'],
                    data['y_test']
                )
                
                # Apply best mitigation strategy
                best_strategy = 'ensemble_learning'  # Default strategy
                error_categories = error_analysis.get('error_categories', {})
                if error_categories.get('class_imbalance', {}).get('count', 0) > 0:
                    best_strategy = 'data_augmentation'
                elif error_categories.get('low_confidence', {}).get('count', 0) > 0:
                    best_strategy = 'confidence_thresholding'
                
                mitigation_result = mitigation_strategies.apply_mitigation(
                    best_strategy,
                    model.model,
                    data['X_train'],
                    data['y_train'],
                    data['X_test'],
                    data['y_test'],
                    error_analysis,
                    confidence_threshold=confidence_threshold
                )
                
                # Display results
                st.success("✅ Demo completed successfully!")
                
                # Results visualization
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("📊 Performance Metrics")
                    
                    # Before/after comparison
                    baseline_acc = baseline_eval['metrics']['accuracy']
                    
                    if 'improved_accuracy' in mitigation_result:
                        improved_acc = mitigation_result['improved_accuracy']
                    elif 'ensemble_accuracy' in mitigation_result:
                        improved_acc = mitigation_result['ensemble_accuracy']
                    elif 'augmented_accuracy' in mitigation_result:
                        improved_acc = mitigation_result['augmented_accuracy']
                    else:
                        improved_acc = baseline_acc
                    
                    improvement = improved_acc - baseline_acc
                    
                    metrics_df = pd.DataFrame({
                        'Metric': ['Accuracy'],
                        'Before': [baseline_acc],
                        'After': [improved_acc],
                        'Improvement': [improvement]
                    })
                    
                    st.dataframe(metrics_df, use_container_width=True)
                    
                    # Improvement visualization
                    fig = go.Figure(data=[
                        go.Bar(name='Before', x=['Accuracy'], y=[baseline_acc], marker_color='red'),
                        go.Bar(name='After', x=['Accuracy'], y=[improved_acc], marker_color='green')
                    ])
                    fig.update_layout(barmode='group', title="Performance Comparison")
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    st.subheader("🎯 Error Analysis")
                    # Error categories
                    error_categories = error_analysis.get('error_categories', {})
                    cat_data = []
                    for category, info in error_categories.items():
                        if info.get('count', 0) > 0:
                            cat_data.append({
                                'Category': category.replace('_', ' ').title(),
                                'Count': info.get('count', 0),
                                'Percentage': info.get('percentage', 0)
                            })
                    if cat_data:
                        cat_df = pd.DataFrame(cat_data)
                        fig = px.pie(cat_df, values='Count', names='Category', 
                                   title="Error Distribution")
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Strategy applied
                    st.info(f"Applied strategy: **{best_strategy.replace('_', ' ').title()}**")
                    st.info(f"Strategy improvement: **{improvement:+.4f}**")
                
                # Detailed results
                with st.expander("📋 Detailed Results"):
                    st.json({
                        'dataset': dataset_name,
                        'model': model_type,
                        'baseline_accuracy': float(baseline_acc),
                        'improved_accuracy': float(improved_acc),
                        'improvement': float(improvement),
                        'strategy_used': best_strategy,
                        'total_errors': monitoring_results['total_errors'],
                        'error_rate': monitoring_results['error_rate']
                    })
                
            except Exception as e:
                st.error(f"Demo failed: {str(e)}")
                st.exception(e)

def show_error_analysis(results_data):
    """Show detailed error analysis."""
    
    st.header("🔍 Error Analysis")
    
    if not results_data['monitoring_results']:
        st.info("No monitoring results available. Run the system to generate error data.")
        return
    
    # Select monitoring session
    session_options = [
        f"Session {i+1} - {result.get('model_name', 'Unknown')} ({result.get('timestamp','')[:19]})"
        for i, result in enumerate(results_data['monitoring_results'])
    ]
    
    selected_session = st.selectbox("Select Monitoring Session", session_options)
    session_idx = int(selected_session.split()[1]) - 1
    
    if session_idx < len(results_data['monitoring_results']):
        monitoring_result = results_data['monitoring_results'][session_idx]
        
        # Overview metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Samples", monitoring_result.get('total_samples', 0))
        
        with col2:
            st.metric("Total Errors", monitoring_result.get('total_errors', 0))
        
        with col3:
            error_rate = monitoring_result.get('error_rate', 0)
            st.metric("Error Rate", f"{error_rate:.2%}")
        
        with col4:
            avg_confidence = monitoring_result.get('average_confidence', 0)
            st.metric("Avg. Confidence", f"{avg_confidence:.3f}")
        
        # Error distribution by confidence
        st.subheader("Error Distribution by Confidence")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Confidence distribution
            confidence_data = {
                'Low Confidence': monitoring_result.get('low_confidence_errors', 0),
                'High Confidence': monitoring_result.get('high_confidence_errors', 0),
                'Medium Confidence': (monitoring_result.get('total_errors', 0) - 
                                    monitoring_result.get('low_confidence_errors', 0) - 
                                    monitoring_result.get('high_confidence_errors', 0))
            }
            
            fig = px.pie(values=list(confidence_data.values()), 
                        names=list(confidence_data.keys()),
                        title="Errors by Confidence Level")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Class-wise error rates
            class_error_rates = monitoring_result.get('class_error_rates', {})
            if class_error_rates:
                classes = list(class_error_rates.keys())
                error_rates = [class_error_rates[cls]['error_rate'] for cls in classes]
                
                fig = px.bar(x=classes, y=error_rates, 
                           title="Error Rate by Class",
                           labels={'x': 'Class', 'y': 'Error Rate'})
                st.plotly_chart(fig, use_container_width=True)
        
        # Confusion matrix
        if 'confusion_matrix' in monitoring_result:
            st.subheader("Confusion Matrix")
            cm = np.array(monitoring_result['confusion_matrix'])
            
            fig = px.imshow(cm, text_auto=True, aspect="auto",
                          title="Confusion Matrix",
                          labels=dict(x="Predicted", y="Actual"))
            st.plotly_chart(fig, use_container_width=True)
        
        # Error details
        if 'error_details' in monitoring_result:
            st.subheader("Error Details")
            
            error_details = monitoring_result['error_details'][:20]  # Show first 20 errors
            
            if error_details:
                error_df = pd.DataFrame([
                    {
                        'Sample Index': error['sample_index'],
                        'True Label': error['true_label'],
                        'Predicted Label': error['predicted_label'],
                        'Confidence': f"{error['confidence']:.3f}",
                        'Error Type': 'High Conf.' if error['confidence'] > 0.8 else 
                                    'Low Conf.' if error['confidence'] < 0.6 else 'Medium Conf.'
                    }
                    for error in error_details
                ])
                
                st.dataframe(error_df, use_container_width=True)
            
            # Download error details
            if st.button("📥 Download Error Details"):
                error_json = json.dumps(error_details, indent=2)
                st.download_button(
                    label="Download JSON",
                    data=error_json,
                    file_name=f"error_details_{session_idx+1}.json",
                    mime="application/json"
                )

def show_mitigation_results(results_data):
    """Show mitigation strategy results."""
    
    st.header("⚡ Mitigation Results")
    
    if not results_data['correction_results']:
        st.info("No correction results available. Run the self-correction system to see mitigation results.")
        return
    
    # Select correction session
    correction_options = [
        f"Correction {i+1} ({res.get('timestamp','')[:19]})"
        for i, res in enumerate(results_data['correction_results'])
    ]
    selected_correction = st.selectbox("Select Correction Session", correction_options)
    correction_idx = int(selected_correction.split()[1]) - 1
    
    if correction_idx < len(results_data['correction_results']):
        correction_result = results_data['correction_results'][correction_idx]
        
        # Overview metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            initial_acc = correction_result.get('initial_accuracy', 0)
            st.metric("Initial Accuracy", f"{initial_acc:.4f}")
        
        with col2:
            final_acc = correction_result.get('final_accuracy', 0)
            st.metric("Final Accuracy", f"{final_acc:.4f}")
        
        with col3:
            total_improvement = correction_result.get('total_improvement', 0)
            st.metric("Total Improvement", f"{total_improvement:+.4f}")
        
        with col4:
            iterations = correction_result.get('iterations_completed', 0)
            st.metric("Iterations", iterations)
        
        # Correction progress
        if 'iterations' in correction_result:
            st.subheader("📈 Correction Progress")
            
            iterations_data = correction_result['iterations']
            
            # Create progress chart
            iteration_numbers = [iter_data['iteration'] for iter_data in iterations_data]
            accuracies_before = [iter_data['accuracy_before'] for iter_data in iterations_data]
            accuracies_after = [iter_data['accuracy_after'] for iter_data in iterations_data]
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=iteration_numbers, y=accuracies_before,
                mode='lines+markers', name='Before Mitigation',
                line=dict(color='red', width=2)
            ))
            fig.add_trace(go.Scatter(
                x=iteration_numbers, y=accuracies_after,
                mode='lines+markers', name='After Mitigation',
                line=dict(color='green', width=2)
            ))
            
            fig.update_layout(
                title="Accuracy Improvement Across Iterations",
                xaxis_title="Iteration",
                yaxis_title="Accuracy",
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Strategy effectiveness
            st.subheader("🎯 Strategy Effectiveness")
            
            strategies_used = [iter_data['best_strategy'] for iter_data in iterations_data]
            improvements = [iter_data['improvement'] for iter_data in iterations_data]
            
            strategy_df = pd.DataFrame({
                'Iteration': iteration_numbers,
                'Strategy': strategies_used,
                'Improvement': improvements
            })
            
            fig = px.bar(strategy_df, x='Iteration', y='Improvement', color='Strategy',
                        title="Improvement by Strategy per Iteration")
            st.plotly_chart(fig, use_container_width=True)
            
            # Detailed iteration results
            with st.expander("📋 Detailed Iteration Results"):
                st.dataframe(strategy_df, use_container_width=True)

def show_model_comparison(results_data):
    """Show model comparison across evaluations."""
    
    st.header("⚖️ Model Comparison")
    
    if not results_data['evaluation_results']:
        st.info("No evaluation results available for comparison.")
        return
    
    # Prepare comparison data
    comparison_data = []
    for i, eval_result in enumerate(results_data['evaluation_results']):
        if 'metrics_comparison' in eval_result:
            metrics = eval_result['metrics_comparison']
            comparison_data.append({
                'Evaluation': eval_result.get('evaluation_name', f'Evaluation {i+1}'),
                'Before Accuracy': metrics['before']['accuracy'],
                'After Accuracy': metrics['after']['accuracy'],
                'Before Precision': metrics['before']['precision'],
                'After Precision': metrics['after']['precision'],
                'Before Recall': metrics['before']['recall'],
                'After Recall': metrics['after']['recall'],
                'Before F1': metrics['before']['f1'],
                'After F1': metrics['after']['f1'],
                'Accuracy Improvement': metrics['improvements']['accuracy'],
                'Precision Improvement': metrics['improvements']['precision'],
                'Recall Improvement': metrics['improvements']['recall'],
                'F1 Improvement': metrics['improvements']['f1']
            })
    
    if not comparison_data:
        st.info("No comparable evaluation data found.")
        return
    
    comparison_df = pd.DataFrame(comparison_data)
    
    # Metrics selection
    metric_to_compare = st.selectbox(
        "Select Metric to Compare",
        ['Accuracy', 'Precision', 'Recall', 'F1']
    )
    
    # Comparison visualization
    col1, col2 = st.columns(2)
    
    with col1:
        # Before vs After comparison
        fig = go.Figure()
        fig.add_trace(go.Bar(
            name='Before',
            x=comparison_df['Evaluation'],
            y=comparison_df[f'Before {metric_to_compare}'],
            marker_color='red',
            opacity=0.7
        ))
        fig.add_trace(go.Bar(
            name='After',
            x=comparison_df['Evaluation'],
            y=comparison_df[f'After {metric_to_compare}'],
            marker_color='green',
            opacity=0.7
        ))
        
        fig.update_layout(
            title=f"{metric_to_compare} Comparison: Before vs After",
            xaxis_title="Evaluation",
            yaxis_title=metric_to_compare,
            barmode='group'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Improvement visualization
        fig = px.bar(
            comparison_df,
            x='Evaluation',
            y=f'{metric_to_compare} Improvement',
            title=f"{metric_to_compare} Improvement by Evaluation",
            color=f'{metric_to_compare} Improvement',
            color_continuous_scale=['red', 'yellow', 'green']
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Summary statistics
    st.subheader("📊 Summary Statistics")
    
    summary_stats = {
        'Metric': ['Accuracy', 'Precision', 'Recall', 'F1'],
        'Avg Before': [
            comparison_df['Before Accuracy'].mean(),
            comparison_df['Before Precision'].mean(),
            comparison_df['Before Recall'].mean(),
            comparison_df['Before F1'].mean()
        ],
        'Avg After': [
            comparison_df['After Accuracy'].mean(),
            comparison_df['After Precision'].mean(),
            comparison_df['After Recall'].mean(),
            comparison_df['After F1'].mean()
        ],
        'Avg Improvement': [
            comparison_df['Accuracy Improvement'].mean(),
            comparison_df['Precision Improvement'].mean(),
            comparison_df['Recall Improvement'].mean(),
            comparison_df['F1 Improvement'].mean()
        ]
    }
    
    summary_df = pd.DataFrame(summary_stats)
    st.dataframe(summary_df, use_container_width=True)
    
    # Detailed comparison table
    with st.expander("📋 Detailed Comparison Data"):
        st.dataframe(comparison_df, use_container_width=True)

def show_explanations():
    """Show model explanation interface."""
    
    st.header("💡 Model Explanations")
    st.markdown("Generate explanations for model predictions and understand why errors occur.")
    
    # Explanation demo
    st.subheader("🎯 Prediction Explanation Demo")
    
    # Generate sample data for demonstration
    if st.button("Generate Sample Explanation"):
        # Create sample explanation data
        sample_explanation = {
            'prediction': 2,
            'confidence': 0.85,
            'top_features': [
                {'name': 'Feature_1', 'value': 0.45, 'importance': 0.3},
                {'name': 'Feature_2', 'value': -0.12, 'importance': 0.25},
                {'name': 'Feature_3', 'value': 1.2, 'importance': 0.2},
                {'name': 'Feature_4', 'value': 0.8, 'importance': 0.15},
                {'name': 'Feature_5', 'value': -0.5, 'importance': 0.1}
            ]
        }
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Prediction Details")
            st.metric("Predicted Class", sample_explanation['prediction'])
            st.metric("Confidence", f"{sample_explanation['confidence']:.3f}")
            
            # Feature importance chart
            features = sample_explanation['top_features']
            feature_names = [f['name'] for f in features]
            importances = [f['importance'] for f in features]
            
            fig = px.bar(
                x=importances, y=feature_names,
                orientation='h',
                title="Feature Importance",
                labels={'x': 'Importance', 'y': 'Features'}
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("Feature Values")
            
            # Feature values table
            feature_df = pd.DataFrame([
                {
                    'Feature': f['name'],
                    'Value': f['value'],
                    'Importance': f['importance']
                }
                for f in features
            ])
            
            st.dataframe(feature_df, use_container_width=True)
            
            # Explanation text
            st.subheader("Explanation")
            st.write(f"""
            The model predicted **Class {sample_explanation['prediction']}** with 
            **{sample_explanation['confidence']:.1%}** confidence.
            
            Key factors in this prediction:
            - **{features[0]['name']}** (value: {features[0]['value']:.3f}) - Most important feature
            - **{features[1]['name']}** (value: {features[1]['value']:.3f}) - Second most important
            - **{features[2]['name']}** (value: {features[2]['value']:.3f}) - Third most important
            
            The combination of these feature values led to the model's prediction.
            """)
    
    # Error explanation section
    st.subheader("❌ Error Explanation")
    
    st.markdown("""
    Error explanations help understand why the model made incorrect predictions.
    Common error types include:
    
    - **Low Confidence Errors**: Model was uncertain about the prediction
    - **High Confidence Errors**: Model was confident but wrong (systematic bias)
    - **Boundary Cases**: Sample near decision boundary between classes
    - **Outliers**: Sample significantly different from training data
    """)
    
    # Interactive error type explanation
    error_type = st.selectbox(
        "Select Error Type for Explanation",
        ["Low Confidence Error", "High Confidence Error", "Boundary Case", "Outlier"]
    )
    
    explanations = {
        "Low Confidence Error": {
            "description": "The model had low confidence in its prediction, indicating uncertainty.",
            "causes": ["Insufficient training data", "Ambiguous features", "Similar classes"],
            "mitigation": "Confidence thresholding, ensemble methods, more training data"
        },
        "High Confidence Error": {
            "description": "The model was confident but made an incorrect prediction.",
            "causes": ["Systematic bias", "Overfitting", "Feature distribution shift"],
            "mitigation": "Model retraining, feature engineering, bias correction"
        },
        "Boundary Case": {
            "description": "The sample was near the decision boundary between classes.",
            "causes": ["Overlapping class distributions", "Insufficient discriminative features"],
            "mitigation": "Ensemble methods, feature selection, more complex models"
        },
        "Outlier": {
            "description": "The sample was significantly different from the training data.",
            "causes": ["Data drift", "Novel patterns", "Data quality issues"],
            "mitigation": "Outlier detection, domain adaptation, data quality checks"
        }
    }
    
    if error_type in explanations:
        exp = explanations[error_type]
        
        st.write(f"**Description**: {exp['description']}")
        st.write(f"**Common Causes**: {', '.join(exp['causes'])}")
        st.write(f"**Recommended Mitigation**: {exp['mitigation']}")

def show_custom_upload():
    """Show custom file upload interface for testing models."""
    
    st.header("📁 Custom File Upload & Testing")
    st.markdown("Upload your own images or data files to test the error mitigation system.")
    
    # File upload section
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📤 Upload File")
        uploaded_file = st.file_uploader(
            "Choose a file to test",
            type=['png', 'jpg', 'jpeg', 'csv', 'npy'],
            help="Supported formats: PNG, JPG, JPEG (images), CSV, NPY (data)"
        )
    
    with col2:
        st.subheader("⚙️ Model Settings")
        model_type = st.selectbox(
            "Select Model Type",
            ["random_forest", "gradient_boosting", "svm", "logistic_regression"]
        )
        
        confidence_threshold = st.slider(
            "Confidence Threshold",
            0.1, 0.9, 0.7, 0.05,
            help="Minimum confidence for accepting predictions"
        )
    
    if uploaded_file is not None:
        st.subheader("🔍 File Analysis")
        
        # Process uploaded file
        if uploaded_file.type.startswith('image/'):
            process_uploaded_image(uploaded_file, model_type, confidence_threshold)
        elif uploaded_file.type == 'text/csv':
            process_uploaded_csv(uploaded_file, model_type, confidence_threshold)
        else:
            st.error("Unsupported file type. Please upload an image or CSV file.")

def process_uploaded_image(uploaded_file, model_type, confidence_threshold):
    """Process uploaded image file."""
    
    # Load and display image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    # Convert to numpy array
    img_array = np.array(image)
    
    # Resize if needed (for digits dataset: 8x8)
    if img_array.shape != (8, 8):
        img_resized = cv2.resize(img_array, (8, 8))
        if len(img_resized.shape) == 3:
            img_resized = cv2.cvtColor(img_resized, cv2.COLOR_RGB2GRAY)
    else:
        img_resized = img_array
    
    # Normalize
    img_normalized = img_resized.astype(np.float32) / 255.0
    img_flattened = img_normalized.flatten()
    
    # Train model and make prediction
    with st.spinner("Training model and making prediction..."):
        try:
            # Load digits dataset for training
            loader = DataLoader('digits')
            data = loader.load_dataset()
            
            # Train model
            model = BaselineModel(model_type)
            model.train(data['X_train'], data['y_train'])
            
            # Make prediction
            prediction = model.predict(img_flattened.reshape(1, -1))[0]
            probabilities = model.predict_proba(img_flattened.reshape(1, -1))[0]
            confidence = probabilities[prediction]
            
            # Display results
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Predicted Class", prediction)
            
            with col2:
                st.metric("Confidence", f"{confidence:.3f}")
            
            with col3:
                if confidence >= confidence_threshold:
                    st.success("✅ High Confidence")
                else:
                    st.warning("⚠️ Low Confidence")
            
            # Confidence thresholding
            if confidence < confidence_threshold:
                st.warning(f"Prediction rejected due to low confidence ({confidence:.3f} < {confidence_threshold})")
                st.info("Consider adjusting the confidence threshold or using ensemble methods.")
            
            # Show probability distribution
            st.subheader("📊 Prediction Probabilities")
            prob_df = pd.DataFrame({
                'Class': [str(i) for i in range(10)],
                'Probability': probabilities
            })
            
            fig = px.bar(prob_df, x='Class', y='Probability', 
                        title="Class Probability Distribution",
                        color='Probability', color_continuous_scale='viridis')
            st.plotly_chart(fig, use_container_width=True)
            
            # XAI explanation if available
            if st.checkbox("Show XAI Explanation"):
                try:
                    xai_visualizer = XAIVisualizer(
                        model.model, data['X_train'], data['y_train'],
                        data.get('feature_names'), data['target_names'], 'image'
                    )
                    
                    explanation = xai_visualizer.explain_prediction_shap(img_flattened.reshape(1, -1))
                    
                    if not explanation.get('error'):
                        st.subheader("🧠 SHAP Explanation")
                        
                        # Show top features
                        top_features = explanation['feature_importance'][:5]
                        feature_df = pd.DataFrame([
                            {
                                'Feature': f['feature'],
                                'SHAP Value': f['shap_value'],
                                'Feature Value': f['feature_value']
                            }
                            for f in top_features
                        ])
                        
                        st.dataframe(feature_df, use_container_width=True)
                        
                        # Create feature importance plot
                        fig = px.bar(feature_df, x='SHAP Value', y='Feature',
                                   orientation='h', title="Top Contributing Features (SHAP)")
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.error("XAI explanation not available")
                        
                except Exception as e:
                    st.error(f"XAI explanation failed: {e}")
        
        except Exception as e:
            st.error(f"Error processing image: {e}")

def process_uploaded_csv(uploaded_file, model_type, confidence_threshold):
    """Process uploaded CSV file."""
    
    # Read CSV
    df = pd.read_csv(uploaded_file)
    st.subheader("📋 CSV Data Preview")
    st.dataframe(df.head(), use_container_width=True)
    
    st.info(f"CSV contains {len(df)} rows and {len(df.columns)} columns")
    
    # For now, show a placeholder for CSV processing
    st.warning("CSV processing feature is under development. Currently supports image uploads.")

def show_xai_explanations():
    """Show XAI explanations interface."""
    
    st.header("🧠 Explainable AI (XAI) Explanations")
    st.markdown("Generate and visualize explanations for model predictions using SHAP and LIME.")
    
    # Configuration
    col1, col2 = st.columns(2)
    
    with col1:
        dataset_name = st.selectbox(
            "Select Dataset",
            ["digits", "breast_cancer", "wine"],
            key="xai_dataset"
        )
    
    with col2:
        model_type = st.selectbox(
            "Select Model Type",
            ["random_forest", "gradient_boosting", "svm"],
            key="xai_model"
        )
    
    # Load data and train model
    if st.button("🚀 Generate XAI Explanations"):
        with st.spinner("Loading data and training model..."):
            try:
                # Load data
                loader = DataLoader(dataset_name)
                data = loader.load_dataset()
                
                # Train model
                model = BaselineModel(model_type)
                model.train(data['X_train'], data['y_train'])
                
                # Initialize XAI visualizer
                xai_visualizer = XAIVisualizer(
                    model.model, data['X_train'], data['y_train'],
                    data.get('feature_names'), data['target_names'], data['data_type']
                )
                
                # Get some test samples
                n_samples = min(5, len(data['X_test']))
                test_indices = np.random.choice(len(data['X_test']), n_samples, replace=False)
                
                st.subheader("🔍 Sample Explanations")
                
                for i, idx in enumerate(test_indices):
                    with st.expander(f"Sample {i+1} (Index {idx})"):
                        sample = data['X_test'][idx:idx+1]
                        true_label = data['y_test'][idx]
                        
                        # Get predictions
                        prediction = model.predict(sample)[0]
                        probabilities = model.predict_proba(sample)[0]
                        confidence = probabilities[prediction]
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("True Class", data['target_names'][true_label])
                        with col2:
                            st.metric("Predicted Class", data['target_names'][prediction])
                        with col3:
                            st.metric("Confidence", f"{confidence:.3f}")
                        
                        # SHAP explanation
                        if st.checkbox(f"Show SHAP Explanation", key=f"shap_{i}"):
                            shap_explanation = xai_visualizer.explain_prediction_shap(sample, idx)
                            
                            if not shap_explanation.get('error'):
                                # Show top features
                                top_features = shap_explanation['feature_importance'][:10]
                                feature_df = pd.DataFrame([
                                    {
                                        'Feature': f['feature'],
                                        'SHAP Value': f['shap_value'],
                                        'Feature Value': f['feature_value']
                                    }
                                    for f in top_features
                                ])
                                
                                st.dataframe(feature_df, use_container_width=True)
                                
                                # Create visualization
                                fig = px.bar(feature_df, x='SHAP Value', y='Feature',
                                           orientation='h', title="SHAP Feature Importance")
                                st.plotly_chart(fig, use_container_width=True)
                            else:
                                st.error("SHAP explanation not available")
                        
                        # LIME explanation
                        if st.checkbox(f"Show LIME Explanation", key=f"lime_{i}"):
                            lime_explanation = xai_visualizer.explain_prediction_lime(sample, idx)
                            
                            if not lime_explanation.get('error'):
                                # Show top features
                                top_features = lime_explanation['feature_importance'][:10]
                                feature_df = pd.DataFrame([
                                    {
                                        'Feature': f['feature'],
                                        'LIME Weight': f['lime_weight'],
                                        'Feature Value': f['feature_value']
                                    }
                                    for f in top_features
                                ])
                                
                                st.dataframe(feature_df, use_container_width=True)
                                
                                # Create visualization
                                fig = px.bar(feature_df, x='LIME Weight', y='Feature',
                                           orientation='h', title="LIME Feature Importance")
                                st.plotly_chart(fig, use_container_width=True)
                            else:
                                st.error("LIME explanation not available")
            
            except Exception as e:
                st.error(f"Error generating XAI explanations: {e}")

def show_strategy_comparison():
    """Show strategy comparison interface."""
    
    st.header("⚖️ Strategy Comparison Dashboard")
    st.markdown("Compare the effectiveness of different mitigation strategies.")
    
    # Configuration
    col1, col2 = st.columns(2)
    
    with col1:
        dataset_name = st.selectbox(
            "Select Dataset",
            ["digits", "breast_cancer", "wine"],
            key="comp_dataset"
        )
    
    with col2:
        baseline_model_type = st.selectbox(
            "Select Baseline Model",
            ["random_forest", "gradient_boosting", "svm"],
            key="comp_model"
        )
    
    # Strategy selection
    st.subheader("🎯 Select Strategies to Compare")
    
    strategies_to_test = []
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.checkbox("Confidence Thresholding"):
            strategies_to_test.append(('confidence_thresholding', {'confidence_threshold': 0.7}))
        
        if st.checkbox("Ensemble Learning"):
            strategies_to_test.append(('ensemble_learning', {'n_estimators': 5}))
    
    with col2:
        if st.checkbox("Data Augmentation"):
            strategies_to_test.append(('data_augmentation', {'augmentation_method': 'smote'}))
        
        if st.checkbox("Class Balancing"):
            strategies_to_test.append(('class_balancing', {'balancing_method': 'class_weight'}))
    
    with col3:
        if st.checkbox("Feature Selection"):
            strategies_to_test.append(('feature_selection', {'selection_method': 'importance'}))
        
        if st.checkbox("Outlier Removal"):
            strategies_to_test.append(('outlier_removal', {'contamination': 0.1}))
    
    if st.button("🚀 Run Strategy Comparison"):
        if not strategies_to_test:
            st.warning("Please select at least one strategy to compare.")
        else:
            with st.spinner("Running strategy comparison..."):
                try:
                    # Load data
                    loader = DataLoader(dataset_name)
                    data = loader.load_dataset()
                    
                    # Train baseline model
                    baseline_model = BaselineModel(baseline_model_type)
                    baseline_model.train(data['X_train'], data['y_train'])
                    baseline_accuracy = baseline_model.evaluate(data['X_test'], data['y_test'])['metrics']['accuracy']
                    
                    # Initialize components
                    error_monitor = ErrorMonitor()
                    error_analyzer = ErrorAnalyzer()
                    mitigation_strategies = MitigationStrategies()
                    strategy_comparator = StrategyComparator(data['target_names'])
                    
                    # Monitor errors
                    baseline_eval = baseline_model.evaluate(data['X_test'], data['y_test'])
                    monitoring_results = error_monitor.monitor_predictions(
                        data['X_test'], data['y_test'],
                        baseline_eval['predictions'], baseline_eval['probabilities'],
                        "baseline"
                    )
                    
                    # Analyze errors
                    error_analysis = error_analyzer.analyze_errors(
                        monitoring_results['error_details'],
                        data['X_test'], data['y_test']
                    )
                    
                    # Test each strategy
                    for strategy_name, params in strategies_to_test:
                        st.write(f"Testing {strategy_name.replace('_', ' ').title()}...")
                        
                        try:
                            result = mitigation_strategies.apply_mitigation(
                                strategy_name, baseline_model.model,
                                data['X_train'], data['y_train'],
                                data['X_test'], data['y_test'],
                                error_analysis, **params
                            )
                            strategy_comparator.add_strategy_result(strategy_name, result)
                        except Exception as e:
                            st.error(f"Strategy {strategy_name} failed: {e}")
                    
                    # Display comparison results
                    st.subheader("📊 Comparison Results")
                    
                    # Create comparison chart
                    comparison_path = strategy_comparator.create_performance_comparison(baseline_accuracy)
                    
                    if comparison_path and os.path.exists(comparison_path):
                        st.image(comparison_path, caption="Strategy Performance Comparison")
                    
                    # Create interactive comparison
                    interactive_path = strategy_comparator.create_interactive_comparison(baseline_accuracy)
                    
                    if interactive_path and os.path.exists(interactive_path):
                        st.subheader("🎮 Interactive Comparison")
                        with open(interactive_path, 'r') as f:
                            html_content = f.read()
                        st.components.v1.html(html_content, height=800)
                    
                    # Generate report
                    report_path = strategy_comparator.generate_comparison_report(baseline_accuracy)
                    
                    if report_path and os.path.exists(report_path):
                        st.subheader("📋 Detailed Report")
                        with open(report_path, 'r') as f:
                            report_content = f.read()
                        st.text_area("Strategy Comparison Report", report_content, height=400)
                
                except Exception as e:
                    st.error(f"Error running strategy comparison: {e}")

def show_error_stories():
    """Show error storytelling interface."""
    
    st.header("📖 Error Stories & Analysis")
    st.markdown("Generate human-readable stories explaining why the model made errors.")
    
    # Configuration
    col1, col2 = st.columns(2)
    
    with col1:
        dataset_name = st.selectbox(
            "Select Dataset",
            ["digits", "breast_cancer", "wine"],
            key="story_dataset"
        )
    
    with col2:
        model_type = st.selectbox(
            "Select Model Type",
            ["random_forest", "gradient_boosting", "svm"],
            key="story_model"
        )
    
    n_error_cases = st.slider("Number of Error Cases to Analyze", 1, 10, 5)
    
    if st.button("📚 Generate Error Stories"):
        with st.spinner("Generating error stories..."):
            try:
                # Load data
                loader = DataLoader(dataset_name)
                data = loader.load_dataset()
                
                # Train model
                model = BaselineModel(model_type)
                model.train(data['X_train'], data['y_train'])
                
                # Initialize components
                error_monitor = ErrorMonitor()
                xai_visualizer = XAIVisualizer(
                    model.model, data['X_train'], data['y_train'],
                    data.get('feature_names'), data['target_names'], data['data_type']
                )
                storyteller = ErrorStoryteller(
                    data['target_names'], data.get('feature_names'), data['data_type']
                )
                
                # Monitor errors
                baseline_eval = model.evaluate(data['X_test'], data['y_test'])
                monitoring_results = error_monitor.monitor_predictions(
                    data['X_test'], data['y_test'],
                    baseline_eval['predictions'], baseline_eval['probabilities'],
                    "baseline"
                )
                
                # Get error cases
                error_cases = monitoring_results['error_details'][:n_error_cases]
                
                if not error_cases:
                    st.warning("No errors found in the test set. Try a different model or dataset.")
                else:
                    st.subheader(f"📖 Error Stories ({len(error_cases)} cases)")
                    
                    # Generate stories for each error case
                    for i, error_case in enumerate(error_cases):
                        with st.expander(f"Error Story #{i+1}: {error_case['true_label']} → {error_case['predicted_label']}"):
                            
                            # Get XAI explanation
                            sample_idx = error_case['sample_index']
                            sample = data['X_test'][sample_idx:sample_idx+1]
                            
                            shap_explanation = xai_visualizer.explain_prediction_shap(sample, sample_idx)
                            
                            # Generate story
                            story = storyteller.generate_error_story(
                                {
                                    'true_class': data['target_names'][error_case['true_label']],
                                    'predicted_class': data['target_names'][error_case['predicted_label']],
                                    'confidence': error_case['confidence'],
                                    'error_type': 'High Confidence Error' if error_case['confidence'] > 0.8 else 'Low Confidence Error',
                                    'sample_idx': sample_idx
                                },
                                shap_explanation if not shap_explanation.get('error') else None
                            )
                            
                            # Display story components
                            st.write("**Title:**", story['story_components']['title'])
                            st.write("**Summary:**", story['story_components']['summary'])
                            st.write("**Detailed Explanation:**", story['story_components']['detailed_explanation'])
                            
                            if st.checkbox(f"Show Full Story", key=f"full_story_{i}"):
                                st.text_area("Complete Story", story['full_story'], height=300)
                    
                    # Generate and save story collection
                    if st.button("💾 Save All Stories"):
                        story_path = storyteller.create_error_story_collection(
                            [{
                                'true_class': data['target_names'][ec['true_label']],
                                'predicted_class': data['target_names'][ec['predicted_label']],
                                'confidence': ec['confidence'],
                                'error_type': 'High Confidence Error' if ec['confidence'] > 0.8 else 'Low Confidence Error',
                                'sample_idx': ec['sample_index']
                            } for ec in error_cases]
                        )
                        st.success(f"Stories saved to: {story_path}")
            
            except Exception as e:
                st.error(f"Error generating error stories: {e}")

if __name__ == "__main__":
    main()