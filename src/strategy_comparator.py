# src/strategy_comparator.py
"""
Strategy Comparison Module
Creates comprehensive comparisons of mitigation strategies with visualizations.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import Dict, List, Any, Tuple, Optional
import os
from datetime import datetime
import json

class StrategyComparator:
    """Comprehensive comparison of mitigation strategies."""
    
    def __init__(self, target_names: List[str]):
        """
        Initialize Strategy Comparator.
        
        Args:
            target_names: Names of target classes
        """
        self.target_names = target_names
        self.comparison_results = {}
        
    def add_strategy_result(self, strategy_name: str, result: Dict[str, Any]):
        """
        Add a strategy result to the comparison.
        
        Args:
            strategy_name: Name of the mitigation strategy
            result: Results from the strategy application
        """
        self.comparison_results[strategy_name] = result
        print(f"Added {strategy_name} to comparison")
    
    def create_performance_comparison(self, baseline_accuracy: float, 
                                   save_path: Optional[str] = None) -> str:
        """
        Create comprehensive performance comparison chart.
        
        Args:
            baseline_accuracy: Baseline model accuracy
            save_path: Path to save the plot
            
        Returns:
            Path to saved plot
        """
        if not self.comparison_results:
            print("No strategy results to compare")
            return None
        
        # Extract performance metrics
        strategies = []
        accuracies = []
        improvements = []
        confidence_scores = []
        
        for strategy_name, result in self.comparison_results.items():
            strategies.append(strategy_name.replace('_', ' ').title())
            
            # Extract accuracy based on strategy type
            if 'improved_accuracy' in result:
                accuracy = result['improved_accuracy']
            elif 'ensemble_accuracy' in result:
                accuracy = result['ensemble_accuracy']
            elif 'augmented_accuracy' in result:
                accuracy = result['augmented_accuracy']
            elif 'balanced_accuracy' in result:
                accuracy = result['balanced_accuracy']
            elif 'selected_accuracy' in result:
                accuracy = result['selected_accuracy']
            elif 'clean_accuracy' in result:
                accuracy = result['clean_accuracy']
            else:
                accuracy = baseline_accuracy
            
            accuracies.append(accuracy)
            improvements.append(accuracy - baseline_accuracy)
            
            # Extract confidence if available
            if 'average_confidence' in result:
                confidence_scores.append(result['average_confidence'])
            else:
                confidence_scores.append(0.5)  # Default confidence
        
        # Create comparison DataFrame
        comparison_df = pd.DataFrame({
            'Strategy': strategies,
            'Accuracy': accuracies,
            'Improvement': improvements,
            'Confidence': confidence_scores
        })
        
        # Sort by improvement
        comparison_df = comparison_df.sort_values('Improvement', ascending=False)
        
        # Create visualization
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. Accuracy comparison
        colors = ['red' if imp < 0 else 'green' for imp in comparison_df['Improvement']]
        bars1 = ax1.bar(range(len(comparison_df)), comparison_df['Accuracy'], 
                       color=colors, alpha=0.7)
        ax1.axhline(y=baseline_accuracy, color='blue', linestyle='--', 
                   label=f'Baseline ({baseline_accuracy:.4f})')
        ax1.set_title('Strategy Performance Comparison', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Mitigation Strategy')
        ax1.set_ylabel('Accuracy')
        ax1.set_xticks(range(len(comparison_df)))
        ax1.set_xticklabels(comparison_df['Strategy'], rotation=45, ha='right')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for i, (bar, acc) in enumerate(zip(bars1, comparison_df['Accuracy'])):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                    f'{acc:.4f}', ha='center', va='bottom', fontsize=10)
        
        # 2. Improvement comparison
        colors2 = ['red' if imp < 0 else 'green' for imp in comparison_df['Improvement']]
        bars2 = ax2.bar(range(len(comparison_df)), comparison_df['Improvement'], 
                       color=colors2, alpha=0.7)
        ax2.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        ax2.set_title('Accuracy Improvement by Strategy', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Mitigation Strategy')
        ax2.set_ylabel('Accuracy Improvement')
        ax2.set_xticks(range(len(comparison_df)))
        ax2.set_xticklabels(comparison_df['Strategy'], rotation=45, ha='right')
        ax2.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for i, (bar, imp) in enumerate(zip(bars2, comparison_df['Improvement'])):
            ax2.text(bar.get_x() + bar.get_width()/2, 
                    bar.get_height() + (0.0005 if imp >= 0 else -0.0005),
                    f'{imp:+.4f}', ha='center', 
                    va='bottom' if imp >= 0 else 'top', fontsize=10)
        
        # 3. Confidence vs Improvement scatter
        scatter = ax3.scatter(comparison_df['Confidence'], comparison_df['Improvement'], 
                            c=comparison_df['Improvement'], cmap='RdYlGn', 
                            s=100, alpha=0.7, edgecolors='black')
        ax3.set_title('Confidence vs Improvement', fontsize=14, fontweight='bold')
        ax3.set_xlabel('Average Confidence')
        ax3.set_ylabel('Accuracy Improvement')
        ax3.grid(True, alpha=0.3)
        
        # Add strategy labels
        for i, strategy in enumerate(comparison_df['Strategy']):
            ax3.annotate(strategy, 
                        (comparison_df['Confidence'].iloc[i], comparison_df['Improvement'].iloc[i]),
                        xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax3)
        cbar.set_label('Improvement')
        
        # 4. Strategy effectiveness ranking
        ranking_df = comparison_df.copy()
        ranking_df['Rank'] = range(1, len(ranking_df) + 1)
        ranking_df['Effectiveness'] = ranking_df['Improvement'].apply(
            lambda x: 'High' if x > 0.01 else 'Medium' if x > 0 else 'Low'
        )
        
        colors4 = {'High': 'green', 'Medium': 'orange', 'Low': 'red'}
        ranking_colors = [colors4[eff] for eff in ranking_df['Effectiveness']]
        
        bars4 = ax4.barh(range(len(ranking_df)), ranking_df['Improvement'], 
                        color=ranking_colors, alpha=0.7)
        ax4.set_title('Strategy Effectiveness Ranking', fontsize=14, fontweight='bold')
        ax4.set_xlabel('Accuracy Improvement')
        ax4.set_ylabel('Strategy Rank')
        ax4.set_yticks(range(len(ranking_df)))
        ax4.set_yticklabels([f"#{r}" for r in ranking_df['Rank']])
        ax4.grid(True, alpha=0.3)
        
        # Add value labels
        for i, (bar, imp) in enumerate(zip(bars4, ranking_df['Improvement'])):
            ax4.text(bar.get_width() + (0.0002 if imp >= 0 else -0.0002), 
                    bar.get_y() + bar.get_height()/2,
                    f'{imp:+.4f}', ha='left' if imp >= 0 else 'right', 
                    va='center', fontsize=10)
        
        plt.tight_layout()
        
        # Save plot
        if save_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = f"results/strategy_comparison_{timestamp}.png"
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return save_path
    
    def create_interactive_comparison(self, baseline_accuracy: float, 
                                   save_path: str = "results/interactive_strategy_comparison.html") -> str:
        """
        Create interactive Plotly comparison dashboard.
        
        Args:
            baseline_accuracy: Baseline model accuracy
            save_path: Path to save the HTML file
            
        Returns:
            Path to saved file
        """
        if not self.comparison_results:
            print("No strategy results to compare")
            return None
        
        # Prepare data
        strategies = []
        accuracies = []
        improvements = []
        confidence_scores = []
        strategy_details = []
        
        for strategy_name, result in self.comparison_results.items():
            strategies.append(strategy_name.replace('_', ' ').title())
            
            # Extract accuracy
            if 'improved_accuracy' in result:
                accuracy = result['improved_accuracy']
            elif 'ensemble_accuracy' in result:
                accuracy = result['ensemble_accuracy']
            elif 'augmented_accuracy' in result:
                accuracy = result['augmented_accuracy']
            elif 'balanced_accuracy' in result:
                accuracy = result['balanced_accuracy']
            elif 'selected_accuracy' in result:
                accuracy = result['selected_accuracy']
            elif 'clean_accuracy' in result:
                accuracy = result['clean_accuracy']
            else:
                accuracy = baseline_accuracy
            
            accuracies.append(accuracy)
            improvements.append(accuracy - baseline_accuracy)
            
            # Extract confidence
            if 'average_confidence' in result:
                confidence_scores.append(result['average_confidence'])
            else:
                confidence_scores.append(0.5)
            
            # Extract additional details
            details = []
            if 'n_estimators' in result:
                details.append(f"Estimators: {result['n_estimators']}")
            if 'confidence_threshold' in result:
                details.append(f"Threshold: {result['confidence_threshold']}")
            if 'augmentation_method' in result:
                details.append(f"Method: {result['augmentation_method']}")
            if 'n_queries' in result:
                details.append(f"Queries: {result['n_queries']}")
            
            strategy_details.append("<br>".join(details) if details else "No additional details")
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Accuracy Comparison', 'Improvement Analysis', 
                          'Confidence vs Improvement', 'Strategy Details'),
            specs=[[{"type": "bar"}, {"type": "bar"}],
                   [{"type": "scatter"}, {"type": "table"}]]
        )
        
        # 1. Accuracy comparison
        fig.add_trace(
            go.Bar(
                x=strategies,
                y=accuracies,
                name='Strategy Accuracy',
                marker_color=['green' if imp > 0 else 'red' for imp in improvements],
                text=[f'{acc:.4f}' for acc in accuracies],
                textposition='auto',
                hovertemplate='<b>%{x}</b><br>Accuracy: %{y:.4f}<extra></extra>'
            ),
            row=1, col=1
        )
        
        # Add baseline line
        fig.add_hline(y=baseline_accuracy, line_dash="dash", line_color="blue",
                     annotation_text=f"Baseline: {baseline_accuracy:.4f}",
                     row=1, col=1)
        
        # 2. Improvement analysis
        fig.add_trace(
            go.Bar(
                x=strategies,
                y=improvements,
                name='Improvement',
                marker_color=['green' if imp > 0 else 'red' for imp in improvements],
                text=[f'{imp:+.4f}' for imp in improvements],
                textposition='auto',
                hovertemplate='<b>%{x}</b><br>Improvement: %{y:+.4f}<extra></extra>'
            ),
            row=1, col=2
        )
        
        # 3. Confidence vs Improvement scatter
        fig.add_trace(
            go.Scatter(
                x=confidence_scores,
                y=improvements,
                mode='markers+text',
                text=strategies,
                textposition='top center',
                marker=dict(
                    size=15,
                    color=improvements,
                    colorscale='RdYlGn',
                    showscale=True,
                    colorbar=dict(title="Improvement")
                ),
                hovertemplate='<b>%{text}</b><br>Confidence: %{x:.3f}<br>Improvement: %{y:+.4f}<extra></extra>'
            ),
            row=2, col=1
        )
        
        # 4. Strategy details table
        fig.add_trace(
            go.Table(
                header=dict(
                    values=['Strategy', 'Accuracy', 'Improvement', 'Confidence', 'Details'],
                    fill_color='lightblue',
                    align='center',
                    font=dict(size=12)
                ),
                cells=dict(
                    values=[strategies, 
                           [f'{acc:.4f}' for acc in accuracies],
                           [f'{imp:+.4f}' for imp in improvements],
                           [f'{conf:.3f}' for conf in confidence_scores],
                           strategy_details],
                    fill_color='white',
                    align='center',
                    font=dict(size=10)
                )
            ),
            row=2, col=2
        )
        
        # Update layout
        fig.update_layout(
            title=dict(
                text="Interactive Strategy Comparison Dashboard",
                x=0.5,
                font=dict(size=20)
            ),
            height=800,
            showlegend=False
        )
        
        # Update axes
        fig.update_xaxes(title_text="Strategy", row=1, col=1)
        fig.update_yaxes(title_text="Accuracy", row=1, col=1)
        fig.update_xaxes(title_text="Strategy", row=1, col=2)
        fig.update_yaxes(title_text="Improvement", row=1, col=2)
        fig.update_xaxes(title_text="Confidence", row=2, col=1)
        fig.update_yaxes(title_text="Improvement", row=2, col=1)
        
        # Save interactive plot
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.write_html(save_path)
        
        return save_path
    
    def generate_comparison_report(self, baseline_accuracy: float, 
                                save_path: str = "results/strategy_comparison_report.txt") -> str:
        """
        Generate comprehensive text report of strategy comparison.
        
        Args:
            baseline_accuracy: Baseline model accuracy
            save_path: Path to save the report
            
        Returns:
            Path to saved file
        """
        if not self.comparison_results:
            print("No strategy results to compare")
            return None
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("MITIGATION STRATEGY COMPARISON REPORT\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Baseline Accuracy: {baseline_accuracy:.4f}\n")
            f.write(f"Number of Strategies Tested: {len(self.comparison_results)}\n\n")
            
            # Sort strategies by improvement
            sorted_strategies = sorted(
                self.comparison_results.items(),
                key=lambda x: self._extract_accuracy(x[1]) - baseline_accuracy,
                reverse=True
            )
            
            f.write("STRATEGY RANKING (by improvement):\n")
            f.write("-" * 50 + "\n")
            
            for i, (strategy_name, result) in enumerate(sorted_strategies, 1):
                accuracy = self._extract_accuracy(result)
                improvement = accuracy - baseline_accuracy
                relative_improvement = (improvement / baseline_accuracy) * 100
                
                f.write(f"{i}. {strategy_name.replace('_', ' ').title()}\n")
                f.write(f"   Accuracy: {accuracy:.4f}\n")
                f.write(f"   Improvement: {improvement:+.4f} ({relative_improvement:+.2f}%)\n")
                
                # Add strategy-specific details
                if 'n_estimators' in result:
                    f.write(f"   Estimators: {result['n_estimators']}\n")
                if 'confidence_threshold' in result:
                    f.write(f"   Confidence Threshold: {result['confidence_threshold']}\n")
                if 'augmentation_method' in result:
                    f.write(f"   Augmentation Method: {result['augmentation_method']}\n")
                if 'n_queries' in result:
                    f.write(f"   Active Learning Queries: {result['n_queries']}\n")
                
                f.write("\n")
            
            # Summary statistics
            improvements = [self._extract_accuracy(result) - baseline_accuracy 
                          for result in self.comparison_results.values()]
            
            f.write("SUMMARY STATISTICS:\n")
            f.write("-" * 30 + "\n")
            f.write(f"Best Improvement: {max(improvements):+.4f}\n")
            f.write(f"Worst Improvement: {min(improvements):+.4f}\n")
            f.write(f"Average Improvement: {np.mean(improvements):+.4f}\n")
            f.write(f"Standard Deviation: {np.std(improvements):.4f}\n")
            f.write(f"Strategies with Positive Improvement: {sum(1 for imp in improvements if imp > 0)}\n")
            f.write(f"Strategies with Negative Improvement: {sum(1 for imp in improvements if imp < 0)}\n\n")
            
            # Recommendations
            f.write("RECOMMENDATIONS:\n")
            f.write("-" * 20 + "\n")
            
            best_strategy = sorted_strategies[0][0]
            best_improvement = sorted_strategies[0][1]
            
            f.write(f"1. Best Performing Strategy: {best_strategy.replace('_', ' ').title()}\n")
            f.write(f"   - Consider this as the primary mitigation approach\n")
            f.write(f"   - Improvement: {self._extract_accuracy(best_improvement) - baseline_accuracy:+.4f}\n\n")
            
            # Find strategies with significant improvement
            significant_strategies = [(name, result) for name, result in sorted_strategies 
                                    if self._extract_accuracy(result) - baseline_accuracy > 0.01]
            
            if len(significant_strategies) > 1:
                f.write("2. Multiple Effective Strategies Found:\n")
                f.write("   - Consider ensemble of top-performing strategies\n")
                f.write("   - Test combination approaches for further improvement\n\n")
            
            # Check for strategies that made things worse
            negative_strategies = [(name, result) for name, result in sorted_strategies 
                                 if self._extract_accuracy(result) - baseline_accuracy < -0.01]
            
            if negative_strategies:
                f.write("3. Strategies with Negative Impact:\n")
                for name, result in negative_strategies:
                    f.write(f"   - {name.replace('_', ' ').title()}: "
                           f"{self._extract_accuracy(result) - baseline_accuracy:+.4f}\n")
                f.write("   - Review these strategies for potential issues\n")
                f.write("   - Consider different parameter settings\n\n")
            
            f.write("4. Next Steps:\n")
            f.write("   - Implement the best-performing strategy in production\n")
            f.write("   - Monitor performance over time\n")
            f.write("   - Consider A/B testing with multiple strategies\n")
            f.write("   - Collect more data to further improve strategies\n")
        
        print(f"Strategy comparison report saved to: {save_path}")
        return save_path
    
    def _extract_accuracy(self, result: Dict[str, Any]) -> float:
        """Extract accuracy from strategy result."""
        if 'improved_accuracy' in result:
            return result['improved_accuracy']
        elif 'ensemble_accuracy' in result:
            return result['ensemble_accuracy']
        elif 'augmented_accuracy' in result:
            return result['augmented_accuracy']
        elif 'balanced_accuracy' in result:
            return result['balanced_accuracy']
        elif 'selected_accuracy' in result:
            return result['selected_accuracy']
        elif 'clean_accuracy' in result:
            return result['clean_accuracy']
        else:
            return 0.5  # Default accuracy
    
    def export_comparison_data(self, baseline_accuracy: float, 
                             save_path: str = "results/strategy_comparison_data.json") -> str:
        """
        Export comparison data as JSON for further analysis.
        
        Args:
            baseline_accuracy: Baseline model accuracy
            save_path: Path to save the JSON file
            
        Returns:
            Path to saved file
        """
        if not self.comparison_results:
            print("No strategy results to compare")
            return None
        
        # Prepare data for export
        export_data = {
            'metadata': {
                'baseline_accuracy': baseline_accuracy,
                'generated_at': datetime.now().isoformat(),
                'total_strategies': len(self.comparison_results)
            },
            'strategies': {}
        }
        
        for strategy_name, result in self.comparison_results.items():
            accuracy = self._extract_accuracy(result)
            improvement = accuracy - baseline_accuracy
            
            export_data['strategies'][strategy_name] = {
                'accuracy': accuracy,
                'improvement': improvement,
                'relative_improvement_percent': (improvement / baseline_accuracy) * 100,
                'result_details': result
            }
        
        # Sort by improvement
        sorted_strategies = sorted(
            export_data['strategies'].items(),
            key=lambda x: x[1]['improvement'],
            reverse=True
        )
        
        export_data['ranking'] = [
            {
                'rank': i + 1,
                'strategy': name,
                'accuracy': data['accuracy'],
                'improvement': data['improvement']
            }
            for i, (name, data) in enumerate(sorted_strategies)
        ]
        
        # Save JSON
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, default=str)
        
        print(f"Strategy comparison data exported to: {save_path}")
        return save_path

# Example usage
if __name__ == "__main__":
    # Example strategy results
    comparator = StrategyComparator(['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'])
    
    # Add some mock results
    comparator.add_strategy_result('ensemble_learning', {
        'ensemble_accuracy': 0.9639,
        'improvement': 0.0139,
        'n_estimators': 5,
        'average_confidence': 0.85
    })
    
    comparator.add_strategy_result('data_augmentation', {
        'augmented_accuracy': 0.9583,
        'improvement': 0.0083,
        'augmentation_method': 'smote',
        'average_confidence': 0.82
    })
    
    # Create comparison
    baseline_acc = 0.95
    comparator.create_performance_comparison(baseline_acc)
    comparator.create_interactive_comparison(baseline_acc)
    comparator.generate_comparison_report(baseline_acc)
