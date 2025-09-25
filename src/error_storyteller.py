# src/error_storyteller.py
"""
Error Storytelling & Analysis Module
Generates human-readable explanations for misclassified samples with storytelling approach.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Tuple, Optional
import os
from datetime import datetime
import json

class ErrorStoryteller:
    """Generates human-readable stories for model errors."""
    
    def __init__(self, target_names: List[str], feature_names: Optional[List[str]] = None,
                 data_type: str = 'tabular'):
        """
        Initialize Error Storyteller.
        
        Args:
            target_names: Names of target classes
            feature_names: Names of features
            data_type: 'tabular', 'image', or 'text'
        """
        self.target_names = target_names
        self.feature_names = feature_names or [f"feature_{i}" for i in range(64)]  # Default for digits
        self.data_type = data_type
        
        # Story templates for different scenarios
        self.story_templates = self._initialize_story_templates()
        
    def _initialize_story_templates(self) -> Dict[str, List[str]]:
        """Initialize story templates for different error scenarios."""
        return {
            'low_confidence': [
                "The model was uncertain about this prediction, showing low confidence ({confidence:.1%}). This suggests the sample lies near the decision boundary between classes.",
                "With only {confidence:.1%} confidence, the model struggled to distinguish between '{true_class}' and '{predicted_class}'. The features were ambiguous.",
                "The model's uncertainty ({confidence:.1%}) indicates this sample has characteristics of multiple classes, making it difficult to classify."
            ],
            'high_confidence_error': [
                "Despite high confidence ({confidence:.1%}), the model incorrectly predicted '{predicted_class}' instead of '{true_class}'. This suggests a systematic bias.",
                "The model was very confident ({confidence:.1%}) but still wrong. This indicates the model has learned incorrect patterns for distinguishing these classes.",
                "High confidence ({confidence:.1%}) but wrong prediction suggests the model is overconfident in its decision boundary between '{true_class}' and '{predicted_class}'."
            ],
            'boundary_case': [
                "This sample lies near the decision boundary between '{true_class}' and '{predicted_class}', making classification challenging.",
                "The features of this sample are similar to both '{true_class}' and '{predicted_class}', causing confusion at the decision boundary.",
                "This is a boundary case where the model's decision boundary doesn't perfectly separate '{true_class}' from '{predicted_class}'."
            ],
            'outlier': [
                "This sample appears to be an outlier with unusual feature values compared to the training data.",
                "The sample has atypical characteristics that don't match the patterns the model learned during training.",
                "This outlier sample has features that deviate significantly from the typical patterns of '{true_class}'."
            ],
            'class_imbalance': [
                "The model may have been influenced by class imbalance, favoring the more common '{predicted_class}' over the less frequent '{true_class}'.",
                "Due to fewer training examples of '{true_class}', the model struggled to learn its distinguishing features.",
                "Class imbalance likely contributed to this error, as the model saw more examples of '{predicted_class}' during training."
            ]
        }
    
    def generate_error_story(self, error_case: Dict[str, Any], 
                           feature_analysis: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Generate a comprehensive story for an error case.
        
        Args:
            error_case: Error case information
            feature_analysis: Optional feature analysis from XAI
            
        Returns:
            Complete error story
        """
        # Extract basic information
        true_class = error_case.get('true_class', 'Unknown')
        predicted_class = error_case.get('predicted_class', 'Unknown')
        confidence = error_case.get('confidence', 0.5)
        error_type = error_case.get('error_type', 'Unknown')
        
        # Generate story components
        story_components = {
            'title': self._generate_title(true_class, predicted_class, confidence),
            'summary': self._generate_summary(true_class, predicted_class, confidence, error_type),
            'detailed_explanation': self._generate_detailed_explanation(error_case, feature_analysis),
            'feature_insights': self._generate_feature_insights(error_case, feature_analysis),
            'recommendations': self._generate_recommendations(error_case, error_type),
            'visual_description': self._generate_visual_description(error_case)
        }
        
        # Combine into full story
        full_story = self._combine_story_components(story_components)
        
        return {
            'error_case': error_case,
            'story_components': story_components,
            'full_story': full_story,
            'generated_at': datetime.now().isoformat()
        }
    
    def _generate_title(self, true_class: str, predicted_class: str, confidence: float) -> str:
        """Generate a compelling title for the error story."""
        if confidence < 0.4:
            return f"Uncertain Model: Predicted '{predicted_class}' instead of '{true_class}'"
        elif confidence > 0.8:
            return f"Confident but Wrong: Model strongly predicted '{predicted_class}' instead of '{true_class}'"
        else:
            return f"Model Confusion: '{true_class}' misclassified as '{predicted_class}'"
    
    def _generate_summary(self, true_class: str, predicted_class: str, 
                         confidence: float, error_type: str) -> str:
        """Generate a brief summary of the error."""
        templates = self.story_templates.get(error_type.lower().replace(' ', '_'), 
                                           self.story_templates['boundary_case'])
        
        # Select template based on confidence
        if confidence < 0.4:
            template = templates[0] if len(templates) > 0 else templates[0]
        elif confidence > 0.8:
            template = templates[1] if len(templates) > 1 else templates[0]
        else:
            template = templates[2] if len(templates) > 2 else templates[0]
        
        return template.format(
            true_class=true_class,
            predicted_class=predicted_class,
            confidence=confidence
        )
    
    def _generate_detailed_explanation(self, error_case: Dict, 
                                     feature_analysis: Optional[Dict]) -> str:
        """Generate detailed explanation of why the error occurred."""
        true_class = error_case.get('true_class', 'Unknown')
        predicted_class = error_case.get('predicted_class', 'Unknown')
        confidence = error_case.get('confidence', 0.5)
        
        explanation_parts = []
        
        # Start with the basic error description
        explanation_parts.append(
            f"The model predicted '{predicted_class}' with {confidence:.1%} confidence, "
            f"but the true class was '{true_class}'."
        )
        
        # Add feature-based insights if available
        if feature_analysis and 'feature_importance' in feature_analysis:
            top_features = feature_analysis['feature_importance'][:3]
            feature_desc = []
            
            for feature in top_features:
                feature_name = feature.get('feature', 'Unknown')
                feature_value = feature.get('feature_value', 0)
                importance = feature.get('shap_value', feature.get('lime_weight', 0))
                
                if self.data_type == 'image' and 'pixel' in feature_name:
                    # Special handling for image features
                    feature_desc.append(
                        f"pixel {feature_name.split('_')[1:]} (value: {feature_value:.2f})"
                    )
                else:
                    feature_desc.append(f"{feature_name} (value: {feature_value:.2f})")
            
            if feature_desc:
                explanation_parts.append(
                    f"The model's decision was primarily influenced by: {', '.join(feature_desc)}."
                )
        
        # Add confidence-based insights
        if confidence < 0.4:
            explanation_parts.append(
                "The low confidence suggests the model was uncertain, possibly due to "
                "ambiguous features or the sample lying near a decision boundary."
            )
        elif confidence > 0.8:
            explanation_parts.append(
                "Despite high confidence, the model was wrong, indicating a potential "
                "systematic bias or overfitting to certain patterns."
            )
        
        # Add class-specific insights
        if self.data_type == 'image':
            explanation_parts.append(
                f"This misclassification between '{true_class}' and '{predicted_class}' "
                f"suggests the model may be confused by similar visual patterns or shapes."
            )
        
        return " ".join(explanation_parts)
    
    def _generate_feature_insights(self, error_case: Dict, 
                                 feature_analysis: Optional[Dict]) -> str:
        """Generate insights about the features that led to the error."""
        if not feature_analysis or 'feature_importance' not in feature_analysis:
            return "Feature analysis not available for this error case."
        
        insights = []
        top_features = feature_analysis['feature_importance'][:5]
        
        for i, feature in enumerate(top_features, 1):
            feature_name = feature.get('feature', 'Unknown')
            feature_value = feature.get('feature_value', 0)
            importance = feature.get('shap_value', feature.get('lime_weight', 0))
            
            # Generate insight based on feature type and value
            if self.data_type == 'image' and 'pixel' in feature_name:
                insight = self._generate_pixel_insight(feature_name, feature_value, importance)
            else:
                insight = f"Feature '{feature_name}' had value {feature_value:.2f} "
                if importance > 0:
                    insight += f"and contributed positively to the '{error_case.get('predicted_class')}' prediction."
                else:
                    insight += f"and contributed negatively to the '{error_case.get('predicted_class')}' prediction."
            
            insights.append(f"{i}. {insight}")
        
        return "\n".join(insights)
    
    def _generate_pixel_insight(self, pixel_name: str, pixel_value: float, importance: float) -> str:
        """Generate insight for pixel-based features (for image data)."""
        # Parse pixel coordinates
        parts = pixel_name.split('_')
        if len(parts) >= 3:
            row, col = parts[1], parts[2]
            location = f"row {row}, column {col}"
        else:
            location = pixel_name
        
        intensity = "bright" if pixel_value > 0.5 else "dark"
        contribution = "supported" if importance > 0 else "opposed"
        
        return f"Pixel at {location} was {intensity} (value: {pixel_value:.2f}) and {contribution} the prediction."
    
    def _generate_recommendations(self, error_case: Dict, error_type: str) -> str:
        """Generate recommendations for improving the model."""
        recommendations = []
        
        if error_type == "Low Confidence Error":
            recommendations.extend([
                "• Collect more training data for similar ambiguous cases",
                "• Consider ensemble methods to reduce uncertainty",
                "• Implement confidence thresholding to reject uncertain predictions"
            ])
        elif error_type == "High Confidence Error":
            recommendations.extend([
                "• Review training data for systematic biases",
                "• Apply regularization to reduce overconfidence",
                "• Use adversarial training to improve robustness"
            ])
        elif error_type == "Boundary Case":
            recommendations.extend([
                "• Fine-tune decision boundaries with more boundary examples",
                "• Use margin-based loss functions",
                "• Consider active learning to focus on boundary cases"
            ])
        else:
            recommendations.extend([
                "• Analyze feature distributions for this class",
                "• Consider data augmentation techniques",
                "• Review model architecture and hyperparameters"
            ])
        
        return "\n".join(recommendations)
    
    def _generate_visual_description(self, error_case: Dict) -> str:
        """Generate a visual description of the error case."""
        if self.data_type == 'image':
            true_class = error_case.get('true_class', 'Unknown')
            predicted_class = error_case.get('predicted_class', 'Unknown')
            
            return (f"Visual Analysis: The image was classified as '{predicted_class}' but "
                   f"actually represents '{true_class}'. This suggests the model may be "
                   f"focusing on incorrect visual features or patterns.")
        else:
            return "Visual analysis not applicable for non-image data."
    
    def _combine_story_components(self, components: Dict[str, str]) -> str:
        """Combine all story components into a cohesive narrative."""
        story = f"""
{components['title']}
{'=' * len(components['title'])}

SUMMARY:
{components['summary']}

DETAILED EXPLANATION:
{components['detailed_explanation']}

FEATURE INSIGHTS:
{components['feature_insights']}

VISUAL DESCRIPTION:
{components['visual_description']}

RECOMMENDATIONS FOR IMPROVEMENT:
{components['recommendations']}
"""
        return story.strip()
    
    def create_error_story_collection(self, error_cases: List[Dict], 
                                    feature_analyses: Optional[List[Dict]] = None,
                                    save_path: str = "results/error_stories.txt") -> str:
        """
        Create a collection of error stories for multiple cases.
        
        Args:
            error_cases: List of error cases
            feature_analyses: Optional list of feature analyses
            save_path: Path to save the story collection
            
        Returns:
            Path to saved file
        """
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("ERROR STORYTELLING & ANALYSIS COLLECTION\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total error cases: {len(error_cases)}\n\n")
            
            # Generate stories for each error case
            for i, error_case in enumerate(error_cases, 1):
                feature_analysis = feature_analyses[i-1] if feature_analyses and i-1 < len(feature_analyses) else None
                
                story = self.generate_error_story(error_case, feature_analysis)
                
                f.write(f"ERROR STORY #{i}\n")
                f.write("-" * 50 + "\n")
                f.write(story['full_story'])
                f.write("\n\n" + "=" * 80 + "\n\n")
        
        print(f"Error story collection saved to: {save_path}")
        return save_path
    
    def create_error_story_json(self, error_cases: List[Dict], 
                              feature_analyses: Optional[List[Dict]] = None,
                              save_path: str = "results/error_stories.json") -> str:
        """
        Create a JSON file with structured error stories.
        
        Args:
            error_cases: List of error cases
            feature_analyses: Optional list of feature analyses
            save_path: Path to save the JSON file
            
        Returns:
            Path to saved file
        """
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        stories_data = {
            'metadata': {
                'generated_at': datetime.now().isoformat(),
                'total_cases': len(error_cases),
                'data_type': self.data_type,
                'target_classes': self.target_names
            },
            'stories': []
        }
        
        for i, error_case in enumerate(error_cases):
            feature_analysis = feature_analyses[i] if feature_analyses and i < len(feature_analyses) else None
            story = self.generate_error_story(error_case, feature_analysis)
            stories_data['stories'].append(story)
        
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(stories_data, f, indent=2, default=str)
        
        print(f"Error stories JSON saved to: {save_path}")
        return save_path

# Example usage
if __name__ == "__main__":
    # Example error case
    error_case = {
        'true_class': '8',
        'predicted_class': '3',
        'confidence': 0.75,
        'error_type': 'High Confidence Error',
        'sample_idx': 42
    }
    
    # Initialize storyteller
    storyteller = ErrorStoryteller(
        target_names=['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'],
        data_type='image'
    )
    
    # Generate story
    story = storyteller.generate_error_story(error_case)
    print("Generated Story:")
    print(story['full_story'])
