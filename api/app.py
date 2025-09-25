# api/app.py
"""
FastAPI Application for Intelligent Error Mitigation System
RESTful API for external access to model predictions and error mitigation.
"""

from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Union
import numpy as np
import pandas as pd
import io
import base64
from PIL import Image
import cv2
import json
import os
import sys
from datetime import datetime
import asyncio
from contextlib import asynccontextmanager

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

try:
    from data_loader import DataLoader
    from baseline_model import BaselineModel
    from error_monitor import ErrorMonitor
    from error_analyzer import ErrorAnalyzer
    from mitigation_strategies import MitigationStrategies
    from xai_visualizer import XAIVisualizer
    from error_storyteller import ErrorStoryteller
    from mlops_logger import MLOpsLogger
except ImportError as e:
    print(f"Warning: Could not import modules: {e}")

# Global variables for model storage
models = {}
mlops_logger = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Startup
    global mlops_logger
    mlops_logger = MLOpsLogger()
    print("FastAPI application started. MLOps logger initialized.")
    
    yield
    
    # Shutdown
    print("FastAPI application shutting down.")

# Initialize FastAPI app
app = FastAPI(
    title="Intelligent Error Mitigation System API",
    description="RESTful API for ML model error detection, analysis, and mitigation",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models for request/response
class PredictionRequest(BaseModel):
    """Request model for single prediction."""
    features: List[float] = Field(..., description="Feature vector for prediction")
    model_name: str = Field(default="random_forest", description="Model type to use")
    dataset_name: str = Field(default="digits", description="Dataset name for training")
    return_confidence: bool = Field(default=True, description="Whether to return confidence scores")
    return_explanation: bool = Field(default=False, description="Whether to return XAI explanation")

class PredictionResponse(BaseModel):
    """Response model for prediction."""
    prediction: int = Field(..., description="Predicted class")
    confidence: float = Field(..., description="Prediction confidence")
    probabilities: List[float] = Field(..., description="Class probabilities")
    model_name: str = Field(..., description="Model used for prediction")
    explanation: Optional[Dict[str, Any]] = Field(None, description="XAI explanation if requested")
    timestamp: str = Field(..., description="Prediction timestamp")

class BatchPredictionRequest(BaseModel):
    """Request model for batch predictions."""
    features_list: List[List[float]] = Field(..., description="List of feature vectors")
    model_name: str = Field(default="random_forest", description="Model type to use")
    dataset_name: str = Field(default="digits", description="Dataset name for training")
    return_confidence: bool = Field(default=True, description="Whether to return confidence scores")

class BatchPredictionResponse(BaseModel):
    """Response model for batch predictions."""
    predictions: List[int] = Field(..., description="List of predicted classes")
    confidences: List[float] = Field(..., description="List of prediction confidences")
    probabilities: List[List[float]] = Field(..., description="List of class probabilities")
    model_name: str = Field(..., description="Model used for predictions")
    timestamp: str = Field(..., description="Prediction timestamp")

class ErrorAnalysisRequest(BaseModel):
    """Request model for error analysis."""
    true_labels: List[int] = Field(..., description="True labels")
    predicted_labels: List[int] = Field(..., description="Predicted labels")
    confidences: List[float] = Field(..., description="Prediction confidences")
    model_name: str = Field(..., description="Model name")
    dataset_name: str = Field(default="digits", description="Dataset name")

class MitigationRequest(BaseModel):
    """Request model for error mitigation."""
    strategy_name: str = Field(..., description="Mitigation strategy to apply")
    model_name: str = Field(default="random_forest", description="Model type")
    dataset_name: str = Field(default="digits", description="Dataset name")
    strategy_params: Dict[str, Any] = Field(default={}, description="Strategy parameters")

class HealthResponse(BaseModel):
    """Health check response model."""
    status: str = Field(..., description="Service status")
    timestamp: str = Field(..., description="Current timestamp")
    version: str = Field(..., description="API version")
    models_loaded: List[str] = Field(..., description="List of loaded models")

# Utility functions
def get_or_load_model(model_name: str, dataset_name: str) -> BaselineModel:
    """Get or load a model."""
    model_key = f"{model_name}_{dataset_name}"
    
    if model_key not in models:
        # Load data
        loader = DataLoader(dataset_name)
        data = loader.load_dataset()
        
        # Train model
        model = BaselineModel(model_name)
        model.train(data['X_train'], data['y_train'])
        
        # Store model and data
        models[model_key] = {
            'model': model,
            'data': data
        }
    
    return models[model_key]['model']

def get_model_data(model_name: str, dataset_name: str) -> Dict:
    """Get model data."""
    model_key = f"{model_name}_{dataset_name}"
    return models[model_key]['data']

# API Endpoints

@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Intelligent Error Mitigation System API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now().isoformat(),
        version="1.0.0",
        models_loaded=list(models.keys())
    )

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """Make a single prediction."""
    try:
        # Get or load model
        model = get_or_load_model(request.model_name, request.dataset_name)
        data = get_model_data(request.model_name, request.dataset_name)
        
        # Convert features to numpy array
        features = np.array(request.features).reshape(1, -1)
        
        # Make prediction
        prediction = model.predict(features)[0]
        probabilities = model.predict_proba(features)[0]
        confidence = probabilities[prediction]
        
        # Generate explanation if requested
        explanation = None
        if request.return_explanation:
            try:
                xai_visualizer = XAIVisualizer(
                    model.model, data['X_train'], data['y_train'],
                    data.get('feature_names'), data['target_names'], data['data_type']
                )
                explanation = xai_visualizer.explain_prediction_shap(features)
            except Exception as e:
                print(f"Explanation generation failed: {e}")
        
        # Log prediction
        if mlops_logger:
            mlops_logger.log_prediction(
                model_name=request.model_name,
                sample_id=f"api_{datetime.now().timestamp()}",
                true_label=None,  # Unknown for new predictions
                predicted_label=prediction,
                confidence=confidence,
                dataset_name=request.dataset_name,
                features=features[0],
                metadata={"api_request": True}
            )
        
        return PredictionResponse(
            prediction=int(prediction),
            confidence=float(confidence),
            probabilities=probabilities.tolist(),
            model_name=request.model_name,
            explanation=explanation,
            timestamp=datetime.now().isoformat()
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(request: BatchPredictionRequest):
    """Make batch predictions."""
    try:
        # Get or load model
        model = get_or_load_model(request.model_name, request.dataset_name)
        
        # Convert features to numpy array
        features = np.array(request.features_list)
        
        # Make predictions
        predictions = model.predict(features)
        probabilities = model.predict_proba(features)
        confidences = np.max(probabilities, axis=1)
        
        # Log predictions
        if mlops_logger:
            for i, (pred, conf, feat) in enumerate(zip(predictions, confidences, features)):
                mlops_logger.log_prediction(
                    model_name=request.model_name,
                    sample_id=f"batch_{i}_{datetime.now().timestamp()}",
                    true_label=None,
                    predicted_label=pred,
                    confidence=conf,
                    dataset_name=request.dataset_name,
                    features=feat,
                    metadata={"api_request": True, "batch_id": i}
                )
        
        return BatchPredictionResponse(
            predictions=predictions.tolist(),
            confidences=confidences.tolist(),
            probabilities=probabilities.tolist(),
            model_name=request.model_name,
            timestamp=datetime.now().isoformat()
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")

@app.post("/upload/image")
async def predict_image(
    file: UploadFile = File(...),
    model_name: str = Form(default="random_forest"),
    dataset_name: str = Form(default="digits"),
    return_explanation: bool = Form(default=False)
):
    """Upload and predict on an image."""
    try:
        # Validate file type
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        # Read and process image
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data))
        
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
        
        # Get or load model
        model = get_or_load_model(model_name, dataset_name)
        data = get_model_data(model_name, dataset_name)
        
        # Make prediction
        features = img_flattened.reshape(1, -1)
        prediction = model.predict(features)[0]
        probabilities = model.predict_proba(features)[0]
        confidence = probabilities[prediction]
        
        # Generate explanation if requested
        explanation = None
        if return_explanation:
            try:
                xai_visualizer = XAIVisualizer(
                    model.model, data['X_train'], data['y_train'],
                    data.get('feature_names'), data['target_names'], data['data_type']
                )
                explanation = xai_visualizer.explain_prediction_shap(features)
            except Exception as e:
                print(f"Explanation generation failed: {e}")
        
        # Log prediction
        if mlops_logger:
            mlops_logger.log_prediction(
                model_name=model_name,
                sample_id=f"image_{datetime.now().timestamp()}",
                true_label=None,
                predicted_label=prediction,
                confidence=confidence,
                dataset_name=dataset_name,
                features=img_flattened,
                metadata={"api_request": True, "file_name": file.filename}
            )
        
        return {
            "prediction": int(prediction),
            "confidence": float(confidence),
            "probabilities": probabilities.tolist(),
            "model_name": model_name,
            "explanation": explanation,
            "timestamp": datetime.now().isoformat(),
            "image_shape": img_array.shape,
            "processed_shape": img_resized.shape
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Image prediction failed: {str(e)}")

@app.post("/analyze/errors")
async def analyze_errors(request: ErrorAnalysisRequest):
    """Analyze prediction errors."""
    try:
        # Get or load model
        model = get_or_load_model(request.model_name, request.dataset_name)
        data = get_model_data(request.model_name, request.dataset_name)
        
        # Initialize error monitor and analyzer
        error_monitor = ErrorMonitor()
        error_analyzer = ErrorAnalyzer()
        
        # Create dummy features for analysis (in real scenario, you'd have actual features)
        n_samples = len(request.true_labels)
        dummy_features = np.random.rand(n_samples, data['X_train'].shape[1])
        
        # Monitor errors
        monitoring_results = error_monitor.monitor_predictions(
            dummy_features,
            np.array(request.true_labels),
            np.array(request.predicted_labels),
            np.column_stack([request.confidences, 1 - np.array(request.confidences)]),  # Dummy probabilities
            request.model_name
        )
        
        # Analyze errors
        error_analysis = error_analyzer.analyze_errors(
            monitoring_results['error_details'],
            dummy_features,
            np.array(request.true_labels)
        )
        
        # Log errors
        if mlops_logger:
            for error_detail in monitoring_results['error_details']:
                mlops_logger.log_error_analysis(
                    model_name=request.model_name,
                    sample_id=f"error_{error_detail['sample_index']}",
                    true_label=error_detail['true_label'],
                    predicted_label=error_detail['predicted_label'],
                    confidence=error_detail['confidence'],
                    error_category=error_detail.get('error_type', 'Unknown'),
                    dataset_name=request.dataset_name,
                    metadata={"api_request": True}
                )
        
        return {
            "error_summary": {
                "total_errors": monitoring_results['total_errors'],
                "error_rate": monitoring_results['error_rate'],
                "low_confidence_errors": monitoring_results['low_confidence_errors'],
                "high_confidence_errors": monitoring_results['high_confidence_errors']
            },
            "error_analysis": error_analysis,
            "timestamp": datetime.now().isoformat()
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error analysis failed: {str(e)}")

@app.post("/mitigate")
async def apply_mitigation(request: MitigationRequest):
    """Apply error mitigation strategy."""
    try:
        # Get or load model
        model = get_or_load_model(request.model_name, request.dataset_name)
        data = get_model_data(request.model_name, request.dataset_name)
        
        # Initialize mitigation strategies
        mitigation_strategies = MitigationStrategies()
        
        # Create dummy error analysis
        dummy_error_analysis = {
            'confidence_analysis': {'low_confidence_errors': 5},
            'class_imbalance_analysis': {'imbalanced_classes': []}
        }
        
        # Apply mitigation strategy
        result = mitigation_strategies.apply_mitigation(
            request.strategy_name,
            model.model,
            data['X_train'],
            data['y_train'],
            data['X_test'],
            data['y_test'],
            dummy_error_analysis,
            **request.strategy_params
        )
        
        # Log mitigation strategy
        if mlops_logger:
            improvement = result.get('improvement', 0)
            mlops_logger.log_mitigation_strategy(
                model_name=request.model_name,
                dataset_name=request.dataset_name,
                strategy_name=request.strategy_name,
                strategy_params=request.strategy_params,
                baseline_accuracy=result.get('original_accuracy', 0.5),
                improved_accuracy=result.get('improved_accuracy', result.get('ensemble_accuracy', 0.5)),
                improvement=improvement,
                execution_time=0.1,  # Placeholder
                success=improvement > 0,
                metadata={"api_request": True}
            )
        
        return {
            "strategy_name": request.strategy_name,
            "result": result,
            "timestamp": datetime.now().isoformat()
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Mitigation failed: {str(e)}")

@app.get("/models")
async def list_models():
    """List available models and datasets."""
    return {
        "available_models": ["random_forest", "gradient_boosting", "svm", "logistic_regression", "naive_bayes"],
        "available_datasets": ["digits", "breast_cancer", "wine", "mnist", "cifar10"],
        "loaded_models": list(models.keys()),
        "timestamp": datetime.now().isoformat()
    }

@app.get("/logs/errors")
async def get_error_logs(days: int = 7, model_name: Optional[str] = None):
    """Get error logs."""
    if not mlops_logger:
        raise HTTPException(status_code=500, detail="MLOps logger not initialized")
    
    try:
        error_summary = mlops_logger.get_error_summary(
            model_name=model_name,
            days=days
        )
        return error_summary
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get error logs: {str(e)}")

@app.get("/logs/performance")
async def get_performance_logs(model_name: str, days: int = 30):
    """Get model performance logs."""
    if not mlops_logger:
        raise HTTPException(status_code=500, detail="MLOps logger not initialized")
    
    try:
        performance_df = mlops_logger.get_model_performance_history(
            model_name=model_name,
            days=days
        )
        return performance_df.to_dict('records')
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get performance logs: {str(e)}")

@app.get("/logs/export")
async def export_logs(days: int = 30):
    """Export logs to CSV files."""
    if not mlops_logger:
        raise HTTPException(status_code=500, detail="MLOps logger not initialized")
    
    try:
        file_paths = mlops_logger.export_logs_to_csv(days=days)
        return {
            "exported_files": file_paths,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to export logs: {str(e)}")

@app.get("/logs/report")
async def generate_report(days: int = 7):
    """Generate MLOps report."""
    if not mlops_logger:
        raise HTTPException(status_code=500, detail="MLOps logger not initialized")
    
    try:
        report_path = mlops_logger.generate_mlops_report(days=days)
        return FileResponse(
            path=report_path,
            filename=f"mlops_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            media_type="text/plain"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate report: {str(e)}")

@app.delete("/models/{model_key}")
async def unload_model(model_key: str):
    """Unload a specific model."""
    if model_key in models:
        del models[model_key]
        return {"message": f"Model {model_key} unloaded successfully"}
    else:
        raise HTTPException(status_code=404, detail=f"Model {model_key} not found")

@app.delete("/models")
async def unload_all_models():
    """Unload all models."""
    models.clear()
    return {"message": "All models unloaded successfully"}

# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Handle HTTP exceptions."""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "timestamp": datetime.now().isoformat(),
            "path": str(request.url)
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Handle general exceptions."""
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "detail": str(exc),
            "timestamp": datetime.now().isoformat(),
            "path": str(request.url)
        }
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
