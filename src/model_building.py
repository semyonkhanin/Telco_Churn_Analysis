# -*- coding: utf-8 -*-

# model_building.py
"""Functions for building, training, and evaluating models."""
import pandas as pd
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, recall_score
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import SGDClassifier
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import joblib
import os

from config import RANDOM_STATE, SCORING, MODEL_PARAMS, TOP_N_MODELS, CV_FOLDS, MODELS_DIR

def create_model_instances():
    """
    Create a dictionary of initialized model instances for classification.

    Returns:
    --------
    dict
        A dictionary where keys are model names and values are corresponding model instances.
    """
    models = {
        "Logistic Regression": LogisticRegression(max_iter=10000, random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(random_state=RANDOM_STATE),
        "XGBoost": XGBClassifier(eval_metric='logloss', random_state=RANDOM_STATE),
        "SGD": SGDClassifier(random_state=RANDOM_STATE, loss='log_loss', max_iter=1000, tol=1e-3)
    }
    
    return models

def save_model_summary(model, name, metrics, params, filepath):
    """Save model summary in a readable text format."""
    with open(filepath, 'w') as f:
        f.write(f"Model: {name}\n")
        f.write("=" * 50 + "\n\n")
        
        f.write("Model Type: " + str(type(model).__name__) + "\n\n")
        
        f.write("Best Parameters:\n")
        for param, value in params.items():
            f.write(f"{param}: {value}\n")
        f.write("\n")
        
        f.write("Performance Metrics:\n")
        for metric, value in metrics.items():
            f.write(f"{metric.capitalize()}: {value:.4f}\n")

def train_tune_model(name, model, params, X_train, y_train, X_val, y_val, preprocessor):
    """
   Train and tune a model using GridSearchCV with SMOTE and preprocessing.

   Parameters:
   -----------
   name : str
       Name of the model.
   model : estimator
       A scikit-learn compatible classifier.
   params : dict
       Hyperparameter grid for GridSearchCV.
   X_train : DataFrame
       Training feature matrix.
   y_train : Series or array-like
       Training target vector.
   X_val : DataFrame
       Validation feature matrix.
   y_val : Series or array-like
       Validation target vector.
   preprocessor : ColumnTransformer
       Preprocessing pipeline for numeric and categorical features.

   Returns:
   --------
   dict
       Dictionary containing the best estimator, best parameters, cross-validation results,
       validation metrics, predictions, and classification report.
   """
    print(f"\nTraining and tuning {name}...")
    
    # Create a pipeline with preprocessing and model using imblearn.pipeline.Pipeline
    pipeline = ImbPipeline([
        ('preprocessor', preprocessor),
        ('smote', SMOTE(random_state=RANDOM_STATE)),
        ('model', model)
    ])
    
    # Create GridSearchCV with stratified k-fold cross-validation
    grid_search = GridSearchCV(
        pipeline,
        params,
        cv=StratifiedKFold(n_splits=CV_FOLDS),
        scoring=SCORING,
        n_jobs=-1,
        return_train_score=True
    )
    
    # Fit the model on training data
    grid_search.fit(X_train, y_train)
    
    # Get best estimator
    best_estimator = grid_search.best_estimator_
    
    # Evaluate on validation set
    y_val_pred = best_estimator.predict(X_val)
    y_val_proba = best_estimator.predict_proba(X_val)[:, 1]
    
    # Calculate validation metrics
    val_metrics = {
        'accuracy': accuracy_score(y_val, y_val_pred),
        'f1': f1_score(y_val, y_val_pred),
        'precision': precision_score(y_val, y_val_pred),
        'recall': recall_score(y_val, y_val_pred),
        'roc_auc': roc_auc_score(y_val, y_val_proba)
    }
    
    # Save the model in both formats
    base_path = os.path.join(MODELS_DIR, name.lower().replace(" ", "_"))
    
    # Save joblib for loading back into Python
    joblib.dump(best_estimator, f"{base_path}.joblib")
    
    # Save readable summary
    save_model_summary(
        model=best_estimator,
        name=name,
        metrics=val_metrics,
        params=grid_search.best_params_,
        filepath=f"{base_path}_summary.txt"
    )
    
    # Store results for this model
    model_results = {
        'best_estimator': best_estimator,
        'best_params': grid_search.best_params_,
        'cv_results': grid_search.cv_results_,
        'validation_metrics': val_metrics,
        'validation_predictions': {
            'y_pred': y_val_pred,
            'y_proba': y_val_proba
        },
        'validation_classification_report': classification_report(y_val, y_val_pred, output_dict=True)
    }
    
    # Print validation metrics
    print(f"\nValidation Metrics for {name}:")
    for metric, value in val_metrics.items():
        print(f"{metric.capitalize()}: {value:.4f}")
    
    # Print classification report
    print(f"\nClassification Report for {name} on Validation Set:")
    print(classification_report(y_val, y_val_pred))
    
    return model_results

def build_and_evaluate_models(X_train, X_val, y_train, y_val, preprocessor):
    """
    Train and evaluate multiple models with hyperparameter tuning.

    Parameters:
    -----------
    X_train : DataFrame
        Training features.
    X_val : DataFrame
        Validation features.
    y_train : Series
        Training labels.
    y_val : Series
        Validation labels.
    preprocessor : ColumnTransformer
        Preprocessing pipeline.

    Returns:
    --------
    results_df : DataFrame
        DataFrame summarizing validation metrics for each model.
    best_models : dict
        Dictionary of best estimators from GridSearchCV for each model.
    all_results : dict
        Detailed results (including metrics and classification reports) for each model.
    """
    # Get model instances
    models = create_model_instances()
    
    # Store results
    all_results = {}
    results = []
    best_models = {}
    
    # Loop through each model and train with hyperparameter optimization
    for name, model in models.items():
        # Get model parameters
        params = MODEL_PARAMS[name]
        
        # Train and evaluate model
        model_results = train_tune_model(
            name, model, params, X_train, y_train, X_val, y_val, preprocessor
        )
        
        # Store best model and results
        best_models[name] = model_results['best_estimator']
        all_results[name] = model_results
        
        # Store results for DataFrame
        results.append((
            name,
            model_results['best_params'],
            model_results['validation_metrics']['accuracy'],
            model_results['validation_metrics']['f1'],
            model_results['validation_metrics']['precision'],
            model_results['validation_metrics']['recall'],
            model_results['validation_metrics']['roc_auc']
        ))
    
    # Create a DataFrame to present the results
    results_df = pd.DataFrame(results, columns=[
        "Model", "Best Parameters", "Validation Accuracy", "Validation F1", 
        "Validation Precision", "Validation Recall", "Validation ROC AUC"
    ])
    
    # Sort by ROC AUC
    results_df = results_df.sort_values(by="Validation Precision", ascending=False)
    
    return results_df, best_models, all_results

def build_stacking_ensemble(X_train, X_val, y_train, y_val, best_models):
    """
   Build a stacking ensemble using the top N models.

   Parameters:
   -----------
   X_train : DataFrame
       Training features.
   X_val : DataFrame
       Validation features.
   y_train : Series
       Training labels.
   y_val : Series
       Validation labels.
   best_models : dict
       Dictionary of best individual model estimators.

   Returns:
   --------
   stacking : StackingClassifier
       Trained stacking ensemble classifier.
   ensemble_results : dict
       Dictionary containing validation metrics and classification report for the ensemble.
   """
    # Select top N models
    top_models = list(best_models.items())[:TOP_N_MODELS]
    
    # Define base models for stacking
    estimators = [(name, model) for name, model in top_models]
    
    # Print models used in the ensemble
    print("\nModels used in the stacking ensemble:")
    for name, model in estimators:
        print(f"- {name}: {model.__class__.__name__}")
    
    # Define meta-model
    meta_model = LogisticRegression(max_iter=10000)
    
    print(f"\nMeta-model used in the stacking ensemble: {meta_model.__class__.__name__}")
    
    # Create stacking classifier
    stacking = StackingClassifier(
        estimators=estimators,
        final_estimator=meta_model,
        cv=5,
        stack_method='predict_proba'
    )
    
    # Train the stacking model on combined train and validation data
    stacking.fit(X_train, y_train)
    
    # Make predictions on validation set
    y_val_pred = stacking.predict(X_val)
    y_val_proba = stacking.predict_proba(X_val)[:, 1]
    
    # Calculate validation metrics
    val_metrics = {
        'accuracy': accuracy_score(y_val, y_val_pred),
        'f1': f1_score(y_val, y_val_pred),
        'precision': precision_score(y_val, y_val_pred),
        'recall': recall_score(y_val, y_val_pred),
        'roc_auc': roc_auc_score(y_val, y_val_proba)
    }
    
    # Save the stacking model in both formats
    base_path = os.path.join(MODELS_DIR, 'stacking_ensemble')
    
    # Save joblib for loading back into Python
    joblib.dump(stacking, f"{base_path}.joblib")
    
    # Save readable summary
    save_model_summary(
        model=stacking,
        name="Stacking Ensemble",
        metrics=val_metrics,
        params={"base_models": [type(model).__name__ for _, model in estimators]},
        filepath=f"{base_path}_summary.txt"
    )
    
    # Store ensemble results
    ensemble_results = {
        'model': stacking,
        'validation_metrics': val_metrics,
        'validation_predictions': {
            'y_pred': y_val_pred,
            'y_proba': y_val_proba
        },
        'validation_classification_report': classification_report(y_val, y_val_pred, output_dict=True)
    }
    
    # Print validation results
    print("\nStacking Ensemble Validation Results:")
    for metric, value in val_metrics.items():
        print(f"{metric.capitalize()}: {value:.4f}")
    
    # Print classification report
    print("\nClassification Report for Stacking Ensemble on Validation Set:")
    print(classification_report(y_val, y_val_pred))
    
    return stacking, ensemble_results

def evaluate_on_holdout_test(models, stacking_model, X_holdout, y_holdout):
    """
    Evaluate individual tuned models and the stacking ensemble on a holdout test set.

    Parameters:
    -----------
    models : dict
        Dictionary of trained base models.
    stacking_model : StackingClassifier
        Trained stacking ensemble.
    X_holdout : DataFrame
        Holdout test features.
    y_holdout : Series
        Holdout test labels.

    Returns:
    --------
    test_results_df : DataFrame
        DataFrame containing performance metrics for each model on the test set.
    """
    print("\n==== Final Evaluation on Holdout Test Set ====")
    
    test_results = []
    
    # Evaluate each individual model
    for name, model in models.items():
        # Make predictions on test set
        y_pred = model.predict(X_holdout)
        y_proba = model.predict_proba(X_holdout)[:, 1]
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_holdout, y_pred),
            'f1': f1_score(y_holdout, y_pred),
            'precision': precision_score(y_holdout, y_pred),
            'recall': recall_score(y_holdout, y_pred),
            'roc_auc': roc_auc_score(y_holdout, y_proba)
        }
        
        # Print results
        print(f"\nTest Metrics for {name}:")
        for metric, value in metrics.items():
            print(f"{metric.capitalize()}: {value:.4f}")
        
        # Print classification report
        print(f"\nClassification Report for {name} on Test Set:")
        print(classification_report(y_holdout, y_pred))
        
        # Store test results for DataFrame
        test_results.append((
            name,
            metrics['accuracy'],
            metrics['f1'],
            metrics['precision'],
            metrics['recall'],
            metrics['roc_auc']
        ))
    
    # Evaluate stacking ensemble
    y_pred = stacking_model.predict(X_holdout)
    y_proba = stacking_model.predict_proba(X_holdout)[:, 1]
    
    # Calculate metrics for stacking model
    metrics = {
        'accuracy': accuracy_score(y_holdout, y_pred),
        'f1': f1_score(y_holdout, y_pred),
        'precision': precision_score(y_holdout, y_pred),
        'recall': recall_score(y_holdout, y_pred),
        'roc_auc': roc_auc_score(y_holdout, y_proba)
    }
    
    # Print results for stacking model
    print("\nTest Metrics for Stacking Ensemble:")
    for metric, value in metrics.items():
        print(f"{metric.capitalize()}: {value:.4f}")
    
    # Print classification report
    print("\nClassification Report for Stacking Ensemble on Test Set:")
    print(classification_report(y_holdout, y_pred))
    
    # Add stacking model to test results
    test_results.append((
        "Stacking Ensemble",
        metrics['accuracy'],
        metrics['f1'],
        metrics['precision'],
        metrics['recall'],
        metrics['roc_auc']
    ))
    
    # Create DataFrame to present test results
    test_results_df = pd.DataFrame(test_results, columns=[
        "Model", "Test Accuracy", "Test F1", "Test Precision", "Test Recall", "Test ROC AUC"
    ])
    
    # Sort by recall
    test_results_df = test_results_df.sort_values(by="Test Precision", ascending=False)
    
    return test_results_df