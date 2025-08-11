#!/usr/bin/env python3
"""
Simple Confusion Matrix Example
This script demonstrates proper confusion matrix analysis with synthetic data.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

def create_sample_data():
    """Create sample data for demonstration."""
    print("Creating sample data for confusion matrix analysis...")
    
    np.random.seed(42)
    n_samples = 200
    
    # Generate features
    feature1 = np.random.normal(0, 1, n_samples)
    feature2 = np.random.normal(0, 1, n_samples)
    feature3 = np.random.normal(0, 1, n_samples)
    
    # Create target with some relationship to features
    target = ((feature1 + feature2 + feature3 + np.random.normal(0, 0.5, n_samples)) > 0).astype(int)
    
    # Create DataFrame
    data = pd.DataFrame({
        'Feature1': feature1,
        'Feature2': feature2,
        'Feature3': feature3,
        'Target': target
    })
    
    print(f"Created dataset with {n_samples} samples")
    print(f"Target distribution:\n{data['Target'].value_counts()}")
    
    return data

def proper_confusion_matrix_analysis():
    """Demonstrate proper confusion matrix analysis."""
    print("\n" + "="*60)
    print("PROPER CONFUSION MATRIX ANALYSIS")
    print("="*60)
    
    # 1. Load/Create Data
    data = create_sample_data()
    
    # 2. Prepare Features and Target
    X = data.drop('Target', axis=1)
    y = data['Target']
    
    print(f"\nFeatures shape: {X.shape}")
    print(f"Target shape: {y.shape}")
    
    # 3. Train-Test Split (with stratification)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\nTraining set: {X_train.shape[0]} samples")
    print(f"Testing set: {X_test.shape[0]} samples")
    
    # 4. Feature Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 5. Train Model
    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(X_train_scaled, y_train)
    
    # 6. Make Predictions
    y_pred = model.predict(X_test_scaled)
    
    # 7. Calculate Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    
    print("\n" + "="*40)
    print("CONFUSION MATRIX")
    print("="*40)
    print("Raw Confusion Matrix:")
    print(cm)
    
    # 8. Calculate Metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    print(f"\nPerformance Metrics:")
    print(f"Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"Precision: {precision:.4f} ({precision*100:.2f}%)")
    print(f"Recall:    {recall:.4f} ({recall*100:.2f}%)")
    print(f"F1-Score:  {f1:.4f} ({f1*100:.2f}%)")
    
    # 9. Visualize Confusion Matrix
    plt.figure(figsize=(12, 5))
    
    # Confusion Matrix Heatmap
    plt.subplot(1, 2, 1)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Predicted 0', 'Predicted 1'],
                yticklabels=['Actual 0', 'Actual 1'])
    plt.title('Confusion Matrix')
    plt.ylabel('Actual Values')
    plt.xlabel('Predicted Values')
    
    # Performance Metrics Bar Plot
    plt.subplot(1, 2, 2)
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    scores = [accuracy, precision, recall, f1]
    
    bars = plt.bar(metrics, scores, color=['skyblue', 'lightgreen', 'lightcoral', 'gold'])
    plt.title('Model Performance Metrics')
    plt.ylabel('Score')
    plt.ylim(0, 1)
    
    # Add value labels on bars
    for bar, score in zip(bars, scores):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{score:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.show()
    
    # 10. Detailed Analysis
    print("\n" + "="*40)
    print("DETAILED ANALYSIS")
    print("="*40)
    
    # Extract confusion matrix components
    tn, fp, fn, tp = cm.ravel()
    
    print(f"True Negatives (TN):  {tn}")
    print(f"False Positives (FP): {fp}")
    print(f"False Negatives (FN): {fn}")
    print(f"True Positives (TP):  {tp}")
    
    # Calculate additional metrics
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    
    print(f"\nSpecificity: {specificity:.4f} ({specificity*100:.2f}%)")
    print(f"Sensitivity: {sensitivity:.4f} ({sensitivity*100:.2f}%)")
    
    # Classification Report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # 11. Model Assessment
    print("\n" + "="*40)
    print("MODEL ASSESSMENT")
    print("="*40)
    
    if accuracy > 0.8:
        print("✅ Model performance is GOOD")
    elif accuracy > 0.6:
        print("⚠️  Model performance is ACCEPTABLE")
    else:
        print("❌ Model performance needs IMPROVEMENT")
    
    if precision > 0.7 and recall > 0.7:
        print("✅ Model has good balance between precision and recall")
    else:
        print("⚠️  Model may need tuning for better precision/recall balance")
    
    # 12. Recommendations
    print("\n" + "="*40)
    print("RECOMMENDATIONS")
    print("="*40)
    
    if accuracy < 0.8:
        print("🎯 Consider:")
        print("   - Feature engineering")
        print("   - Trying different algorithms")
        print("   - Hyperparameter tuning")
    
    if precision < 0.7 or recall < 0.7:
        print("🎯 For better precision/recall:")
        print("   - Adjust classification threshold")
        print("   - Use class weights")
        print("   - Collect more balanced data")
    
    print("\n✅ Confusion Matrix Analysis Complete!")
    print("This demonstrates the proper workflow for classification model evaluation.")

def compare_with_original_approach():
    """Compare proper approach with common mistakes."""
    print("\n" + "="*60)
    print("COMPARISON: PROPER vs ORIGINAL APPROACH")
    print("="*60)
    
    print("\n❌ ORIGINAL APPROACH ISSUES:")
    print("1. No data preprocessing (scaling, cleaning)")
    print("2. No train-test split")
    print("3. No cross-validation")
    print("4. Only basic confusion matrix")
    print("5. Missing performance metrics")
    print("6. No error handling")
    print("7. Insufficient data (only 4 unique samples)")
    
    print("\n✅ PROPER APPROACH:")
    print("1. Complete data preprocessing pipeline")
    print("2. Stratified train-test split")
    print("3. Feature scaling")
    print("4. Multiple evaluation metrics")
    print("5. Comprehensive visualization")
    print("6. Error handling and validation")
    print("7. Sufficient data (200+ samples)")
    
    print("\n📊 KEY DIFFERENCES:")
    print("- Data Quality: 4 samples vs 200 samples")
    print("- Metrics: 1 metric vs 6+ metrics")
    print("- Validation: None vs Cross-validation")
    print("- Visualization: Basic vs Comprehensive")
    print("- Reliability: Low vs High")

if __name__ == "__main__":
    print("🚀 Confusion Matrix Analysis Example")
    print("This demonstrates proper implementation vs common mistakes")
    
    # Run proper analysis
    proper_confusion_matrix_analysis()
    
    # Show comparison
    compare_with_original_approach()
    
    print("\n🎉 Example completed!")
    print("Use this as a template for your own confusion matrix analysis.")
