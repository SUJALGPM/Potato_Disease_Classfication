import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import numpy as np

def evaluate_model(model, val_gen):
    # Getting true and predicted labels
    y_true = val_gen.classes
    y_pred = model.predict(val_gen)
    y_pred_classes = np.argmax(y_pred, axis=1)

    # Print Classification Report
    print("Classification Report:")
    print(classification_report(y_true, y_pred_classes, target_names=list(val_gen.class_indices.keys())))

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred_classes)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=val_gen.class_indices.keys(),
                yticklabels=val_gen.class_indices.keys())
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.show()

    # Additional Detailed Evaluation
    print("\nConfusion Matrix:")
    print(cm)

    print("\nDetailed Classification Report:")
    print(classification_report(y_true, y_pred_classes, target_names=list(val_gen.class_indices.keys())))
