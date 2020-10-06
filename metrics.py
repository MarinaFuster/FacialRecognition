from sklearn import metrics
import numpy as np

def print_metrics(labels_pred, names, labels_test, labels_test_mapped_to_labels_train, names_test,
                  testing_with_training_dataset, show_testing_metrics):
    corrects = 0
    for i in range(len(labels_pred)):
        print(f"Predicting label: {names_test[labels_test[i]]}. Face belongs to ... {names[int(labels_pred[i])]}")
        if testing_with_training_dataset and names_test[labels_test[i]] == names[int(labels_pred[i])]:
            corrects = corrects + 1
    if testing_with_training_dataset:
        print(f"{corrects} out of {labels_pred.shape[0]} were predicted properly")
        if show_testing_metrics:
            print("Final metrics for testing purposes:")
            print(metrics.classification_report(labels_test_mapped_to_labels_train, labels_pred, labels=np.unique(labels_pred)))
