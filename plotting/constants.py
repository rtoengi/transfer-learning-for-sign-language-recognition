from core.constants import Metric

# Dictionary of the labels used in plots
_LABELS = {
    'loss_accuracy': ['Loss', 'Accuracy'],
    'train_validation': ['Train', 'Validation'],
    'dataset_sizes': ['Large', 'Medium', 'Small']
}

# List of the metrics used in the learning curves plot
_LOSS_ACCURACY_COLUMNS = [
    [Metric.LOSS.value, Metric.VAL_LOSS.value],
    [Metric.ACCURACY.value, Metric.VAL_ACCURACY.value]
]

# List of the columns used in the compare training plot
COMPARE_TRAINING_COLUMNS = [
    ['baseline_accuracy', 'finetuned_accuracy'],
    ['baseline_val_accuracy', 'finetuned_val_accuracy']
]

# List of the columns used in the compare test scores plot
COMPARE_TEST_SCORES_COLUMNS = [
    ['baseline_loss', 'finetuned_loss'],
    ['baseline_accuracy', 'finetuned_accuracy']
]
