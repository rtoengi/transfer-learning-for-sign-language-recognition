from training.constants import Metric

# Dictionary of the labels used in plots
_LABELS = {
    'loss_accuracy': ['Loss', 'Accuracy'],
    'train_validation': ['Train', 'Validation'],
    'learning_curves': 'Learning curves of the {model_type} model when trained on the {dataset_size} target dataset',
    'learning_curves_without_validation': 'Learning curves of the pre-trained model trained on the source dataset'
}

# List of the metrics used in the learning curves plot
_LOSS_ACCURACY_COLUMNS = [
    [Metric.LOSS.value, Metric.VAL_LOSS.value],
    [Metric.ACCURACY.value, Metric.VAL_ACCURACY.value]
]

# List of the columns used in the compare training plot
BASELINE_FINETUNED_COLUMNS = [
    ['baseline_accuracy', 'finetuned_accuracy'],
    ['baseline_val_accuracy', 'finetuned_val_accuracy']
]
