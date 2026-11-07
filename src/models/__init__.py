"""Classical baseline models and evaluation utilities."""
from .classical_svm import (
    create_svm_classifier,
    evaluate_class_weight_strategies,
    predict_and_score,
    save_model_artifacts,
    train_svm_model,
)
from .evaluate import (
    compute_classification_metrics,
    plot_comparative_curves,
    plot_confusion_matrix,
    plot_curves,
    plot_kernel_matrix_heatmap,
)
from .quantum_kernel import (
    QuantumKernelClassifier,
    compute_fidelity_kernel_matrix,
    compute_statevector_matrix,
    create_zz_feature_map,
    run_tiny_pilot_benchmark,
    validate_kernel_matrix,
)

__all__ = [
    "create_svm_classifier",
    "evaluate_class_weight_strategies",
    "predict_and_score",
    "save_model_artifacts",
    "train_svm_model",
    "compute_classification_metrics",
    "plot_confusion_matrix",
    "plot_curves",
    "plot_comparative_curves",
    "plot_kernel_matrix_heatmap",
    "create_zz_feature_map",
    "compute_statevector_matrix",
    "compute_fidelity_kernel_matrix",
    "validate_kernel_matrix",
    "run_tiny_pilot_benchmark",
    "QuantumKernelClassifier",
]
