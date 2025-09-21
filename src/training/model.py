"""
Multitask TCN Model Module

Defines a unified Temporal Convolutional Network (TCN) architecture that performs 
both exercise type classification and exercise repetition segmentation simultaneously.

Architecture Overview:
1. Shared TCN Backbone: Processes temporal sequences of joint angles
2. Classification Head: Global pooling + dense layers → exercise type
3. Segmentation Head: Per-frame dense layers → repetition probability
4. Multi-task loss: Weighted combination of classification and segmentation losses
"""

import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, metrics
from tensorflow.keras.losses import SparseCategoricalCrossentropy, BinaryCrossentropy
from tensorflow.keras.utils import plot_model
import numpy as np


def ResidualTCNBlock(x, filters, kernel_size, dilation_rate, dropout_rate=0.0, name_prefix="tcn"):
    """
    Functional implementation of a TCN residual block with better architecture.
    
    Architecture:
      - Conv1D (causal) → LayerNorm → ReLU → Dropout
      - Conv1D (causal) → LayerNorm → Dropout  
      - 1×1 Conv on residual path if channel dims differ
      - Add & ReLU
    
    Args:
        x: Input tensor
        filters: Number of output filters
        kernel_size: Size of the convolution kernel
        dilation_rate: Dilation rate for the convolution
        dropout_rate: Dropout rate (default: 0.0)
        name_prefix: Prefix for layer names
    
    Returns:
        Output tensor after residual block
    """
    # First conv + LayerNorm + ReLU + Dropout
    y = layers.Conv1D(
        filters, kernel_size,
        dilation_rate=dilation_rate,
        padding='causal',
        name=f'{name_prefix}_conv1d_1'
    )(x)
    y = layers.LayerNormalization(name=f'{name_prefix}_ln1')(y)
    y = layers.Activation('relu', name=f'{name_prefix}_relu1')(y)
    y = layers.Dropout(dropout_rate, name=f'{name_prefix}_dropout1')(y)

    # Second conv + LayerNorm + Dropout
    y = layers.Conv1D(
        filters, kernel_size,
        dilation_rate=dilation_rate,
        padding='causal',
        name=f'{name_prefix}_conv1d_2'
    )(y)
    y = layers.LayerNormalization(name=f'{name_prefix}_ln2')(y)
    y = layers.Dropout(dropout_rate, name=f'{name_prefix}_dropout2')(y)

    # Residual connection (1×1 conv if needed)
    if x.shape[-1] != filters:
        x = layers.Conv1D(filters, 1, padding='same', name=f'{name_prefix}_residual_conv')(x)

    out = layers.add([x, y], name=f'{name_prefix}_add')
    out = layers.Activation('relu', name=f'{name_prefix}_relu_out')(out)
    return out


def AttentionBlock(x, name_prefix="attention"):
    """
    Self-attention mechanism to help the model focus on important temporal features.
    
    Args:
        x: Input tensor of shape (batch, time, features)
        name_prefix: Prefix for layer names
        
    Returns:
        Attention-weighted features
    """
    # Multi-head self-attention
    attention_output = layers.MultiHeadAttention(
        num_heads=4,
        key_dim=x.shape[-1] // 4,
        name=f'{name_prefix}_mha'
    )(x, x)
    
    # Add & Norm
    x = layers.Add(name=f'{name_prefix}_add')([x, attention_output])
    x = layers.LayerNormalization(name=f'{name_prefix}_ln')(x)
    
    return x


def build_multitask_tcn_model(
    input_dim=25,
    num_classes=4,
    window_size=30,
    # Shared backbone parameters
    backbone_filters=64,
    backbone_layers=6,
    kernel_size=3,
    dropout_rate=0.2,
    # Head-specific parameters
    classification_units=[128, 64],
    segmentation_units=[64, 32],
    use_attention=True,
    # Loss weights
    classification_weight=1.0,
    segmentation_weight=5.0,
    # Optimizer parameters
    learning_rate=1e-3
):
    """
    Builds a unified multitask TCN model for exercise classification and segmentation.
    
    The model has three main components:
    1. Shared TCN Backbone: Processes temporal sequences with dilated convolutions
    2. Classification Head: Predicts exercise type (4 classes)
    3. Segmentation Head: Predicts per-frame repetition probability
    
    Args:
        input_dim: Number of input features per frame (25 joint angles)
        num_classes: Number of exercise classes (4: push-ups, squats, pull-ups, dips)
        window_size: Expected input sequence length (30 frames)
        backbone_filters: Number of filters in backbone TCN layers
        backbone_layers: Number of TCN layers in backbone
        kernel_size: Convolution kernel size
        dropout_rate: Dropout rate for regularization
        classification_units: List of hidden units in classification head
        segmentation_units: List of hidden units in segmentation head
        use_attention: Whether to use attention mechanism
        classification_weight: Weight for classification loss
        segmentation_weight: Weight for segmentation loss
        learning_rate: Learning rate for optimizer
    
    Returns:
        Compiled Keras model with two outputs: classification and segmentation
    """
    
    # Input layer
    inputs = layers.Input(shape=(window_size, input_dim), name='input_features')
    
    # ============ SHARED TCN BACKBONE ============
    x = inputs
    
    # Initial feature mapping
    x = layers.Conv1D(
        backbone_filters, 1, 
        padding='same', 
        name='backbone_initial_conv'
    )(x)
    
    # Stack TCN layers with exponential dilation
    for i in range(backbone_layers):
        dilation = 2 ** i
        x = ResidualTCNBlock(
            x,
            filters=backbone_filters,
            kernel_size=kernel_size,
            dilation_rate=dilation,
            dropout_rate=dropout_rate,
            name_prefix=f'backbone_tcn_{i}'
        )
    
    # Optional attention mechanism
    if use_attention:
        x = AttentionBlock(x, name_prefix='backbone_attention')
    
    # Shared features for both tasks
    shared_features = x  # Shape: (batch, window_size, backbone_filters)
    
    # ============ CLASSIFICATION HEAD ============
    # Global pooling to aggregate temporal information
    cls_features = layers.GlobalAveragePooling1D(name='cls_global_pool')(shared_features)
    
    # Dense layers for classification
    for i, units in enumerate(classification_units):
        cls_features = layers.Dense(
            units, activation='relu', 
            name=f'cls_dense_{i}'
        )(cls_features)
        cls_features = layers.Dropout(
            dropout_rate, 
            name=f'cls_dropout_{i}'
        )(cls_features)
    
    # Final classification output
    classification_output = layers.Dense(
        num_classes, 
        activation='softmax',
        name='classification_output'
    )(cls_features)
    
    # ============ SEGMENTATION HEAD ============
    seg_features = shared_features
    
    # Dense layers for per-frame processing
    for i, units in enumerate(segmentation_units):
        seg_features = layers.TimeDistributed(
            layers.Dense(units, activation='relu'),
            name=f'seg_dense_{i}'
        )(seg_features)
        seg_features = layers.Dropout(
            dropout_rate,
            name=f'seg_dropout_{i}'
        )(seg_features)
    
    # Final segmentation output (per-frame probabilities)
    segmentation_output = layers.TimeDistributed(
        layers.Dense(1, activation='sigmoid'),
        name='segmentation_output'
    )(seg_features)
    
    # ============ MODEL CREATION ============
    model = models.Model(
        inputs=inputs,
        outputs=[classification_output, segmentation_output],
        name='multitask_tcn_model'
    )
    
    # ============ CUSTOM LOSS COMPILATION ============
    
    # Classification loss (sparse categorical crossentropy)
    classification_loss = SparseCategoricalCrossentropy(from_logits=False)
    
    # Segmentation loss (binary crossentropy for better probability calibration)
    segmentation_loss = BinaryCrossentropy(from_logits=False)
    
    # Optimizer
    optimizer = optimizers.Adam(learning_rate=learning_rate)
    
    # Compile with multi-task losses
    model.compile(
        optimizer=optimizer,
        loss={
            'classification_output': classification_loss,
            'segmentation_output': segmentation_loss
        },
        loss_weights={
            'classification_output': classification_weight,
            'segmentation_output': segmentation_weight
        },
        metrics={
            'classification_output': [
                metrics.SparseCategoricalAccuracy(name='cls_accuracy'),
                metrics.SparseTopKCategoricalAccuracy(k=2, name='cls_top2_accuracy')
            ],
            'segmentation_output': [
                metrics.AUC(name='seg_auc'),
                metrics.Precision(name='seg_precision'),
            ]
        }
    )
    
    return model


def create_inference_model(trained_model):
    """
    Create an inference-optimized version of the model.
    
    Args:
        trained_model: Trained multitask model
        
    Returns:
        Inference model with both outputs
    """
    # Extract the trained weights but create a new model structure optimized for inference
    inference_model = models.Model(
        inputs=trained_model.input,
        outputs=trained_model.output,
        name='multitask_tcn_inference'
    )
    
    # Copy weights
    inference_model.set_weights(trained_model.get_weights())
    
    return inference_model


def get_model_summary(
    input_dim=25, 
    num_classes=4, 
    window_size=30,
    backbone_filters=64,
    backbone_layers=6
):
    """
    Get a summary of the multitask model architecture.
    
    Args:
        input_dim: Number of input features
        num_classes: Number of exercise classes
        window_size: Input sequence length
        backbone_filters: Number of backbone filters
        backbone_layers: Number of backbone layers
        
    Returns:
        Model summary string
    """
    model = build_multitask_tcn_model(
        input_dim=input_dim,
        num_classes=num_classes,
        window_size=window_size,
        backbone_filters=backbone_filters,
        backbone_layers=backbone_layers
    )
    return model.summary()


def save_model_plot(
    input_dim=25, 
    num_classes=4, 
    window_size=30,
    output_path='multitask_tcn_model.png',
    **kwargs  # Accept any additional arguments
):
    """
    Save a visual plot of the model architecture.
    
    Args:
        input_dim: Number of input features
        num_classes: Number of exercise classes  
        window_size: Input sequence length
        output_path: Path to save the plot
    """
    model = build_multitask_tcn_model(
        input_dim=input_dim,
        num_classes=num_classes,
        window_size=window_size
    )
    
    plot_model(
        model,
        to_file=output_path,
        show_shapes=True,
        expand_nested=True,
        show_layer_names=True
    )
    print(f"Model architecture plot saved to: {output_path}")


def analyze_model_complexity(
    input_dim=25,
    num_classes=4, 
    window_size=30,
    backbone_filters=64,
    backbone_layers=6,
    **kwargs  # Accept any additional arguments
):
    """
    Analyze the computational complexity of the multitask model.
    
    Args:
        input_dim: Number of input features
        num_classes: Number of exercise classes
        window_size: Input sequence length  
        backbone_filters: Number of backbone filters
        backbone_layers: Number of backbone layers
        
    Returns:
        Dictionary with model complexity metrics
    """
    model = build_multitask_tcn_model(
        input_dim=input_dim,
        num_classes=num_classes,
        window_size=window_size,
        backbone_filters=backbone_filters,
        backbone_layers=backbone_layers
    )
    
    # Calculate model complexity
    total_params = model.count_params()
    trainable_params = sum([tf.keras.backend.count_params(w) for w in model.trainable_weights])
    non_trainable_params = total_params - trainable_params
    
    # Estimate memory usage (rough calculation)
    input_size = window_size * input_dim * 4  # 4 bytes per float32
    
    complexity_info = {
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'non_trainable_parameters': non_trainable_params,
        'estimated_input_size_bytes': input_size,
        'model_size_mb': total_params * 4 / (1024 * 1024),  # Rough estimate
        'backbone_layers': backbone_layers,
        'backbone_filters': backbone_filters,
        'receptive_field': 2 ** backbone_layers - 1  # Theoretical max receptive field
    }
    
    return complexity_info


if __name__ == '__main__':
    """Example usage and model analysis."""
    print("Building multitask TCN model...")
    
    # Configuration
    config = {
        'input_dim': 25,
        'num_classes': 4,
        'window_size': 30,
        'backbone_filters': 64,
        'backbone_layers': 6,
        'use_attention': True
    }
    
    # Create model
    model = build_multitask_tcn_model(**config)
    
    # Print summary
    print("\n" + "="*60)
    print("MULTITASK TCN MODEL SUMMARY")
    print("="*60)
    model.summary()
    
    # Analyze complexity
    print("\n" + "="*60)
    print("MODEL COMPLEXITY ANALYSIS")
    print("="*60)
    complexity = analyze_model_complexity(**config)
    for key, value in complexity.items():
        print(f"{key.replace('_', ' ').title()}: {value:,}")
    
    # Save architecture plot
    save_model_plot(**config, output_path='multitask_tcn_architecture.png')
    
    # Test with dummy data
    print("\n" + "="*60)
    print("TESTING WITH DUMMY DATA")
    print("="*60)
    
    dummy_input = tf.random.normal((2, 30, 25))  # Batch of 2 sequences
    dummy_cls_labels = tf.constant([0, 1])  # Classification labels
    dummy_seg_labels = tf.random.uniform((2, 30, 1), 0, 1)  # Segmentation labels
    
    # Forward pass
    cls_pred, seg_pred = model(dummy_input)
    print(f"Classification output shape: {cls_pred.shape}")
    print(f"Segmentation output shape: {seg_pred.shape}")
    print(f"Classification prediction example: {cls_pred[0].numpy()}")
    print(f"Segmentation prediction range: [{seg_pred.numpy().min():.3f}, {seg_pred.numpy().max():.3f}]")
    
    print("\nModel created and tested successfully!") 