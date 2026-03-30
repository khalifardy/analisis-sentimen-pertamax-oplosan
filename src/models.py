"""
Models Module
=============
Definisi semua model:
- MLP Baseline (sederhana)
- MLP Advanced (BatchNormalization + Dropout)
- MLP Keras Tuner (hyperparameter tuning)
- Model ML Tradisional (Naive Bayes, SVM, Random Forest, Logistic Regression)
"""

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression


# ============= CALLBACKS =============

def get_callbacks():
    """Return list callbacks untuk training MLP."""
    return [
        EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=0
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=0
        )
    ]


# ============= MLP BASELINE =============

def create_mlp_baseline(input_dim, num_classes=3):
    """
    Model MLP sederhana sebagai baseline.

    Arsitektur:
    - Dense(128, relu) -> Dropout(0.3)
    - Dense(64, relu) -> Dropout(0.3)
    - Dense(3, softmax)
    """
    model = Sequential([
        Input(shape=(input_dim,)),
        Dense(128, activation='relu'),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(num_classes, activation='softmax')
    ])

    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model


# ============= MLP ADVANCED =============

def create_mlp_advance(input_dim, num_classes=3,
                       hidden_layers=[256, 128, 64],
                       dropout_rate=0.4,
                       learning_rate=0.001):
    """
    Model MLP advance dengan BatchNormalization.

    Arsitektur:
    - [Dense -> BatchNorm -> Dropout] x len(hidden_layers)
    - Dense(3, softmax)
    """
    model = Sequential()
    model.add(Input(shape=(input_dim,)))

    for i, units in enumerate(hidden_layers):
        model.add(Dense(units, activation='relu', name=f'dense_{i+1}'))
        model.add(BatchNormalization(name=f'bn_{i+1}'))
        model.add(Dropout(dropout_rate, name=f'dropout_{i+1}'))

    model.add(Dense(num_classes, activation='softmax', name='output'))

    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model


# ============= KERAS TUNER BUILDER =============

def make_tuner_builder(input_dim, num_classes=3):
    """
    Factory function yang return builder function untuk Keras Tuner.

    Harus dipanggil dengan input_dim spesifik karena Keras Tuner
    memerlukan fungsi yang hanya menerima parameter `hp`.
    """
    def build_model_tuner(hp):
        model = Sequential()
        model.add(Input(shape=(input_dim,)))

        for i in range(hp.Int('num_layers', 1, 4)):
            model.add(Dense(
                units=hp.Choice(f'units{i}', values=[64, 128, 256, 512]),
                activation='relu'
            ))
            model.add(BatchNormalization())
            model.add(Dropout(hp.Float('dropout', 0.2, 0.5, step=0.1)))

        model.add(Dense(num_classes, activation='softmax'))

        model.compile(
            optimizer=Adam(
                learning_rate=hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])
            ),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        return model

    return build_model_tuner


# ============= SKLEARN MODELS =============

def create_naive_bayes():
    """Multinomial Naive Bayes."""
    return MultinomialNB()


def create_svm(random_state=42):
    """SVM dengan kernel linear."""
    return SVC(kernel='linear', probability=True, random_state=random_state)


def create_random_forest(random_state=42):
    """Random Forest Classifier."""
    return RandomForestClassifier(n_estimators=100, random_state=random_state)


def create_logistic_regression(random_state=42):
    """Logistic Regression."""
    return LogisticRegression(max_iter=500, random_state=random_state)


# ============= MODEL REGISTRY =============

MODEL_CONFIGS = {
    'MLP Baseline':        'mlp_baseline',
    'MLP Advanced':        'mlp_advance',
    'MLP Keras Tuner':     'mlp_tuner',
    'Naive Bayes':         'naive_bayes',
    'SVM':                 'svm',
    'Random Forest':       'random_forest',
    'Logistic Regression': 'logistic_regression',
}
