import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam, RMSprop, SGD, Adagrad, AdamW
from tensorflow.keras.regularizers import l1, l2
from tensorflow.keras.callbacks import EarlyStopping

def build_model(num_classes, optimizer_type="adam_l1"):
    if optimizer_type == "adam_l1":
        model = Sequential([
            Dense(64, activation='relu', kernel_regularizer=l1(0.001)),
            Dropout(0.1),
            Dense(48, activation='relu', kernel_regularizer=l1(0.001)),
            Dropout(0.1),
            Dense(32, activation='relu', kernel_regularizer=l1(0.001)),
            Dense(num_classes, activation='softmax')
        ])
        model.compile(optimizer=Adam(learning_rate=0.0017), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    elif optimizer_type == "rmsprop_l2":
        model = Sequential([
            Dense(64, activation='relu', kernel_regularizer=l2(0.005)),
            Dropout(0.3),
            Dense(48, activation='relu', kernel_regularizer=l2(0.005)),
            Dropout(0.3),
            Dense(32, activation='relu', kernel_regularizer=l2(0.005)),
            Dense(num_classes, activation='softmax')
        ])
        model.compile(optimizer=RMSprop(learning_rate=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    elif optimizer_type == "sgd_l1":
        model = Sequential([
            Dense(64, activation='relu', kernel_regularizer=l1(0.005)),
            Dense(48, activation='relu', kernel_regularizer=l1(0.005)),
            Dense(32, activation='relu', kernel_regularizer=l1(0.005)),
            Dense(num_classes, activation='softmax')
        ])
        model.compile(optimizer=SGD(learning_rate=0.005, momentum=0.8), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    elif optimizer_type == "adagrad_bn":
        model = Sequential([
            Dense(64, activation='relu'),
            BatchNormalization(),
            Dropout(0.2),
            Dense(48, activation='relu'),
            BatchNormalization(),
            Dense(32, activation='relu'),
            BatchNormalization(),
            Dense(num_classes, activation='softmax')
        ])
        model.compile(optimizer=Adagrad(learning_rate=0.01), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    elif optimizer_type == "adamw_clip":
        model = Sequential([
            Dense(64, activation='relu', kernel_regularizer=l2(0.01)),
            Dropout(0.2),
            Dense(48, activation='relu', kernel_regularizer=l2(0.01)),
            Dropout(0.2),
            Dense(32, activation='relu', kernel_regularizer=l2(0.01)),
            Dense(num_classes, activation='softmax')
        ])
        model.compile(optimizer=AdamW(learning_rate=0.001, weight_decay=0.004, clipnorm=1.0),
                      loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def train_model(model, X_train, y_train, X_val, y_val, epochs=250, batch_size=32):
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_val, y_val), verbose=0,
                        callbacks=[EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)])
    return model, history