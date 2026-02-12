from tensorflow import keras



def CNNmodel(model):
    # Define the input shape of the frames (assumed to be the same for all frames)
    input_shape = (224, 224, 3)

    # Define the number of output nodes (2 for pitch and yaw)
    num_outputs = 2

    # Define the CNN model
    model = keras.Sequential([
        keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape),
        keras.layers.MaxPooling2D(pool_size=(2, 2)),
        keras.layers.Conv2D(64, kernel_size=(3, 3), activation='relu'),
        keras.layers.MaxPooling2D(pool_size=(2, 2)),
        keras.layers.Conv2D(128, kernel_size=(3, 3), activation='relu'),
        keras.layers.MaxPooling2D(pool_size=(2, 2)),
        keras.layers.Flatten(),
        keras.layers.Dense(256, activation='relu'),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(num_outputs, activation='linear')
    ])

    # Compile the model with the mean squared error loss and the Adam optimizer
    model.compile(loss='mse', optimizer='adam', metrics=['mae'])
    
    model.summary()
    
    #loss='mse', optimizer='adam', metrics=['mae']
    # model.compile(
    #     optimizer=keras.optimizers.RMSprop(),  # Optimizer
    #     # Loss function to minimize
    #     loss=keras.losses.SparseCategoricalCrossentropy(),
    #     # List of metrics to monitor
    #     metrics=[keras.metrics.SparseCategoricalAccuracy()],
    # )

    return model

#--------------------------------------------------------------------------------------------------------------------------------------#

def VGG16(model):
    # Define the input shape of the frames (assumed to be the same for all frames)
    input_shape = (224, 224, 3)

    # Define the number of output nodes (2 for pitch and yaw)
    num_outputs = 2

    # Define the CNN model
    model = keras.Sequential([
        keras.layers.Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same', input_shape=input_shape),
        keras.layers.Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same'),
        keras.layers.MaxPooling2D(pool_size=(2, 2)),
        keras.layers.Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same'),
        keras.layers.Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same'),
        keras.layers.MaxPooling2D(pool_size=(2, 2)),
        keras.layers.Conv2D(256, kernel_size=(3, 3), activation='relu', padding='same'),
        keras.layers.Conv2D(256, kernel_size=(3, 3), activation='relu', padding='same'),
        keras.layers.Conv2D(256, kernel_size=(3, 3), activation='relu', padding='same'),
        keras.layers.MaxPooling2D(pool_size=(2, 2)),
        keras.layers.Conv2D(512, kernel_size=(3, 3), activation='relu', padding='same'),
        keras.layers.Conv2D(512, kernel_size=(3, 3), activation='relu', padding='same'),
        keras.layers.Conv2D(512, kernel_size=(3, 3), activation='relu', padding='same'),
        keras.layers.MaxPooling2D(pool_size=(2, 2)),
        keras.layers.Conv2D(512, kernel_size=(3, 3), activation='relu', padding='same'),
        keras.layers.Conv2D(512, kernel_size=(3, 3), activation='relu', padding='same'),
        keras.layers.Conv2D(512, kernel_size=(3, 3), activation='relu', padding='same'),
        keras.layers.MaxPooling2D(pool_size=(2, 2)),
        keras.layers.Flatten(),
        keras.layers.Dense(1024, activation='relu'),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(1024, activation='relu'),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(num_outputs, activation='linear')
    ])

    # Compile the model with the mean squared error loss and the Adam optimizer
    model.compile(loss='mse', optimizer='adam', metrics=['mse'])

    return model


#--------------------------------------------------------------------------------------------------------------------------------------#

import tensorflow as tf
from sklearn.model_selection import train_test_split

def train(frames_array_labeled, angles_array_labeled, model):
    # frees up any memory that was used during last session
    tf.keras.backend.clear_session()

    # Split the labeled video data into training and validation sets
    x_train, x_val, y_train, y_val = train_test_split(frames_array_labeled, angles_array_labeled, test_size=0.2, random_state=1)

    # Use GPU
    with tf.device('/GPU:0'):
        # Train the model on the training set using the fit() function
        history = model.fit(x_train, y_train, batch_size=64, epochs=100, validation_data=(x_val, y_val))

        # Evaluate the performance of the model on the validation set using the evaluate() function
        loss = model.evaluate(x_val, y_val)
        print('Validation Loss:', loss)
