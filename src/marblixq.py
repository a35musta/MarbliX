import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import pairwise_distances
from matplotlib import pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Flatten, Lambda
import tensorflow.keras.backend as K


def load_embeddings(samples_dir, image_dir, seq_dir):
    """
    Load image and sequence embeddings based on file names in a CSV file.

    Args:
        samples_dir (str): Path to samples csv file with 'file_name' and 'label' columns.
        image_dir (str): Directory containing image embedding .npy files.
        seq_dir (str): Directory containing sequence embedding .npy files.

    Returns:
        tuple: (image_embeddings, seq_embeddings, labels)
    """
    df = pd.read_csv(samples_dir)
    file_names = df['file_name'].tolist()
    labels = df['label'].tolist()

    image_embeddings, seq_embeddings = [], []

    for file_name in file_names:
        try:
            image_features = np.load(os.path.join(image_dir, file_name + '-features.npy'))
            seq_features = np.load(os.path.join(seq_dir, file_name + '-features.npy'))

            image_embeddings.append(image_features)
            seq_embeddings.append(seq_features)
        except FileNotFoundError as e:
            print(f"Error loading embeddings for {file_name}: {e}")

    return np.array(image_embeddings), np.array(seq_embeddings), np.array(labels)


def split_and_normalize(image_embeddings, seq_embeddings, labels):
    """
    Split data into training/testing sets and normalize.

    Returns:
        Tuple: normalized embeddings and labels split for train and test
    """
    img_train, img_test, seq_train, seq_test, y_train, y_test = train_test_split(
        image_embeddings, seq_embeddings, labels, test_size=0.5, random_state=3)

    scaler = MinMaxScaler()
    train_concat = np.concatenate((img_train, seq_train), axis=1)
    test_concat = np.concatenate((img_test, seq_test), axis=1)

    train_norm = scaler.fit_transform(train_concat)
    test_norm = scaler.transform(test_concat)

    split = len(img_train[0])
    return (train_norm[:, :split], test_norm[:, :split],
            train_norm[:, split:], test_norm[:, split:],
            y_train, y_test)


def build_autoencoder(input_size):
    """
    Build a symmetric autoencoder model.

    Args:
        input_size (int): Size of the input vector.

    Returns:
        Keras Model: Autoencoder model.
    """
    input_layer = layers.Input(shape=(input_size,), name='input')
    x = layers.Dense(512, activation='linear')(input_layer)
    x = layers.Dense(256, activation='linear')(x)
    encoded = layers.Dense(128, activation='linear', name='bottleneck')(x)
    x = layers.Dense(256, activation='linear')(encoded)
    x = layers.Dense(512, activation='linear')(x)
    output = layers.Dense(input_size, activation='linear')(x)

    model = models.Model(inputs=input_layer, outputs=output)
    model.compile(optimizer=Adam(learning_rate=0.0001), loss='mse', metrics=['cosine_similarity'])
    return model


def train_autoencoder(model, x_train, y_train, x_val=None, y_val=None):
    """
    Train an autoencoder.

    Returns:
        History: Training history object.
    """
    return model.fit(
        x_train, y_train,
        epochs=30,
        batch_size=32,
        shuffle=True,
        validation_data=(x_val, y_val) if x_val is not None else None
    )


def extract_bottlenecks(model, data, layer_name='bottleneck'):
    """
    Extract encoded features from bottleneck layer.

    Args:
        model (Model): Trained autoencoder.
        data (np.ndarray): Input data.

    Returns:
        np.ndarray: Bottleneck features.
    """
    bottleneck_model = Model(inputs=model.input, outputs=model.get_layer(layer_name).output)
    return bottleneck_model.predict(data)


def generate_hard_triplets(vectors, labels):
    """
    Generate hard triplets using farthest positive and closest negative.

    Returns:
        np.ndarray: Array of triplets.
    """
    triplets = []
    dist_matrix = pairwise_distances(vectors, metric='euclidean')
    labels = np.array(labels)

    for i in range(len(vectors)):
        pos_idx = np.where(labels == labels[i])[0]
        neg_idx = np.where(labels != labels[i])[0]

        far_pos = pos_idx[np.argmax(dist_matrix[i, pos_idx])]
        close_neg = neg_idx[np.argmin(dist_matrix[i, neg_idx])]

        triplets.append([vectors[i], vectors[far_pos], vectors[close_neg]])
    return np.array(triplets)


def triplet_loss(_, y_pred):
    """
    Custom triplet loss function using Euclidean distance.
    """
    anchor, positive, negative = tf.split(y_pred, num_or_size_splits=3, axis=1)
    pos_dist = tf.reduce_sum(tf.square(anchor - positive), axis=1)
    neg_dist = tf.reduce_sum(tf.square(anchor - negative), axis=1)
    margin = 0.3
    return tf.reduce_mean(tf.maximum(pos_dist - neg_dist + margin, 0.0))


def build_triplet_model(img_dim, seq_dim):
    """
    Build multimodal triplet network model.

    Returns:
        Model: Compiled triplet network.
    """
    def outer(x): return tf.matmul(tf.expand_dims(x[0], -1), tf.expand_dims(x[1], 1))
    def binarize(x): return tf.where(x < 0, 0, 1)

    def feature_branch(name_prefix, img_input, seq_input):
        merged = Lambda(outer, name=f'{name_prefix}_outer')([img_input, seq_input])
        x = Flatten()(merged)
        x = Dense(1024, activation='tanh')(x)
        x = Dense(256, activation='tanh')(x)
        x = Dense(64, activation='tanh')(x)
        return x

    # Inputs
    a_img = Input((img_dim,), name='a_img')
    p_img = Input((img_dim,), name='p_img')
    n_img = Input((img_dim,), name='n_img')
    a_seq = Input((seq_dim,), name='a_seq')
    p_seq = Input((seq_dim,), name='p_seq')
    n_seq = Input((seq_dim,), name='n_seq')

    # Outputs
    a_out = feature_branch('a', a_img, a_seq)
    p_out = feature_branch('p', p_img, p_seq)
    n_out = feature_branch('n', n_img, n_seq)

    merged = Lambda(lambda x: K.concatenate(x, axis=1), name='concat')([a_out, p_out, n_out])

    model = Model(inputs=[a_img, p_img, n_img, a_seq, p_seq, n_seq], outputs=merged)
    model.compile(optimizer=Adam(learning_rate=0.00001), loss=triplet_loss)
    return model


def extract_features(model, img_data, seq_data, binarize=False):
    """
    Extract features from trained model.

    Args:
        model (Model): Trained model.
        img_data, seq_data (np.ndarray): Input data.

    Returns:
        np.ndarray: Extracted features.
    """
    outputs = model.output
    if binarize:
        outputs = Lambda(lambda x: tf.where(x < 0, 0, 1))(outputs)
    extractor = Model(inputs=model.input, outputs=outputs)

    features = [extractor.predict([img.reshape(1, -1), seq.reshape(1, -1)], verbose=0)
                for img, seq in zip(img_data, seq_data)]
    return np.squeeze(np.array(features))


def plot_binary_matrix(matrix, index=0):
    """
    Plot a binary matrix (8x8).

    Args:
        matrix (np.ndarray): 3D array of binary matrices.
    """
    plt.imshow(matrix[index], cmap='binary', extent=[0, 8, 0, 8])
    plt.title(f'Binary Matrix at Index {index}')
    plt.show()


def main():
    samples_dir = '/path/to/samples/file.csv'
    image_path = '/path/to/image/embeddings'
    seq_path = '/path/to/sequence/embeddings'

    # Load data
    image_emb, seq_emb, labels = load_embeddings(samples_dir, image_path, seq_path)
    img_train, img_test, seq_train, seq_test, y_train, y_test = split_and_normalize(image_emb, seq_emb, labels)

    # Autoencoders
    ae_img = build_autoencoder(input_size=768)
    ae_img.fit(img_train, seq_train, epochs=30, batch_size=32, validation_data=(img_test, seq_test))

    ae_seq = build_autoencoder(input_size=768)
    ae_seq.fit(seq_train, img_train, epochs=30, batch_size=32)

    # Extract bottlenecks
    encoded_img_train = extract_bottlenecks(ae_img, img_train)
    encoded_img_test = extract_bottlenecks(ae_img, img_test)
    encoded_seq_train = extract_bottlenecks(ae_seq, seq_train)
    encoded_seq_test = extract_bottlenecks(ae_seq, seq_test)

    # Triplet training
    concat_train = np.hstack((encoded_img_train, encoded_seq_train))
    triplets = generate_hard_triplets(concat_train, y_train)

    model = build_triplet_model(encoded_img_train.shape[1], encoded_seq_train.shape[1])
    model.fit(
        [triplets[:, 0][:, :768], triplets[:, 1][:, :768], triplets[:, 2][:, :768],
         triplets[:, 0][:, 768:], triplets[:, 1][:, 768:], triplets[:, 2][:, 768:]],
        np.zeros((triplets.shape[0], 1)),
        epochs=60, batch_size=32)

    # Extract features
    multimodal_features = extract_features(Model(inputs=[model.input[0], model.input[3]], outputs=model.layers[-2].output),
                                     encoded_img_test, encoded_seq_test)
    multimodal_binary_features = extract_features(Model(inputs=[model.input[0], model.input[3]], outputs=model.layers[-1].output),
                                       encoded_img_test, encoded_seq_test, binarize=True)

    binary_matrices = np.reshape(multimodal_binary_features, (-1, 8, 8))
    plot_binary_matrix(binary_matrices, index=13)


if __name__ == "__main__":
    main()
