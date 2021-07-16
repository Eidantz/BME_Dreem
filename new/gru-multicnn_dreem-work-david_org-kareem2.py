import h5py  # Read and write HDF5 files from Python
import matplotlib
import matplotlib.pyplot as plt
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import tensorflow as tf
from keras import Input, Model
from scipy.interpolate import interp1d
from scipy.signal import stft
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Lambda, BatchNormalization, MaxPooling1D, Dense, Conv1D, Dropout, GRU, \
    TimeDistributed, \
    GlobalAveragePooling1D
from tensorflow.keras.layers import Layer
from tensorflow.keras.layers import concatenate
from tensorflow.keras.optimizers import Adam

data_path = r"C:\Users\eidan\Documents\BME_Dreem\new"
file_xtrain = data_path + r"/X_train.h5"
file_xtest = data_path + r"/X_test.h5"
file_ytrain = data_path + r"/y_train.csv"


def normalize_data(eeg_array):
    """normalize signal between 0 and 1"""
    normalized_array = np.clip(eeg_array, -150, 150)
    normalized_array = normalized_array / 150

    return normalized_array


def stft_preprocessing(data, mean=None, var=None):
    """Transform the signal in input in his STFT version, add a dimension"""

    zxx = stft(data, fs=50, nperseg=128, nfft=128, noverlap=64, axis=1)[2]
    zxx = np.log(np.abs(zxx))
    cliped = np.clip(zxx, -20, 20)
    cliped = np.swapaxes(cliped, 1, 2)
    cliped = np.swapaxes(cliped, 2, 3)

    newdata = cliped
    print(newdata.shape)
    if mean is None:
        mean = newdata.mean()

    newdata = newdata - mean

    if var is None:
        var = newdata.var()

    newdata = newdata / var

    return newdata, mean, var


def coordinate_preprocessing(data, mean=None, var=None):
    if mean is None:
        mean = data.mean(axis=(0, 1))
    else:
        data = data - mean

    if var is None:
        var = data.var(axis=(0, 1))

    else:
        data = data / var

    data = np.diff(data, axis=1)
    return data, mean, var


def positional_embeding(data):
    data = np.array(data)

    pos = np.zeros([data.shape[0], 6])

    pos[:, 0] = data / 1200

    angle = [30, 60, 90, 120, 150]
    for i in range(5):
        pos[:, i + 1] = np.cos((data * np.pi) / angle[i])

    return pos


def split_data(input_list):
    with h5py.File(file_xtrain, "r") as fi:
        if len(input_list) == 1:
            x_data = fi[input_list[0]][()]
        else:
            x_data = np.zeros([24688, 1500, len(input_list)], dtype=np.float64)
            for i in range(0, len(input_list)):
                if 'x' in input_list[i] or 'y' in input_list[i] or 'z' in input_list[i]:
                    f1 = interp1d(np.arange(0, 300), fi[input_list[i]][()], axis=1)
                    xnew = np.linspace(0, 30, num=1500)
                    x_data[0:24688, 0:1500, i] = f1(xnew)
                else:
                    x_data[0:24688, 0:1500, i] = fi[input_list[i]][()]

        y_data_org = pd.read_csv(file_ytrain)['sleep_stage'].to_numpy()
        metadata = fi["index_window"][()]

        mask_eeg = [0, 1, 2, 3]

        x_eeg_raw_data = normalize_data(x_data[:, :, mask_eeg])
        x_stft_data = np.copy(x_eeg_raw_data)
        print('kill them allllll')
        for subjects in range(0, x_stft_data.shape[0]):
            for channels in range(0, x_stft_data.shape[2]):
                if np.random.uniform() < 0.1:
                    x_stft_data[subjects, :, channels] = x_stft_data[subjects, :, channels] * 0
        x_eeg_raw_data = np.copy(x_stft_data)
        print('JK just 10% of them')

        x_stft_data, mean_eeg, var_eeg = stft_preprocessing(x_stft_data)

        x_emb_pos_data = positional_embeding(metadata)
        x_emb_pos_data = np.append(x_emb_pos_data[0:-1, :], x_emb_pos_data[1:, :], axis=1)
        x_emb_pos_data = x_emb_pos_data.reshape(24687, 2, 6)
        x_stft_data = np.append(x_stft_data[0:-1, :, :, :], x_stft_data[1:, :, :, :], axis=1)
        x_stft_data = x_stft_data.reshape(24687, 2, 4, 25, 65)

        x_eeg_raw_data = np.append(x_eeg_raw_data[0:-1, :, :], x_eeg_raw_data[1:, :, :], axis=1)

        y_data_org = y_data_org[1:]

        print('data ready')
    return x_eeg_raw_data, x_stft_data, x_emb_pos_data, y_data_org


input_signals_list = ['eeg_4', 'eeg_5', 'eeg_6', 'eeg_7', 'x', 'y', 'z']

x_eeg_raw, x_stft, x_emb_pos, y_data = split_data(input_signals_list)

# In[3]:


print('y_data.shape = ', y_data.shape)
print('x_eeg.shape = ', x_stft.shape)
print('x_eeg_normalized.shape = ', x_eeg_raw.shape)
print('x_meta.shape = ', x_emb_pos.shape)
shapes = np.array([y_data.shape[0], x_stft.shape[0], x_eeg_raw.shape[0], x_emb_pos.shape[0]])
if np.sum(np.abs(np.diff(shapes))) != 0:
    print("shape error")
optimizer = Adam(learning_rate=0.001)


def custom_loss(ytrue, ypred):
    # this is SparseCategoricalCrossentropy with weights
    scce = tf.keras.losses.SparseCategoricalCrossentropy()

    weight = tf.constant([[0.94355903, 2.70683738, 0.96141876, 0.55233263, 0.62986756]])
    #     weight = tf.constant([[1.92961025, 2.68815953, 0.96311429, 1.17516681, 1.21891314]])

    new_y = tf.expand_dims(ypred, axis=1)

    new_weight = tf.matmul(weight, new_y, transpose_b=True)

    score = scce(ytrue, ypred, sample_weight=new_weight)

    return score


# Define the K-fold Cross Validator if we want to use sklearn
kfold = KFold(n_splits=30, shuffle=True)
fold_no = 1


# In[8]:


class Attention(Layer):

    def __init__(self, input_dim, context_size=25):
        super(Attention, self).__init__()  # constructor of Base Layer class
        init = tf.initializers.GlorotUniform()
        self.context_matrix = tf.Variable(init(shape=(context_size, input_dim)), trainable=True)
        self.context_bias = tf.Variable(init(shape=(1, context_size)), trainable=True)
        self.context_vector = tf.Variable(init(shape=(context_size, 1)), trainable=True)

    def get_config(self):
        config = super().get_config()
        config.update({
            'context_matrix': self.context_matrix,
            'context_bias': self.context_bias,
            'context_vector': self.context_vector
        })
        return config

    def call(self, x, **kwargs):
        """
        x (tensor: batch_size,sequence length,input_dim):
        returns x (tensor: batch_size,input_dim),  attention_weights (tensor: batch_size,sequence_length)
        """
        batch_size, length, n_features = x.shape
        x_att = tf.reshape(x, [-1, n_features])
        u = tf.linalg.matmul(x_att, tf.transpose(self.context_matrix, perm=[1, 0])) + self.context_bias
        u = tf.nn.tanh(u)
        uv = tf.linalg.matmul(u, self.context_vector)
        uv = tf.reshape(uv, [-1, length])
        alpha = tf.nn.softmax(uv, axis=1)
        alpha = tf.expand_dims(alpha, axis=-1)
        x_out = alpha * x
        x_out = tf.math.reduce_sum(x_out, axis=1)
        #         if np.random.uniform() < 0.2:
        #             x_out = x_out * 0

        return x_out


def custom_Kfold(n):
    # makes sure we are training and validated on different subjects
    with h5py.File(file_xtrain, "r") as fi:
        x_metadata = fi['index'][()]

    x_metadata = x_metadata[1:]
    indice = []

    for i in range(n):
        train_indices, val_indices = np.argwhere(x_metadata % n != 0 + i), np.argwhere(x_metadata % n == 0 + i)
        np.random.shuffle(train_indices)
        np.random.shuffle(val_indices)
        indice.append({"train": np.squeeze(train_indices), "validation": np.squeeze(val_indices)})

    return indice


dic_kfold = custom_Kfold(5)


def backend_reshape(x):
    # reshape with the batch size
    return tf.keras.backend.reshape(x, (-1, n_channels, spectrogram_length, filter_size))


hidden = None
# filter and channel redudcution
C = 4
filter_reduc = 40
# attention layer contaxt sizes
att_context = 20

# number of units GRU
m1 = 128
m2 = 128
# for dropouts
p1 = 0.2
p2 = 0.2
_, temporal_context, n_channels, spectrogram_length, filter_size = x_stft.shape
number_channels = n_channels
inputA = Input(shape=(temporal_context, n_channels, spectrogram_length, filter_size))
inputC = Input(shape=(2, 6))
inputD = Input(shape=134)
inputH = Input(shape=(3000, number_channels))

k = Lambda(backend_reshape)(inputA)
#  filter redudcution
k = TimeDistributed(Dense(filter_reduc))(k)
k = tf.transpose(k, perm=[0, 3, 2, 1])
#  Channel redudcution
k = TimeDistributed(Dense(C))(k)
k = tf.transpose(k, perm=[0, 3, 2, 1])
k = tf.keras.layers.Dropout(p1)(k)

k = tf.keras.layers.Reshape((-1, spectrogram_length))(k)
k = tf.transpose(k, perm=[0, 2, 1])
# bidirectional GRU
k = tf.keras.layers.Bidirectional(GRU(m1, return_sequences=True, go_backwards=False))(k)
k = tf.keras.layers.Dropout(p1)(k)
k = Attention(m1 * 2, att_context)(k)
k = tf.reshape(k, [-1, temporal_context, 2 * m1])
model_A = Model(inputs=inputA, outputs=k)

# embedded position
m = Dense(6, activation=tf.nn.relu)(inputC)
m = Dense(6, activation=tf.nn.relu)(m)
m = Dense(6, activation=tf.nn.relu)(m)
m = Dense(6, activation=tf.nn.relu)(m)
m = Dense(6, activation=tf.nn.relu)(m)
model_C = Model(inputs=inputC, outputs=m)

# SkipGru
combined_input = concatenate([model_A.output, model_C.output])
combined = tf.keras.layers.Reshape((combined_input.shape[1], -1))(combined_input)
x_residual = Dense(m1 * 2)(combined)
x_lstm = tf.keras.layers.Bidirectional(GRU(m2, return_state=True, return_sequences=True))(combined,
                                                                                          initial_state=hidden)  # 1
hidden = x_lstm[1:]
x_lstm = x_lstm[0]
# x_lstm = tf.transpose(x_lstm,perm=[0,2,1])
x_cat = (x_residual + x_lstm) / 2
combined = Dropout(p2)(x_cat)
x_residual = Dense(m1 * 2)(combined)
x_lstm = tf.keras.layers.Bidirectional(GRU(m2, return_state=True, return_sequences=True))(combined,
                                                                                          initial_state=hidden)  # 2
hidden = x_lstm[1:]
x_lstm = x_lstm[0]
x_cat = (x_residual + x_lstm) / 2
combined = Dropout(p2)(x_cat)
combined = GlobalAveragePooling1D()(combined)

combined = Model(inputs=[model_A.input, model_C.input], outputs=combined)

# raw data cnn
raws = BatchNormalization()(inputH)
raws = Conv1D(filters=60, kernel_size=20, padding='SAME', activation=tf.nn.leaky_relu)(raws)
raws = MaxPooling1D(pool_size=6)(raws)
raws = Conv1D(filters=80, kernel_size=15, padding='SAME', activation=tf.nn.leaky_relu)(raws)
raws = MaxPooling1D(pool_size=6)(raws)
raws = Dropout(0.25)(raws)
raws = Conv1D(filters=80, kernel_size=10, padding='SAME', activation=tf.nn.leaky_relu)(raws)
raws = MaxPooling1D(pool_size=6)(raws)
raws = Conv1D(filters=80, kernel_size=10, padding='SAME', activation=tf.nn.leaky_relu)(raws)
raws = MaxPooling1D(pool_size=4)(raws)
raws = Dense(128, activation=tf.nn.leaky_relu)(raws)
z = GlobalAveragePooling1D()(raws)

z = Model(inputs=inputH, outputs=z)

last_layer = concatenate([combined.output, z.output])

v = Dense(5, activation="softmax")(last_layer)

model = Model(inputs=[model_A.input, model_C.input, z.input], outputs=v)
print(model.summary())

# use one subject out CV
dic_kfold = custom_Kfold(30)
#run model with Kfold
for dic in dic_kfold:
    train, test = dic.values()
    # for train, test in kfold.split(x_eeg, y_data):
    if fold_no == 16:
        print(test.shape)
        print(f'Training for fold {fold_no} ...')
        model.compile(loss=custom_loss,
                      optimizer=optimizer,
                      metrics=['accuracy'])
        history = model.fit([x_stft[train], x_emb_pos[train], x_eeg_raw[train]], y_data[train],
                            batch_size=32,
                            validation_data=([x_stft[test], x_emb_pos[test], x_eeg_raw[test]], y_data[test])
                            , callbacks=[EarlyStopping(monitor='val_loss', mode='min', verbose=0, patience=15)],
                            epochs=200, verbose=1)
        print('------------------------------------------------------------------------')
        print(f'DONE training for fold {fold_no} ...')
    # Increase fold number
    fold_no = fold_no + 1

if history is not None:
    # summarize history for accuracy
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()


def test_data(input_list):
    # get the test data the same as the traning data
    with h5py.File(file_xtest, "r") as fi:
        index = fi['index_absolute'][()]
        metadata = fi["index_window"][()]
        if len(input_list) == 1:
            x_data = fi[input_list[0]][()]
        else:
            x_data = np.zeros([24980, 1500, len(input_list)], dtype=np.float64)
            for i in range(0, len(input_list)):
                if 'x' in input_list[i] or 'y' in input_list[i] or 'z' in input_list[i]:
                    f1 = interp1d(np.arange(0, 300), fi[input_list[i]][()], axis=1)
                    xnew = np.linspace(0, 30, num=1500)
                    x_data[0:24980, 0:1500, i] = f1(xnew)
                else:
                    x_data[0:24980, 0:1500, i] = fi[input_list[i]][()]

    # EEG extraction
    mask_eeg = [0, 1, 2, 3]

    x_eeg_normalized = normalize_data(x_data[:, :, mask_eeg])
    x_eeg = np.copy(x_eeg_normalized)
    x_eeg, mean_eeg, var_eeg = stft_preprocessing(x_eeg)
    x_meta = positional_embeding(metadata)
    x_meta = np.append(x_meta[0:-1, :], x_meta[1:, :], axis=1)
    x_meta = x_meta.reshape(24979, 2, 6)
    x_eeg = np.append(x_eeg[0:-1, :, :, :], x_eeg[1:, :, :, :], axis=1)
    x_eeg = x_eeg.reshape(24979, 2, 4, 25, 65)
    x_eeg_normalized = np.append(x_eeg_normalized[0:-1, :, :], x_eeg_normalized[1:, :, :], axis=1)

    print('data ready')

    return x_eeg_normalized, x_eeg, x_meta, index


x_eeg_raw_test, x_stft_test, x_emb_pos_test, index = test_data(input_signals_list)
# predict the test data and save it as CSV
y_test = model.predict([x_stft_test, x_emb_pos_test, x_eeg_raw_test])
y_test = np.argmax(y_test, axis=1)
y_test = np.append(y_test[0], y_test)
if index.shape == y_test.shape:
    df = pd.DataFrame(data={'index': index, 'sleep_stage': y_test})
    df.to_csv('model_num_' + str(fold_no) + '_our_result.csv', index=False)
    print('you can save the results')
else:
    print('there is an error in the shape of y_test')


def heatmap(data, row_labels, col_labels, ax=None,
            cbar_kw=None, cbarlabel="", **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Parameters
    ----------
    data
        A 2D numpy array of shape (N, M).
    row_labels
        A list or array of length N with the labels for the rows.
    col_labels
        A list or array of length M with the labels for the columns.
    ax
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
        not provided, use current axes or create a new one.  Optional.
    cbar_kw
        A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
    cbarlabel
        The label for the colorbar.  Optional.
    **kwargs
        All other arguments are forwarded to `imshow`.
    """

    if cbar_kw is None:
        cbar_kw = {}
    if not ax:
        ax = plt.gca()

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    # We want to show all ticks...
    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_yticks(np.arange(data.shape[0]))
    # ... and label them with the respective list entries.
    ax.set_xticklabels(col_labels)
    ax.set_yticklabels(row_labels)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=-30, ha="right",
             rotation_mode="anchor")

    # Turn spines off and create white grid.
    for edge, spine in ax.spines.items():
        spine.set_visible(False)

    ax.set_xticks(np.arange(data.shape[1] + 1) - .5, minor=True)
    ax.set_yticks(np.arange(data.shape[0] + 1) - .5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar


def annotate_heatmap(im, data=None, valfmt="{x:.2f}",
                     textcolors=None,
                     threshold=None, **textkw):
    """
    A function to annotate a heatmap.

    Parameters
    ----------
    im
        The AxesImage to be labeled.
    data
        Data used to annotate.  If None, the image's data is used.  Optional.
    valfmt
        The format of the annotations inside the heatmap.  This should either
        use the string format method, e.g. "$ {x:.2f}", or be a
        `matplotlib.ticker.Formatter`.  Optional.
    textcolors
        A list or array of two color specifications.  The first is used for
        values below a threshold, the second for those above.  Optional.
    threshold
        Value in data units according to which the colors from textcolors are
        applied.  If None (the default) uses the middle of the colormap as
        separation.  Optional.
    **kwargs
        All other arguments are forwarded to each call to `text` used to create
        the text labels.
    """

    if textcolors is None:
        textcolors = ["black", "white"]
    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max()) / 2.

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center",
              verticalalignment="center")
    kw.update(textkw)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
            text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
            texts.append(text)

    return texts


def plot_heatmap(y_true, y_pred):
    mat = confusion_matrix(y_true, y_pred)
    mat = mat / np.sum(mat, axis=0) * 100
    mat = mat.astype(np.int)

    fig, ax = plt.subplots()

    im, cbar = heatmap(mat, ['Wake', 'NREM1', 'NREM2', 'NREM3', 'REM'], ['Wake', 'NREM1', 'NREM2', 'NREM3', 'REM'],
                       ax=ax,
                       cmap="YlGn", cbarlabel="Pourcentage de classification")
    # texts = annotate_heatmap(im, valfmt="{x:} %")

    fig.tight_layout()
    plt.show()


y_estimate = model.predict([x_stft[test], x_emb_pos[test]])
y_estimate = np.argmax(y_estimate, axis=1)
# y_estimate=np.append(y_estimate[0],y_estimate)
plot_heatmap(y_data[test], y_estimate)

# In[ ]:


# alpha = tf.constant(np.arange(2, 32*65, dtype=np.int32), shape=[32, 65, 1])
# x_out= tf.constant(np.arange(2, 32*65*100, dtype=np.int32), shape=[32, 65, 100])


# In[ ]:


# init = tf.initializers.GlorotUniform()
# context_matrix = tf.Variable(init(shape=(25,50*2)),trainable=True)
# context_bias = tf.Variable(init(shape=(1,25)),trainable=True)
# context_vector = tf.Variable(init(shape=(25,1)),trainable=True)

# x = tf.random.uniform(shape=[64,65,100])
# batch_size, length, n_features = x.shape
# x_att = tf.reshape(x,[-1, n_features])
# print('x_att.shape', x_att.shape)
# u = tf.linalg.matmul(x_att,tf.transpose(context_matrix,perm=[1,0])) + context_bias
# u = tf.nn.tanh(u)
# print('u.shape',u.shape)
# uv = tf.linalg.matmul(u,context_vector)
# print('uv.shape',uv.shape)
# uv = tf.reshape(uv, [-1, length])
# print('uv.shape',uv.shape)
# alpha = tf.nn.softmax(uv,axis=1)
# print('alpha.shape',alpha.shape)
# alpha = tf.expand_dims(alpha, axis=-1)
# print('alpha.shape',alpha.shape)
# x_out = alpha * x
# print('x_out.shape',x_out.shape)
# x_out=tf.math.reduce_sum(x_out, axis=1)
# print('x_out.shape',x_out.shape)
# print(x_out)


# In[ ]:
