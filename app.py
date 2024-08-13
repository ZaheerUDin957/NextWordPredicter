import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Embedding, Input, Layer
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model

# Display the original training corpus
st.title("Text Generation using LSTM with Attention")
st.subheader("Original Training Corpus")
sentences = [
    "Machine learning algorithms are powerful tools.",
    "I enjoy exploring new algorithms.",
    "Learning about AI is captivating.",
    "Deep learning models can be complex.",
    "I love understanding how neural networks work.",
]

for i, sentence in enumerate(sentences, 1):
    st.write(f"{i}. {sentence}")

# Tokenization and Preprocessing
st.subheader("Tokenization and Preprocessing")

tokenizer = Tokenizer()
tokenizer.fit_on_texts(sentences)
total_words = len(tokenizer.word_index) + 1

st.write("Total words:", total_words)
st.write("Word Index:", tokenizer.word_index)

input_sequences = []
for line in sentences:
    token_list = tokenizer.texts_to_sequences([line])[0]
    for i in range(1, len(token_list)):
        n_gram_sequence = token_list[:i + 1]
        input_sequences.append(n_gram_sequence)

max_sequence_len = max([len(x) for x in input_sequences])
input_sequences = np.array(pad_sequences(input_sequences, maxlen=max_sequence_len, padding='pre'))

X = input_sequences[:, :-1]
y = input_sequences[:, -1]
y = tf.keras.utils.to_categorical(y, num_classes=total_words)

# Display Preprocessed Data
st.subheader("Preprocessed Data")
st.write("Input Sequences (X):")
st.write(X)

st.write("Labels (y):")
st.write(y)

# Define the Attention Layer
class AttentionLayer(Layer):
    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(shape=(input_shape[-1], input_shape[-1]), initializer='glorot_uniform', trainable=True)
        self.b = self.add_weight(shape=(input_shape[-1],), initializer='zeros', trainable=True)
        self.u = self.add_weight(shape=(input_shape[-1], 1), initializer='glorot_uniform', trainable=True)
        super(AttentionLayer, self).build(input_shape)

    def call(self, x):
        u_it = tf.tanh(tf.tensordot(x, self.W, axes=1) + self.b)
        a_it = tf.nn.softmax(tf.tensordot(u_it, self.u, axes=1), axis=1)
        return tf.reduce_sum(a_it * x, axis=1)

# Define the Model
embedding_dim = 128
lstm_units = 150

input = Input(shape=(max_sequence_len - 1,))
x = Embedding(total_words, embedding_dim, input_length=max_sequence_len - 1)(input)
lstm_output = LSTM(lstm_units, return_sequences=True)(x)

attention_output = AttentionLayer()(lstm_output)
output = Dense(total_words, activation='softmax')(attention_output)

model = Model(inputs=input, outputs=output)

# Compile the Model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the Model
st.subheader("Training the Model")
st.write("Training... (This may take a while depending on your system)")
history = model.fit(X, y, epochs=50, verbose=1)
st.write("Training completed!")

# Prediction Function
def predict_next_word(seed_text, next_words=2):
    for _ in range(next_words):
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        token_list = pad_sequences([token_list], maxlen=max_sequence_len - 1, padding='pre')
        predicted = model.predict(token_list, verbose=0)
        predicted_word_index = np.argmax(predicted, axis=-1)[0]
        predicted_word = tokenizer.index_word[predicted_word_index]
        seed_text += " " + predicted_word
    return seed_text

# Input Box for User Input
st.subheader("Generate Text")
user_input = st.text_input("Enter a seed text:", "Machine Learning can improve")
num_words = st.slider("Number of words to predict:", 1, 10, 2)

if st.button("Generate"):
    output_text = predict_next_word(user_input, next_words=num_words)
    st.write("Generated Text:", output_text)
