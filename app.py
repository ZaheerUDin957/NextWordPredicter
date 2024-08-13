import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Embedding, Input, Layer
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from collections import Counter
import seaborn as sns
from textblob import TextBlob
bright_colors = ['#FF1493', '#00FFFF', '#FF4500', '#7FFF00', '#9932CC', '#00CED1', '#FFD700']
# Display the original training corpus
st.title("Text Generation using LSTM with Attention")
st.subheader("Original Training Corpus")
sentences = [
    "Machine learning algorithms are powerful tools.",
    "I enjoy exploring new algorithms.",
    "Learning about AI is captivating.",
    "Deep learning models can be complex.",
    "I love understanding how neural networks work.",
    "The field of data science is evolving rapidly.",
    "I find artificial intelligence intriguing.",
    "Studying computer vision is exciting.",
    "Natural language processing is a fascinating domain.",
    "I enjoy coding in Python for data science projects.",
    "Building models is a creative process.",
    "I love experimenting with different machine learning techniques.",
    "The potential of AI to transform industries is amazing.",
    "I enjoy staying updated with the latest tech trends.",
    "Learning about reinforcement learning is interesting.",
    "I find predictive modeling to be very useful.",
    "I love solving problems with data analysis.",
    "Data preprocessing is a crucial step in machine learning.",
    "I enjoy reading research papers on deep learning.",
    "I find optimization techniques fascinating.",
    "Understanding algorithms helps in developing better solutions.",
    "I love the challenge of debugging code.",
    "Machine learning applications are diverse and impactful.",
    "I enjoy collaborating with others on tech projects.",
    "Learning new programming languages is fun.",
    "I love working with large datasets.",
    "I find feature engineering to be an art.",
    "Model evaluation is an essential part of machine learning.",
    "I enjoy attending tech conferences.",
    "Learning about big data technologies is exciting.",
    "I love experimenting with neural network architectures.",
    "I find the theory behind machine learning algorithms interesting.",
    "I enjoy visualizing data insights.",
    "Machine learning models can make accurate predictions.",
    "I love the creativity involved in data storytelling.",
    "I find unsupervised learning techniques intriguing.",
    "I enjoy automating tasks with AI.",
    "Learning about AI ethics is important.",
    "I love the problem-solving aspect of machine learning.",
    "I find cloud computing technologies fascinating.",
    "I enjoy using machine learning for real-world applications.",
    "I love experimenting with different data preprocessing techniques.",
    "I find transfer learning to be a powerful approach.",
    "I enjoy working on machine learning projects.",
    "Learning about data privacy is crucial.",
    "I love the innovation happening in the AI field.",
    "I find data visualization tools useful.",
    "I enjoy testing and validating machine learning models.",
    "I love discovering new machine learning applications.",
    "I find ensemble methods to be effective.",
    "I enjoy learning from data.",
    "Machine learning can provide valuable insights.",
    "I love the interdisciplinary nature of AI.",
    "I find recommendation systems interesting.",
    "I enjoy participating in hackathons.",
    "Learning about neural networks is fascinating.",
    "I love the potential of AI to solve complex problems.",
    "I find sentiment analysis intriguing.",
    "I enjoy implementing machine learning algorithms.",
    "I love the excitement of discovering patterns in data.",
    "I find time series analysis challenging.",
    "I enjoy exploring different types of data.",
    "Machine learning is transforming various industries.",
    "I love working on predictive analytics.",
    "I find anomaly detection to be useful.",
    "I enjoy studying the mathematics behind machine learning.",
    "I love the hands-on experience of building models.",
    "I find clustering techniques interesting.",
    "I enjoy exploring open-source machine learning libraries.",
    "Machine learning can automate complex tasks.",
    "I love the flexibility of machine learning models.",
    "I find computer vision applications fascinating.",
    "I enjoy solving real-world problems with AI.",
    "I love the continuous learning aspect of AI.",
    "I find reinforcement learning to be challenging.",
    "I enjoy experimenting with hyperparameter tuning.",
    "Machine learning can improve decision-making processes.",
    "I love the creativity involved in feature selection.",
    "I find generative models to be fascinating.",
    "I enjoy reading about the latest AI advancements.",
    "Machine learning can enhance user experiences.",
    "I love the diversity of machine learning applications.",
    "I find natural language generation intriguing.",
    "I enjoy working with text data.",
    "Machine learning can optimize business processes.",
    "I love the innovation in AI research.",
    "I find the concept of machine learning interpretability interesting.",
    "I enjoy creating machine learning workflows.",
    "Machine learning can uncover hidden patterns.",
    "I love the impact of AI on society.",
    "I find deep reinforcement learning fascinating.",
    "I enjoy developing custom machine learning solutions.",
    "Machine learning can improve customer experiences.",
    "I love the potential of AI in healthcare.",
    "I find the scalability of machine learning models intriguing.",
    "I enjoy applying machine learning to finance.",
    "Machine learning can enhance security measures.",
    "I love the possibilities of AI in creative industries.",
    "I find the ethical implications of AI important.",
    "I enjoy sharing knowledge about machine learning.",
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

# Visualization
st.title("Text Analysis and Visualization")

# Tokenization for Visualization
all_words = [word for sentence in sentences for word in sentence.split()]
filtered_tokens = [word for word in all_words if word.isalpha()]
word_freq = Counter(filtered_tokens)

# Word Frequency Distribution
st.subheader("Word Frequency Distribution")
fig, ax = plt.subplots(figsize=(10, 5))
most_common_words = dict(word_freq.most_common(20))
ax.bar(most_common_words.keys(), most_common_words.values(), color=bright_colors)
ax.set_xlabel('Words')
ax.set_ylabel('Frequency')
plt.xticks(rotation=90)
st.pyplot(fig)

# Word Cloud
st.subheader("Word Cloud")
wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(word_freq)
fig, ax = plt.subplots(figsize=(10, 5))
ax.imshow(wordcloud, interpolation='bilinear')
ax.axis('off')
st.pyplot(fig)

# Sentiment Analysis (Pie Chart)
st.subheader("Sentiment Analysis")
sentiment_labels = ['Positive', 'Neutral', 'Negative']
sentiment_values = [sum(1 for s in sentences if TextBlob(s).sentiment.polarity > 0),
                    sum(1 for s in sentences if TextBlob(s).sentiment.polarity == 0),
                    sum(1 for s in sentences if TextBlob(s).sentiment.polarity < 0)]
fig, ax = plt.subplots()
ax.pie(sentiment_values, labels=sentiment_labels, autopct='%1.1f%%', startangle=90, colors=bright_colors)
ax.set_title('Sentiment Distribution')
st.pyplot(fig)

# Part-of-Speech Tagging (Bar Chart)
st.subheader("Part-of-Speech Tagging")
pos_tags = [TextBlob(sentence).tags for sentence in sentences]
pos_flat = [tag for sublist in pos_tags for tag in sublist]
pos_freq = Counter(tag[1] for tag in pos_flat)

fig, ax = plt.subplots(figsize=(10, 5))
sns.barplot(x=list(pos_freq.keys()), y=list(pos_freq.values()), palette=bright_colors, ax=ax)
ax.set_xlabel('Part-of-Speech Tags')
ax.set_ylabel('Frequency')
plt.xticks(rotation=45)
st.pyplot(fig)

# Word Length Distribution (Histogram)
st.subheader("Word Length Distribution")
word_lengths = [len(word) for word in filtered_tokens]
fig, ax = plt.subplots(figsize=(10, 5))
ax.hist(word_lengths, bins=range(1, max(word_lengths) + 1), edgecolor='black', color=bright_colors[1])
ax.set_xlabel('Word Length')
ax.set_ylabel('Frequency')
st.pyplot(fig)

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

# Plot Training History
st.subheader("Training History")
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(history.history['loss'], label='Loss', color=bright_colors[0])
ax.plot(history.history['accuracy'], label='Accuracy', color=bright_colors[2])
ax.set_xlabel('Epoch')
ax.set_ylabel('Value')
ax.legend()
st.pyplot(fig)

# Model Evaluation
st.subheader("Model Evaluation")
loss, accuracy = model.evaluate(X, y, verbose=0)
st.write(f"Loss: {loss:.4f}")
st.write(f"Accuracy: {accuracy:.4f}")

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
user_input = st.text_input("Enter a seed text:", "Machine learning algorithms")
num_words = st.slider("Number of words to predict:", 1, 10, 2)

if st.button("Generate"):
    output_text = predict_next_word(user_input, next_words=num_words)
    st.write("Generated Text:", output_text)
