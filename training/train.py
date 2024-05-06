import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.models import Sequential

# Sample text data for training
text = "Hello, how are you? I hope you are doing well."

# Create a character mapping (char to index and index to char)
chars = sorted(list(set(text)))
char_to_idx = {char: idx for idx, char in enumerate(chars)}
idx_to_char = {idx: char for char, idx in char_to_idx.items()}

# Length of vocabulary
vocab_size = len(chars)

# Prepare training data
max_len = 40  # Length of input sequence
step = 3      # Step size for creating sequences

sequences = []
next_chars = []
for i in range(0, len(text) - max_len, step):
    sequences.append(text[i : i + max_len])
    next_chars.append(text[i + max_len])

# Vectorization: Convert sequences into input and output data
X = np.zeros((len(sequences), max_len, vocab_size), dtype=np.bool)
y = np.zeros((len(sequences), vocab_size), dtype=np.bool)

for i, sequence in enumerate(sequences):
    for t, char in enumerate(sequence):
        X[i, t, char_to_idx[char]] = 1
    y[i, char_to_idx[next_chars[i]]] = 1

# Build the model
model = Sequential([
    LSTM(128, input_shape=(max_len, vocab_size)),
    Dense(vocab_size, activation='softmax')
])

model.compile(loss='categorical_crossentropy', optimizer='adam')

# Train the model
model.fit(X, y, batch_size=128, epochs=50)

# Function to sample the next character from the model's predictions
def sample(preds, temperature=1.0):
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

# Generate text using the trained model
def generate_text(seed_text, num_chars=100, temperature=1.0):
    generated_text = seed_text
    for _ in range(num_chars):
        x_pred = np.zeros((1, max_len, vocab_size))
        for t, char in enumerate(seed_text):
            x_pred[0, t, char_to_idx[char]] = 1.0
        preds = model.predict(x_pred, verbose=0)[0]
        next_index = sample(preds, temperature)
        next_char = idx_to_char[next_index]
        generated_text += next_char
        seed_text = seed_text[1:] + next_char
    return generated_text

# Generate text starting from a seed text
seed_text = "Hello, how are"
generated_text = generate_text(seed_text, num_chars=200, temperature=0.5)
print(generated_text)
