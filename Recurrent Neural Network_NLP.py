#!/usr/bin/env python
# coding: utf-8

# # DL Engineer At BBC
Problem Statement: BBC wants to auto categorize the news into various categories which will also help in recommending the right news articles to it users at a later stage.

Note: BBC - British Broadcasting CorporationA sample of 2225 news article and their relavant category.
# In[1]:


import pandas as pd
import numpy as np
from IPython.display import Image
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, accuracy_score
import re
import matplotlib.pyplot as plt


# In[2]:


df = pd.read_csv('/Users/abhishekmishra/Downloads/Datasets/bbc-news-data.csv',sep="\t")


# In[3]:


print(df.shape)


# In[ ]:





# In[4]:


df.info()


# In[5]:


df = df[((~df.title.isnull()) & (~df.content.isnull()))].reset_index(drop=True)


# In[6]:


df.shape


# In[7]:


plt.bar(df.category.value_counts().index, df.category.value_counts().values)
plt.title('Category distribution')
plt.xlabel("Category")
plt.ylabel("Count")


# In[8]:


ind=1807
print(f'Title: {df.title[ind]}')
print(f'Category: {df.category[ind]}')

Code Implementation
# In[9]:


from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Dense, Embedding, Input, InputLayer, RNN, SimpleRNN, LSTM, GRU, TimeDistributed
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow as tf

import string

import nltk
nltk.download('stopwords')

from nltk.corpus import stopwords
stop_words = stopwords.words('english')

from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
import seaborn as sns

Text cleaning
# In[10]:


def data_cleaning(text):

    # Lower the words in the sentence
    cleaned = text.lower()

    # Replace the full stop with a full stop and space
    cleaned = cleaned.replace(".", ". ")

    # Remove the stop words : optional pre-processing step
    tokens = [word for word in cleaned.split() if not word in stop_words]

    # Remove the punctuations
    tokens = [tok.translate(str.maketrans(' ', ' ', string.punctuation)) for tok in tokens]

    # Joining the tokens back to form the sentence
    cleaned = " ".join(tokens)

    # Remove any extra spaces
    cleaned = cleaned.strip()

    return cleaned


# In[14]:


for index, data in tqdm(df.iterrows(), total=df.shape[0]):
    df.loc[index, 'title'] = data_cleaning(data['title'])

Fixing the sequence length
# In[16]:


sns.boxplot(df['title'].str.split(" ").str.len())

As we can see that maximum number of words in the title is 7, we can set the sequence length to be the max words in the longest sentence.
# In[30]:


max_sentence_len = df['title'].str.split(" ").str.len().max()
total_classes = df.category.nunique()

print(f"Maximum sequence length: {max_sentence_len}")
print(f"Total classes: {total_classes}")

Splitting the data to train and test
# In[31]:


np.random.seed(100)
train_X, test_X, train_Y, test_Y = train_test_split(df['title'],
                                                    df['category'],
                                                    test_size=0.2,
                                                    random_state=100)
train_X = train_X.reset_index(drop=True)
test_X = test_X.reset_index(drop=True)
train_Y = train_Y.reset_index(drop=True)
train_Y = train_Y.reset_index(drop=True)


# In[32]:


train_X.shape, train_Y.shape, test_X.shape, test_Y.shape

One hot Encode the labels
# In[33]:


train_Y = pd.get_dummies(train_Y).values
test_Y = pd.get_dummies(test_Y).values


# In[34]:


validation = test_Y.argmax(axis=1)

Tokenize the input text and pad them
# In[35]:


def tokenize_and_pad(inp_text, max_len, tok):

    text_seq = tok.texts_to_sequences(inp_text)
    text_seq = pad_sequences(text_seq, maxlen=max_len, padding='post')

    return text_seq

text_tok = Tokenizer()
text_tok.fit_on_texts(train_X)
train_text_X = tokenize_and_pad(inp_text=train_X, max_len=max_sentence_len, tok=text_tok)
test_text_X = tokenize_and_pad(inp_text=test_X, max_len=max_sentence_len, tok=text_tok)
vocab_size = len(text_tok.word_index)+1

print("Overall text vocab size", vocab_size)

Choose the latent dimension and embedding dimension:
    Latent dimension: Dimension of the weight matrix U, V, W
    Embedding dimension: Dimension of the word embeddings at the embedding layer
# In[36]:


latent_dim=50
embedding_dim=100

Define RNN Model Architecture
# In[37]:


seed = 56
tf.random.set_seed(seed)
np.random.seed(seed)

model = Sequential()
model.add(Embedding(vocab_size, embedding_dim, trainable=True))
model.add(SimpleRNN(latent_dim, recurrent_dropout=0.2, return_sequences=False, activation='tanh'))
model.add(Dense(total_classes, activation='softmax'))
model.summary()

Model Training:
    Optimizer: Adam
    Loss: Categorical cross-entrophy since it is a multiclass classification probem
    Early stopping: Used to stop training if validation accuracy does not improve while training to avoid overfitting
    
# In[38]:


tf.random.set_seed(seed)
np.random.seed(seed)

model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['acc'])

early_stopping = EarlyStopping(monitor='val_acc',
                               mode='max',
                               verbose=1,
                               patience=5)

model.fit(x=train_text_X, y=train_Y,
          validation_data=(test_text_X, test_Y),
          batch_size=64,
          epochs=10,
          callbacks=[early_stopping])


# In[39]:


seed=56
tf.random.set_seed(seed)
np.random.seed(seed)

model_clipping = Sequential()
model_clipping.add(Embedding(vocab_size, embedding_dim, trainable=True))
model_clipping.add(SimpleRNN(latent_dim, recurrent_dropout=0.2, return_sequences=False, activation='tanh'))
model_clipping.add(Dense(total_classes, activation='softmax'))
model_clipping.summary()


# In[44]:


tf.random.set_seed(seed)
np.random.seed(seed)
# Define a maximum gradient norm threshold
max_norm = 1.0

# Compile the model with a loss function and optimizer
model_clipping.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['acc'])

# Define a callback to clip the gradients during training
class GradientClipping(tf.keras.callbacks.Callback):
    def on_batch_end(self, batch, logs=None):
        grads = self.model_clipping.optimizer.get_gradients(self.model_clipping.total_loss, self.model_clipping.trainable_variables)
        clipped_grads, _ = tf.clip_by_global_norm(grads, max_norm)
        self.model_clipping.optimizer.apply_gradients(zip(clipped_grads, self.model_clipping.trainable_variables))



# In[47]:


# Fit the model to the training data
model_clipping.fit(x=train_text_X, y=train_Y,
          validation_data=(test_text_X, test_Y),
          batch_size=64,
          epochs=10)

Saved the Trained Model
# In[48]:


model.save("BCC_classifier.h5")
model_clipping.save("BCC_classifier_clipping.hs")

Load the saved Model
# In[49]:


model = tf.keras.models.load_model("BCC_classifier.h5")

Make predictions on the test dataset
# In[51]:


prediction = model.predict(test_text_X)
prediction = prediction.argmax(axis=1)
print(f"Accuracy: {accuracy_score(prediction, validation)}")

Confusion matrix of the prediction and actual
# In[53]:


cm = confusion_matrix(validation, prediction)

# print("")
# plt.figure(figsize=(15,15))
sns.heatmap(cm, annot=True, cmap='Oranges')
plt.xlabel("Prediction")
plt.ylabel("Actual")
plt.title("Confusion Matrix")

