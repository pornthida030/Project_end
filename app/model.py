import pandas as pd
import sklearn
import numpy as np
from IPython.display import display
import matplotlib.pyplot as plt
# from tqdm import tqdm
from sklearn.model_selection import train_test_split
import pickle
from gensim.models import KeyedVectors
from pythainlp.tokenize import word_tokenize
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
import tensorflow as tf

df=pd.read_csv('app\data-project2.csv',names=['input_text','labels'])
data_df=pd.DataFrame.from_records(df)

#เปลี่ยนตัวอักษรตัวใหญ่เป็นตัวเล็ก
data_df['cleaned_labels']=data_df['labels'].str.lower()

#เอา labels ออก
data_df.drop('labels',axis=1,inplace=True)

data_df=data_df[data_df['cleaned_labels']!='garbage']


cleaned_input_text=data_df['input_text'].str.strip()
cleaned_input_text=cleaned_input_text.str.lower()

data_df['cleaned_input_text']=cleaned_input_text
data_df.drop('input_text',axis=1,inplace=True)

data_df=data_df.drop_duplicates("cleaned_input_text",keep="first")

input_text=data_df["cleaned_input_text"].tolist()
labels=data_df["cleaned_labels"].tolist()
train_text,test_text,train_labels,test_labels=train_test_split(input_text,labels,test_size=0.2,random_state=42)

#เก็บ train set และ test set ลงในไฟล์
with open('train_set.pkl','wb') as f:
  pickle.dump([train_text,train_labels],f)
with open('test_set.pkl','wb') as f:
  pickle.dump([test_text,test_labels],f)

word2vec_model = KeyedVectors.load_word2vec_format('LTW2V_v0.1.bin',binary=True,unicode_errors='ignore')

for word, sim_score in word2vec_model.most_similar("หนู"):
  print(word,sim_score)

print("Embedding size:", word2vec_model.vector_size)
print("Vocab size:", len(word2vec_model.vocab))


# Display first 5 entries of train set
for text, label in zip(train_text[:5], train_labels[:5]):
  print("Label:", label, "     \t", "Text:", text)


tokenized_train_text = [word_tokenize(text) for text in train_text]
tokenized_test_text = [word_tokenize(text) for text in test_text]

def map_word_index(word_seq):
 
  indices = []
  for word in word_seq:
    if word in word2vec_model.vocab:
      indices.append(word2vec_model.vocab[word].index + 1)
    else:
      indices.append(1)
  return indices

train_word_indices = [map_word_index(word_seq) for word_seq in tokenized_train_text]
test_word_indices = [map_word_index(word_seq) for word_seq in tokenized_test_text]

# Find maxlen
max_leng = 0
for word_seq in tokenized_train_text:
  if len(word_seq) > max_leng:
    max_leng = len(word_seq)

print("Maximum word length:", max_leng)

# pad sequence using keras library


train_padded_wordinds = pad_sequences(train_word_indices, maxlen=max_leng, value=0)
test_padded_wordinds = pad_sequences(test_word_indices, maxlen=max_leng, value=0)

print("Word sequence:", tokenized_train_text[0])
print("Prepared word indices:", train_padded_wordinds[0])


# Prepare map between label index and label name
unique_labels = set(train_labels)
index_to_label = [label for label in sorted(unique_labels)]
label_to_index = {label:i for i, label in enumerate(sorted(unique_labels))}

# Mapping label to index
train_label_indices = [label_to_index[label] for label in train_labels]
test_label_indices = [label_to_index[label] for label in test_labels]

# Create dummy code of label index
train_dummy_label = to_categorical(train_label_indices, num_classes=len(unique_labels))
test_dummy_label = to_categorical(test_label_indices, num_classes=len(unique_labels))

def model_use(text):
    loaded_model = tf.keras.models.load_model('content\my_model')
    # tokenize
    word_seq = word_tokenize(text)
    # map index
    word_indices = map_word_index(word_seq)
    # padded to max_leng
    padded_wordindices = pad_sequences([word_indices], maxlen=max_leng, value=0)
    # predict to get logit
    logit = loaded_model.predict(padded_wordindices, batch_size=32)
    # get prediction
    predict = [ index_to_label[pred] for pred in np.argmax(logit, axis=1) ][0]
    print(predict)
    return predict
# model_use("กินอะไรดี")