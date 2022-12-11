from flask import Flask, render_template, request
import pandas as pd
import tensorflow
from tensorflow import keras
import pickle
import numpy as np
import json
from keras.models import Sequential, Model
from tensorflow.keras.layers import Embedding
from keras.layers import Input, Activation, Dense, Permute, Dropout
from keras.layers import add, dot, concatenate
from keras.layers import LSTM
from keras_preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer

import flask
app = Flask(__name__)

def vectorize_text(data, word_index, max_story_len, max_question_len):
    '''
    INPUT:
    data: consisting of Stories,Queries,and Answers
    word_index: word index dictionary from tokenizer
    max_story_len: the length of the longest story (used for pad_sequences function)
    max_question_len: length of the longest question (used for pad_sequences function)
    OUTPUT:
    Vectorizes the stories,questions, and answers into padded sequences. We first loop for every story, query , and
    answer in the data. Then we convert the raw words to an word index value. Then we append each set to their appropriate
    output list. Then once we have converted the words to numbers, we pad the sequences so they are all of equal length.
    
    Returns this in the form of a tuple (X,Xq,Y) (padded based on max lengths)
    '''
    # X = STORIES
    X = []
    # Xq = QUESTION
    Xq = []
    # Y = CORRECT ANSWER
    Y = []
    
    for story, query, answer in data:
        # Grab the word index for every word in story
        x = [word_index[word.lower()] for word in story]
        # Grab the word index for every word in query
        xq = [word_index[word.lower()] for word in query]
        # Grab the Answers (either Yes/No so we don't need to use list comprehension here)
        # Index 0 is reserved so we're going to use + 1
        y = np.zeros(len(word_index) + 1)
        # Now that y is all zeros and we know its just Yes/No , we can use numpy logic to create this assignment
        y[word_index[answer]] = 1
        
        # Append each set of story,query, and answer to their respective holding lists
        X.append(x)
        Xq.append(xq)
        Y.append(y)
        
        # print(type(X))
        xp = pad_sequences(X, maxlen=max_story_len)
        qp = pad_sequences(Xq, maxlen=max_question_len)
        yn = np.array(Y)
    # Finally, pad the sequences based on their max length so the RNN can be trained on uniformly long sequences.
    return (xp, qp, yn)


# Unpickling file for training data
with open("train_qa.txt", "rb") as fp:
    train_data = pickle.load(fp)

with open("test_qa.txt", "rb") as fp:
    test_data = pickle.load(fp)

vocab = set()
all_data = train_data + test_data

for story, question , answer in all_data:
    vocab = vocab.union(set(story))
    vocab = vocab.union(set(question))

vocab.add('no')
vocab.add('yes')
vocab_len = len(vocab) + 1
max_story_len = max([len(data[0]) for data in all_data])
max_question_len = max([len(data[1]) for data in all_data])

# Reserve 0 for pad_sequences
vocab_size = len(vocab) + 1

# integer encode sequences of words
tokenizer = Tokenizer(filters=[])
tokenizer.fit_on_texts(vocab)

train_story_text = []
train_question_text = []
train_answers = []

for story,question,answer in train_data:
    train_story_text.append(story)
    train_question_text.append(question)

train_story_seq = tokenizer.texts_to_sequences(train_story_text)

# inputs_train, queries_train, answers_train = vectorize_text(train_data)
train_stories, train_questions, train_answers = vectorize_text(train_data, tokenizer.word_index, max_story_len, max_question_len)  

# inputs_test, queries_test, answers_test = vectorize_text(test_data)
test_stories, test_questions, test_answers = vectorize_text(test_data, tokenizer.word_index, max_story_len, max_question_len)

# Create LSTM Model

story_sequence = Input((max_story_len,))
question = Input((max_question_len,))

# Input gets embedded to a sequence of vectors
story_encoder_m = Sequential()
story_encoder_m.add(Embedding(input_dim = vocab_size, output_dim = 64))
story_encoder_m.add(Dropout(0.3))

# embed the input into a sequence of vectors of size query_maxlen
story_encoder_c = Sequential()
story_encoder_c.add(Embedding(input_dim = vocab_size, output_dim = max_question_len))
story_encoder_c.add(Dropout(0.3))

# embed the question into a sequence of vectors
question_encoder = Sequential()
question_encoder.add(Embedding(input_dim = vocab_size,
                            output_dim = 64,
                            input_length = max_question_len))
question_encoder.add(Dropout(0.3))

# encode input sequence & questions (which are indices) to sequences of dense vectors
story_encoded_m = story_encoder_m(story_sequence)
story_encoded_c = story_encoder_c(story_sequence)
question_encoded = question_encoder(question)

# shape: `(samples, story_maxlen, query_maxlen)`
match = dot([story_encoded_m, question_encoded], axes=(2, 2))
match = Activation('softmax')(match)

# add the match matrix with the second input vector sequence
response = add([match, story_encoded_c])
response = Permute((2, 1))(response)

# concatenate the match matrix with the question vector sequence
answer = concatenate([response, question_encoded])

# Reduce with RNN (LSTM)
answer = LSTM(32)(answer)  # (samples, 32)

# Regularization with Dropout
answer = Dropout(0.5)(answer)
answer = Dense(vocab_size)(answer)  # (samples, vocab_size)

# we output a probability distribution over the vocabulary
answer = Activation('softmax')(answer)

# build final model
model = Model([story_sequence, question], answer)
model.compile(optimizer='rmsprop', loss='categorical_crossentropy',
            metrics=['accuracy'])

# train the model
history = model.fit([train_stories, train_questions], train_answers, batch_size=32, epochs=40, 
                    validation_data = ([test_stories, test_questions], test_answers))

# save the model
filename = 'chatbot_120_epochs.h5'
model.save(filename)


@app.route('/')
def index():
    return render_template('trybot.html')
    # return render_template('index.html')

@app.route('/yes')
def pos():
    return render_template('yes.html')

@app.route('/no')
def neg():
    return render_template('no.html')

@app.route('/predict', methods=['POST'])
def predict():
    # get the input content
    input_cont = request.form.to_dict()

    # load save model
    filename = 'chatbot_120_epochs.h5'
    model.load_weights(filename)

    # input for prediction
    # my_story = "John left the kitchen . Sandra dropped the football in the garden ."
    story =  input_cont['story']
    story =  story.replace(".", " .")
    # my_question = "Is the football in the garden ?"
    question = input_cont['question']
    question = question.replace("?", " ?")

    if input_cont['answer'] == 'noip':
        answer = 'yes'
    else:
        answer = input_cont['answer']

    data = [(story.split(), question.split(), answer)]
    # mydata = [(my_story.split(),my_question.split(),'no')]
    my_story,my_ques,my_ans = vectorize_text(data, tokenizer.word_index, max_story_len, max_question_len)
    pred = model.predict(([my_story, my_ques]))

    #Generate prediction from model
    val_max = np.argmax(pred[0])

    for key, val in tokenizer.word_index.items():
        if val == val_max:
            k = key

    prob_yes = pred[0][tokenizer.word_index['yes']]
    prob_no = pred[0][tokenizer.word_index['no']]
    print(prob_yes > prob_no)
    print("Prediction is: ", k)
    print("Probability: ", pred[0][val_max])

    # if possibility of yes > no
    if (prob_yes > prob_no):
        return render_template('yes.html', conf = round(pred[0][val_max]*100, 2))
    else:
        return render_template('no.html', conf = round(pred[0][val_max]*100, 2))

    return render_template('err.html')


if __name__ == '__main__':
    print("Loading the model, please wait...")
    app.run(host='0.0.0.0', port=8080)
