from flask import Flask, render_template, request
import pandas as pd
import tensorflow
from tensorflow import keras
import pickle
import numpy as np
import json
import ast
from keras.models import Sequential, Model
from tensorflow.keras.layers import Embedding
from keras.layers import Input, Activation, Dense, Permute, Dropout
from keras.layers import add, dot, concatenate
from keras.layers import LSTM
from keras_preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import model_from_json

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


# =======================

with open('vocab.txt','r') as f:
   vocab = ast.literal_eval(f.read())

vocab_len = len(vocab) + 1

with open('long_rec.txt') as f:
    ddata = f.read()
   
# reconstructing the data as a dictionary
len_dict = ast.literal_eval(ddata)
# max_story_len = max([len(data[0]) for data in all_data])
max_story_len = int(len_dict['max_story_len'])
# max_question_len = max([len(data[1]) for data in all_data])
max_question_len = int(len_dict['max_question_len'])

# Reserve 0 for pad_sequences
vocab_size = len(vocab) + 1

# load json and create model
json_file = open('chatbot.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("chatbot_80_epochs.h5")
loaded_model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

@app.route('/')
def index():
    return render_template('trybot.html')

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
    with open('word_index.txt','r') as f:
        word_index = ast.literal_eval(f.read())

    my_story,my_ques,my_ans = vectorize_text(data, word_index, max_story_len, max_question_len)
    pred = loaded_model.predict(([my_story, my_ques]))

    #Generate prediction from model
    val_max = np.argmax(pred[0])

    for key, val in word_index.items():
        if val == val_max:
            k = key

    prob_yes = pred[0][word_index['yes']]
    prob_no = pred[0][word_index['no']]
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
    app.run(host='0.0.0.0', port=8080)
