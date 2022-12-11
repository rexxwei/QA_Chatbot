
from flask import Flask, render_template, request
import pandas as pd
import tensorflow
from tensorflow import keras
import pickle
import numpy as np
from keras.models import Sequential, Model
from tensorflow.keras.layers import Embedding
from keras.layers import Input, Activation, Dense, Permute, Dropout
from keras.layers import add, dot, concatenate
from keras.layers import LSTM
from keras_preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer


import flask
app = Flask(__name__)

def vectorize_stories(data, word_index, max_story_len, max_question_len):
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
    # Xq = QUERY/QUESTION
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
        
    # Finally, pad the sequences based on their max length so the RNN can be trained on uniformly long sequences.
        
    # RETURN TUPLE FOR UNPACKING
    return (pad_sequences(X, maxlen=max_story_len),pad_sequences(Xq, maxlen=max_question_len), np.array(Y))


@app.route('/')
def index():
    return render_template('trybot.html')
    # return render_template('index.html')

@app.route('/fastinput')
def fastinput():
    return render_template('fastinput.html')


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

    vocab = {'.', '?', 'Daniel', 'Is', 'John', 'Mary', 'Sandra', 'apple', 'back', 'bathroom', 'bedroom',
            'discarded', 'down', 'dropped', 'football', 'garden', 'got', 'grabbed', 'hallway',
            'in', 'journeyed', 'kitchen', 'left', 'milk', 'moved', 'no', 'office', 'picked', 'put', 
            'the', 'there', 'to', 'took', 'travelled', 'up','went', 'yes'}

    # integer encode sequences of words
    tokenizer = Tokenizer(filters=[])
    tokenizer.fit_on_texts(vocab)

    word_index = tokenizer.word_index

    max_story_len = 156
    max_question_len = 6
    vocab_size = 38

    # creat model
    input_sequence = Input((max_story_len,))
    question = Input((max_question_len,))
    input_encoder_m = Sequential()
    input_encoder_m.add(Embedding(input_dim=vocab_size,output_dim=64))
    input_encoder_m.add(Dropout(0.3))
    input_encoder_c = Sequential()
    input_encoder_c.add(Embedding(input_dim=vocab_size,output_dim=max_question_len))
    input_encoder_c.add(Dropout(0.3))
    # embed the question into a sequence of vectors
    question_encoder = Sequential()
    question_encoder.add(Embedding(input_dim=vocab_size,
                                output_dim=64,
                                input_length=max_question_len))
    question_encoder.add(Dropout(0.3))
    # encode input sequence and questions (which are indices)
    # to sequences of dense vectors
    input_encoded_m = input_encoder_m(input_sequence)
    input_encoded_c = input_encoder_c(input_sequence)
    question_encoded = question_encoder(question)
    # shape: `(samples, story_maxlen, query_maxlen)`
    match = dot([input_encoded_m, question_encoded], axes=(2, 2))
    match = Activation('softmax')(match)
    # add the match matrix with the second input vector sequence
    response = add([match, input_encoded_c])
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

    # build the final model
    model = Model([input_sequence, question], answer)
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy',
                metrics=['accuracy'])

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
    my_story,my_ques,my_ans = vectorize_stories(data, word_index, max_story_len, max_question_len)
    pred = model.predict(([my_story, my_ques]))

    #Generate prediction from model
    val_max = np.argmax(pred[0])

    for key, val in tokenizer.word_index.items():
        if val == val_max:
            k = key

    prob_yes = pred[0][tokenizer.word_index['yes']]
    prob_no = pred[0][tokenizer.word_index['no']]
    print(str( prob_yes > prob_no))
    # print("Predicted answer is: ", k)
    # print("Probability was: ", pred[0][val_max])

    # if possibility of yes > no
    if (prob_yes > prob_no):
        return render_template('yes.html', conf = round(pred[0][val_max]*100, 2))
    else:
        return render_template('no.html', conf = round(pred[0][val_max]*100, 2))

    return render_template('err.html')


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
