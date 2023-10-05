import tensorflow as tf
import numpy as np
import json
import re
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Dense, GRU, LSTM, Masking
from keras.preprocessing.text import tokenizer_from_json

with open('D:/aaaaaaaaaaaaaaaaa/23-24/5112/project2/simple-seq2seq-chatbot-master/simple-seq2seq-chatbot-master/preprocessed_data/questions.json', 'r') as f:
    json_data = json.load(f)
    question_corpus = tokenizer_from_json(json_data)
    f.close()

with open('D:/aaaaaaaaaaaaaaaaa/23-24/5112/project2/simple-seq2seq-chatbot-master/simple-seq2seq-chatbot-master/preprocessed_data/answers.json', 'r') as f:
    json_data = json.load(f)
    answer_corpus = tokenizer_from_json(json_data)
    f.close()

npzfile = np.load('D:/aaaaaaaaaaaaaaaaa/23-24/5112/project2/simple-seq2seq-chatbot-master/simple-seq2seq-chatbot-master/preprocessed_data/data.npz') 

q_word2ind={e:i for e,i in question_corpus.word_index.items() if i <= 2500}
q_ind2word={e:i for i,e in q_word2ind.items()}
a_word2ind={e:i for e,i in answer_corpus.word_index.items() if i <= 2500}
a_ind2word={e:i for i,e in a_word2ind.items()}

def create_encoder(inputdim, embeddingsize, inputlen, n_units):

    encoder_input = Input((inputlen,))
    encoder_embed = Embedding(inputdim+1, embeddingsize)(encoder_input)
    encoder_mask = Masking()(encoder_embed)
    encoder = LSTM(n_units, return_state = True)
    _, encoder_h, encoder_c = encoder(encoder_mask)
    
    encoder=Model(encoder_input, [encoder_h,encoder_c])
    
    return encoder

def create_decoder(inputdim, embeddingsize, n_units):
    # the size of input at here is 1 because we want to predict the answer step by step, each time only input 1 word
    decoder_input = Input((1,))
    initial_stateh = Input((n_units,))
    initial_statec = Input((n_units,))
    encoder_state = [initial_stateh,initial_statec]
    decoder_embed = Embedding(inputdim+1, embeddingsize,input_length = 1)(decoder_input)
    decoder_mask = Masking()(decoder_embed)
    decoder = LSTM(n_units, return_sequences = True, return_state = True)
    # in training model, we dont use the state h & c. but in inference model, we do
    decoder_output, decoder_h, decoder_c = decoder(decoder_mask,initial_state = encoder_state)
    decoder_dense = Dense(inputdim, activation = 'softmax')
    decoder_output_ = decoder_dense(decoder_output)
    
    decoder=Model([decoder_input,initial_stateh,initial_statec],[decoder_output_,decoder_h,decoder_c])
    
    return decoder

N_Unit = 256
EmbeddingSize = 128
VocabSize = 2500 
QuestionLen = npzfile['arr_0'].shape[1]
AnswerLen = npzfile['arr_1'].shape[1]

encoder=create_encoder(VocabSize,EmbeddingSize,QuestionLen,N_Unit)

encoder.load_weights('D:/aaaaaaaaaaaaaaaaa/23-24/5112/project2/simple-seq2seq-chatbot-master/simple-seq2seq-chatbot-master/trained_model/lstm_enc_test.h5')

decoder=create_decoder(VocabSize,EmbeddingSize,N_Unit)

decoder.load_weights('D:/aaaaaaaaaaaaaaaaa/23-24/5112/project2/simple-seq2seq-chatbot-master/simple-seq2seq-chatbot-master/trained_model/lstm_dec_test.h5')

def clean_text(text):

    # remove unnecessary characters in sentences
    
    text = text.lower().strip()
    text = re.sub(r"i'm", "i am", text)
    text = re.sub(r"he's", "he is", text)
    text = re.sub(r"she's", "she is", text)
    text = re.sub(r"it's", "it is", text)
    text = re.sub(r"that's", "that is", text)
    text = re.sub(r"what's", "what is", text)
    text = re.sub(r"where's", "where is", text)
    text = re.sub(r"there's", "there is", text)
    text = re.sub(r"how's", "how is", text)
    text = re.sub(r"\'ll", " will", text)
    text = re.sub(r"\'ve", " have", text)
    text = re.sub(r"\'re", " are", text)
    text = re.sub(r"\'d", " would", text)
    text = re.sub(r"\'re", " are", text)
    text = re.sub(r"won't", "will not", text)
    text = re.sub(r"can't", "cannot", text)
    text = re.sub(r"n't", " not", text)
    text = re.sub(r"n'", "ng", text)
    text = re.sub(r"'bout", "about", text)
    text = re.sub(r"'til", "until", text)
    text = re.sub(r'[" "]+', " ", text)
    text = re.sub(r"[-()\"#/@;:<>{}`+=~|.!?,]", "", text)
    
    return text

def evaluate(sentence):
    
    sentence=clean_text(sentence) # clean the input text
    encoder_inputs=[]
    # convert the input text to index sequence and use unk replace the word not in vocabulary
    for word in sentence.split():
        if word in q_word2ind:
            encoder_inputs.append(q_word2ind[word])
        elif word not in q_word2ind:
            encoder_inputs.append(q_word2ind['unk'])

    encoder_inputs=tf.keras.preprocessing.sequence.pad_sequences([encoder_inputs],maxlen=QuestionLen,padding='post')
    encoder_inputs = tf.convert_to_tensor(encoder_inputs)
    encoder_h,encoder_c=encoder(encoder_inputs)
    
    # initialize the decoder input
    decoder_inputs=tf.expand_dims([a_word2ind['bos']], 0)
    hidden_h,hidden_c=encoder_h,encoder_c
    
    result=''
    for t in range(AnswerLen):
        pred,state_h,state_c=decoder([decoder_inputs,hidden_h,hidden_c])
        pred=np.squeeze(pred)
        pred_ind=tf.math.argmax(pred).numpy()+1
 
        if a_ind2word[pred_ind]=='eos': # once we get the eos symbol, stop the loop
            return result
        result+=a_ind2word[pred_ind] + ' '
        decoder_inputs=tf.expand_dims([pred_ind],0) # pass the predict index and state vectors to the next input       
        hidden_h,hidden_c=state_h,state_c
    return result

while True:
    inputs = input('User :> ')
    if inputs == 'quit':
        break

    result = evaluate(inputs)
    result_without_bos = ' '.join(word for word in result.split() if word != 'bos')

    print('Bot :> ' + result_without_bos)