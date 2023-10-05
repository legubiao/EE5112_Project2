import random
import json
import pickle
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
import tensorflow
from tensorflow.keras.models import Sequential,load_model
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.optimizers import SGD
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')
lemmatizer = WordNetLemmatizer()

intents = json.loads(open('D:/aaaaaaaaaaaaaaaaa/23-24/5112/project2/Chat-Bot-for-medical-usecase-main/Chat-Bot-for-medical-usecase-main/intents.json').read())

words = pickle.load(open('D:/aaaaaaaaaaaaaaaaa/23-24/5112/project2/Chat-Bot-for-medical-usecase-main/Chat-Bot-for-medical-usecase-main/words1.pkl','rb'))
classes = pickle.load(open('D:/aaaaaaaaaaaaaaaaa/23-24/5112/project2/Chat-Bot-for-medical-usecase-main/Chat-Bot-for-medical-usecase-main/classes1.pkl','rb'))
model = load_model('D:/aaaaaaaaaaaaaaaaa/23-24/5112/project2/Chat-Bot-for-medical-usecase-main/Chat-Bot-for-medical-usecase-main/chatbot_model1.h5')

def clean_up_sentence(sentence):
  sentence_words = nltk.word_tokenize(sentence) # message by user converted to tokens.
  sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words] # lemmatize the words to its grammatically simplest form.
  return sentence_words
def bag_of_words(sentence):
  sentence_words = clean_up_sentence(sentence) # consisits of tokens created from users message
  bag = [0] * len(words) # initialized the list with tokens 
  for w in sentence_words: 
    for i,word in enumerate(words): # enumerate adds a counter to an iterable eg. (words,start=0)
      if(word == w):
        bag[i] = 1 # updating the bag to 1 where we get the user token and words(tokens) as equal 
  return np.array(bag)
def predict_class(sentence):
  bow = bag_of_words(sentence)
  res = model.predict(np.array([bow]))[0]
  ERROR_THRESHOLD = 0.25 # to bypass a specified number of pointer errors without terminating
  results  = [[i,r] for i,r in enumerate(res) if r>ERROR_THRESHOLD]

  results.sort(key=lambda x: x[1],reverse=True)
  return_list = [] 
  for r in results:
    return_list.append({'intent':classes[r[0]],'probability':str(r[1])})
  return return_list
def get_response(intents_list, intents_json): # comparing the two the output of predict_class and intents_json
  tag = intents_list[0]['intent']
  list_of_intents = intents_json['intents']
  for i in list_of_intents:
    if i['tag'] == tag: 
      result = random.choice(i['responses']) # if get matched output randomly from the responses
      break
  return result

bye = ("cya", "See you later", "Goodbye","goodbye","good bye", "I am Leaving", "Have a Good day", "bye", "cao", "see ya")


k=1
# global s
# s=0
while (k):
  message = input("")
  ints = predict_class(message)
  res = get_response(ints,intents)
  for i in bye:
    if message==i:
      k=0
  print(res)