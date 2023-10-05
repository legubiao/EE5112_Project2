import random
import json
import pickle
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
import tensorflow
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.optimizers import legacy
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')

lemmatizer = WordNetLemmatizer() # create an instance of WordNetLemmatizer

intents = json.loads(open('D:/aaaaaaaaaaaaaaaaa/23-24/5112/project2/Chat-Bot-for-medical-usecase-main/Chat-Bot-for-medical-usecase-main/intents.json').read())

words = []
classes = []
documents = []
ignore_letters = ['?','!','.',',']

for intent in intents['intents']:
  for pattern in intent['patterns']:
    word_list = nltk.word_tokenize(pattern)  # tokenize the sentence i.e. "I am John" to "I","am","John"
    words.extend(word_list) #appending the tokenized word in the word_list
    documents.append((word_list,intent['tag'])) # in documents we stored the the tokenized form of patterns along with the tag.
    if(intent['tag'] not in classes): 
      classes.append(intent['tag'])  #storing the unique tags in the classes list
# print(documents)
#lemmatize 
words = [lemmatizer.lemmatize(word) for word in words if word not in ignore_letters] # lemmatize the word to its simplest meaningful form
words = sorted(set(words)) # removing the duplicates in words
# print(words)
 
classes = sorted(set(classes)) # removing the duplicates classes (mostly not req.coz classes are unique and user defined)

pickle.dump(words, open('D:/aaaaaaaaaaaaaaaaa/23-24/5112/project2/Chat-Bot-for-medical-usecase-main/Chat-Bot-for-medical-usecase-main/words1.pkl','wb'))
pickle.dump(classes, open('D:/aaaaaaaaaaaaaaaaa/23-24/5112/project2/Chat-Bot-for-medical-usecase-main/Chat-Bot-for-medical-usecase-main/classes1.pkl','wb'))

# converting the textual data into numeric data to feed into the neural network
training =[]
output_empty = [0] * len(classes) # initialized the list with the number of tags in intents.json

for document in documents:  # document = (['How','are','you'],'greeting')
  bag = []
  word_patterns = document[0]  # document[0] consists of tokens eg. ['How','are','you']
  word_patterns = [lemmatizer.lemmatize(word.lower()) for word in word_patterns] # lemmatizing the word along with converting it in lowercase.
  for word in words: # words consists of tokens in lower as well as upper case.
    bag.append(1) if word in word_patterns else bag.append(0) # bag is created with columns equal to size of words list 
    # 1 is appended where there is no need to check for mapping since the words are already in lowercase. 
  output_row = list(output_empty)
  output_row[classes.index(document[1])] = 1  # document[1] consists of tags of that particular set of tokens
  # output_row store 1 mapping to index of tag in classes.
  training.append([bag,output_row]) #input is bag(tokens) and output is output_row(tags)


random.shuffle(training)
training = np.array(training)

train_x = list(training[:,0]) # data corresponding to all rows and just the first column(bag)
train_y = list(training[:,1]) # data corresponding to all rows and just the second column(output_row)

# training of model8
model = Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]),),activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]),activation= 'softmax'))

sgd = legacy.SGD(learning_rate= 0.01,decay= 1e-6,momentum=0.9,nesterov=True)
model.compile(loss='categorical_crossentropy',optimizer=sgd,metrics=['accuracy'])

hist = model.fit(np.array(train_x),np.array(train_y),epochs=200,batch_size=5,verbose=1)

model.save('D:/aaaaaaaaaaaaaaaaa/23-24/5112/project2/Chat-Bot-for-medical-usecase-main/Chat-Bot-for-medical-usecase-main/chatbot_model1.h5',hist)
print('done')

# Access the training history
loss = hist.history['loss']
accuracy = hist.history['accuracy']

# Create a range of epochs for x-axis
epochs = range(1, len(loss) + 1)

# Plot loss
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(epochs, loss, 'r', label='Training loss')
plt.title('Training Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

# Plot accuracy
plt.subplot(1, 2, 2)
plt.plot(epochs, accuracy, 'g', label='Training accuracy')
plt.title('Training Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

# Display both plots
plt.tight_layout()
plt.show()
