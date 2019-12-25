
# **********************************************************************************************
# **********************************************************************************************
#this runs on python 3.6, 3.7 has some errorfor the tflearn library
# install 3.6 and select 3.6 from conda or cmd as below
# conda create -n chatbot python=3.6.8



# the currenrt tflearn 0.3.2 is compartible with python 3.6.8 and tensorflow 1.15

# pip list - to view versions of installed packages
# pip install tensorflow==1.15 - to downgrade tensorflow

# activate chatbot
# pip install nltk
# pip install numpy
# pip install tflearn
# pip install tensorflow

# initiate this file from powershell with the command PYTHON Chat.py
# **********************************************************************************************
# **********************************************************************************************


#import libraries and packages
import nltk
nltk.download('punkt')
from nltk.stem.lancaster import LancasterStemmer   #this wil be used to stem our words
nls = LancasterStemmer()
import numpy 
import tflearn 
#import tensorflow.compat.v1 as tf
#tf.disable_v2_behavior()
import tensorflow as tf
import random as rnd 
import json #we will sue this to read our json file 
import pickle


# to read the the json file
with open("Convo.json") as file:
	data = json.load(file)

# to view our json file
# print(data)
try:
	#trigger
	#uncomment trigger to rerun the preprocessing without checking pickle for saved processing
	with open("data.pickle","rb") as saved_processing:
		words, labels, training, output = pickle.load(saved_processing)

except:
	# asssign a ablank list to various groupings
	words=[]
	labels=[]
	doc_x=[]
	doc_y=[] 

	# to loop through all the dictionaries and lists in the master json dictonary "convos"
	# and tokenize (getting every word in the list using nltk)
	for convo in data["convos"]:
		for user in convo["users"]:
			wrds = nltk.word_tokenize(user) #to tokenize the users words
			words.extend(wrds) #because words is already a list we simply extend to add tokenized list
			doc_x.append(wrds) #to add (by appending) to our docs the typical user words
			doc_y.append(convo["subject"]) #to contain subject detail for which a users conversation belongs to

			# to add the unique subjects to the label list
		if convo["subject"] not in labels:
			labels.append(convo["subject"])

	# then stem (get the root words, remove extra characters like apostrophy and question marks)the words
	# and remove duplicates
	words = [nls.stem(w.lower()) for w in words if w != "?"]
	# now make the word list and make it a set to remove duplicates and convert back to sorted list
	words = sorted(list(set(words)))

	#sorting our labels as well
	labels=sorted(labels)

	# to create the training and testing data
	# we need to create a bag of words 
	# which represents any type of word available in the user list and we will use this to train our model

	training = []
	output = []

	out_empty = [0 for _ in range(len(labels))]

	for x, doc in enumerate(doc_x):
		bag = []

		wording = [nls.stem(w.lower()) for w in doc]

		for w in words:
			if w in wording:
				bag.append(1)
			else:
				bag.append(0)

		output_row = out_empty[:]
		output_row[labels.index(doc_y[x])] = 1

		training.append(bag)
		output.append(output_row)

	training = numpy.array(training)
	output = numpy.array(output)


# to save the pickle file if it hasn't been saved before using wb (write byte) 
with open("data.pickle","wb") as saved_processing:
	pickle.dump((words, labels, training, output), saved_processing)


# modelling

tf.reset_default_graph()

net = tflearn.input_data(shape=[None, len(training[0])])
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(output[0]), activation = 'softmax')
net = tflearn.regression(net)

model = tflearn.DNN(net)

try: 
	#trigger
	#uncomment trigger to retrain the model each time the code runs
	model.load("model.tflearn")

except:
	model.fit(training, output, n_epoch = 500, batch_size= 8, show_metric = True)
	model.save("model.tflearn")                                                     

# to solve Could not load dynamic library 'cudart64_100.dll'; dlerror: cudart64_100.dll not found
# https://www.joe0.com/2019/10/19/how-resolve-tensorflow-2-0-error-could-not-load-dynamic-library-cudart64_100-dll-dlerror-cudart64_100-dll-not-found/


#to save our model so we don't do the preprocessing everytime, so we need to save the preprocessed data and model using pickle and some try and except clause above

# the try: will see if the model is saved, else the except: will run and save the entire code


# now  to process the users input, we create the funtion below
def  users_words(new, words):
	bag =[0 for _ in range(len(words))] # an empty list that will iterate if the word isn't existing already

	new_words = nltk.word_tokenize(new)
	new_words = [nls.stem(word.lower()) for word in new_words]

	for given in new_words:
		for i, existing in enumerate(words):
			if existing == given:
				bag[i] = 1

	return numpy.array(bag)

# input prompt function

def chatting():
	print("chat with me, I'm here to help or type 'no thanks' for me to leave you")
	while True:
		entr = input("You: ")
		if entr.lower() == "no thanks": # to help us end when we want to
			print("Anny: ", "Okay, bye")
			break

		response = model.predict([users_words(entr,words)])[0] #fyi because model.predict requires a list we'll put in words just to have a list
		# [0] then selects the first entity of the list entr, kinda tricking the model.predict
		# print(response) the response will just show a probability of possible responses
		# we need to choose the index of the highest probability as below
		response_inx = numpy.argmax(response)
		# to select the possible subject
		user_subject = labels[response_inx]
		#print(user_subject) uncomment to see subject of conversation


		# to provide smarter reponses based on the chosen probability treshold
		if response[response_inx] > 0.6:
			for entity in data["convos"]:
				if entity["subject"] == user_subject:
					reply = entity["bots"]

			print("Anny: ", rnd.choice(reply))
		else:
			print("Anny: ", "I'm not sure I quite understand, please rephrase or as another question")

chatting()
