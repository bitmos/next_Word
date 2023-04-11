from django.shortcuts import render
from django.http import HttpResponse, response
import pickle
import json
import numpy as np
# Importing the Libraries

from tensorflow.keras.models import load_model
def initialize():
    global __model

    with open("house_price/banglore_home_prices_model.pickle", 'rb') as f:
        __model = pickle.load(f)
    return __model

model = load_model('house_price/nextword1.h5')
tokenizer = pickle.load(open('house_price/tokenizer1.pkl', 'rb'))

def Predict_Next_Words(text):
    """
        In this function we are using the tokenizer and models trained
        and we are creating the sequence of the text entered and then
        using our model to predict and return the the predicted word.
    
    """
    for i in range(3):
        sequence = tokenizer.texts_to_sequences([text])[0]
        sequence = np.array(sequence)
        
        preds = np.argmax(model.predict(sequence), axis=-1)
#         print(preds)
        predicted_word = ""
        
        for key, value in tokenizer.word_index.items():
            if value == preds:
                predicted_word = key
                break
        
        print(predicted_word)
        return predicted_word
# def index(request):
#     location=initialize()
#     data={
#         'locations':__locations
#     }
#     return render(request,'houseprice.html',{'data':data})
def estimate(request):
    if request.method=="GET":
       print("Hi")
       return render(request,"house_price/houseprice.html",{})
    else:
        print("pOST METHOD")
        text=str(request.POST.get('text'))
        print(text)
        return render(request,'house_price/houseprice.html',{'data': Predict_Next_Words(text),'text':text})
         
    