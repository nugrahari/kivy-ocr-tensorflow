import os
import sys
import numpy as np
import pandas as pd
from keras.models import load_model
import json
from kivy.app import App
from kivy.uix.widget import Widget
from kivy.properties import ObjectProperty
from kivy.lang import Builder
from kivy.core.window import Window
from csv import writer
import cv2
from dotenv import load_dotenv
load_dotenv('support/setting.kcl')

from kivy.config import Config
# Config.set('kivy','window_icon','template/logo.png')

# load gui
Builder.load_file('template/GUI.kv')
# get file directory
if getattr(sys, 'frozen', False):
    application_path = os.path.dirname(os.path.realpath(sys.executable))
elif __file__:
    application_path = os.path.dirname(os.path.realpath(__file__))
    
from kivy.uix.popup import Popup
class MyPopup(Popup):
    
    def message(self, message):
        self.ids.labelPOPUP.text = message


class MyGridLayout(Widget):
    
    def __init__(self, **kwargs):
        self.model_path  = os.path.join(application_path, os.getenv('MODEL'))
        self.model  = load_model(self.model_path)
        self.classes = ['A', 'BA', 'CA', 'DA', 'GA', 'HA', 'JA', 'KA', 'LA', 'MA', 'MPA', 'NA', 'NCA', 'NGA', 'NGKA', 'NRA', 'NYA', 'PA', 'RA', 'SA', 'TA', 'WA', 'YA']
        self.popups = MyPopup()
        self.history = ''
        self.counter = 0
        super(MyGridLayout, self).__init__(**kwargs)

    def start(self, index):
        print('========================================')
        file_path    = self.ids.file_path.text
        if file_path == '':
            self.popup("Error, File path is invalid")
        else:
            image = cv2.imread(file_path)
            feature = self.image_to_feature_vector(image) / 255.0

            probs = self.model.predict(feature)[0]
            prediction = probs.argmax(axis=0)
            self.popup(F"Sukses, Gambar Terprediksi {self.classes[prediction]}")
            
            self.history += f'{self.classes[prediction]}, '

        self.ids.labelHistory.text = self.history
        if self.counter == 18:
            self.counter = 0
            self.history = ''
        else:
            self.counter += 1

    def image_to_feature_vector(self, image, size=(32, 32)):
        # resize the image to a fixed size, then flatten the image into
        # a list of raw pixel intensities
        return np.array([cv2.resize(image, size).flatten()])

    def popup(self, message):
        
        self.popups.message(message)
        self.popups.open()



class SetupApp(App):
    icon = 'template/logo.png'
    def build(self):
        self.title = os.getenv('TITLE_aps')
        self.icon = 'template/logo.png'
        Window.size = (650, 650)
        return MyGridLayout()




if __name__ == '__main__':
    SetupApp().run()