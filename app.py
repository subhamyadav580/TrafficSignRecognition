import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import gradio as gr
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

classes = { 0:'Speed limit (20km/h)',
            1:'Speed limit (30km/h)', 
            2:'Speed limit (50km/h)', 
            3:'Speed limit (60km/h)', 
            4:'Speed limit (70km/h)', 
            5:'Speed limit (80km/h)', 
            6:'End of speed limit (80km/h)', 
            7:'Speed limit (100km/h)', 
            8:'Speed limit (120km/h)', 
            9:'No passing', 
            10:'No passing veh over 3.5 tons', 
            11:'Right-of-way at intersection', 
            12:'Priority road', 
            13:'Yield', 
            14:'Stop', 
            15:'No vehicles', 
            16:'Veh > 3.5 tons prohibited', 
            17:'No entry', 
            18:'General caution', 
            19:'Dangerous curve left', 
            20:'Dangerous curve right', 
            21:'Double curve', 
            22:'Bumpy road', 
            23:'Slippery road', 
            24:'Road narrows on the right', 
            25:'Road work', 
            26:'Traffic signals', 
            27:'Pedestrians', 
            28:'Children crossing', 
            29:'Bicycles crossing', 
            30:'Beware of ice/snow',
            31:'Wild animals crossing', 
            32:'End speed + passing limits', 
            33:'Turn right ahead', 
            34:'Turn left ahead', 
            35:'Ahead only', 
            36:'Go straight or right', 
            37:'Go straight or left', 
            38:'Keep right', 
            39:'Keep left', 
            40:'Roundabout mandatory', 
            41:'End of no passing', 
            42:'End no passing veh > 3.5 tons' }


def transform_images(img):
    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Resize((30, 30)),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )
    return transform(img)


model = torch.jit.load('traffic_sign_cpu.pt')
model.eval()

def classify_image(img):
    image = transform_images(img).to(device)
    outputs = model(image)
    _, predicted = torch.max(outputs.data, 1)
    return classes[int(predicted[0])]


image = gr.inputs.Image(shape=(30,30))
label = gr.outputs.Label()
examples = ['002_0003_j.png', '054_0024_j.png', '056_1_0001_1_j.png', '003_1_0009_1_j.png', '055_1_0005_1_j.png', '056_1_0013_1_j.png']

intf = gr.Interface(fn=classify_image, inputs=image, outputs=label, examples=examples)
intf.launch(inline=False)