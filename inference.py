#!/usr/bin/env python 

import pyaudio
import librosa
import numpy as np
from numpy_ringbuffer import RingBuffer
import matplotlib.pyplot as plt
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import datasets, transforms, models
import time
from threading import Thread
import cv2
import pickle
import argparse
from IPython.display import clear_output, display
from pythonosc.udp_client import SimpleUDPClient

model=None
classes=None
ringBuffer=None
input_resolution=None
sample_rate = 44100
SpectrumVariables=None
input_type=None
ringBuffer = RingBuffer(28672*2) #??
pa = None
frames_per_second=2
chunk_size = np.int(sample_rate/frames_per_second)  # divide sample_rate by n updates per second if needed, still need to add to final call 
predict=None
prob=None
stream = None
mean = None
std = None

def load_model(model_path):
    global model
    global SpectrumVariables
    global classes
    global input_resolution
    global input_type 
    global mean
    global std
    
    model_data = torch.load(model_path, map_location='cpu')
    input_resolution = model_data['resolution']
    SpectrumVariables = model_data['SpectrumVariables']
    classes = model_data['classes']
    input_type = model_data['inputType']
    mean = model_data['mean']
    std = model_data['std']
    
    found_model = False

    model = models.densenet121()
    model.classifier = nn.Linear(1024, len(classes)) 
    found_model = True
    print('model is '+model_data['modelType'])
    if not found_model:
        print('could not find requested model:', model_data['modelType'])

    model.load_state_dict(model_data['model']) 
    model.eval() # moved to here 25/2/2020 
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("device is cuda")
    else:
        device = torch.device("cpu")
        print("device is cpu")
    model.to(device)
    print('classes are: '+str(classes))


def callback(data, frame_count, time_info, flag):
    audio_data = np.frombuffer(data, dtype=np.float32)
    ringBuffer.extend(audio_data)
    return None, pyaudio.paContinue


def start_audio():
    global stream
    global pa
    print('opening audio stream')
    cv2.startWindowThread()
    pa = pyaudio.PyAudio()
    stream = pa.open(format=pyaudio.paFloat32,
                    channels=1,
                    rate=sample_rate,
                    output=False,
                    input=True,
                    frames_per_buffer=chunk_size,
                    stream_callback=callback)
    stream.start_stream()


def stop_audio():
    global pa
    global stream
    stream.close() 
    cv2.destroyAllWindows()

def infer_class_from_audio():
    global predict
    global prob
    global classes
    global input_type
    global mean
    global std
    if(not ringBuffer.is_full):
        return;
    input_resolution=SpectrumVariables["RESOLUTION"]
    
        
    cqt = librosa.core.cqt(np.array(ringBuffer), sr=SpectrumVariables["SAMPLE_RATE"], n_bins=SpectrumVariables["N_BINS"], bins_per_octave=SpectrumVariables["BPO"], hop_length=SpectrumVariables["HOP_LENGTH"])
    cqt_db = np.float32(librosa.amplitude_to_db(cqt, ref=np.max))
        
    image=cqt_db[0:input_resolution,0:input_resolution]
    image -= image.min() # ensure minimal value is 0.0
    image /= image.max() # maximum value in image is 1.0
    image*=256
    image = image.astype(np.uint8)
    color_image = cv2.applyColorMap(image, cv2.COLORMAP_BONE)
    output_image = cv2.resize(color_image[:,-input_resolution:,:], (input_resolution, input_resolution))
    cv2.imshow("rolling spectrogram", output_image) 
    cv2.waitKey(100)
    image_tensor = transforms.Compose([
        transforms.ToPILImage(), #?
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)])(output_image)   
    image_tensor = Variable(image_tensor, requires_grad=False)
    test_image = image_tensor.unsqueeze(0)

    if torch.cuda.is_available():
        test_image = test_image.cuda()
    output = model(test_image)
    output = F.softmax(output, dim=1)
    prob, predict = torch.topk(output, len(classes))
    prob = str(list(prob[0].detach().cpu().numpy())) 
    predict = str(predict[0].cpu().numpy())
    new_classes = str(list(classes))
    print(predict, prob, new_classes)
    client = SimpleUDPClient("127.0.0.1", 1337) 
    client.send_message("/neuralnet1", [predict, prob, new_classes])
    return;


def run_inference(opt): #runtime is seconds
    print("loading model data")
    load_model(opt.model_path)
    start_audio()
    t0 = time.time()
    print('running inference')
    while stream.is_active():
        infer_class_from_audio()
        if (opt.runtime > 0) and ((time.time()-t0) >= opt.runtime):
            print('stopping...')
            stop_audio()
            print("stopped and done")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--runtime', type=int)
    parser.add_argument('--model_path', type=str)
    opt = parser.parse_args()
    run_inference(opt)
