#!/usr/bin/env python 

import os
from pathlib import Path
import argparse
import wave
from pydub import AudioSegment
from pydub.utils import make_chunks
import numpy as np
import glob
import librosa
import librosa.display
import matplotlib.pyplot as plt
import cv2

# new concatenate script - works up to 4GB 

def concatenate(opt):

    import wave
    infiles = []

    import os
    for root, dirs, files in os.walk(opt.src_path):
        for file in files:
            if file.endswith(".wav"):
                infiles.append(os.path.join(root, file)) 
                
    data= []
    for infile in infiles:
        w = wave.open(infile, 'rb')
        data.append( [w.getparams(), w.readframes(w.getnframes())] )
        w.close()

    output = wave.open(opt.src_path+'/concat.wav', 'wb')
    output.setparams(data[0][0])
    for i in range(len(data)): 
        output.writeframes(data[i][1])
    print('concatenated audio files')

def resample(opt):

    print('resampling audio (this can take a while)')
    sound = AudioSegment.from_file(opt.src_path+'/concat.wav', format='wav')
    sound = sound.set_frame_rate(opt.sr)
    sound.export(opt.src_path+'/concat_{}.wav'.format(opt.sr), format='wav')
    #del(opt.src_path+'/concat.wav')
    print('resampled audio')

def slice_audio(opt):
    
    src_file = AudioSegment.from_wav(opt.src_path+'/concat_{}.wav'.format(opt.sr))
    chunks = make_chunks(src_file, opt.slice_len)
    del chunks[-1] # delete last leftover file 
    slice_path = opt.src_path+'/sliced_{}/'.format(opt.slice_len)
    if not os.path.exists(slice_path):
        os.makedirs(slice_path)
        print(slice_path + ' directory created')
    for i, chunk in enumerate(chunks):
        chunk_name = slice_path+'{}.wav'.format(i)
        chunk.export(chunk_name, format='wav')
    print('sliced into {} ms chunks'.format(opt.slice_len))
    #del(opt.src_path+'/concat_22050.wav')

def generate_spectrograms(opt): # not sure of best val for dpi

    dst_path=''
    files = Path(opt.src_path+'/sliced_{}/'.format(opt.slice_len)).glob('*.wav')
    for filename in files:
        
        if opt.image_type == 'chroma':            

            y, sample_rate = librosa.load(filename, sr=None)
            chroma = librosa.feature.chroma_stft(y, 
                                                 sr=opt.sr, 
                                                 n_fft=1024, 
                                                 hop_length=512, 
                                                 n_chroma=12) #halves sr unless declare sr=None
            chroma = np.float32(chroma, ref=np.max) # not sure if ref=np.max necessary 
            
            image = chroma;
            image -= image.min() # ensure the minimal value is 0.0
            image /= image.max() # maximum value in image is now 1.0
            image*=opt.image_res
            image = image.astype(np.uint8)
            color_image = cv2.applyColorMap(image, cv2.COLORMAP_BONE)
            output_image = cv2.resize(color_image[:,-opt.mage_res:,:],(opt.image_res, opt.image_res)) # David wrote this 
            plt.axis('off')
            dst_path = opt.src_path+'/chromagrams/'
            if not os.path.exists(dst_path):
                os.makedirs(dst_path)
                print('created new directory:', dst_path)
            plt.imsave(dst_path+'{}.png'.format(filename.name), output_image)
            
        if opt.image_type == 'cqt':
            
            y, sr = librosa.load(filename,mono=True, sr=None) # will halve sr unless specified as sr=None
            cqt = librosa.core.cqt(y, 
                                   sr=opt.sr, 
                                   fmin=90, 
                                   n_bins=168, 
                                   bins_per_octave=24, 
                                   hop_length=512)
            cqt_db = np.float32(librosa.amplitude_to_db(cqt, ref=np.max))
            
            image = cqt_db;
            image -= image.min() # ensure the minimal value is 0.0
            image /= image.max() # maximum value in image is now 1.0
            image*=opt.image_res
            image = image.astype(np.uint8)
            color_image = cv2.applyColorMap(image, cv2.COLORMAP_BONE)
            output_image = cv2.resize(color_image[:,-opt.image_res:,:],(opt.image_res, opt.image_res))
            plt.axis('off')
            dst_path = opt.src_path+'/cqt_spectrograms/'
            if not os.path.exists(dst_path):
                os.makedirs(dst_path)
                print('created new directory:', dst_path)
            plt.imsave(dst_path+'{}.png'.format(filename.name), output_image)
            
        if opt.image_type == 'mel':

            y, sample_rate = librosa.load(filename, mono=True, sr=None)
            spectrogram = librosa.feature.melspectrogram(y, 
                                                         sr=opt.sr, 
                                                         n_fft=1024, 
                                                        hop_length=512, 
                                                        n_mels=128, 
                                                         power=2.0,
                                                       fmin=40, 
                                                         fmax=22050) 
            log_spectrogram = np.float32(librosa.power_to_db(spectrogram, ref=np.max))
            
            image = log_spectrogram;
            image -= image.min() # ensure the minimal value is 0.0
            image /= image.max() # maximum value in image is now 1.0
            image*=opt.image_res
            image = image.astype(np.uint8)
            color_image = cv2.applyColorMap(image, cv2.COLORMAP_BONE)
            output_image = cv2.resize(color_image[:,-opt.image_res:,:],(opt.image_res, opt.image_res))
            plt.axis('off')
            dst_path = opt.src_path+'/mel_spectrograms/' # name of new folder 
            if not os.path.exists(dst_path):
                os.makedirs(dst_path)
                print('created new directory:', dst_path)
                print('...')
            plt.imsave(dst_path+'{}.png'.format(filename.name), output_image)
    print('done creating spectrograms')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--src_path', type=str, help='path to folder of wav files')
    parser.add_argument('--sr', type=int, default=44100, help='sample_rate')
    parser.add_argument('--slice_len', type=int, default=1000, help='length of chunked audio, in milliseconds')
    parser.add_argument('--image_type', type=str, help='spectrogram type, choose from cqt (constant q transform), mel (mel spectrogram), chroma (chromagram)', required=True)
    parser.add_argument('--image_res', type=int, default=256, help='image resolution, recommend 256')
    opt = parser.parse_args()
    concatenate(opt)
    resample(opt)
    slice_audio(opt)
    generate_spectrograms(opt)
