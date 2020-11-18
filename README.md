# Audio-CNN

# Three stages of Convolutional Neural Network adapted for use with real-time audio, with script for mel / cqt / chromagram generation from wav files, model training and inference for live input.

Partly based on https://github.com/FAR-Lab/RealtimeAudioClassification

It's recommended to have Anaconda installed - https://docs.anaconda.com/anaconda/install/

To install dependencies, create an Anaconda environment from the root of this repository:
```
conda create -n audio-cnn python=3.7 pytorch==1.4.0 torchvision==0.5.0 cudatoolkit=10.1 -c pytorch 
conda install pyaudio -c nwani
pip install -r requirements.txt

```
The above works for Windows and Linux, Mac users should omit `cudatoolkit=10.1` from the first line.  

`data preprocessing.py` will need to be run at least twice, once for each data class. 
each class should be in a subfolder of `--data_path`, you have to do this manually for now. 
Spectrogram settings are at sensible defaults for ~1000ms-length audio files but will need 
to be adjusted according to use case

`train.py` allows for variation of many useful parameters, run `train.py -h` to see 

`inference.py` takes a trained model file (`--model_path`) and runs for `--runtime` seconds with a live input. it prints its output as well as sends it via OSC. 

Included is an example SuperCollider patches for OSC communication with a the trained model. 

In `Audio-CNN_responder.scd`, model outputs are recieved by a SuperCollider patch via OSC in which samples are triggered according to class (you have to provide your own links to samples).

If you already have SuperCollider installed, you can run this patch and a provided model together by running `sh CNN_SC_parallel.sh` from inside this repository (might not work on Windows).




