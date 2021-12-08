import keyboard
import numpy as np
from sklearn.preprocessing import normalize
import os
from scipy.io import wavfile
import warnings
warnings.filterwarnings('ignore')

import threading
import time
import pandas as pd

from time import perf_counter


# Recording signals from guitar in loop
from threading import Thread, Lock
from time import sleep
lock = Lock()
stop_thread = False
def infinit_worker():
    print("Start infinit_worker()")
    while True:
        recognision_1(recording())

        lock.acquire()
        if stop_thread is True:
            break
        lock.release()
        sleep(0.01)
    print("Stop infinit_worker()")
# Create and start thread
th = Thread(target=infinit_worker)
th.start()
sleep(300)
# Stop thread
lock.acquire()
stop_thread = True
lock.release()


# Calculating derivative for a sample to separate sound samples from regions with zero volume
def peak_f(k, window: int, min_der: int):
    der = []
    for _ in range(1, k.shape[0], window):
        der.append(np.array([abs(k[i])-abs(k[i-1]) for i in range(_-window, _)]).mean())
    
    dotes = []
    for t, z in enumerate(der):
        if abs(z) > min_der:
            dotes.append(t)
    
    return der, dotes 

# Returns list of samples ready for calculating wavelength
def autosampling(frames):
    der, dotes = peak_f(frames[0], 10, 30)
    if dotes == []:
        print('Empty frame!')
        
        pass
        
    series = []
    ser = []
    dotes = [10*t for t in dotes]   
    
    for _ in range(len(dotes)):

        if dotes[_] - dotes[_-1] < 10000:
            ser.append(dotes[_])
        else:
            if ser != []:
                series.append(ser)
                ser = []
                pass
    
    sounds = []
    
    for _ in series:
        obj = frames[0][_[0]:_[-1]]
        sounds.append(obj)

   
    return sounds



# Calculating rolling mean helps to decrese wave sharpness (works like lowpass filter)
def rolling_mean(series, n):
    dotes = []
    if not isinstance(series, pd.Series):
        series = pd.Series(series)

    for _ in range(len(series)):
        try:
            
            dotes.append(series.loc[_:_+(n-1)].values.mean())
        except:
            pass
    return dotes





# Calculating wavelength
def search_trooth(wave):
    test_fg = []

    for el in range(len(wave[:1000])):
        if 0.01 < wave[el]:
            test_fg.append(wave[:1000][el])
        else:
            test_fg.append(0)
  

    are = []
    art = []
    coords = []
    coorbs = []
    for i in range(1, len(test_fg)):
    
        if test_fg[i] == 0:
      
            if are != []:
                art.append(are)
                coorbs.append(coords)
            are = []
            coords = []
      
        else:
            are.append(test_fg[i])
            coords.append(i)


    max_values = []

    for t in range(len(art)):
        for kg in range(len(art[t])):
            if art[t][kg] == max(art[t]):
                max_values.append(coorbs[t][kg])

    KL = []
    for el in range(1, len(max_values)):
        KL.append(max_values[el] - max_values[el-1])
    KL = np.array(KL)
    wavelength = KL.mean()
    return 44100/wavelength




def recognision_1(frames):

    dickt = {0: "B6", 1:"C#6", 2:"A#6", 3:"D6", 4:"D#6", 5:"C#6", 6:"G#1", 7:"A#1", 
            8:"C#1", 9:"D#1", 10:"B2", 11:"C2", 12:"C#2", 13:"D2", 14:"D#2", 15:"E2", 16:"F2",
            17:"F#2", 18:"G2", 19:"G#2", 20:"A2", 21:"A#2", 22:"G3", 23:"G#3", 24:"A3", 25:"A#3",
            26:"B3", 27:"C3", 28:"C#3", 29:"D3", 30:"D#3", 31:"E3", 32:"F3", 33:"F#3", 34:"D4",
            35:"D#4", 36:"E4", 37:"F4", 38:"F#4", 39:"G4", 40:"G#4", 41:"A4", 42:"A#4", 43:"B4", 44:"C4",
            45:"C#4", 46:"A5", 47:"A#5", 48:"B5", 49:"C5", 50:"C#5", 51:"D5", 52:"D#5", 53:"E5", 54:"F5",
            55:"F#5", 56:"G5", 57:"G#5", 58:"G#6", 59:"A6", 60:"C6", 61:"E6", 62:"F6", 63:"F#6", 64:"G6",
            65:"A1", 66:"B1", 67:"C1", 68:"D1", 69:"E1", 70:"F1", 71:"G1"}

    keys_rus = {"E1": '1', "F1": '2', "F#1": '3', "G1": '4', "G#1": '5', "A1": '6', "A#1": '7', "B1": '8', "C1": '9', "C#1": '0', "D1": '=', "D#1": 'backspace',
            "B2": 'tab', "C2": 'й', "C#2": 'ц', "D2": 'у', "D#2": 'к', "E2": 'е', "F2": 'н', "F#2": 'г', "G2": 'ш', "G#2": 'щ', "A2": 'з', "A#2": 'х',
            "G3": 'ф', "G#3": 'ы', "A3": 'в', "A#3": 'а', "B3": 'п', "C3": 'р', "C#3": 'о', "D3": 'л', "D#3": 'д', "E3": 'ж', "F3": 'э', "F#3": 'enter',
            "D4": 'я', "D#4": 'ч', "E4": 'с', "F4": 'м', "F#4": 'и', "G4": 'т', "G#4": 'ь', "A4": 'б', "A#4": 'ю', "B4": '\.', "C4": 'shift',
             "A5": 'ctrl',  "A#5": 'fn', "B5": 'alt', "C5": 'space'}

    keys_eng = {"E1": '1', "F1": '2', "F#1": '3', "G1": '4', "G#1": '5', "A1": '6', "A#1": '7', "B1": '8', "C1": '9', "C#1": '0', "D1": '=', "D#1": 'backspace',
            "B2": 'tab', "C2": 'q', "C#2": 'w', "D2": 'e', "D#2": 'r', "E2": 't', "F2": 'y', "F#2": 'u', "G2": 'i', "G#2": 'o', "A2": 'p', "A#2": '\\',
            "G3": 'a', "G#3": 's', "A3": 'd', "A#3": 'f', "B3": 'g', "C3": 'h', "C#3": 'j', "D3": 'k', "D#3": 'l', "E3": ':', "F3": "'", "F#3": 'enter',
            "D4": 'z', "D#4": 'x', "E4": 'c', "F4": 'v', "F#4": 'b', "G4": 'n', "G#4": 'm', "A4": '\,', "A#4": '\.', "B4": '?', "C4": 'shift',
             "A5": 'ctrl',  "A#5": 'fn', "B5": 'alt', "C5": 'space'}

    keys_gaming = {"E1": '1', "F1": '2', "F#1": '3', "G1": '4', "G#1": '5', "A1": '8', "A#1": '9', "B1": 'num lock', "C1": '6', "C#1": '7', "D1": 'space', "D#1": 'esc',
            "B2": 'num lock', "C2": '6', "C#2": '7', "D2": 'space', "D#2": 'esc', "E2": '1', "F2": '2', "F#2": '3', "G2": '4', "G#2": '5', "A2": '8', "A#2": '9',
            "G3": '4', "G#3": '5', "A3": '8', "A#3": '9', "B3": 'num lock', "C3": '6', "C#3": '7', "D3": 'space', "D#3": 'esc', "E3": '1', "F3": "2", "F#3": '3',
            "D4": 'space', "D#4": 'esc', "E4": '1', "F4": '2', "F#4": '3', "G4": '4', "G#4": '5', "A4": '8', "A#4": '9', "B4": 'num lock', "C4": '6', "C#4": '7',
             "A5": '8',  "A#5": '9', "B5": 'num lock', "C5": '6', "C#5": '7', "D5": 'space', "D#5": 'esc',"E6": 's',  "F6": 'd', "F#6": 'x', "G6": 'c', "G#6": 'm', "A6": 'space', 'E1_2': 'm'}

    frames = autosampling(frames)

    z_z = None
    k_k = None

    for el in frames:
        
        if el[1000:].mean() == 0:
            pass
        else:

            value = search_trooth(rolling_mean(el, 48))
            try:
                value = int(value)
            except:
                pass
            if value in range(640, 678):
                z_z = 'E1_2'
                k_k = 'E1_2'
                print('E1_2')

            elif value in range(604, 639):
                z_z = 'D#1'
                k_k = 'E1_2'
                print('D#1')

            elif value in range(571, 603):
                z_z = 'D1'
                k_k = 'D#1'
                print('D1')

            elif value in range(538, 570):
                z_z = 'C#1'
                k_k = 'D1'
                print('C#1')

            elif value in range(509, 537):
                z_z = 'C1'
                k_k = 'C#1'
                print('C1')

            elif value in range(480, 508):
                z_z = 'B1'
                k_k = 'C1'
                print('B1')

            elif value in range(453, 479):
                z_z = 'A#1'
                k_k = 'G1'
                print('A#1 or A#2')

            elif value in range(428, 452):
                z_z = 'A1'
                k_k = 'A#1'
                print('A1 or A2')

            elif value in range(404, 427):
                z_z = 'G#1'
                k_k = 'A1'
                print('G#1 or G#2')

            elif value in range(381, 403):
                z_z = 'G1'
                k_k = 'G#1'
                print('G1 or G2')

            elif value in range(360, 380):
                z_z = 'F#1'
                k_k = 'G1'
                print('F#1 or F#2 or F#3')

            elif value in range(340, 359):
                z_z = 'F1'
                k_k = 'F#1'
                print('F1 or F2 or F3')

            elif value in range(321, 339):
                z_z = 'E1'
                k_k = 'F1'
                print('E1 or E2 or E3')

            elif value in range(303, 320):
                z_z = 'D#1'
                k_k = 'E1'
                print('D#1 or D#2')

            elif value in range(286, 302):
                z_z = 'D2'
                k_k = 'D#2'
                print('D2 or D3')

            elif value in range(270, 285):
                z_z = 'C#2'
                k_k = 'E2'
                print('C#2 or C#3 or C#4')

            elif value in range(255, 269):
                z_z = 'C2'
                k_k = 'C#2'
                print('C2 or C3 or C4')

            elif value in range(240, 254):
                z_z = 'B2'
                k_k = 'C2'
                print('B2 or B3 or B4')

            elif value in range(227, 239):
                z_z = 'A#3'
                k_k = 'B3'
                print('A#3 or A#4')

            elif value in range(215, 226):
                z_z = 'A3'
                k_k = 'A#3'
                print('A3 or A4')

            elif value in range(202, 214):
                z_z = 'G#3'
                k_k = 'A3'
                print('G#3 or G#4 or G#5')

            elif value in range(191, 201):
                z_z = 'G3'
                k_k = 'G#3'
                print('G3 or G4 or G5')

            elif value in range(181, 190):
                z_z = 'F#4'
                k_k = 'G4'
                print('F#4 or F#5')

            elif value in range(170, 180):
                z_z = 'F4'
                k_k = 'F#4'
                print('F4 or F5')

            elif value in range(161, 169):
                z_z = 'E4'
                k_k = 'F4'
                print('E4 or E5')

            elif value in range(152, 160):
                z_z = 'D#4'
                k_k = 'E4'
                print('D#4 or D#5 or D#6')

            elif value in range(144, 151):
                z_z = 'D4'
                k_k = 'D#4'
                print('D4 or D5 or D6')

            elif value in range(136, 143):
                z_z = 'C#5'
                k_k = 'D4'
                print('C#5 or C#6')

            elif value in range(127, 135):
                z_z = 'C5'
                k_k = 'C#5'
                print('C5 or C6')

            elif value in range(121, 126):
                z_z = 'B5'
                k_k = 'C5'
                print('B5 or B6')

            elif value in range(114, 120):
                z_z = 'A#5'
                k_k = 'B5'
                print('A#5 or A#6')

            elif value in range(107, 113):
                z_z = 'A5'
                k_k = 'A#5'
                print('A5 or A6')

            elif value in range(101, 106):
                z_z = 'G#6'
                k_k = 'A5'
                print('G#6')

            elif value in range(95, 100):
                z_z = 'G6'
                k_k = 'G#6'
                print('G6')

            elif value in range(90, 94):
                z_z = 'F#6'
                k_k = 'G6'
                print('F#6')

            elif value in range(85, 89):
                z_z = 'F6'
                k_k = 'F#6'
                print('F6')

            elif value in range(79, 84):
                z_z = 'E6'
                k_k = 'F6'
                print('E6')

            if k_k != None:
                print(keys_gaming[k_k])


            try:
                if k_k not in  ['A1', 'A2', 'A3', 'A5', 'A6', 'A#1', 'A#2', 'A#3', 'A#5', 'A#6']:
                    keyboard.press(f'{keys_gaming[dickt[recog]]}')
                    keyboard.release(f'{keys_gaming[dickt[recog]]}')
                else:
                    keyboard.send(f'{keys_gaming[dickt[recog]]}', do_press=True, do_release=False)
                    time.sleep(0.1)
                    keyboard.send(f'{keys_gaming[dickt[recog]]}', do_press=False, do_release=True)
            except:
                print('key unregistered')
                pass