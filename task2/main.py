import os
import time

import cv2
import numpy as np
import win32api
from PIL import ImageGrab

import torch
import torch.nn as nn
import torchvision
from torch.autograd import Variable
from torchvision import transforms
import time
import os

from task2.moves import *

key_values = {
    'A': {
        'action': left,
    },
    'D': {
        'action': right,
    },
    'S': {
        'action': stop,
    },
    'W': {
        'action': forward,
    },
    # 'AW': {
    #     'action': forward_left,
    # },
    # 'DW': {
    #     'action': forward_right,
    # },

    # 'idle': {
    #     'action': None,
    # },
}


def collect_pressed_keys():
    keys_to_grab = list('WASD')

    keys = []
    for key in keys_to_grab:
        if win32api.GetAsyncKeyState(ord(key)):
            keys.append(key)
    if win32api.GetAsyncKeyState(ord('Q')):
        return None

    keys.sort()
    pressed = ''.join(keys)
    if pressed in key_values.keys():
        return pressed
    else:
        return 'idle'


def collect_data(delay=5, ops=5):
    while delay != 0:
        print(delay)
        time.sleep(1)
        delay -= 1

    collected_data = {key: [] for key in key_values.keys()}

    last_time = time.time()
    while True:
        screen = np.array(ImageGrab.grab(bbox=(0, 40, 800, 640)))
        delta = time.time() - last_time
        if 1 / ops > delta:
            time.sleep(1 / ops - delta)
        delta = time.time() - last_time

        print(f'{int(1 / delta)} OPS')
        pressed_key = collect_pressed_keys()
        print(pressed_key)

        if pressed_key == 'idle':
            print('\t\tSKIP')
            continue

        if pressed_key is not None:
            screen = screen[:-160, :]
            screen = cv2.resize(screen, (224, 224))
            screen = cv2.cvtColor(screen, cv2.COLOR_BGR2RGB)
            collected_data[pressed_key].append(screen)
        else:
            main_path = os.getcwd() + '/data/'
            for key, images in collected_data.items():
                dir_path = main_path + key + '/'
                os.makedirs(os.path.dirname(dir_path))
                for index, image in enumerate(images):
                    cv2.imwrite(f'{dir_path}{index + 1}.jpg', image)
            break

        last_time = time.time()
        cv2.imshow('window', screen)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break


def start_self_driving(delay=5, ops=5):
    model = torchvision.models.resnet18()
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, len(key_values.keys()))
    model.load_state_dict(torch.load('model'))
    model.eval()

    while delay != 0:
        print(delay)
        time.sleep(1)
        delay -= 1

    last_time = time.time()
    while True:
        screen = np.array(ImageGrab.grab(bbox=(0, 40, 800, 640)))
        delta = time.time() - last_time
        if 1 / ops > delta:
            time.sleep(1 / ops - delta)
        delta = time.time() - last_time

        screen = screen[:-160, :]
        screen = cv2.resize(screen, (224, 224))
        screen = cv2.cvtColor(screen, cv2.COLOR_BGR2RGB)

        to_pil = transforms.ToPILImage()
        test_transforms = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
        ])

        image_tensor = test_transforms(to_pil(screen)).float()
        image_tensor = image_tensor.unsqueeze_(0)
        output = model(Variable(image_tensor))
        print(output)

        try:
            key = list(key_values.keys())[torch.argmax(output, dim=1)[0].item()]
            action = key_values[key]['action']
            print(key)
            print(action)
            if action is not None:
                action()
        except Exception as e:
            print(e)

        last_time = time.time()
        cv2.imshow('window', screen)
        # cv2.waitKey(0)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break


# collect_data(delay=15)
start_self_driving(delay=10, ops=5)
