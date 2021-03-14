import random
import time

from task2.keyboard import press_key, release_key, W, A, S, D


def forward():
    press_key(W)
    time.sleep(0.3)
    release_key(W)


def left():
    press_key(A)
    time.sleep(0.3)
    release_key(A)


def right():
    press_key(D)
    time.sleep(0.3)
    release_key(D)


def stop():
    press_key(S)
    time.sleep(0.3)
    release_key(S)


def forward_left():
    press_key(W)
    press_key(A)
    time.sleep(0.3)
    release_key(W)
    release_key(A)


def forward_right():
    press_key(W)
    press_key(D)
    time.sleep(0.3)
    release_key(W)
    release_key(D)


# def reverse_left():
#     press_key(S)
#     press_key(A)
#     release_key(W)
#     release_key(D)
#
#
# def reverse_right():
#     press_key(S)
#     press_key(D)
#     release_key(W)
#     release_key(A)
#
#
# def no_keys():
#     if random.randrange(0, 3) == 1:
#         press_key(W)
#     else:
#         release_key(W)
#     release_key(A)
#     release_key(S)
#     release_key(D)
