'''
Purpose:
This file creates the frontend graphics for the game

Features:
- Drums circles
- Scoring labels
- Lives/ Hearts x 5
- Falling objects

Source:
- https://www.youtube.com/watch?v=c2sgaT4SNBo

'''

import pygame
import cv2
import numpy as np
# import HandCV

pygame.init()
WIDTH, HEIGHT = 1470, 810
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Smach the Perfect Day")
running = True

# Setting up the webcam
cap = cv2.VideoCapture(0)  # 0 is the default camera

# Creating a function that adds the opencv frame to the pygame screen
# Using Numpy array for the colour, axis rotation, and direct surface generation
def cv2_frame_to_pygame_surface(frame):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_rgb = np.rot90(frame_rgb)  
    # cv2 and pygame disagree on axis order

    surface = pygame.surfarray.make_surface(frame_rgb)
    return surface

while running:
    ret, frame = cap.read() #reading the next frame from the webcam
    if not ret: # Create fallback to prevent crashing
        break
    frame = cv2.resize(frame, (WIDTH, HEIGHT))
    surface = cv2_frame_to_pygame_surface(frame)
    screen.blit(surface, (0, 0))   # draw camera feed as background
    pygame.display.flip() # Display the flipped screen

    # Handling the quiting event, so webcam can be closed
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

cap.release() # To release the webcam
pygame.quit() # To close the pygame window