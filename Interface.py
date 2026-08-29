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
import random
# import HandCV

pygame.init()
WIDTH, HEIGHT = 1470, 810
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Smach the Perfect Day")
running = True

#------------------- Set up graphics -------------------#
# Music beats
BPM = 115
seconds_per_beat = 60 / BPM
NUM_BEATS = 100
beat_times = [i * seconds_per_beat for i in range(1, NUM_BEATS + 1) if i % 2 == 0]
spawned_indices = set()
elapsed_ms = 0  

# Load the image
roundDrums1_image = pygame.image.load("images/roundDrums.png")
roundDrums1_image = pygame.transform.scale(roundDrums1_image, (400, 400))  # Resize the image to 200x200 pixels
roundDrums1_x, roundDrums1_y = 0, 450

roundDrums2_image = pygame.image.load("images/roundDrums.png")
roundDrums2_image = pygame.transform.scale(roundDrums2_image, (400, 400))  # Resize the image to 200x200 pixels
roundDrums2_x, roundDrums2_y = 1060, 450

hamburger_image = pygame.image.load("images/hamburger.png")
hamburger_image = pygame.transform.scale(hamburger_image, (150, 150))  # Resize the image to 200x200 pixels

image_2 = pygame.image.load("images/pizza.png")
image_2 = pygame.transform.scale(image_2, (150, 150))  # Resize the image to 200x200 pixels

image_3 = pygame.image.load("images/cat.png")
image_3 = pygame.transform.scale(image_3, (150, 150))  # Resize the image to 200x200 pixels

image_4 = pygame.image.load("images/dog.png")
image_4 = pygame.transform.scale(image_4, (150, 150))  # Resize the image to 200x200 pixels

image_5 = pygame.image.load("images/nintendo.png")
image_5 = pygame.transform.scale(image_5, (200, 150))  # Resize the image to 200x200 pixels

object_images = [hamburger_image, image_2, image_3, image_4, image_5]

# Heart/ Lives
heart_images = [
    pygame.image.load("images/heart_frame1.png"),
    pygame.image.load("images/heart_frame2.png"),
    ]
heart_images = [pygame.transform.scale(img, (40, 40)) for img in heart_images]

current_frame = 0
frame_timer = 0
clock = pygame.time.Clock()   

# Create five of them
lives = 5

# Listing out the falling objects
falling_objects = []
FALL_SPEED = 30
spawn_count = 0

def spawn_object():
    global spawn_count
    x_positions = [100, 1250]
    spawn_x = x_positions[spawn_count % 2]
    chosen_image = random.choice(object_images) # Make them random
    falling_objects.append({
        "x": spawn_x,
        "y": -200,
        "image": chosen_image,
    })
    spawn_count += 1

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

def get_hand_state(frame):
    return {
        "left_hand":  {"x": 400, "y": 300, "moving_down": True},
        "right_hand": {"x": 800, "y": 300, "moving_down": True},
    }

while running:
    ret, frame = cap.read() #reading the next frame from the webcam
    if not ret: # Create fallback to prevent crashing
        break
    frame = cv2.resize(frame, (WIDTH, HEIGHT))
    surface = cv2_frame_to_pygame_surface(frame)
    screen.blit(surface, (0, 0))   # draw camera feed as background

    # Adding the images/ graphics to the screen
    screen.blit(roundDrums1_image, (roundDrums1_x, roundDrums1_y))
    screen.blit(roundDrums2_image, (roundDrums2_x, roundDrums2_y))

    # Animate the heart so it is bumping
    frame_timer += 1
    if frame_timer >= 10:          # change frame every 10 game loops
        current_frame = (current_frame + 1) % len(heart_images)
        frame_timer = 0        

    heart_image = heart_images[current_frame]

    for index in range(lives):
        screen.blit(heart_image, (10 + index * 40, 0))

    # ----------------------- Set up falling objects ----------------------- #
    elapsed_ms += clock.get_time()   # milliseconds since the last frame
    elapsed_sec = elapsed_ms / 1000

    for index, beat_t in enumerate(beat_times):
        if index not in spawned_indices and elapsed_sec >= beat_t:
            spawn_object()
            spawned_indices.add(index)

    # Iterate through all the falling items in my list
    for obj in falling_objects:
        obj["y"] += FALL_SPEED

    # Prevents the object to fall out
    falling_objects = [obj for obj in falling_objects if obj["y"] < HEIGHT]  # drop off-screen ones

    # looping through the remaining item
    for obj in falling_objects:
        screen.blit(obj["image"], (obj["x"], obj["y"]))

    pygame.display.flip() # Display the flipped screen
    clock.tick(30)   # locks the loop to a steady 30 frames per second

    # Handling the quiting event, so webcam can be closed
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

cap.release() # To release the webcam
pygame.quit() # To close the pygame window