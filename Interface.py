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
import HandCV

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
roundDrums1_image = pygame.transform.scale(roundDrums1_image, (400, 400))
roundDrums1_x, roundDrums1_y = 0, 450

roundDrums2_image = pygame.image.load("images/roundDrums.png")
roundDrums2_image = pygame.transform.scale(roundDrums2_image, (400, 400))
roundDrums2_x, roundDrums2_y = 1060, 450

hamburger_image = pygame.image.load("images/hamburger.png")
hamburger_image = pygame.transform.scale(hamburger_image, (150, 150))

image_2 = pygame.image.load("images/pizza.png")
image_2 = pygame.transform.scale(image_2, (150, 150))

image_3 = pygame.image.load("images/cat.png")
image_3 = pygame.transform.scale(image_3, (150, 150))

image_4 = pygame.image.load("images/dog.png")
image_4 = pygame.transform.scale(image_4, (150, 150))

image_5 = pygame.image.load("images/nintendo.png")
image_5 = pygame.transform.scale(image_5, (200, 150))

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

# Score tracking
score = 0
font = pygame.font.SysFont(None, 60)

# Listing out the falling objects
falling_objects = []
FALL_SPEED = 30
spawn_count = 0
HIT_RADIUS = 100  # how close a hand needs to be to an object to count as a hit

# Drum hit zones — a hit only counts if the object AND the hand are both
# inside the same drum's rectangle (matches the visible drum image area)
DRUM_ZONES = [
    pygame.Rect(roundDrums1_x, roundDrums1_y, 400, 400),  # left drum
    pygame.Rect(roundDrums2_x, roundDrums2_y, 400, 400),  # right drum
]


def spawn_object():
    global spawn_count
    x_positions = [100, 1250]
    spawn_x = x_positions[spawn_count % 2]
    chosen_image = random.choice(object_images)
    falling_objects.append({
        "x": spawn_x,
        "y": -200,
        "image": chosen_image,
    })
    spawn_count += 1


# Setting up the webcam
cap = cv2.VideoCapture(0)


def cv2_frame_to_pygame_surface(frame):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_rgb = np.rot90(frame_rgb)
    surface = pygame.surfarray.make_surface(frame_rgb)
    return surface


while running:
    ret, frame = cap.read()
    if not ret:
        break

    hand_state = HandCV.get_hand_state(frame)

    orig_h, orig_w = frame.shape[:2]  # capture ORIGINAL camera size before resizing

    frame = cv2.resize(frame, (WIDTH, HEIGHT))

    # --- draw hand markers (debug: big red circles) ---
    scale_x = WIDTH / orig_w
    scale_y = HEIGHT / orig_h

    for hand_key in ["left_hand", "right_hand"]:
        hand = hand_state[hand_key]
        if hand["present"]:
            for (lx, ly) in hand["landmarks"]:
                hx = int(lx * scale_x)
                hy = int(ly * scale_y)
                cv2.circle(frame, (hx, hy), 8, (0, 255, 255), -1)

    surface = cv2_frame_to_pygame_surface(frame)
    screen.blit(surface, (0, 0))

    # Adding the images/graphics to the screen
    screen.blit(roundDrums1_image, (roundDrums1_x, roundDrums1_y))
    screen.blit(roundDrums2_image, (roundDrums2_x, roundDrums2_y))

    # Animate the heart so it is bumping
    frame_timer += 1
    if frame_timer >= 10:
        current_frame = (current_frame + 1) % len(heart_images)
        frame_timer = 0

    heart_image = heart_images[current_frame]

    for index in range(lives):
        screen.blit(heart_image, (10 + index * 40, 0))

    # ----------------------- Falling objects ----------------------- #
    elapsed_ms += clock.get_time()
    elapsed_sec = elapsed_ms / 1000

    for index, beat_t in enumerate(beat_times):
        if index not in spawned_indices and elapsed_sec >= beat_t:
            spawn_object()
            spawned_indices.add(index)

    for obj in falling_objects:
        obj["y"] += FALL_SPEED

    # --- collision check: hit only counts inside a drum zone ---
    for obj in falling_objects:
        if obj.get("hit"):
            continue
        obj_center_x = obj["x"] + 75  # objects are 150x150, so center is +75
        obj_center_y = obj["y"] + 75

        # which drum zone (if any) is this object currently inside?
        obj_zone = None
        for zone in DRUM_ZONES:
            if zone.collidepoint(obj_center_x, obj_center_y):
                obj_zone = zone
                break
        if obj_zone is None:
            continue  # object hasn't reached a drum yet, can't be hit

        for hand_key in ["left_hand", "right_hand"]:
            hand = hand_state[hand_key]
            if not hand["present"] or not hand["moving_down"]:
                continue
            hx = int(hand["x"] * scale_x)
            hy = int(hand["y"] * scale_y)

            # hand must ALSO be inside that same drum zone
            if not obj_zone.collidepoint(hx, hy):
                continue

            distance = ((hx - obj_center_x) ** 2 + (hy - obj_center_y) ** 2) ** 0.5
            if distance < HIT_RADIUS:
                obj["hit"] = True
                score += 1
                break

    # remove hit objects immediately; remove missed ones that fell off-screen (lose a life)
    still_falling = []
    for obj in falling_objects:
        if obj.get("hit"):
            continue  # disappears immediately on hit
        if obj["y"] >= HEIGHT:
            lives -= 1
            continue  # missed, slides off, lose a life
        still_falling.append(obj)
    falling_objects = still_falling

    for obj in falling_objects:
        screen.blit(obj["image"], (obj["x"], obj["y"]))

    # --- score display ---
    score_text = font.render(f"Score: {score}", True, (255, 255, 255))
    screen.blit(score_text, (WIDTH - 220, 20))

    pygame.display.flip()
    clock.tick(30)

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

cap.release()
pygame.quit()