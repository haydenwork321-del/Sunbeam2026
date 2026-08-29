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

# The game was designed at this resolution — every fixed pixel value below
# (image sizes, drum positions, spawn x's, HUD placement) is scaled relative
# to it, so the layout looks proportionally the same on any screen size.
BASE_WIDTH, BASE_HEIGHT = 1470, 810

_display_info = pygame.display.Info()
WIDTH, HEIGHT = _display_info.current_w, _display_info.current_h

# Uniform scale factor: keeps images/spacing proportional (no stretching).
# Whichever axis is more constrained relative to the base design sets the
# scale, so nothing overflows the screen on very wide/narrow/tall monitors.
UI_SCALE = min(WIDTH / BASE_WIDTH, HEIGHT / BASE_HEIGHT)


def scale_val(v):
    """Scale a single length (in design-resolution px) to the real screen."""
    return int(v * UI_SCALE)


def scale_pos(x, y):
    """Scale an (x, y) position from the design resolution to the real screen."""
    return scale_val(x), scale_val(y)


screen = pygame.display.set_mode((WIDTH, HEIGHT), pygame.FULLSCREEN)
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

# Object sprite size at the design resolution — used to scale every falling
# object image and to compute their centers consistently everywhere below.
OBJ_SIZE = scale_val(150)
OBJ_HALF = OBJ_SIZE // 2

# Load the image
roundDrums1_image = pygame.image.load("images/roundDrums.png")
roundDrums1_image = pygame.transform.scale(roundDrums1_image, (scale_val(400), scale_val(400)))
roundDrums1_x, roundDrums1_y = scale_pos(0, 450)

roundDrums2_image = pygame.image.load("images/roundDrums.png")
roundDrums2_image = pygame.transform.scale(roundDrums2_image, (scale_val(400), scale_val(400)))
roundDrums2_x, roundDrums2_y = scale_pos(1060, 450)

hamburger_image = pygame.image.load("images/hamburger.png")
hamburger_image = pygame.transform.scale(hamburger_image, (OBJ_SIZE, OBJ_SIZE))

image_2 = pygame.image.load("images/pizza.png")
image_2 = pygame.transform.scale(image_2, (OBJ_SIZE, OBJ_SIZE))

image_3 = pygame.image.load("images/cat.png")
image_3 = pygame.transform.scale(image_3, (OBJ_SIZE, OBJ_SIZE))

image_4 = pygame.image.load("images/dog.png")
image_4 = pygame.transform.scale(image_4, (OBJ_SIZE, OBJ_SIZE))

image_5 = pygame.image.load("images/nintendo.png")
image_5 = pygame.transform.scale(image_5, (scale_val(200), OBJ_SIZE))

object_images = [hamburger_image, image_2, image_3, image_4, image_5]

# Heart/ Lives
heart_images = [
    pygame.image.load("images/heart_frame1.png"),
    pygame.image.load("images/heart_frame2.png"),
]
heart_images = [pygame.transform.scale(img, (scale_val(40), scale_val(40))) for img in heart_images]

current_frame = 0
frame_timer = 0
clock = pygame.time.Clock()

# Create five of them
lives = 5

# Score tracking
score = 0
font = pygame.font.SysFont(None, scale_val(60))

# Listing out the falling objects
falling_objects = []
FALL_SPEED = scale_val(30)
spawn_count = 0
HIT_RADIUS = scale_val(100)  # unused by the current zone-based check, kept for reference

# Drum hit zones — matches the visible drum image area (used for drawing)
DRUM_ZONES = [
    pygame.Rect(roundDrums1_x, roundDrums1_y, scale_val(400), scale_val(400)),  # left drum
    pygame.Rect(roundDrums2_x, roundDrums2_y, scale_val(400), scale_val(400)),  # right drum
]

# Detection zones — same drums, but padded outward so the hit check has a much
# wider, more forgiving column/top-edge than the exact drum artwork. Used only
# for collision checks, not for drawing.
DETECTION_MARGIN = scale_val(120)  # extra room added on every side of each drum
DETECTION_ZONES = [
    pygame.Rect(
        zone.left - DETECTION_MARGIN,
        zone.top - DETECTION_MARGIN,
        zone.width + 2 * DETECTION_MARGIN,
        zone.height + 2 * DETECTION_MARGIN,
    )
    for zone in DRUM_ZONES
]


def spawn_object():
    global spawn_count
    x_positions = [scale_val(100), scale_val(1250)]
    spawn_x = x_positions[spawn_count % 2]
    chosen_image = random.choice(object_images)
    falling_objects.append({
        "x": spawn_x,
        "y": -OBJ_SIZE,
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
        screen.blit(heart_image, (scale_val(10) + index * scale_val(40), 0))

    # ----------------------- Falling objects ----------------------- #
    elapsed_ms += clock.get_time()
    elapsed_sec = elapsed_ms / 1000

    for index, beat_t in enumerate(beat_times):
        if index not in spawned_indices and elapsed_sec >= beat_t:
            spawn_object()
            spawned_indices.add(index)

    for obj in falling_objects:
        obj["y"] += FALL_SPEED

    # --- collision check: hit only counts on the drums ---
    # An object is "on" a drum once it's in that drum's horizontal column
    # AND has fallen at/past the drum's top edge (its y coordinate).
    # A hit needs a hand in that same column, also at/past that same top
    # edge, moving down — no distance/radius math needed.
    for obj in falling_objects:
        if obj.get("hit"):
            continue
        obj_center_x = obj["x"] + OBJ_HALF
        obj_center_y = obj["y"] + OBJ_HALF

        # which drum column (if any) does this object line up with?
        obj_zone = None
        for zone in DETECTION_ZONES:
            if zone.left <= obj_center_x <= zone.right:
                obj_zone = zone
                break
        if obj_zone is None:
            continue  # object isn't over a drum, can't be hit

        # object must have reached the drum's top edge
        if obj_center_y < obj_zone.top:
            continue

        for hand_key in ["left_hand", "right_hand"]:
            hand = hand_state[hand_key]
            if not hand["present"] or not hand["moving_down"]:
                continue
            hx = int(hand["x"] * scale_x)
            hy = int(hand["y"] * scale_y)

            # hand must be in the same drum column and at/past its top edge
            if not (obj_zone.left <= hx <= obj_zone.right):
                continue
            if hy < obj_zone.top:
                continue

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
    screen.blit(score_text, (WIDTH - scale_val(220), scale_val(20)))

    pygame.display.flip()
    clock.tick(30)

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

cap.release()
pygame.quit()