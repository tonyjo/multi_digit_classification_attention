# coding: utf-8
import cv2
import string
import random
import numpy as np
from PIL import Image
from PIL.ImageDraw import Draw
from captcha import ImageCaptcha
from captcha import random_color
# Seed
np.random.seed(8964)
#--------------------------------Functions--------------------------------------
# Random dots generator
def create_noise_dots(image, color, width=2, number=125):
    draw = Draw(image)
    w, h = image.size
    while number:
        x1 = random.randint(0, w)
        y1 = random.randint(0, h)
        draw.line(((x1, y1), (x1 - 1, y1 - 1)), fill=color, width=width)
        number -= 1
    return image
# Random curve generator
def create_noise_curve(image, color):
    w, h = image.size
    x1 = random.randint(10, 15)
    x2 = random.randint(w - 10, w)
    y1 = random.randint(20, 35)
    y2 = random.randint(y1, 60)
    points = [x1, y1, x2, y2]
    end = random.randint(180, 200)
    start = random.randint(0, 20)
    Draw(image).arc(points, start, end, fill=color)
    return image
# Caption generator
image = ImageCaptcha(width=64, height=64, font_sizes=(64, 64, 64))
#-------------------------------------------------------------------------------
# Get characters
az = string.ascii_lowercase
AZ = string.ascii_uppercase
nm = string.digits
#-------------------------------------------------------------------------------
# Append all characters
all_selections = []
for i in range(len(az)):
    all_selections.append(az[i])
for i in range(len(AZ)):
    all_selections.append(AZ[i])
for i in range(len(nm)):
    all_selections.append(nm[i])
#-------------------------------------------------------------------------------

#----------------------------------MAIN-----------------------------------------
count = 0
max_chars = 6
train_set = []
dump_train_file = './datasets/captcha/train.txt'

# Atleast have all characters appear once in trainset-- generate 100 for each
for select in all_selections:
    for _ in range(1000):
        all_data = []
        all_lbls = []
        # Randomly generate a placeholder to put in the char
        insert_id  = random.randint(0, max_chars)
        # Generate numbers
        for t in range(max_chars):
            if t == insert_id:
                lbl  = select
                data = image.generate_image(select)
            else:
                idx  = random.randint(0, len(all_selections)-1)
                lbl  = all_selections[idx]
                data = image.generate_image(lbl)
            # Append
            all_lbls.append(lbl)
            all_data.append(data)
        # Get max width
        total_w = 0
        for i in all_data:
            pix = np.array(i)
            h, w, _ = pix.shape
            total_w += w
        # Get max height
        highest_h = 50
        for i in all_data:
            pix = np.array(i)
            h, w, _ = pix.shape
            if h < highest_h:
                highest_h = h
        # Begin painting
        canvas = np.ones((highest_h + 35, total_w + 35, 3)).astype(np.uint8) * 255
        prev_w = 10
        all_bbox = []
        try:
            for i in all_data:
                pix = np.array(i)
                h, w, _ = pix.shape
                # print(pix.shape)
                # Get BBox's
                h1 = 10
                w1 = prev_w
                h2 = h
                w2 = w
                all_bbox.append([w1, h1, w2, h2])
                # Paint Canvas
                canvas[10:10+h, prev_w+2:prev_w+2+w, :] = pix
                prev_w += w
            # Append to training set
            train_set.append([all_lbls, all_bbox])
            # Convert to PIL Image
            im = Image.fromarray(canvas)
            # Generate different colored dots
            color = random_color(10, 200, random.randint(220, 255))
            im = create_noise_dots(im, color)
            color = random_color(10, 200, random.randint(220, 255))
            im = create_noise_dots(im, color)
            color = random_color(10, 200, random.randint(220, 255))
            im = create_noise_curve(im, color)
            # Convert to numpy array
            canvas = np.array(im)
            # Save image
            cv2.imwrite('./datasets/captcha/%d.png' % count, canvas)
            # Increment
            count += 1
            # Progress
            if count%1000 == 0:
                print('Creating training set --Progress: ', count)
        except ValueError:
            print('Skipping......')
            continue

# Create random data afterwards
for _ in range(1400):
    all_data = []
    all_lbls = []
    # Generate numbers
    for t in range(max_chars):
        idx  = random.randint(0, len(all_selections)-1)
        lbl  = all_selections[idx]
        data = image.generate_image(lbl)
        # Append
        all_lbls.append(lbl)
        all_data.append(data)
    # Get max width
    total_w = 0
    for i in all_data:
        pix = np.array(i)
        h, w, _ = pix.shape
        total_w += w
    # Get max height
    highest_h = 50
    for i in all_data:
        pix = np.array(i)
        h, w, _ = pix.shape
        if h < highest_h:
            highest_h = h
    # Begin painting
    canvas = np.ones((highest_h + 35, total_w + 35, 3)).astype(np.uint8) * 255
    prev_w = 10
    all_bbox = []
    try:
        for i in all_data:
            pix = np.array(i)
            h, w, _ = pix.shape
            # print(pix.shape)
            # Get BBox's
            h1 = 10
            w1 = prev_w
            h2 = h
            w2 = w
            all_bbox.append([w1, h1, w2, h2])
            # Paint Canvas
            canvas[10:10+h, prev_w+2:prev_w+2+w, :] = pix
            prev_w += w
        # Append to training set
        train_set.append([all_lbls, all_bbox])
        # Convert to PIL Image
        im = Image.fromarray(canvas)
        # Generate different colored dots
        color = random_color(10, 200, random.randint(220, 255))
        im = create_noise_dots(im, color)
        color = random_color(10, 200, random.randint(220, 255))
        im = create_noise_dots(im, color)
        color = random_color(10, 200, random.randint(220, 255))
        im = create_noise_curve(im, color)
        # Convert to numpy array
        canvas = np.array(im)
        # Save image
        cv2.imwrite('./datasets/captcha/%d.png' % count, canvas)
        # Increment
        count += 1
        # Progress
        if count%1000 == 0:
            print('Creating training set --Progress: ', count)
    except ValueError:
        print('Skipping......')
        continue
print('Training Set Completed ',  len(train_set), ' ', count)

with open(dump_train_file, 'w') as f:
    for q in range(len(train_set)):
        all_data_interm, all_bbox_interm = train_set[q]
        for t in range(max_chars):
            char_i = all_data_interm[t]
            w1, h1, w2, h2 =  all_bbox_interm[t]
            f.write('%d.png, %s, %d, %d, %d, %d\n' % (q, char_i, w1, h1, w2, h2))
f.close()

# Free memory
del train_set
#-------------------------------------------------------------------------------
valid_set = []
dump_valid_file = './datasets/captcha/val.txt'
# Create random data afterwards
for _ in range(1500):
    all_data = []
    all_lbls = []
    # Generate numbers
    for t in range(max_chars):
        idx  = random.randint(0, len(all_selections)-1)
        lbl  = all_selections[idx]
        data = image.generate_image(lbl)
        # Append
        all_lbls.append(lbl)
        all_data.append(data)
    # Get max width
    total_w = 0
    for i in all_data:
        pix = np.array(i)
        h, w, _ = pix.shape
        total_w += w
    # Get max height
    highest_h = 50
    for i in all_data:
        pix = np.array(i)
        h, w, _ = pix.shape
        if h < highest_h:
            highest_h = h
    # Begin painting
    canvas = np.ones((highest_h + 35, total_w + 35, 3)).astype(np.uint8) * 255
    prev_w = 10
    all_bbox = []
    try:
        for i in all_data:
            pix = np.array(i)
            h, w, _ = pix.shape
            # print(pix.shape)
            # Get BBox's
            h1 = 10
            w1 = prev_w
            h2 = h
            w2 = w
            all_bbox.append([w1, h1, w2, h2])
            # Paint Canvas
            canvas[10:10+h, prev_w+2:prev_w+2+w, :] = pix
            prev_w += w
        # Append to training set
        train_set.append([all_lbls, all_bbox])
        # Convert to PIL Image
        im = Image.fromarray(canvas)
        # Generate different colored dots
        color = random_color(10, 200, random.randint(220, 255))
        im = create_noise_dots(im, color)
        color = random_color(10, 200, random.randint(220, 255))
        im = create_noise_dots(im, color)
        color = random_color(10, 200, random.randint(220, 255))
        im = create_noise_curve(im, color)
        # Convert to numpy array
        canvas = np.array(im)
        # Save image
        cv2.imwrite('./datasets/captcha/%d.png' % count, canvas)
        # Increment
        count += 1
        # Progress
        if count%1000 == 0:
            print('Creating validation set --Progress: ', count)
    except ValueError:
        print('Skipping......')
        continue
print('Training Set Completed ',  len(valid_set), ' ', count)

with open(dump_val_file, 'w') as f:
    for q in range(len(train_set)):
        all_data_interm, all_bbox_interm = train_set[q]
        for t in range(max_chars):
            char_i = all_data_interm[t]
            w1, h1, w2, h2 =  all_bbox_interm[t]
            f.write('%d.png, %s, %d, %d, %d, %d\n' % (q, char_i, w1, h1, w2, h2))
f.close()
#-------------------------------------------------------------------------------

test_set = []
dump_test_file = './datasets/captcha/test.txt'
# Create random data afterwards
for _ in range(1500):
    all_data = []
    all_lbls = []
    # Generate numbers
    for t in range(max_chars):
        idx  = random.randint(0, len(all_selections)-1)
        lbl  = all_selections[idx]
        data = image.generate_image(lbl)
        # Append
        all_lbls.append(lbl)
        all_data.append(data)
    # Get max width
    total_w = 0
    for i in all_data:
        pix = np.array(i)
        h, w, _ = pix.shape
        total_w += w
    # Get max height
    highest_h = 50
    for i in all_data:
        pix = np.array(i)
        h, w, _ = pix.shape
        if h < highest_h:
            highest_h = h
    # Begin painting
    canvas = np.ones((highest_h + 35, total_w + 35, 3)).astype(np.uint8) * 255
    prev_w = 10
    all_bbox = []
    try:
        for i in all_data:
            pix = np.array(i)
            h, w, _ = pix.shape
            # print(pix.shape)
            # Get BBox's
            h1 = 10
            w1 = prev_w
            h2 = h
            w2 = w
            all_bbox.append([w1, h1, w2, h2])
            # Paint Canvas
            canvas[10:10+h, prev_w+2:prev_w+2+w, :] = pix
            prev_w += w
        # Append to training set
        train_set.append([all_lbls, all_bbox])
        # Convert to PIL Image
        im = Image.fromarray(canvas)
        # Generate different colored dots
        color = random_color(10, 200, random.randint(220, 255))
        im = create_noise_dots(im, color)
        color = random_color(10, 200, random.randint(220, 255))
        im = create_noise_dots(im, color)
        color = random_color(10, 200, random.randint(220, 255))
        im = create_noise_curve(im, color)
        # Convert to numpy array
        canvas = np.array(im)
        # Save image
        cv2.imwrite('./datasets/captcha/%d.png' % count, canvas)
        # Increment
        count += 1
        # Progress
        if count%500 == 0:
            print('Creating testing set --Progress: ', count)
    except ValueError:
        print('Skipping......')
        continue
print('Testing Set Completed ',  len(test_set), ' ', count)

with open(dump_test_file, 'w') as f:
    for q in range(len(train_set)):
        all_data_interm, all_bbox_interm = train_set[q]
        for t in range(max_chars):
            char_i = all_data_interm[t]
            w1, h1, w2, h2 =  all_bbox_interm[t]
            f.write('%d.png, %s, %d, %d, %d, %d\n' % (q, char_i, w1, h1, w2, h2))
f.close()
#-------------------------------------------------------------------------------
