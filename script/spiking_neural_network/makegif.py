from PIL import Image
import glob
import time

# create an empty list called images
images = []

# get the current time to use in the filename
timestr = time.strftime("%Y%m%d-%H%M%S")

# get all the images in the 'images for gif' folder
for filename in sorted(glob.glob('makegif/*.png')): # loop through all png files in the folder
    im = Image.open(filename) # open the image
    #im_small = im.resize((1200, 1500), resample=0) # resize them to make them a bit smaller
    images.append(im) # add the image to the list

# calculate the frame number of the last frame (ie the number of images)
last_frame = (len(images)) 

# create 10 extra copies of the last frame (to make the gif spend longer on the most recent data)
for x in range(0, 9):
    im = images[last_frame-1]
    images.append(im)

# save as a gif   
images[0].save('../../data/ReSuMetest_3.gif', save_all=True, append_images=images[1:], optimize=False, duration=500, loop=0)
