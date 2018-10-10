"""
Concatenate images
"""
import os
from PIL import Image, ImageDraw, ImageFont

scoreList = ['lower limb movement', 'sensory', 'SCIM']
atlasNameList = ['Brodmann', 'AAL', 'AICHA', 'Brainnetome']
fileParts = 'C:/Users/geyx/Desktop/2018 paper/distance score/SCI sensorimotor 2-1/Diff 2-1 %s %s.png'
imagePath = []
for atlas in atlasNameList:
	for score in scoreList:
		imagePath.append(fileParts % (score, atlas))
# each column is a score
# each row is one atlas

images = map(Image.open, imagePath)
images = list(images)
width, height = images[0].size

wMargin = 400
hMargin = 100
numScore = len(scoreList)
numAtlas = len(atlasNameList)
new_im = Image.new('RGB', (width * numScore + wMargin, height * numAtlas + hMargin), 'white')
x_offset = wMargin
y_offset = hMargin

counter = 0
for im in images:
	new_im.paste(im, (x_offset, y_offset))
	x_offset += im.size[0]
	counter += 1
	if counter % numScore == 0:
		y_offset += im.size[1]
		x_offset = wMargin

draw = ImageDraw.Draw(new_im)
font = ImageFont.truetype('arial.ttf', 64)

# decorate image - automated
for row in range(len(scoreList)):
	w, h = draw.textsize(scoreList[row], font = font)
	draw.text((wMargin + (2*row+1)*width/2 - w/2, hMargin/2 - h/2), scoreList[row], (0, 0, 0), font=font)

for column in range(len(atlasNameList)):
	w, h = draw.textsize(atlasNameList[column], font = font)
	draw.text((wMargin/2 - w/2, hMargin + (2*column+1)*height/2 - h/2), atlasNameList[column], (0, 0, 0), font=font)

# decorate image - SCI
# w, h = draw.textsize('Lower limb', font = font)
# draw.text((wMargin + width/2 - w/2, hMargin/2 - h/2), 'Lower limb', (0, 0, 0), font=font)

# w, h = draw.textsize('Sensory', font = font)
# draw.text((wMargin + 3*width/2 - w/2, hMargin/2 - h/2), 'Sensory', (0, 0, 0), font=font)

# w, h = draw.textsize('SCIM', font = font)
# draw.text((wMargin + 5*width/2 - w/2, hMargin/2 - h/2), 'SCIM', (0, 0, 0), font=font)

# w, h = draw.textsize('Brodmann', font = font)
# draw.text((wMargin/2 - w/2, hMargin + height/2 - h/2), 'Brodmann', (0, 0, 0), font=font)

# # w, h = draw.textsize('Brodmann_ce', font = font)
# # draw.text((wMargin/2 - w/2, hMargin + 3*height/2 - h/2), 'Brodmann_ce', (0, 0, 0), font=font)

# w, h = draw.textsize('AAL', font = font)
# draw.text((wMargin/2 - w/2, hMargin + 3*height/2 - h/2), 'AAL', (0, 0, 0), font=font)

# w, h = draw.textsize('AICHA', font = font)
# draw.text((wMargin/2 - w/2, hMargin + 5*height/2 - h/2), 'AICHA', (0, 0, 0), font=font)

# w, h = draw.textsize('Brainnetome', font = font)
# draw.text((wMargin/2 - w/2, hMargin + 7*height/2 - h/2), 'Brainnetome', (0, 0, 0), font=font)

new_im.save('C:/Users/geyx/Desktop/2018 paper/distance score/SCI sensorimotor 2-1/Diff 2-1 all.png')
