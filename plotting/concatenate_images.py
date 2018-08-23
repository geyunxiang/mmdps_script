"""
Concatenate images
"""
import os
from PIL import Image, ImageDraw, ImageFont

# CS: daihuachen, guojiye, tanenci, wangjingying
# SCI: guoqian, hongqingfeng, kangcuiping, mahuibo, qiutongyu
imagePath = ['C:/Users/geyx/Desktop/2018 paper/distance score/SCI/' + imageFileName for imageFileName in sorted(os.listdir('C:/Users/geyx/Desktop/2018 paper/distance score/SCI/'))]
print(imagePath)

images = map(Image.open, imagePath)
images = list(images)
width, height = images[0].size

margin = 100
numScore = 3
numSubject = 4
new_im = Image.new('RGB', (width * numScore + margin, height * numSubject + margin), 'white')
x_offset = margin
y_offset = margin

counter = 0
for im in images:
	new_im.paste(im, (x_offset, y_offset))
	x_offset += im.size[0]
	counter += 1
	if counter % numScore == 0:
		y_offset += im.size[1]
		x_offset = margin

draw = ImageDraw.Draw(new_im)
font = ImageFont.truetype('arial.ttf', 72)

# decorate image - SCI
w, h = draw.textsize('SCIM', font = font)
draw.text((margin + width/2 - w/2, margin/2 - h/2), 'SCIM', (0, 0, 0), font=font)

w, h = draw.textsize('Lower limb', font = font)
draw.text((margin + 3*width/2 - w/2, margin/2 - h/2), 'Lower limb', (0, 0, 0), font=font)

w, h = draw.textsize('Sensory', font = font)
draw.text((margin + 5*width/2 - w/2, margin/2 - h/2), 'Sensory', (0, 0, 0), font=font)

w, h = draw.textsize('A', font = font)
draw.text((margin/2 - w/2, margin + height/2 - h/2), 'A', (0, 0, 0), font=font)

w, h = draw.textsize('B', font = font)
draw.text((margin/2 - w/2, margin + 3*height/2 - h/2), 'B', (0, 0, 0), font=font)

w, h = draw.textsize('C', font = font)
draw.text((margin/2 - w/2, margin + 5*height/2 - h/2), 'C', (0, 0, 0), font=font)

w, h = draw.textsize('D', font = font)
draw.text((margin/2 - w/2, margin + 7*height/2 - h/2), 'D', (0, 0, 0), font=font)

w, h = draw.textsize('E', font = font)
draw.text((margin/2 - w/2, margin + 9*height/2 - h/2), 'E', (0, 0, 0), font=font)

# decorate image - CS
# w, h = draw.textsize('ARAT', font = font)
# draw.text((margin + width/2 - w/2, margin/2 - h/2), 'ARAT', (0, 0, 0), font=font)

# w, h = draw.textsize('FMA', font = font)
# draw.text((margin + 3*width/2 - w/2, margin/2 - h/2), 'FMA', (0, 0, 0), font=font)

# w, h = draw.textsize('WOLF', font = font)
# draw.text((margin + 5*width/2 - w/2, margin/2 - h/2), 'WOLF', (0, 0, 0), font=font)

# w, h = draw.textsize('A', font = font)
# draw.text((margin/2 - w/2, margin + height/2 - h/2), 'A', (0, 0, 0), font=font)

# w, h = draw.textsize('B', font = font)
# draw.text((margin/2 - w/2, margin + 3*height/2 - h/2), 'B', (0, 0, 0), font=font)

# w, h = draw.textsize('C', font = font)
# draw.text((margin/2 - w/2, margin + 5*height/2 - h/2), 'C', (0, 0, 0), font=font)

# w, h = draw.textsize('D', font = font)
# draw.text((margin/2 - w/2, margin + 7*height/2 - h/2), 'D', (0, 0, 0), font=font)

new_im.save('C:/Users/geyx/Desktop/2018 paper/distance score/SCI/test.png')
