#!/usr/bin/python


from PIL import Image
import sys
import glob
import os

the_dir = './png'

class Unroller:
	def __init__(self, im):
		self.name = im
		self.image = []
		# get digit from filename:
		self.digit = os.path.basename(im)[:-4] 
		im = Image.open(im)
		pix = im.load()
		for i in range(im.size[0]):
			for j in range(im.size[1]):
				# black is 1
				# white is 0
				if 0 in pix[i,j]:
					self.image.append(1)
				else:
					self.image.append(0)
	def print_it(self):
		print("y(%s,:) = %s;\nX(%s,:) = %s;" % (self.digit, self.digit, self.digit, self.image))

arrays = []
png_files = glob.glob('%s/*.png' % the_dir )

print("function [X y] = custom()")
print("X = zeros(10,400);")
print("y = zeros(10,1);")
for png in png_files:
	Unroller(png).print_it()
print("X = repmat(X,50,1);")
print("y = repmat(y,50,1);")
print("\nend")

