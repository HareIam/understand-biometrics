from PIL import Image
from PIL import ImageFilter
from pylab import *
#Q1
#plt.subplot(431)
img = Image.open('lena.png')
img_copy = img
gray()
img.show()

#plt.subplot(432)
img_re = img_copy.crop((100, 100, 400, 400))
# img_re.show()
rot_Image = img_re.rotate(45)
img_copy.paste(rot_Image, (100, 100, 400, 400))
gray()
img_copy.show()
img_copy.save ('Q2-1.jpg')

#Q2
im = array(Image.open('lena.png').convert('L'))
#subplot(434)
hist(im.flatten(),256)
show()
imhist,bins = histogram(im.flatten(),256,normed=True)
cdf = imhist.cumsum()
cdf = cdf*255/cdf[-1]
im2 = interp(im.flatten(),bins[:256],cdf)
im2 = im2.reshape(im.shape)
#subplot(435)
hist(im2.flatten(),256)
show()


#Q3
#plt.subplot(4,3,7)
im_Max=Image.open('lena.png').filter(ImageFilter.MaxFilter())
im_Max.show()
#plt.subplot(4,3,8)
im_Min = Image.open('lena.png').filter(ImageFilter.MinFilter())
im_Min.show()
#plt.subplot(4,3,9)
im_Medi = Image.open('lena.png').filter(ImageFilter.MedianFilter())
im_Medi.show()
im_Max.save ( 'Q2-2.jpg' )
im_Min.save ( 'Q2-3.jpg' )
im_Medi.save ( 'Q2-4.jpg' )

#Q4
#plt.subplot(4,3,10)
im_Gauss3 = Image.open('lena.png').filter(ImageFilter.GaussianBlur(radius=3))
im_Gauss3.show()
#plt.subplot(4,3,11)
im_Gauss5 = Image.open('lena.png').filter(ImageFilter.GaussianBlur(radius=5))
im_Gauss5.show()
im_Gauss3.save ( 'Q2-5.jpg' )
im_Gauss5.save ( 'Q2-6.jpg' )
#show()