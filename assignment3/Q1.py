
from PIL import Image
from PIL import ImageOps
import numpy
from numpy import linalg
from matplotlib import pyplot

pyplot.subplot(231)
flower = Image.open("flower.jpg")
pyplot.imshow(flower)
pyplot.xlabel('original')

pyplot.subplot(232)
flower = ImageOps.grayscale(flower)
aflower = numpy.asarray(flower)
aflower = numpy.float32(aflower)
U, S, Vt = linalg.svd(aflower)
pyplot.plot(S,'b.')
pyplot.xlabel(' singular values ')

pyplot.subplot(233)
K = 20
Sk = numpy.diag(S[:K])
Uk = U[:, :K]
Vtk = Vt[:K, :]
aImk = numpy.dot(Uk, numpy.dot( Sk, Vtk))
Imk = Image.fromarray(aImk)
if Imk.mode!='RGB':
    newImk=Imk.convert('RGB')
newImk.save('Q5-K=20.jpg')
pyplot.imshow(Imk)
pyplot.xlabel('K=20')

pyplot.subplot(234)
K = 50
Sk = numpy.diag(S[:K])
Uk = U[:, :K]
Vtk = Vt[:K, :]
aImk = numpy.dot(Uk, numpy.dot( Sk, Vtk))
Imk = Image.fromarray(aImk)
if Imk.mode!='RGB':
    newImk=Imk.convert('RGB')
newImk.save('Q5-K=50.jpg')
pyplot.imshow(Imk)
pyplot.xlabel('K=50')

pyplot.subplot(235)
K = 100
Sk = numpy.diag(S[:K])
Uk = U[:, :K]
Vtk = Vt[:K, :]
aImk = numpy.dot(Uk, numpy.dot( Sk, Vtk))
Imk = Image.fromarray(aImk)
if Imk.mode!='RGB':
    newImk=Imk.convert('RGB')
newImk.save('Q5-K=100.jpg')
pyplot.imshow(Imk)
pyplot.xlabel('K=100')

pyplot.subplot(236)
K = 200
Sk = numpy.diag(S[:K])
Uk = U[:, :K]
Vtk = Vt[:K, :]
aImk = numpy.dot(Uk, numpy.dot( Sk, Vtk))
Imk = Image.fromarray(aImk)
if Imk.mode!='RGB':
    newImk=Imk.convert('RGB')
newImk.save('Q5-K=200.jpg')
pyplot.imshow(Imk)
pyplot.xlabel('K=200')
pyplot.savefig("Q5-all.ps")
pyplot.show()

