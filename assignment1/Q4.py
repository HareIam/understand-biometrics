L=[]
print (L)
L=L+[5]+[10]+[3]
print (L)
L.insert(0,9)
print (L)
L=L+L
print (L)
for item in L[::]:
	if item == 10:
		L.remove(item)
print (L)
L.reverse()
print (L)