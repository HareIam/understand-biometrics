import random

def InsertSort( l ):
    for i in range(len(l)):
        for j in range(i + 1, len(l)):
            if l[i] > l[j]:
                l[j], l[i] = l[i], l[j]
    return l
testseq = []
for ii in range(20):
    testseq . append(random. randint(1, 200))
print(testseq)
print(InsertSort(testseq))
