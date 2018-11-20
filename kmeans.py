from random import randint
import operator
import random

def toMatrixArray(file):
    with open(file, "r") as f:
        line = f.readline()
        matrixArray = []
        while(line != ''):
            line_as_array = line.split()
            line_in_matrixArray = []
            for i in range(len(line_as_array)):
                line_in_matrixArray.append(float(line_as_array[i]))
            matrixArray.append(line_in_matrixArray)
            line = f.readline()
        return matrixArray


def createClasses(number_of_classes):
    classes = []
    mean_of_classes = []
    for _ in range(number_of_classes):
        classes.append([])
    return classes, mean_of_classes


def assignDataToRandomClass(matrixArray, classes):
    for i in range(len(matrixArray)):
        random_class = random.randrange(0, len(classes))
        classes[random_class].append(matrixArray[i])

# def assignDataToRandomClass(matrixArray, classes):
#     for i in range(len(matrixArray)):
#         k = i%len(classes)
#         print(k)
#         classes[k].append(matrixArray[i])


def assingDataToClass(data, _class, classNumber):
    _class[classNumber].append(data)


def findMinIndex(_list):
    min_val = min(_list)
    index = _list.index(min_val)
    return index


def mean(_class):
    mean = []
    _sum = []
    for j in range(len(_class[0])):
        _sum.append(int(0))
        mean.append(0)
    for i in range(len(_class)):
        for j in range(len(_class[0])):
            _sum[j] += _class[i][j]
    for i in range(len(_class[0])):
        mean[i] = _sum[i]/len(_class)
    return mean


matrixArray = toMatrixArray("data")
number_of_classes = 2
classes, mean_of_classes = createClasses(number_of_classes)
assignDataToRandomClass(matrixArray, classes)

new_classes, mean_of_new_classes = createClasses(number_of_classes)

for i in range(len(classes)):
    mean_of_classes.append(0.5)
    mean_of_classes[i] = mean(classes[i])
mean_of_old_classes = []
for i in range(len(classes)):
    mean_of_old_classes.append(0.5)
    mean_of_old_classes[i] = mean(classes[i])

dist = []
for i in range(len(classes)):
    dist.append(0)

flag = 0
while(flag != 1):
    new_classes, a = createClasses(number_of_classes)
    
    for i in range(len(matrixArray)):
        for k in range(len(classes)):
            for j in range(len(matrixArray[0])):
                dist[k] += ((matrixArray[i][j] - mean_of_classes[k][j])**2)
   
        index = findMinIndex(dist)
        dist = []
        
        for m in range(len(classes)):
            dist.append(0)
        assingDataToClass(matrixArray[i], new_classes, index)    
    
    for i in range(len(classes)):
        mean_of_old_classes[i] = mean(classes[i])
    classes = new_classes
    
    for i in range(len(classes)):
        mean_of_classes[i] = mean(classes[i])
    diff = []

    for i in range(len(mean_of_classes)):
        for j in range(len(mean_of_classes[0])):
            diff.append(mean_of_classes[i][j] - mean_of_old_classes[i][j])

    for i in range(len(diff)):
        if diff[i] == 0 :
            flag = 1
for i in range(len(matrixArray)):
    for j in range(len(classes)):
        if((classes[j].count(matrixArray[i]) == 1)):
            print(i, j)