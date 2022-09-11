import numpy as np
import cv2
from tensorflow.keras.datasets import mnist


# file_name = "A_Z Handwritten Data.csv"

# def load_az_dataset(file_name,j):
#     i = 0
#     k = 0
#     print("reached boi")
#     for row in open(file_name):
#         row = row.split(",")
#         label = int(row[0])
#         t = f"D:\\PES\\Semester - 3\\Mini -Project\\Final Text OCR Detection\\{j}\\image{i}.png"
#         if(label == j and k<=1000):
#             img = np.array([int(x) for x in row[1:]],dtype = "uint8")
#             img = img.reshape((28,28))
#             cv2.imwrite(t,img)
#             i+=1
#             k+=1
#         elif(label == j+1):
#             break

#     print("DONW BOI", j+1)

def load_mnist_dataset():
    images = []
    ((trainData,trainLabels), (testData, testLabels)) = mnist.load_data()
    data = np.vstack([trainData, testData])
    labels = np.hstack([trainLabels, testLabels])
    print(len(labels))
    dic = {}
    for i in range(0,len(data)):
        
        if labels[i] in dic:
            if(dic[labels[i]] <= 1000):
                dic[labels[i]] = dic[labels[i]] + 1
                t = f"D:\\PES\\Semester - 3\\Mini -Project\\Final Text OCR Detection\\0-9\\{labels[i]}\\image{dic[labels[i]]}.png"
                cv2.imwrite(t,data[i])

        else:
            dic[labels[i]] = 0
            t = f"D:\\PES\\Semester - 3\\Mini -Project\\Final Text OCR Detection\\0-9\\{labels[i]}\\image{dic[labels[i]]}.png"
            cv2.imwrite(t,data[i])
            
    print("DONE")
    print("\n\n", dic)

load_mnist_dataset()
