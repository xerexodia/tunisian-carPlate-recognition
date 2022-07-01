import cv2
import numpy as np
from matplotlib import pyplot as plt
import matplotlib as mpl
import matplotlib.pyplot as plt
import tensorflow as tf 




def recog(img):
    img = cv2.resize( img, (800,300))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret,thresh = cv2.threshold(gray,120,255,cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    #extract white contour
    for c in contours:
        x,y,w,h=cv2.boundingRect(c)
        if (w>600 and h>220):
            crop = thresh[y+32:y+h-20, x+23:x+w-20]
    BLACK = [0, 0, 0]
    # Add black border to the image
    crop = cv2.copyMakeBorder(crop, 3, 3, 3, 3, cv2.BORDER_CONSTANT, value=BLACK)
    digit = []

    nb = np.array(crop)
    x_sum = cv2.reduce(nb, 0, cv2.REDUCE_SUM, dtype=cv2.CV_32S)# compute the sommation
    y_sum = cv2.reduce(nb, 1, cv2.REDUCE_SUM, dtype=cv2.CV_32S)
    x_sum = x_sum.transpose()# rotate the vector x_sum
    x = nb.shape[1]# get height and weight
    y = nb.shape[0]
    x_sum = x_sum / y# division the result by height and weight
    y_sum = y_sum / x
    x_arr = np.arange(x)
    y_arr = np.arange(y)
    z = np.array(x_sum)# convert x_sum to numpy array
    w = np.array(y_sum)# convert y_arr to numpy array
    z[z < 15] = 0 # convert to zero small details
    z[z > 15] = 1
    w[w < 20] = 0# convert to zero small details and 1 for needed details
    w[w > 20] = 1
    test = z.transpose() * nb # vertical histogram
    test = w * test # horizontal histogram
    horizontal = plt.plot(w, y_arr)
    plt.show()
    vertical = plt.plot(x_arr ,z)
    plt.show()

    f = 0
    ff = z[0]
    t1 = list()
    t2 = list()
    for i in range(z.size):
        if z[i] != ff:
            f += 1
            ff = z[i]
            t1.append(i)
    rect_h = np.array(t1)

    f = 0
    ff = w[0]
    for i in range(w.size):
        if w[i] != ff:
            f += 1
            ff = w[i]
            t2.append(i)
    rect_v = np.array(t2)

    # take the appropriate height
    rectv = []
    rectv.append(rect_v[0])
    rectv.append(rect_v[1])
    max = int(rect_v[1]) - int(rect_v[0])
    for i in range(len(rect_v) - 1):
        diff2 = int(rect_v[i + 1]) - int(rect_v[i])

        if diff2 > max:
            rectv[0] = rect_v[i]
            rectv[1] = rect_v[i + 1]
            max = diff2

    # extract caracter
    for i in range(len(rect_h) - 1):

        # eliminate slice that can't be a digit, a digit must have width bigger then 8
        diff1 = int(rect_h[i + 1]) - int(rect_h[i])

        if (diff1 > 5) and (z[rect_h[i]] == 1):
            # cutting nb (image) and adding each slice to the list caracrter_list_image
            curr_num = nb[rectv[0]:rectv[1], rect_h[i]:rect_h[i + 1]]
            
            digit.append(curr_num)
        
            # draw rectangle on digits
            cv2.rectangle(crop, (rect_h[i], rectv[0]), (rect_h[i + 1], rectv[1]), (255, 255, 255), 1)
    #Show segmentation result
    plt.imshow(crop, cmap='gray')
    plt.show()

    resultat=[]
    categorie = ["0","1","2","3","4","5","6","7","8","9","TN"]
    #loading the model
    model = tf.keras.models.load_model("model.model")
    for i in range(len(digit)):
        #resizing each digit 
        digit[i] = cv2.resize(digit[i],(40,40))
        #reshaping and feeding into the model 
        digit[i] = np.reshape(digit[i],(-1,40,40,1))
        #precting each digit
        prediction = model.predict(digit[i]) 
        resultat.append(categorie[np.argmax(prediction)])

    resultat=' '.join(resultat)
    return print(resultat)