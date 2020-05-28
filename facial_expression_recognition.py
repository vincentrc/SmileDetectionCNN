#!/usr/bin/env python

import os
import numpy as np
import random as rand
import function2 as f
import classes3 as c
import conv as conv
import pool as pool
from PIL import Image

'''CHANGE VALUES HERE TO ADJUST TESTS'''
#show_img
#True  -> shows the image at each step of the process
#False -> does not show images
show_img = False

#show_data
#True  -> show data for each iteration
#False -> don't show data
show_data = True

#random_filters
#True  -> use randomized filters
#False -> use the filters defined below
random_filters = True

#num_filters -> used if random_filters is false
num_filters = 2

#learn_rate -> adjust how quickly the nn learns(?)
learn_rate = 0.005

#folder_name -> specify folder for use in training/testing
folder_name = 'faces'

#face list name -> specify the text file with list of ALL faces
face_list = 'FACES_list2.txt'

#limit -> limit the number of tests (-1 for no limit)
#limit is now set later, based on the total number of faces available (if limit > num faces)
limit = -1

#these determine number of learning and test operations
#even if greater than the number of possible faces, this will stop the operations
#before reaching limit
learn_limit = 500
test_limit = 100
'''END TEST ADJUSTMENT AREA'''

'''DEFINE FILTERS HERE IF NEEDED'''
#list of filters (change first value if more filters needed)
filters = np.zeros((2,3,3))
#vertical edge detection
filters[0,:,:]=np.array([[[-1,0,1],[-1,0,1],[-1,0,1]]])
#horizontal edge detection
filters[1,:,:]=np.array([[[1,1,1],[0,0,0],[-1,-1,-1]]])
'''END FILTER DEFINITION'''

#current working directory
cwd = os.getcwd()

try:
    os.chdir(cwd+'/'+folder_name)
except:
    print("An error occurred, the input folder could not be found.")
    quit()

#General Variables
test_num = 0
correct_count = 0
incorrect_count = 0
smile_guess = 0
nonsmile_guess = 0
smile_correct = [0,0]
nonsmile_correct = [0,0]

#Learning Variables
go_learn = True
learn_correct_count = 0
learn_incorrect_count = 0
learn_smile_guess = 0
learn_nonsmile_guess = 0
learned = 1

#Testing Variables
go_test = True
test_correct_count = 0
test_incorrect_count = 0
test_smile_guess = 0
test_nonsmile_guess = 0
tested = 1

faces = (open(cwd + '/' + face_list))
face_lines = faces.readlines()
num_faces = len(face_lines)
indices = list(range(1,num_faces + 1))

if limit > num_faces:
    limit = num_faces

#Instantiate convolution layer
if random_filters == True:
    cv = conv.Conv(num_filters,None,random_filters)
else:
    cv = conv.Conv(filters.shape[0],filters,random_filters)

#Instantiate convolution layer 2
if random_filters == True:
    cv2 = conv.Conv(num_filters,None,random_filters)
else:
    cv2 = conv.Conv(filters.shape[0],filters,random_filters)

#Instantiate max pooling
mp = pool.Pool()
mp2 = pool.Pool()

#FULLY CONNECTED
#Create fully-connected layer
fc = c.FullyConnected(961*num_filters)

#loops until both learning and testing are complete
while(go_learn or go_test):
    test = rand.choice(indices)
    indices.remove(rand.choice(indices))
    fn = face_lines[test-1]
    fn = fn.strip('\n')
    fn = fn.split('.')[0]+'.pgm'
	#if f.check(fn,cwd) == -1:
	#	continue

    try:
        img = Image.open(fn)
    except:
        continue

    #preprocessing - recolor, resize, reshape image if needed
    img = f.prep(img)

    #show preprocessed image if indicated by show_img
    if show_img:
    	img.show()

    #encoding - convert image to numpy array
    np_img = np.array(img)
    
    #CONVOLUTION
    feature_map = cv.forward((np_img/255)-0.5)

    #show post-convolution image if indicated by show_img
    if show_img:
        #new_img = Image.fromarray(np.uint(feature_map_to_add[0]))
        #only display the last of the feature maps
        new_img = Image.fromarray(np.uint(feature_map[-1]))
        new_img.show()

    #RELU - add relu maps to relu map list
    relu_map = f.relu(feature_map)

    #show post-relu image if indicated by show_img
    if show_img:
        #only display the last of the relu maps
        new_img_r = Image.fromarray(np.uint(relu_map[-1]))
        new_img_r.show()

    #MAX POOLING
    pool_map = mp.forward(relu_map)

    #show post-pooling image if indicated by show_img
    if show_img:
        #only display the last of the pooling maps
        new_img_p = Image.fromarray(np.uint(pool_map[-1]))
        new_img_p.show()

    '''flat_result_array = np.zeros((len(pool_map),pool_map[-1].shape[0],pool_map[-1].shape[1]))
    for a in range(0,len(pool_map)):
        flat_result_array[a]=pool_map[a]  

    result = flat_result_array'''
    result = pool_map
    output = fc.forward(result)
	
    test_num += 1

    if show_data:
        print('______________________________________________')
    
    if go_learn:
        print('LEARN NUMBER ',learned,'\t ', fn)
    elif go_test:
        print('TEST NUMBER ',tested,'\t',fn)

    if show_data:
        print("output", output)

    
	#Check output
    answer = np.array([f.check(fn,cwd)])

    if show_data:
        print("answer = ", answer)

    guess = np.where(output == np.amax(output))
	
    if answer[0] == 1:
        if go_learn:
            smile_correct[0] += 1
        elif go_test:
            smile_correct[1] += 1
    elif answer[0] == 0:
        if go_learn:
            nonsmile_correct[0] += 1
        elif go_test:
            nonsmile_correct[1] += 1

    if show_data:
        print(guess)

    if guess[0][0] == 0:
        smile_guess += 1
        if go_learn:
            learn_smile_guess += 1
        elif go_test:
            test_smile_guess += 1
    elif guess[0][0] == 1:
        nonsmile_guess += 1
        if go_learn:
            learn_nonsmile_guess += 1
        elif go_test:
            test_nonsmile_guess += 1

    if guess[0][0] == answer[0]: #np.array_equal(answer,guess[0]):
        correct = 1
        correct_count += 1
        if go_learn:
            learn_correct_count += 1
        elif go_test:
            test_correct_count += 1

        print("CORRECT!")
        #correct_amt += output[np.amax(output)]
    else:
        correct = 0
        incorrect_count += 1
        if go_learn:
            learn_incorrect_count += 1
        elif go_test:
            test_incorrect_count += 1

        
        print("INCORRECT!")
        #incorrect_amt += output[np.amin(output)]

    

	#Calculate cross-entropy loss
    loss = -np.log(output[answer[0]])

    if show_data:
        print("loss", loss)
	
	#Calculate dLoss_dOut (initital gradient) to backprop into fully-connected layer
	#dLoss_dOut is zero except in the correct output element
    gradient = np.zeros(2)
    gradient[answer[0]] = -1 / output[answer[0]]	#Derivative of -ln(output) with respect to output
    
    if show_data:
        print("gradient", gradient)
	
    if go_learn:
        gradient = fc.backward(gradient, answer[0],learn_rate)
        dL_dInput = mp.backprop(gradient)	
        res_filters = cv.backprop(dL_dInput,learn_rate)

    if limit != -1 and test_num >= limit:
    	break

    if go_learn:
        learned += 1
        if learned > learn_limit:
            go_learn = False
    elif go_test:
        tested += 1
        if tested > test_limit:
            go_test = False

#Avoid division by zero
if learned == 0:
    learned = 1
if tested == 0:
    tested = 1
if test_num == 0:
    test_num = 1

if show_data:
    print("_______________________________________________________________________")
    print("RESULTS -> Learn Rate = ", learn_rate)
    print("Number Correct   = ", correct_count, "     Percentage = ", correct_count/test_num)
    print("Number Incorrect = ", incorrect_count, "     Percentage = ", incorrect_count/test_num)

    print("Smile Guesses    = ", smile_guess)
    print("NonSmile Guesses = ", nonsmile_guess)

    print("Result of Fully Connected Backprop:")
    print(gradient)
    print("Result of Pooling Backprop:")
    print(dL_dInput)
    #print("Results of Convolution Backprop:")
    #print(res_filters)

print("__________________________________________________________")
print("1 LAYER CNN RESULTS -> Learn Rate = ", learn_rate, " Number of Filters = ", num_filters)
print("Number of Training Iterations = ", learn_limit, " Testing Iterations = ", test_limit)
print("CATEGORY            ", "\t LEARNING", "\t TESTING", "\t TOTAL")
print("Number Correct     =\t", learn_correct_count, "\t\t", test_correct_count, "\t\t", correct_count)
print("Percent Correct    =\t", round(learn_correct_count/learned,3), "\t\t", round(test_correct_count/tested,3), "\t\t", round(correct_count/test_num,3)) 
print("Number Incorrect   =\t", learn_incorrect_count, "\t\t", test_incorrect_count, "\t\t", incorrect_count)
print("Percent Incorrect  =\t", round(learn_incorrect_count/learned,3), "\t\t", round(test_incorrect_count/tested,3), "\t\t", round(incorrect_count/test_num,3)) 
print("Smile Guesses      =\t", learn_smile_guess, "\t\t", test_smile_guess, "\t\t", smile_guess)
print("Smile Guess %      =\t", round(learn_smile_guess/learned,3), "\t\t", round(test_smile_guess/tested,3), "\t\t", round(smile_guess/test_num,3))
print("Smile Correct      =\t", smile_correct[0], "\t\t", smile_correct[1], "\t\t", smile_correct[0] + smile_correct[1])
print("Smile Correct %    =\t", round(smile_correct[0]/learned,3), "\t\t", round(smile_correct[1]/tested,3), "\t\t", round((smile_correct[0]+smile_correct[1])/test_num,3))
print("NonSmile Guesses   =\t", learn_nonsmile_guess, "\t\t", test_nonsmile_guess, "\t\t", nonsmile_guess)
print("NonSmile Guess %   =\t", round(learn_nonsmile_guess/learned,3), "\t\t", round(test_nonsmile_guess/tested,3), "\t\t", round(nonsmile_guess/test_num,3)) 
print("NonSmile Correct   =\t", smile_correct[0], "\t\t", smile_correct[1], "\t\t", smile_correct[0] + smile_correct[1])
print("NonSmile Correct % =\t", round(nonsmile_correct[0]/learned,3), "\t\t", round(nonsmile_correct[1]/tested,3), "\t\t", round((nonsmile_correct[0]+nonsmile_correct[1])/test_num,3))
