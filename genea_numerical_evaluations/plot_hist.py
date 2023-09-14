"""
Plots the experimental results after calculating motion statistics
Expects that calc_distance was run before this script

@author: Taras Kucherenko
"""

import matplotlib.pyplot as plt
import csv
import numpy as np

def read_joint_names(filename):
    with open(filename, 'r') as f:
        org = f.read()
        joint_names = org.split(',')

    return joint_names

def read_csv(filename):

    x=[]
    y=[]
    total_sum = 0
    with open(filename, 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        next(reader, None)  # skip the headers
        for row in reader:
            next_val = float(row[-2]) + float(row[-5]) # average over left and right wrists
            x.append(row[0])
            y.append(next_val)
            total_sum+=next_val
            # Crop on 40
            if float(row[0]) >= 50:
                break

    return np.array(x), np.array(y)*100 / total_sum


def get_average(feature_name):

    feature_filename = 'result/Used/'+feature_name+'/1/hmd_' + type + '_1.csv'
    _, feature_1 = read_csv(feature_filename)
    feature_filename = 'result/Used/'+feature_name+'/2/hmd_' + type + '_1.csv'
    _, feature_2 = read_csv(feature_filename)
    feature_filename = 'result/Used/'+feature_name+'/3/hmd_' + type + '_1.csv'
    _, feature_3 = read_csv(feature_filename)
    # average
    feature = np.mean(np.array([feature_1, feature_2, feature_3]), axis=0)

    return feature


plt.rcParams.update({'font.size': 36})

type = "vel"

import os
print("pwd=" + os.getcwd())

original_filename = 'result/GT/hmd_'+type+'_1.csv'

x,original = read_csv(original_filename)

# Get Full model results
feature_filename = 'result/FullModel/hmd_' + type + '_1.csv'
_, full_model = read_csv(feature_filename)


# Get No Autoregr
_,no_autoregr = read_csv('result/NoAutoregression/hmd_' + type + '_1.csv')

# Get No FiLM
_,no_FiLM = read_csv('result/NoFiLM/hmd_' + type + '_1.csv')

_,no_Vel_pen = read_csv('result/NoVelPenalty/hmd_' + type + '_1.csv')


# Get No Speech
_,no_speech = read_csv('result/NoSpeech/hmd_' + type + '_1.csv')

# Get No Text
_,no_Text = read_csv('result/NoText/hmd_' + type + '_1.csv')



_,no_PCA = read_csv('result/NoPCA/hmd_' + type + '_1.csv')



plt.plot(x,original,linewidth=7.0, label='Ground Truth', color='Purple')

plt.plot(x,full_model , label='Proposed Model',linewidth=7.0)


"""
plt.plot(x,no_autoregr , label='No Autoregression',linewidth=7.0, color='C6')

plt.plot(x,no_FiLM , label='No FiLM',linewidth=7.0, color='C1')

plt.plot(x,no_Vel_pen , label='No Velocity Loss',linewidth=7.0, color='C3')

"""

plt.plot(x,no_PCA , label='No PCA',linewidth=7.0, color='Blue')

plt.plot(x, no_Text , label='No Text',linewidth=7.0, color='C2')

plt.plot(x, no_speech , label='No Audio',linewidth=7.0, color='C5')



plt.xlabel("Velocity ($cm$/$s$)", size=50)
plt.ylabel('Frequency (%)', size=50)
#plt.title('Average Velocity Histogram')



plt.xticks(np.arange(0,51,5), size=50)#, ('Tom', 'Dick', 'Harry', 'Sally', 'Sue'))


leg = plt.legend(prop={'size': 42})



#plt.show()

figure = plt.gcf() # get current figure
figure.set_size_inches(25.5, 13)
# when saving, specify the DPI
plt.savefig("myplot.png", dpi = 100)