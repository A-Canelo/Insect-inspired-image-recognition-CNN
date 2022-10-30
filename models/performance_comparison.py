# Plotting performance CNNs comparison

# Angel Canelo 2022.10.26

###### import ####################
import numpy as np
import matplotlib.pyplot as plt
from pymatreader import read_mat
import seaborn as sns
##################################
######## Load data ###############
data = read_mat('../data/FlyDrosNet_perf.mat')
acc = data['hist_acc']
acctest = data['hist_testacc']
topac = data['topmax']
topactest = data['topmax_test']
topac_lite = data['topmax_lite']
print('FlyDrosNet \n' 'Top acc =', topac, '\n' 'Top acc test =', topactest, '\n' 'Top acc test (8bit) =', topac_lite, '\n')
data2 = read_mat('../data/ResNet101_perf.mat')
acc2 = data2['hist_acc']
acctest2 = data2['hist_testacc']
topac2 = data2['topmax']
topactest2 = data2['topmax_test']
print('ResNet101 \n' 'Top acc =', topac2, '\n' 'Top acc test =', topactest2, '\n')
data3 = read_mat('../data/MobileNetV2_perf.mat')
acc3 = data3['hist_acc']
acctest3 = data3['hist_testacc']
topac3 = data3['topmax']
topactest3 = data3['topmax_test']
print('MobileNetV2 \n' 'Top acc =', topac3, '\n' 'Top acc test =', topactest3, '\n')
#################################
########### Plotting ############
sns.set()
plt.figure()
plt.plot(acc,color="blue", linestyle='dotted', alpha=0.5); plt.plot(acctest,color="blue")
plt.plot(acc2,color="red", linestyle='dotted', alpha=0.5); plt.plot(acctest2,color="red", alpha=0.5)
plt.plot(acc3,color="green", linestyle='dotted', alpha=0.5); plt.plot(acctest3,color="green", alpha=0.5)
plt.title('Performance on pattern dataset \n (3000/300 frames train/test)'); plt.ylabel('Accuracy (%)'); plt.xlabel('Iterations')
plt.legend(['FlyDrosNet train', 'FlyDrosNet test', 'ResNet101 train', 'ResNet101 test',
            'MobileNetV2 train', 'MobileNetV2 test'], loc='lower right')
plt.show()