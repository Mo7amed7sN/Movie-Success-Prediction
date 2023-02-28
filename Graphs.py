import matplotlib.pyplot as plt; plt.rcdefaults()
import numpy as np
import matplotlib.pyplot as plt

objects = ('Dtree_adaboost', 'SVM', 'Dtree', 'KNN')
y_pos = np.arange(len(objects))
performance = [77, 72, 75, 73]

plt.bar(y_pos, performance, align='center', alpha=0.5)
plt.xticks(y_pos, objects)
plt.ylabel('Accuracy')
plt.title('Acc Graph')

plt.show()
