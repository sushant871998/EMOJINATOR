
# coding: utf-8

# In[1]:


import csv
r = csv.reader(open('C:\\Users\\Sushant\\EMOJINATOR\\gestures\\train_foo1.csv')) # Here your csv file
lines = list(r)
line=[item[0] for item in lines]
val=[i[-1] for i in line]
line=[i for i in val]
i=0
for item in lines:
    item[0]=line[i]
    i=i+1 

writer = csv.writer(open('C:\\Users\\Sushant\\EMOJINATOR\\gestures\\train_foo1.csv', 'w'))
writer.writerows(lines)

