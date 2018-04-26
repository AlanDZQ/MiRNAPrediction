# ! //usr/bin/env python
# Constructed by Xiuqin Liu


# text file transpose


import sys

if len(sys.argv) < 3:
    sys.stderr.write("\n It is needed 2 paras:\n %s <input file>  <output file>\n" % sys.argv[0])
    sys.exit(1)

# myfile=open('feaure.txt')
myfile = open(sys.argv[1])
row = []
for ss in myfile.readlines():
    column = []
    line = ss.split()
    for field in line:
        column.append(field)
    row.append(column)
myfile.close()

temp1 = [[r[col] for r in row] for col in range(len(row[0]))]

# f=open('feaure1.txt','w')
f = open(sys.argv[1][:-4] + '1' + sys.argv[1][-4:], 'w')
for i in range(len(temp1)):

    for j in range(len(temp1[i])):
        f.write(temp1[i][j])
        f.write(' ')
    f.write('\n')
f.close()

# change the feature vector into the format for SVM

# train.txt is the training set, the row of which is a samples, the column of which is features.
# f = open( 'feaure1.txt' )
f = open(sys.argv[1][:-4] + '1' + sys.argv[1][-4:])
temp = ''.join(f.readlines()).splitlines()
f.close()

# f=open('predict_format.txt','w')
f = open(sys.argv[2], 'w')
for i in range(len(temp)):
    tempi = temp[i].rsplit()
    f.write(str(1))
    f.write(' ')
    for j in range(len(tempi)):
        f.write(' ')
        f.write(str(j + 1))
        f.write(':')
        f.write(tempi[j])
    f.write('\n')
f.close()