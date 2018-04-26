# ! //usr/bin/env python
# Constructed by Xiuqin Liu

# computing feature vectors

import re
import sys

if len(sys.argv) < 4:
    sys.stderr.write("\n It is needed 3 paras:\n %s <input file> <motif1300.txt> <output file>\n " % sys.argv[0])
    sys.exit(1)

# f = open( 'human_pre-miRNA_secondary_structure.txt' )
f = open(sys.argv[1])
temp = ''.join(f.readlines()).splitlines()
f.close()

# f = open( 'human_pre-miRNA_secondary_structure1.txt','w' )
f = open(sys.argv[1][:-4] + '1' + sys.argv[1][-4:], 'w')
for i in range(int(len(temp) / 3)):
    f.write(temp[3 * i])
    f.write('\n')

    for j in range(len(temp[3 * i + 1])):
        f.write(temp[3 * i + 1][j])
        if temp[3 * i + 2][j] == '(':
            f.write('L')
        elif temp[3 * i + 2][j] == '.':
            f.write('D')
        elif temp[3 * i + 2][j] == ')':
            f.write('R')

    f.write('\n')

f.close()

# f = open( 'human_pre-miRNA_secondary_structure1.txt' )
f = open(sys.argv[1][:-4] + '1' + sys.argv[1][-4:])
temp1 = ''.join(f.readlines()).splitlines()
f.close()

# f = open( 'motif1300.txt' )
f = open(sys.argv[2])
temp2 = ''.join(f.readlines()).splitlines()
f.close()

# f = open( 'feaure.txt' , 'w' )
f = open(sys.argv[3], 'w')

for i in range(len(temp2)):
    # f.write( temp2[ i ] )
    # f.write('(')

    P = re.compile(temp2[i])

    for m in range(int(len(temp1) / 2)):

        num1 = 0
        for n in range(len(temp1[2 * m + 1]) - len(temp2[i]) + 1):
            s = temp1[2 * m + 1][n:]
            if re.match(P, s):
                num1 = num1 + 1
            else:
                num1 = num1
        f.write(str(num1))
        f.write(' ')

    f.write('\n')
f.close()

