import re

arff_path = 'D:/Github/datasets/emotions/emotions-train.arff'
gold_path = 'D:/Github/datasets/emotions/emotions-train.gold'
size_attrib = 72
size_label = 6

#arff_path = 'D:/Github/datasets/flags/flags-train.arff'
#gold_path = 'D:/Github/datasets/flags/flags-train.gold'
#size_attrib = 19
#size_label = 7

#arff_path = 'D:/Github/datasets/scene/scene-train.arff'
#gold_path = 'D:/Github/datasets/scene/scene-train.gold'
#size_attrib = 294
#size_label = 6

#arff_path = 'D:/Github/datasets/yeast/yeast-train.arff'
#gold_path = 'D:/Github/datasets/yeast/yeast-train.gold'
#size_attrib = 103
#size_label = 14

instance_id = 1

arff_file = open(arff_path)
gold_file = open(gold_path, 'w')
begin_instance = False
for line in arff_file:
    strs = re.split(',|\s', line.strip())
    if begin_instance == True:
        if (strs != None) and (len(strs) != 0):
            print(len(strs))
            if (len(strs) == (size_attrib + size_label)):
                for i in range(1, size_label + 1):
                    gold_file.write(str(instance_id) + '\t' + str(i) + '\t' + str(int(strs[size_attrib + i - 1])+1) + '\n')
                instance_id += 1
    else:
        if (strs != None) and (len(strs) != 0):
            if (strs[0] == '@data') or (strs[0] == '@DATA'):
                begin_instance = True
gold_file.close()
arff_file.close()