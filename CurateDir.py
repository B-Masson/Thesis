# Transform a given input file list into a statified, curated list that can be used in order
# Richard Masson
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from collections import Counter

imgname = "/home/mssric004/Directories/test_adni_1_images.txt"
labname = "/home/mssric004/Directories/test_adni_1_labels.txt"
path_file = open(imgname, "r")
path = path_file.read()
path = path.split("\n")
path_file.close()
label_file = open(labname, 'r')
labels = label_file.read()
labels = labels.split("\n")
labels = [ int(i) for i in labels]
label_file.close()
#print("Data distribution:", Counter(labels))

x_train, x_val, y_train, y_val = train_test_split(path, labels, stratify=labels, shuffle=True, test_size=0.5)
x_val, x_test, y_val, y_test = train_test_split(x_val, y_val, stratify=y_val, test_size=0.5)
print("Training distribution:", Counter(y_train))
print("Validation distribution:", Counter(y_val))
print("Test distribution:", Counter(y_test))

choice = input("Continue? y/n\n")
if choice == 'y':
    newline = ''
    i = open("/home/mssric004/Directories/curated_images.txt", 'w')
    l = open("/home/mssric004/Directories/curated_labels.txt", 'w')
    for h in range(len(x_train)):
        i.write(newline+x_train[h])
        l.write(newline+str(y_train[h]))
        newline = '\n'
    for j in range(len(x_val)):
        i.write(newline+x_val[j])
        l.write(newline+str(y_val[j]))
    for k in range(len(x_test)):
        i.write(newline+x_test[k])
        l.write(newline+str(y_test[k]))
    i.close()
    l.close()

    print("All done.")
elif choice == 'n':
    print("We should try again.")
else:
    print("That wasn't even a valid answer you dingus.")
