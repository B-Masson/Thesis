# Generate box-plots using recorded mean accuracies
# Richard Masson
from statistics import mean
import matplotlib
from matplotlib import testing
matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np
import os
print("Imports done.")

print("Generating box plot for all stored means...")

testing_mode = False # Write to sep file
run_all = False # Let's you pick and choose

# Sort out variables
loc = "Means"
mean_files = os.listdir(loc)
mean_files.sort()
accs = []
losses = []
names = []
save_loc = "Box-Plots/"
logname = "Advanced3D"
re_ord = False

# Narrow down the list if necessary
picks = ["K_V6-advanced-true.npz"] # Needs to have .npz
if not run_all:
    removals = []
    if picks:
        print(mean_files)
        for m in mean_files:
            print("Accessing", m, end='')
            if m not in picks:
                print(" | removed", end='')
                removals.append(m)
            print("")
    for r in removals:
        mean_files.remove(r)

# Append to dict
for m in mean_files:
    names.append(str(m))
    means = np.load(loc+"/"+m)
    accs.append(means['arr_0'])
    losses.append(means['arr_1'])
    del means
# Meta data
print("Names len: ", len(names), ", Acc len: ", len(accs), ", Loss len: ", len(losses), sep='')
for i in range(len(names)):
    print("Set", i+1, ":", names[i], ":", round(mean(accs[i])),"\t|", accs[i], "|", losses[i])

proc = input("Proceed? (y/n)\n")
if proc == 'n':
    exit()

count = 0
for name in names:
    newname = input("Enter a label name for " + name + "\n")
    names[count] = newname
    count += 1

print("Labels are now:")
for nameagain in names:
    print(nameagain)

# Generate plots
def make_unique(file_name, extension):
    if os.path.isfile(file_name):
            expand = 1
            while True:
                new_file_name = file_name.split(extension)[0] + str(expand) + extension
                if os.path.isfile(new_file_name):
                    expand += 1
                    continue
                else:
                    file_name = new_file_name
                    break
    else:
        print("")
    print("Saving to", file_name)
    return file_name

# Acc
fig = plt.figure(figsize =(10, 7))
plt.boxplot(accs, meanline=True, showmeans=True)
ticks = np.arange(1, len(names)+1, 1)
plt.xticks(ticks, names)
plt.title("Accuracy Distribution across several experiments")
plt.ylabel("Accuracy (%)")
if testing_mode:
    mod = "test_"
else:
    mod = ""
savename = save_loc+mod+logname+"_acc.png"
plt.savefig(make_unique(savename, ".png"))
plt.clf()
# Loss
fig = plt.figure(figsize =(10, 7))
plt.boxplot(losses, meanline=True, showmeans=True)
plt.xticks(ticks, names)
plt.title("Loss Distribution across several experiments")
plt.ylabel("Loss")
savename = save_loc+mod+logname+"_loss.png"
plt.savefig(make_unique(savename, ".png"))
plt.clf()

print("Done!")