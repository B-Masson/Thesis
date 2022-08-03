import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np
import os
print("Imports done.")

print("Generating box plot for all stored means...")

# Sort out variables
loc = "Means"
mean_files = os.listdir(loc)
mean_files.sort()
accs = []
losses = []
names = []
save_loc = "Box-Plots/"

# Append to dict
for m in mean_files:
    names.append(str(m))
    means = np.load(loc+"/"+m)
    accs.append(means['arr_0'])
    losses.append(means['arr_1'])
    del means

# Articificially expand arrays
#accs.append(accs[0])
#losses.append(losses[0])

# Meta data
print("Names len: ", len(names), ", Acc len: ", len(accs), ", Loss len: ", len(losses), sep='')
for i in range(len(names)):
    print("Set", i+1, ":", names[i], "|", accs[i], "|", losses[i])

# Generate plots
# Acc
#print(accs)
fig = plt.figure(figsize =(10, 7))
#ax = fig.add_axes([0, 0, 1, 1])
plt.boxplot(accs)
ticks = np.arange(1, len(names)+1, 1)
plt.xticks(ticks, names)
plt.title("Accuracy Distribution across several experiments")
plt.savefig(save_loc+"test_acc.png")
plt.clf()
# Loss
fig = plt.figure(figsize =(10, 7))
#ax = fig.add_axes([0, 0, 1, 1])
plt.boxplot(losses)
plt.xticks(ticks, names)
plt.title("Loss Distribution across several experiments")
plt.savefig(save_loc+"test_loss.png")
plt.clf()

# Create a golden standard to compare output to
data = np.random.normal(100, 20, 200)
fig = plt.figure(figsize =(10, 7))
plt.boxplot(data)
plt.savefig(save_loc+"standard.png")

print("Done!")