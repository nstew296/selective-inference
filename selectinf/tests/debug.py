import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
np.random.seed(5)


def set_box_color(bp, color, linestyle):
    plt.setp(bp['boxes'], color=color, linestyle=linestyle, linewidth=2.5)
    plt.setp(bp['whiskers'], color=color, linestyle=linestyle, linewidth=2.5)
    plt.setp(bp['caps'], color=color, linewidth=2.5)
    plt.setp(bp['medians'], color=color, linewidth=2.5)


fig = plt.figure(figsize=(17, 14))
ax1 = fig.add_subplot(111)

plt.sca(ax1)
first = plt.boxplot(np.asarray([.1,.2,.3,.4,.5,.6,.7]), positions=np.asarray([1]), sym='', widths=0.3)
second = plt.boxplot(np.asarray([.1,.2,.3,.4,.5,.6,.7]), positions=np.asarray([1.3]), sym='', widths=0.3)
fourth = plt.boxplot(np.asarray([.1,.2,.3,.4,.5,.6,.7]), positions=np.asarray([1.8]), sym='', widths=0.3)
fifth = plt.boxplot(np.asarray([.1,.2,.3,.4,.5,.6,.7]), positions=np.asarray([2.1]), sym='', widths=0.3)
set_box_color(first, '#2b8cbe', 'solid')  # colors are from http://colorbrewer2.org/
set_box_color(second, '#6baed6', '--')
set_box_color(fourth, '#238443', 'solid')
set_box_color(fifth, '#31a354', '--')
plt.xlim(0.7, 2.4)
plt.tight_layout()
plt.plot([], c='#2b8cbe', label='Randomized Multi-Task Lasso 0.7', linewidth=2.5)
plt.plot([], c='#6baed6', label='Randomized Multi-Task Lasso 1.0', linestyle='--', linewidth=2.5)
plt.plot([], c='#238443', label='Data Splitting 67/33', linewidth=2.5)
plt.plot([], c='#31a354', label='Data Splitting 50/50', linestyle='--', linewidth=2.5)
plt.legend()
plt.ylabel('Interval Length', fontsize=20)

ax1.set_title("Distribution of Interval Lengths", y=1.01, fontsize=24)
ax1.legend(loc='lower left', bbox_to_anchor=(0.319, -0.225), fontsize=20)
ax1.set_xticklabels([])
ax1.set_xticks([])


def common_format(ax):
    ax.grid(True, which='both', color='#f0f0f0')
    ax.set_xlabel('Method', fontsize=20)
    return ax

common_format(ax1)

plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
plt.savefig('real_data_lengths.png', bbox_inches='tight')


intervals1 = {0:[1,2,3,4],1:[2,4,5],2:[1,2,4,5,7]}
lengths1 = [1,2,1,2,3,1,3,2,4,1,2,1]
intervals2 = {0:[1,2,3,4],1:[2,4,5],2:[1,2,4,5,7]}
lengths2 = [8,3,2,3,4,2,4,3,5,2,3,-7]
match_length_indx = {}
start = 0
for i in range(3):
    match_length_indx[i] = lengths1[start:start+len(intervals1[i])]
    start += len(intervals1[i])
match_length_indx2 = {}
start2 = 0
for i in range(3):
    match_length_indx2[i] = lengths2[start2:start2 + len(intervals2[i])]
    start2 += len(intervals2[i])

common = {i:np.intersect1d(intervals1[i],intervals2[i]) for i in range(3)}
print(common)
common_lengths = []
for i in range(3):
    for predictor in common[i]:
        diff_length = match_length_indx2[i][np.argwhere(intervals2[i]==predictor)[0][0]]-match_length_indx[i][np.argwhere(intervals1[i]==predictor)[0][0]]
        common_lengths.append(diff_length)
print(common_lengths)

def set_box_color(bp, color, linestyle):
    plt.setp(bp['boxes'], color=color, linestyle=linestyle, linewidth=2.5)
    plt.setp(bp['whiskers'], color=color, linestyle=linestyle, linewidth=2.5)
    plt.setp(bp['caps'], color=color, linewidth=2.5)
    plt.setp(bp['medians'], color=color, linewidth=2.5)

fig = plt.figure(figsize=(17, 14))
ax1 = fig.add_subplot(111)

plt.sca(ax1)
first = plt.boxplot([common_lengths_67], positions=np.asarray([1]), sym='', widths=0.3)
second = plt.boxplot([common_lengths], positions=np.asarray([1.6]), sym='', widths=0.3)
set_box_color(first, '#2c7fb8', 'solid')  # colors are from http://colorbrewer2.org/
set_box_color(second, '#2c7fb8', '--')
plt.xlim(0.7, 1.9)
plt.tight_layout()
plt.plot([], c='#2c7fb8', label='57/33 Split')
plt.plot([], c='#2c7fb8', label='50/50 Split', linestyle='--', linewidth=2.5)
plt.legend()
plt.ylabel('Difference in Interval Length', fontsize=20)

ax1.set_title("Difference in Confidence Interval Length between Data Splitting and Selective Inference", y=1.01 ,fontsize=24)
ax1.legend(loc='lower left', bbox_to_anchor=(0.41, -0.125), fontsize=20)
ax1.set_xticklabels([])
ax1.set_xticks([])


def common_format(ax):
    ax.grid(True, which='both', color='#f0f0f0')
    ax.set_xlabel('Method', fontsize=20)
    return ax

common_format(ax1)

plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
plt.savefig('real_data_lengths2.png', bbox_inches='tight')