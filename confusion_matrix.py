from textwrap import wrap
import re
import itertools
import matplotlib
import numpy as np
from sklearn.metrics import confusion_matrix



def plot_confusion_matrix(
    correct_labels,
    predict_labels,
    labels,
    title='Confusion matrix',
    normalize=False
):
    cm = confusion_matrix(correct_labels, predict_labels, labels=labels)

    if normalize:
        cm = cm.astype('float') * 10 / cm.sum(axis=1)[:, np.newaxis]
        cm = np.nan_to_num(cm)
        cm = cm.astype('int')

    fig = matplotlib.figure.Figure(figsize=(7, 7), dpi=320)
    ax = fig.add_subplot(1, 1, 1)
    ax.imshow(cm, cmap='Oranges')
    ax.set_title(title)

    classes = [re.sub(r'([a-z](?=[A-Z])|[A-Z](?=[A-Z][a-z]))', r'\1 ', x) for x in labels]
    classes = ['\n'.join(wrap(l, 40)) for l in classes]

    tick_marks = np.arange(len(classes))
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    ax.set_xticklabels(classes, fontsize=4, rotation=-90)
    ax.set_yticklabels(classes, fontsize=4)

    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        ax.text(j, i, cm[i, j] if cm[i, j] != 0 else '.',
                ha="center", va="center", fontsize=6)

    fig.tight_layout()
    return fig

