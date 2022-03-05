"""XXX
"""
import string

CLASSES = list(string.ascii_lowercase) + ['blank', 'other']

LABELS_TO_CLASSES = {l: i for i, l in enumerate(CLASSES)}
CLASSES_TO_LABELS = {v: k for k, v in LABELS_TO_CLASSES.items()}
