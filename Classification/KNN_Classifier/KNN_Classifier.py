import numpy as np
from collections import Counter
import warnings


def k_nearest_neighbours(data, predict, k=3):
    if len(data) >= k:
        warnings.warn('K is set to a value less than total voting groups!
    distance = []
    for class_name in data:
        for features in data[class_name]:
            euclidian_distance = np.linalg.norm(np.array(features) - np.array(predict))
            distance.append([euclidian_distance,class_name])

    votes = [i[1] for i in sorted(distance)[:k]]
    vote_result = Counter(votes).most_common(1)[0][0]
    confidence = Counter(votes).most_common(1)[0][1]/k
    return vote_result,confidence

