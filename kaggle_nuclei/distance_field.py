from itertools import count

import numpy as np
import skfmm


def distance_field(masks, clip=(-1, 1)):
    maxdist = np.full(masks.shape, -np.inf, dtype=np.float32)
    dists = []
    for layer in count():
        mask = masks == (layer + 1)
        if mask.sum() == 0:
            break
        phi = np.ones(masks.shape)
        phi[mask > 0] = 1
        phi[mask <= 0] = -1
        dist = skfmm.distance(phi)
        dists.append(dist)

    mean_max_dist = np.max(dists, axis=(1, 2)).mean()
    for dist in dists:
        dist[dist > 0] /= dist.max()
        dist[dist < 0] /= mean_max_dist
        np.maximum(maxdist, dist, out=maxdist)

    return maxdist.clip(*clip)