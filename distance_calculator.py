import numpy as np


def point_to_line_dist(self, cpos, bpos, bvel):
    pdiff = np.subtract(cpos, bpos)
    nval = np.linalg.norm(bvel)
    if (nval > 0):
        dist = np.linalg.norm(np.cross(pdiff, bvel)) / nval
    else:
        # if vector direction (velocity) is 0 then distance is distance between the two points
        dist = np.linalg.norm(pdiff)
    return dist

# line to line give position and direction (velocity) representation -- if velocity is 0 return 9999 as distance (its not well defined  but this will make it look novel)


def line_to_line_dist(self, apos, avel, bpos, bvel):

    magA = np.linalg.norm(avel)
    magB = np.linalg.norm(bvel)
    if (magA == 0 or magB == 0):
        return 9999

    nA = avel / magA
    nB = bvel / magB

    cross = np.cross(nA, nB)
    denom = np.linalg.norm(cross)**2
    t = (bpos-apos)
    d0 = np.dot(nA, t)

    if (denom == 0):  # parallel lines
        dist = np.linalg.norm(((d0*nA)+apos)-bpos)
    else:

        # skew lines: Calculate the projected closest points
        t = (bpos - apos)
        detA = np.linalg.det([t, nB, cross])
        detB = np.linalg.det([t, nA, cross])

        t0 = detA/denom
        t1 = detB/denom

        pA = apos + (nA * t0)  # Projected closest point on line A
        pB = bpos + (nB * t1)  # Projected closest point on line B
        dist = np.linalg.norm(pA-pB)

    return dist
