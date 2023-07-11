#! /usr/bin/python

import sys
import os
import random

seed = sys.argv[1]
nSamples = int(sys.argv[2])
nVariants = int(sys.argv[3])
root = sys.argv[4]

random.seed(seed)


def homRef(maf):
    return (1.0 - maf) * (1.0 - maf)


def het(maf):
    return 2 * maf * (1.0 - maf)


def homAlt(maf):
    return maf * maf


def randomGen(missingRate):
    gps = []
    for j in range(nSamples):
        if random.random() < missingRate:
            gps += [0, 0, 0]
        else:
            d1 = random.random()
            d2 = random.uniform(0, 1.0 - d1)
            gps += [d1, d2, 1.0 - d1 - d2]
    return gps


def hweGen(maf, missingRate):
    bb = homAlt(maf)
    aa = homRef(maf)
    gps = []
    for j in range(nSamples):
        gt = random.random()
        missing = random.random()
        if missing < missingRate:
            gps += [0, 0, 0]
        else:
            d1 = 1.0 - random.uniform(0, 0.01)
            d2 = random.uniform(0, 1.0 - d1)
            d3 = 1.0 - d1 - d2

            if gt < aa:
                gps += [d1, d2, d3]
            elif gt >= aa and gt <= 1.0 - bb:
                gps += [d2, d1, d3]
            else:
                gps += [d3, d2, d1]

    return gps


def constantGen(triple, missingRate):
    gps = []
    for j in range(nSamples):
        if random.random() < missingRate:
            gps += [0, 0, 0]
        else:
            gps += triple
    return gps


variants = {}
for i in range(nVariants * 0, nVariants * 1):
    variants[i] = randomGen(0.0)

for i in range(nVariants * 1, nVariants * 2):
    missingRate = random.random()
    variants[i] = randomGen(missingRate)

for i in range(nVariants * 2, nVariants * 3):
    maf = random.random()
    variants[i] = hweGen(maf, 0.0)

for i in range(nVariants * 3, nVariants * 4):
    maf = random.random()
    missingRate = random.random()
    variants[i] = hweGen(maf, missingRate)

for i in range(nVariants * 4, nVariants * 5):
    missingRate = random.random()
    variants[i] = constantGen([1, 0, 0], missingRate)

for i in range(nVariants * 5, nVariants * 6):
    missingRate = random.random()
    variants[i] = constantGen([0, 1, 0], missingRate)

for i in range(nVariants * 6, nVariants * 7):
    missingRate = random.random()
    variants[i] = constantGen([0, 0, 1], missingRate)

variants[i + 1] = constantGen([0, 0, 0], 0.0)
variants[i + 2] = constantGen([1, 0, 0], 0.0)
variants[i + 3] = constantGen([0, 1, 0], 0.0)
variants[i + 4] = constantGen([0, 0, 1], 0.0)


def transformDosage(dx):
    w0 = dx[0]
    w1 = dx[1]
    w2 = dx[2]

    sumDx = w0 + w1 + w2

    try:
        l0 = int(w0 * 32768 / sumDx + 0.5)
        l1 = int((w0 + w1) * 32768 / sumDx + 0.5) - l0
        l2 = 32768 - l0 - l1
    except:
        print(dx)
        sys.exit()
    return [l0 / 32768.0, l1 / 32768.0, l2 / 32768.0]


def calcInfoScore(gps):
    nIncluded = 0
    e = []
    f = []
    altAllele = 0.0
    totalDosage = 0.0

    for i in range(0, len(gps), 3):
        dx = gps[i : i + 3]
        if sum(dx) != 0.0:
            dxt = transformDosage(dx)
            nIncluded += 1
            e.append(dxt[1] + 2 * dxt[2])
            f.append(dxt[1] + 4 * dxt[2])
            altAllele += dxt[1] + 2 * dxt[2]
            totalDosage += sum(dxt)

    z = zip(e, f)
    z = [fi - ei * ei for (ei, fi) in z]

    if totalDosage == 0.0:
        infoScore = None
    else:
        theta = altAllele / totalDosage
        if theta != 0.0 and theta != 1.0:
            infoScore = 1.0 - (sum(z) / (2 * float(nIncluded) * theta * (1.0 - theta)))
        else:
            infoScore = 1.0

    return (infoScore, nIncluded)


genOutput = open(root + ".gen", 'w')
sampleOutput = open(root + ".sample", 'w')
resultOutput = open(root + ".result", 'w')

sampleOutput.write("ID_1 ID_2 missing\n0 0 0\n")
for j in range(nSamples):
    id = "sample" + str(j)
    sampleOutput.write(" ".join([id, id, "0"]) + "\n")

for v in variants:
    genOutput.write("01 SNPID_{0} RSID_{0} {0} A G ".format(v) + " ".join([str(d) for d in variants[v]]) + "\n")
    (infoScore, nIncluded) = calcInfoScore(variants[v])
    resultOutput.write(" ".join(["01:{0}:A:G SNPID_{0} RSID_{0}".format(v), str(infoScore), str(nIncluded)]) + "\n")

genOutput.close()
sampleOutput.close()
resultOutput.close()
