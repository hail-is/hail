import sys
import subprocess as sp
import numpy
from random import shuffle

# python split.py N_APPLICATIONS REVIEWERS

reviewers = sys.argv[2].split(',')

apps = [x for x in range(1, int(sys.argv[1]) + 1)]
shuffle(apps)
for idx, line in enumerate(numpy.array_split(apps, len(reviewers))):
    pdfs = [f'build/split.d/{x}.pdf' for x in line]
    sp.run(['gs',
            '-q',
            '-dNOPAUSE',
            '-dBATCH',
            '-sDEVICE=pdfwrite',
            f'-sOutputFile=build/packets.d/{reviewers[idx]}.pdf',
            *pdfs],
           check=True)
