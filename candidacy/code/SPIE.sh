#!/bin/sh
#SBATCH --job-name=SPIE
#SBATCH -t 10:59:00
#SBATCH -D /gpfs/u/home/CGHI/CGHIarch/barn/candidacy

cd /gpfs/u/home/CGHI/CGHIarch/barn/candidacy
python SPIE_candidacy_CCI.py
