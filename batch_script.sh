#!/bin/bash
#
#
#SBATCH --ntasks=1
#SBATCH --time=2:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=125GB

module load miniconda
module load cudnn
source activate py37
python word2vec_model.py
