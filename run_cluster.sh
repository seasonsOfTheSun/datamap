                                                                                                                                         #!/bin/sh                                                                                                                                                                                 

module load build-env/2020
module load parallel/20190222-gcccore-7.3.0
source datamap/bin/activate

methods=(
    '3-Dimensional Autoencoder'
    '4-Dimensional Autoencoder'
    '5-Dimensional Autoencoder'
    'Spectral Clustering 10 Dimensions'
    'k-Means 10'
    'Optics'
    'Louvain'
    'GreedyModularity'
)

dataset=$1
version=$(git rev-parse --short HEAD)
mkdir data/processed/${dataset}_${version}
parallel srun python src/clustering.py {1} {2} data/processed/${dataset}_${version}  ::: "${methods[@]}" ::: data/synthetic/${dataset}/*
python src/zip.py data/processed/synthetic_clustering_${version}


















