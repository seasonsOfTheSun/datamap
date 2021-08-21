source datamap/bin/activate

moo=(
#     '2-Dimensional Autencoder'
     'Spectral Clustering 8 Dimensions'
     'Spectral Clustering 9 Dimensions'
#     'k-Means 1'
#     'k-Means 2'
#     'k-Means 3'
#     'k-Means 4'
#     'k-Means 5'
#     'k-Means 6'
#     'k-Means 7'
#     'k-Means 8'
     'k-Means 9'
     'k-Means 10'
     'k-Means 11'
     'k-Means 12'
     'k-Means 13'
     'k-Means 14'
     'k-Means 15'
     'k-Means 16'
     'k-Means 17'
     'k-Means 18'
     'k-Means 19'
#     'Optics'
     'Louvain'
     'GreedyModularity'
   )
#
#for i in "${moo[@]}"
#do
#    for j in data/synthetic/size/*
#    do
#        python src/clustering.py "$i" $j data/processed/size/${i// /_}_${j////_}
#    done
#done

#
#for i in "${moo[@]}"
#do
#    for j in data/synthetic/scale/*
#    do
#        python src/clustering.py "$i" $j data/processed/scale/${i// /_}_${j////_}
#    done
#done



for i in "${moo[@]}"
do
    for j in data/synthetic/dimension/*
    do
        python src/clustering.py "$i" $j data/processed/dimension/${i// /_}_${j////_}
    done
done