Utilisation de classif.py : python3 classif.py --data [donnees de test]

Par défaut, le classifieur utilisé est le SVC et le fichier d'entrainement doit
être situé, à partir du dossier racine de classif.py, dans Data/train_32x32.mat.
De plus, le nombre d'images utilisés pour l'entrainement est limité à 10 000.

Pour plus de détails sur les différentes options utilisables, il suffit d'executer
le script sans arguments ou avec une mauvaise syntaxe, dans quel cas une brève
explication des options utilisables apparaitra (en anglais).

Voici le manuel fourni en cas d'erreur :

Usage : python classif.py --data [file_location]
Possible options :
--trainset [path_of_trainset] : replaces the usual training dataset by the one given
--classifier [name of classifier] : chooses the used classifier from the following ones :
'DMIN', 'SVC', 'LinearSVC', 'KNN', 'NearestNeighbors' (default : SVC)
--PCA [n] : uses PCA reduction with n components or, if 0 < n < 1, the variance for the PCA reduction (default : no PCA)
--conf : Displays confusion matrix with matplotlib.pyplot (default : False)
--traincap [n] : Sets the maximum number of pictures taken from train dataset to n

