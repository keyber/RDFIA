1) L'erreur en train est inférieure à l'erreur en test.
Par grid search de la valeur de C optimale trouvée est 10^-3. Au delà de ce seuil, le modèle est moins tolérant aux fautes sur l'ensemble de train et
commence à sur-apprendre (la précision en train tend vers 1, et la précision en test diminue)

2)
On ne peut pas utiliser l'ensemble de test pour le grid search
car utiliser l'ensemble de test pour l'apprentissage biaiserait les scores obtenus.

3)
étant donné une nouvelle image on utilise la chaîne de traitement suivante :
    1-  Déterminer les points d'intérêt en trouvant les angles de l'image (détecteur de HARRIS) (on ne l'a pas fait dans ce TP)
    2-  On extrait les caractéristiques de l'image au niveau des points d'intérêt avec des SIFTs.
        On obtient une représentation en Bag Of Features
    3-  En utilisant le dictionnaire de mots visuels déjà calculé sur la base de données d'apprentissage, on détermine la
        Représentation Bag Of (Visual) Words en affectant les sifts de l'image aux mots du dictionnaire dont il est proche.
        Dans notre cas, on fait du "hard coding" puis du "sum pooling" mais il existe d'autres méthodes
    4-  On utilise ensuite notre SVM préentraîné afin de déterminer la classe de l'image.
        Le cas multi-classe est traité en "ovr", c'est à dire en "Un Contre Tous"

4) On peut améliorer notre chaîne de traitement de différentes manières,
    - Pour l'extraction de features, on pourrait
        - déterminer les points d'intérêt de l'image
        - rendre nos SIFT scale invariant
        - faire un "soft assignation" au lieu d'un "hard coding" afin de mieux gérer les points à égale distance de deux mots du dictionnaire par exemple.
    - Pour la classification, on pourrait
        - essayer d'autres modèles, par exemple un "Random Forest" car ce type de modèle gère bien le multi-classe ou un réseau de neurones.