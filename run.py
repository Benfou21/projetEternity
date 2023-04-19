
import matplotlib.pyplot as plt
from algo import lecture
from algo import recherche
from analyse import tunning


#Read the file

arr,h,l = lecture("benchEternity2WithoutHint.txt")
#arr,h,l = lecture("pieces_04x04.txt")




#---------------------------------------------------
#TEST UNITAIRE


def test_unitaire(nb_voisins,iterations) :


    #Lancement l'alogithme
    #----------------------
    #Pour afficher tous les scores des évaluations
    list,list_t,solution = recherche(arr,nb_voisins,iterations,h,l)    #les points rouges sont les solutions selectionnée à chaque voisinnage7

    #Pour afficher seulement les score de la solution selectionnée à chaque voisinnage
    #list_t,list,solution = recherche(arr,nb_voisins,iterations,h,l)     #ne pas faire attention aux couleurs
    #--------------------

    
    plt.plot(list)  
    plt.scatter([i for i in range(len(list))],list)  
    plt.scatter([i for i in range(len(list)) if i % (nb_voisins+1) == 0],  
                [list[i] for i in range(len(list)) if i % (nb_voisins+1) == 0],  
                color='red')     
    plt.title("paramètres : nb_voisins : "+str(nb_voisins)+", nb_itérations : "+str(iterations)+", taille puzzle :"+str(h)+"*"+str(l))       
    plt.xlabel("voisinage")       
    plt.ylabel("Score")       
    plt.show()         


nb_voisins = 4
   
iterations = 20000

#To uncomment
#test_unitaire(nb_voisins,iterations)


#------------------------------------------------
#Comparaison selon voisin 
#renvoie des résultats moyens sur 5 itérations des algorithmes selon le nombre de voisins

#To uncomment 
list_voisins = [4,10,30]
iterations_test2 = 2000
#tunning(arr,list_voisins,iterations_test2)


#------------------------------------------------




