import numpy as np
import pandas as pd
import random
import sys
import copy
import matplotlib.pyplot as plt
import statistics
from scipy.optimize import curve_fit
import time

from algo import lecture
from algo import recherche
from algo import solAleatoire
from algo import selectionSol
from algo import verify
from algo import score_eval
from algo import puzzle

#Read the file

arr,h,l = lecture("benchEternity2WithoutHint.txt")
#arr,h,l = lecture("pieces_04x04.txt")







def exponential_func(x, a, b, c):
    return a * np.exp(-b * x) + c




#-------------------------------------------------------------------------------





#Analyse selon le nombre d'itérations    [Analyse avec demie-vie et coefficient de varation]

def analyse_i(arr,iterations,nb_voisins,h,l):
    list_coef = []
    list_dv = []
    for i in iterations :
        list,list_sol,solution = recherche(arr,nb_voisins,i,h,l)
        # Calcul de la moyenne
        moyenne = statistics.mean(list_sol)
        # Calcul de l'écart-type
        ecart_type = statistics.stdev(list_sol)
        # Calcul du coefficient de variation
        coeff_variation = ecart_type/moyenne
        list_coef.append(coeff_variation)

        #Vitesse de convergence (calcul demie vie)
        a= [i for i in range(len(list_sol))]
        # Ajustement de la fonction exponentielle à nos données
        popt, pcov = curve_fit(exponential_func, a, list_sol)

        # Calcul de la demi-vie du modèle ajusté
        dv = np.log(2) / popt[1]
        list_dv.append(dv)

    # plt.scatter(a,list_sol)
    b = [i for i in range(len(list_coef))]
    plt.scatter(b,list_coef)
    plt.scatter(b,list_dv)
    plt.plot(list_coef,label =" coef_var")
    plt.plot(list_dv, label = "demi-vie")
    plt.legend(loc="upper right")
    plt.ylabel("score")
    plt.title("Coef de varation / demi-vie")
    plt.show()
    

#analyse_i(arr,[50,200,2000,20000],10,h,l)
#Recherche V1.2 : 
# On remarque en générale une demi vie très faible, signe d'une très rapide convergence. On se plafonne très vite. (environ de 0,0050)
# On remarque que le coef de varation est le plus grand pour des petites itérations. Montrant la encore que notre algorithme plafonne sur le long terme.
#Cela suit les résultats observé sur les graphes




#-------------------------------------------------------------------------------



#Analyse selon nombre de voisins    [Analyse avec demie-vie et coefficient de varation]

def analyse_nb_voisin(arr,iterations,nb_voisins,h,l):
    list_coef = []
    list_dv = []
    for n in nb_voisins :
        list,list_sol,solution = recherche(arr,n,iterations,h,l)
        # Calcul de la moyenne
        moyenne = statistics.mean(list_sol)
        # Calcul de l'écart-type
        ecart_type = statistics.stdev(list_sol)
        # Calcul du coefficient de variation
        coeff_variation = ecart_type/moyenne
        list_coef.append(coeff_variation)

        #Vitesse de convergence (calcul demie vie)
        a= [i for i in range(len(list_sol))]
        # Ajustement de la fonction exponentielle à nos données
        popt, pcov = curve_fit(exponential_func, a, list_sol)

        # Calcul de la demi-vie du modèle ajusté
        dv = np.log(2) / popt[1]
        list_dv.append(dv)

    # plt.scatter(a,list_sol)
    b = [i for i in range(len(list_coef))]
    plt.scatter(b,list_coef)
    plt.scatter(b,list_dv)
    plt.plot(list_coef,label =" coef_var")
    plt.plot(list_dv, label = "demi-vie")
    plt.legend(loc="upper right")
    plt.ylabel("score")
    plt.title("Coef de varation / demi-vie")
    plt.show()


#(arr,iterations,list_nb_voisins,h,l)

#analyse_nb_voisin(arr,2000,[4,10,20],h,l)

#Recherche V1.2
#On obersve une demi vie constante, montrant que la convergence est proportielement identique. Le nb de voisins n'influ pas sur la rapidité de la convergence 
#On observe un coef de variation qui diminue lorsque que le nb de voisins augmentent. Ce qui indique qu'augmenter le nb de voisin n'impacte pas la variation du score de la solution dans l'état actuel de l'algorithme. (Peut être quand rajoutant plus de mutation sur le voisin ceci peut jouer)
#Upadate Recherche V1.3 (avec changement au centre) le coef de variation remonte avec plus de voisin, form de cuve ( forte var petit voisin, forte var beaucoup de voisin)




#-------------------------------------------------------------------------------



#Affichage de statisques selon le nombre de voisins
def stat(arr,nb_voisins,iteration,n):
    labels = []
    for nb in nb_voisins :
        list_score = []
        for i in range(n):
            list,list_sol,solution = recherche(arr,nb,iteration,h,l)
            list_score.append(solution.score)
        m = max(list_score)
        moyenne = statistics.mean(list_score)

        ecart_type = statistics.stdev(list_score)

        label ="nb_voisins : +" + str(nb) + 'max : '+str(m) +" moyenne = "+str(moyenne) + "ecart-type + "+str(ecart_type)

        labels.append(label)

    for label in labels:
        print(label)

#n = nombre de fois que l'algo tourne pour chaque voisin
#(arr,list_nb_voisins,iteration,n)

#stat(arr,[4,10,20,30],2000,5)

# nb_voisins : 4 , max : 216 moyenne = 201.8 ecart-type + 11.77709641634983
# nb_voisins : 10 , max : 222 moyenne = 209 ecart-type + 10.41633332799983
# nb_voisins : 20 , max : 247 moyenne = 215.2 ecart-type + 19.57549488518745
# nb_voisins : 30 , max : 220 moyenne = 214.4 ecart-type + 6.268971207462992

#Avoir plus de voisins permet de converger plus rapidement
#Mais ce résultat nous avance pas sur les résultats long terme




#-------------------------------------------------------------------------------





#Tunning

# def tunning_V1(arr,parametres,iteration):
    
#     for p in parametres :

#             list,list_sol,solution = recherche(arr,10,iteration,h,l)
#             # plt.scatter(a,list_sol)
#             # b = [i for i in range(len(list_sol))]
#             # plt.scatter(b,list_sol)
#             plt.plot(list_sol, label = "v = "+str(p))
#             plt.legend(loc="lower right")
#     plt.ylabel("score")
#     plt.title("Score selon nb_voisin")
#     plt.show()


#Affichage graphique des scores des algo selon le nombre de voisins    (moyenne des scores sur 5 itérations de l'algo)
def tunning(arr,parametres,iteration):

    for p in parametres :
        list_sol = []
        for i in range(5) :
            
            list,list_s,solution = recherche(arr,p,iteration,h,l)
            # plt.scatter(a,list_sol)
            #Somme
            if (i>0):
                for i in range(len(list_sol)):
                    list_sol[i] = (list_sol[i] + list_s[i])
            else:
                list_sol =list_s
            
        #moyenne
        for i in range(len(list_sol)):
                    list_sol[i] = list_sol[i] / 5
        
        b = [i for i in range(len(list_sol))]
        
        #plt.scatter(b,list_sol)
        plt.plot(list_sol, label = "nb_voisins : "+str(p))
        plt.legend(loc="lower right")
    plt.ylabel("score")
    plt.title("Score moyen (5 itérations) selon nombre de voisins")
    plt.show()


#(arr,list_nb_voisins,iteration)
#tunning(arr,[4,10,30],5000)




#-------------------------------------------------------------------------------





#Modification de l'algo pour qu'on puisse tester différentes configurations sur le nombre de mutation lors de la génération d'un voisin


def recherche_tunning(arr,nb_voisins,iterations,h,l,pcentre,pbords,pcoins):
    t1 = time.time()
    list =[]
    list_sol =[]
    # pieces = solAleatoire(arr,h,l)
    # score = score_eval(arr,pieces,h,l)
    # print("first : "+ str(score))
    # solution = puzzle(pieces,score)
    solution = solAleatoire(arr,h,l)
    score = solution.score
    i = 0
    nb_i = 0
    while(i < iterations ):
        
        
        voisins = voisinage_tunning(nb_voisins,solution,h,l,pcentre,pbords,pcoins)
        for v in voisins:
            list.append(v.score)
        
        new_solution = selectionSol(voisins)
        # print([i.score for i in voisins ])
        # print(new_solution.score)
        # voisins.remove(new_solution)
        list_sol.append(new_solution.score)

        #Perturbation    #A améliorer 
        # if(len(list_sol)>10000) :     #FAIRE SELON LA DIVERGENCE,  si il y a eu déjà une forte variation du score
        #    if(nb_i>3000):
        #         perturbation = calcul_ecart_type(list_sol[-3000:],1)    #lorsque la valeur varie de moins de 1
        #         if(perturbation == True):
        #             new_solution = selection_perturbation(voisins)
        #             list_sol.remove(list_sol[-1])  #retirer la solution ajouté plus haut
        #             list_sol.append(new_solution.score)   
        #             nb_i =0  #Réinitialisation du compteur avant perturbation si e <1   
               
        
        #list.append(new_solution.score)
        solution = puzzle( new_solution.matrice,new_solution.score)
        print("new solution  :"+ str(solution.score))
        nb_i += 1
        i +=1
    list.append(solution.score)

    #list représente la list de tous les scores (voisins inclus)
    #list_sol représente la list de tous les scores des solutions retenues


    print("Random solution : score for :" + str(score))
    print("Solution for :" + str(solution.score))
    t2= time.time()
    t = t2-t1
    print(str(t))
    return list,list_sol,solution


#Générer un voisin
def voisin_tunning(sol,h,l,pcentre,pbords,pcoins):
    
    n = l*h  #n total
    
    corners = [0,l-1,l*(h-1),l*h-1]  #Positions des coins
    
    #Position des bords
    top_border = [i for i in range(1,l-1)] 
    bottom_border = [i for i in range(l*(h-1)+1, l*h-1)]
    left_border = [i for i in range(l, l*h -l, l) ]
    right_border = [ i for i in range(l-1 +l, l*h -l, l) ]
    borders = top_border + bottom_border + left_border + right_border
    
    res = copy.deepcopy(sol)

    #1 ROTATION CENTRE

    #Récupération des indices des pièces du centre
    inside = []
    k=0
    for i in range(h):
        for j in range(l):
            if(i in range(1,h-1)):
                if(j in range(1,l-1)):
                    inside.append(k)
            k += 1
    
    for i in range(pcentre):  #Plus p est grand plus l'algorithme converge rapidement vers un top

            #print("rotation centre")
            j = np.random.choice(inside)
            inside.remove(j)
            colors = [arr[res.matrice[j][0]][i] for i in range(4)]
            #Rotation centre
            if '0' not in colors : #inutile mais renforcement de la sécuritéS
                res.matrice[j][1] = (sol.matrice[j][1]+2)%4   #rotation de 90°
        

    #Changement de piece centre (même rotation)
    
    for i in range(pcentre):  #Plus r est grand plus l'algorithme converge rapidement vers un top
        
            #print("changement centre")
            j = np.random.choice(inside)
            inside.remove(j)
            colors = [arr[res.matrice[j][0]][i] for i in range(4)]
            #Rotation centre
            if '0' not in colors : #inutile mais renforcement de la sécurité
                k = np.random.choice(inside)
                inside.remove(k)
                colors = [arr[res.matrice[k][0]][i] for i in range(4)]
                #Rotation centre
                if '0' not in colors : #inutile mais renforcement de la sécurité
                    #Changement
                    id = sol.matrice[j][0]
                    sol.matrice[j][0] = res.matrice[k][0]
                    res.matrice[k][0] =  id

    
    #2 CHANGEMENT DE PIECE Bords/ Coins
    
    for i in range(pbords) :
    
            #print("changement bords")
            #gestion bords  
            j = np.random.choice(borders)
            borders.remove(j)
            k = np.random.choice(borders)
            #Changement
            id_j = sol.matrice[j][0]
            sol.matrice[j][0] = res.matrice[k][0]
            res.matrice[k][0] =  id_j
            
    for i in range(pcoins) :
            #print("changement coins")
            #gestion coins  
            j = np.random.choice(corners) #Sans remise pour les tests
            
            k = np.random.choice(corners)
            if (k != j) : 
                #Changement
                id_j = sol.matrice[j][0]
                sol.matrice[j][0] = res.matrice[k][0]
                res.matrice[k][0] =  id_j
    

    res.score = score_eval(arr,res.matrice,h,l)
    
    verify(arr,res,h,l)
    
    return res.matrice,res.score


def voisinage_tunning(nb_voisins,sol,h,l,pcentre,pbords,pcoins):
    vois = [puzzle(sol.matrice,sol.score)]

    SOLUTION = copy.deepcopy(sol)
        
    for i in range(nb_voisins):
        
        v_matrice,v_score = voisin_tunning(SOLUTION,h,l,pcentre,pbords,pcoins)  #BLEMs
        print("Solution de base :"+str(SOLUTION.score))
        print("voisin :"+str(i)+" , score : "+str(v_score))
        
        vois.append(puzzle(v_matrice,v_score))
        
    return vois



#Affichage des résultats des algos selon le nombre de mutation


def tunning_p(arr,parametres,iteration):

    for p in parametres :
        list_sol = []
        for i in range(5) :
            pcentre = p[0] 
            pbords =p[1]
            pcoins = p[2]
            list,list_s,solution = recherche_tunning(arr,10,iteration,h,l,pcentre,pbords,pcoins)
            # plt.scatter(a,list_sol)
            #Somme
            if (i>0):
                for i in range(len(list_sol)):
                    list_sol[i] = (list_sol[i] + list_s[i])
            else:
                list_sol =list_s
            
        #moyenne
        for i in range(len(list_sol)):
                    list_sol[i] = list_sol[i] / 5

        b = [i for i in range(len(list_sol))]

        #plt.scatter(b,list_sol)
        plt.plot(list_sol, label = "nb_centre = "+str(pcentre) +" nb_bords = "+str(pbords)+" nb_coins = "+str(pcoins))
        plt.legend(loc="lower right")
    plt.ylabel("score")
    plt.title("Score moyen (5 itérations) selon nombre de changement")
    plt.show()





#TEST

#paramètres = [list_pcentre,list_pbords,list_pcoins]
parametres = [[2,1,1],[4,3,2],[8,6,3],[5,2,1]]

#tunning_p(arr,parametres,2000)


#Résultat : 
#On remarque que l'algorithme avec le moins de changement possède de meilleurs performances (courbes bleus)
#Solution mettre le moins possible de mutation par voisin, plus facteur d'aléatoire