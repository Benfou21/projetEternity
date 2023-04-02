import numpy as np
import pandas as pd
import random
import sys
import copy
import matplotlib.pyplot as plt
import statistics
from scipy.optimize import curve_fit

class piece:
    def __init__(self,id,colors,rot):
        self.id = id
        self.colors = colors
        self.rot = rot

class puzzle:

    def __init__(self,matrice,score):
        self.matrice =  matrice
        self.score = score
    




#Read the file

def lecture(source):
    file = open(source, "r")
    contents = file.read()
    file.close()

    #Turn into dataframe
    rows = contents.split("\n")
    arr = [row.split(" ") for row in rows]
    size = arr[0]
    print(size[0])
    arr = arr[1:-1]
    h = int(size[0])
    l = int( size[1])
    


    
    df =pd.DataFrame(arr)
    df["id" ] = [i for i in range(df[0].size)]

    arr = df.values
    return arr,h,l
    
    #print(type(df[0][0])0)   #Str variables


arr,h,l = lecture("benchEternity2WithoutHint.txt")
#arr,h,l = lecture("pieces_04x04.txt")
print(arr)



#Coder une première solution aléatoire

#corners of an n*n array :   top_left = 1 ; top_right = n ; bottom_left = n*n -n ; bottom_right = n*n
#Nb de bord avec coin = (2 x nombre de pièces en largeur – 2) + (2 x nombre de pièces en hauteur – 2) 
#Nb de bord = Nb de bord avec coin  - 4
#ex : 2*3-2+2*3-2 = 8


def verify(pieces,solution,h,l):
    #Verif if solution is valid
    #verif top border
    for i in range(l):
        
        if pieces[solution.matrice[i][0]][(6-solution.matrice[i][1])%4] != '0':
            
            print("Top border -> INVALID")
            sys.exit(1)
    #print("Top border -> VALID")

    #verif bottom border
    for i in range(l*(h-1), l*h):
        if pieces[solution.matrice[i][0]][(4-solution.matrice[i][1])%4] != '0':
            print("Bottom border -> INVALID")
            sys.exit(1)
    #print("Bottom border -> VALID")

    #verif left border
    for i in range(0, l*h, l): #0 to l*h with step l
        
        if pieces[solution.matrice[i][0]][(5-solution.matrice[i][1])%4] != '0':
            print("Left border -> INVALID")
            sys.exit(1)
    #print("Left border -> VALID")

    #verif right border
    for i in range(l-1, l*h, l):   #step l
        
        if pieces[solution.matrice[i][0]][(3-solution.matrice[i][1])%4] != '0':
            print("Right border -> INVALID")
            sys.exit(1)
    #print("right border -> VALID")



def solAleatoire(arr,n,m):

    
    sol = []
    for i in range(n):
        sol.append([])
        for j in range(m):
            sol[i].append( 0)

    

    #sol  = np.zeros((n,m))
    #angle = np.zeros((n,m))
    
    nB = 2*n-2 +2*m-2    #Nb de bords
    
    corners = [i for i in range(0,4)] #Les quatres premiers sont les coins

    
    borders = [i for i in range(4,nB)]#Puis les bords
    
    
    #fill the corners
    c = random.choice(corners)
    corners.remove(c)
    sol[0][0] = piece(arr[c][4],[arr[c][0],arr[c][1],arr[c][2],arr[c][3]],1)  #id, colors, rot
    #angle[0][0] = 1

    c = random.choice(corners)
    corners.remove(c)
    sol[0][m-1] = piece(arr[c][4],[arr[c][0],arr[c][1],arr[c][2],arr[c][3]],2)
    

    c = random.choice(corners)
    corners.remove(c)
    
    sol[n-1][0] = piece(arr[c][4],[arr[c][0],arr[c][1],arr[c][2],arr[c][3]],0)
    


    c = random.choice(corners)
    corners.remove(c)
    sol[n-1][m-1] = piece(arr[c][4],[arr[c][0],arr[c][1],arr[c][2],arr[c][3]],3)
    
    
    # fill top and bottom borders with specific value
    for i in range(1,m-1):
        j = random.choice(borders)
        borders.remove(j)
        sol[0][i] = piece(arr[j][4],[arr[j][0],arr[j][1],arr[j][2],arr[j][3]],2)
        
        j = random.choice(borders)
        borders.remove(j)
        sol[n-1][i] = piece(arr[j][4],[arr[j][0],arr[j][1],arr[j][2],arr[j][3]],0)
        
        
    # fill left and right borders with specific value
    for i in range(1,n-1):
        j = random.choice(borders)
        borders.remove(j)
        sol[i][0] = piece(arr[j][4],[arr[j][0],arr[j][1],arr[j][2],arr[j][3]],1)
        
        j = random.choice(borders)
        borders.remove(j)
        sol[i][m-1] = piece(arr[j][4],[arr[j][0],arr[j][1],arr[j][2],arr[j][3]],3)
        
        
    inside=[i for i in range(nB,n*m)]
    
    for i in range(1, n-1):
        
        for k in range(1, m-1):
            
            j = random.choice(inside)
            inside.remove(j)
            rot = random.choice([0,1,2,3])
            sol[i][k] = piece(arr[j][4],[arr[j][0],arr[j][1],arr[j][2],arr[j][3]],rot)
            j+=1
    
    

    for row in sol :
        for value in row:
            print("("+str(value.id)+"_" + str(value.rot)+")" +"-", end='')
        print("")
    print("---")
    

    list = []
    #Turn to solution
    for i in range(len(sol)):
        for j in range(len(sol[0])):
            list.append([sol[i][j].id,sol[i][j].rot])

    
    
    score = score_eval(arr,list,h,l)
    
    solution = puzzle(list,score)
    verify(arr,solution,h,l)
    
    
    return solution    #(id,angle)








#Algo évaluation

#Score_matrice = nb de match de face  (!= de pièces)   score max = 2*n*(n-1)    pour 16*16 = 480

def score_eval(pieces,solution,h,l):
    score = 0

    #print("Verif horizontal:")
    #print("-----------------")
    for i in range(l*h-1):
        if i%l != l-1:
            #print(i, ",", i+1, end='')
            f = pieces[solution[i][0]][(3-solution[i][1])%4]   # 3 : coté droite
            g =pieces[solution[i+1][0]][(5-solution[i+1][1])%4] # 5 : coté gauche
            if f == g:
                score += 1
                #print(" -> ok")
            #else:
                #print(" -> not ok")

    #print("Verif veritcal:")
    #print("---------------")

    for i in range(l*(h-1)):
        #print(i, ",", i+l, end='')
        a= pieces[solution[i][0]][(4-solution[i][1])%4]  # 4 : coté bottom
        b =pieces[solution[i+l][0]][(6-solution[i+l][1])%4] # 6 : coté top
        if a == b:
            score += 1
            #print(" -> ok")
        #else:
            #print(" -> not ok")
    
    
    return score



#Score = nombre de match id     
#Eval 
def eval(real,sol):
    score = 0
    for i in range(len(real)):
        if(real[i][1] == sol[i][1] and real[i][0] == sol[i][0] ) :
            score +=1
    return score
#print(eval(ex,test))


# ex = [(1.0, 1.0), (4.0, 2.0), (3.0, 2.0), (5.0, 1.0), (8.0, 0.0), (6.0, 3.0), (0.0, 3.0), (7.0, 0.0), (2.0, 0.0)]
# test = solAleatoire(arr,3,3)

# print(score_eval(arr,test,3,3))




#population

def population(n,h,l):
    pop =[]
    for i in range(n):
        p = solAleatoire(arr,h,l)
        # score = score_eval(arr,pieces,h,l)
        # p = puzzle(pieces,score)
        pop.append((p))
    return pop


#population(5,h,l)  #OK

#Recherche locale

#Générer un voisin
def voisin(sol,h,l):
    
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
    #print(inside)
    
    r = np.random.choice([1,2,3,4])  #tirage du potentiel nombre de changement

    for i in range(1):  #Plus r est grand plus l'algorithme converge rapidement vers un top

        a = np.random.choice([1,2,3,5]) #3/4

        if(a%2 !=0):
            #print("rotation centre")
            j = np.random.choice(inside)
            inside.remove(j)
            colors = [arr[res.matrice[j][0]][i] for i in range(4)]
            #Rotation centre
            if '0' not in colors : #inutile mais renforcement de la sécuritéS
                res.matrice[j][1] = (sol.matrice[j][1]+2)%4   #rotation de 90°
        

    #Changement de piece centre (même rotation)
    s = np.random.choice([1,2,3,4])  #tirage du potentiel nombre de changement 

    for i in range(r):  #Plus r est grand plus l'algorithme converge rapidement vers un top

        a = np.random.choice([1,2,3,5]) #3/4
        j = np.random.choice(inside)
        inside.remove(j)
        k = np.random.choice(inside)
        inside.remove(k)
        id = sol.matrice[j][0]
        sol.matrice[j][0] = res.matrice[k][0]
        res.matrice[k][0] =  id

        # if(a%2 !=0):
        #     #print("changement centre")
        #     j = np.random.choice(inside)
        #     inside.remove(j)
        #     colors = [arr[res.matrice[j][0]][i] for i in range(4)]
        #     #Rotation centre
        #     if '0' not in colors : #inutile mais renforcement de la sécurité
        #         k = np.random.choice(inside)
        #         inside.remove(k)
        #         colors = [arr[res.matrice[k][0]][i] for i in range(4)]
        #         #Rotation centre
        #         if '0' not in colors : #inutile mais renforcement de la sécurité
        #             #Changement
        #             id = sol.matrice[j][0]
        #             sol.matrice[j][0] = res.matrice[k][0]
        #             res.matrice[k][0] =  id

    
    #2 CHANGEMENT DE PIECE 
    
    b = np.random.choice([1,2,3]) #2/3
    c = np.random.choice([1,2])  #1/2
    
    if(b%2 !=0):
        #print("changement bords")
        #gestion bords  
        j = np.random.choice(borders)
        borders.remove(j)
        k = np.random.choice(borders)
        #Changement
        id_j = sol.matrice[j][0]
        sol.matrice[j][0] = res.matrice[k][0]
        res.matrice[k][0] =  id_j
        
    if(c %2 !=0) :
        #print("changement coins")
        #gestion coins  
        j = np.random.choice(corners)
        corners.remove(j)
        k = np.random.choice(corners)
        #Changement
        id_j = sol.matrice[j][0]
        sol.matrice[j][0] = res.matrice[k][0]
        res.matrice[k][0] =  id_j
    

    res.score = score_eval(arr,res.matrice,h,l)
    
    verify(arr,res,h,l)
    
    return res.matrice,res.score

# sol = solAleatoire(arr,h,l)
# v = voisin(sol,h,l)
    
#pop = population(1,h,l)

# print(pop[0].score )
# print(pop[0].matrice)
# v = voisin(pop[0],h,l)
# print(v.matrice)
# print(v.score)

#voisins[0] = solution à i-1
def voisinage(nb_voisins,sol,h,l):
    vois = [puzzle(sol.matrice,sol.score)]

    SOLUTION = copy.deepcopy(sol)
        
    for i in range(nb_voisins):
        
        v_matrice,v_score = voisin(SOLUTION,h,l)  #BLEMs
        print("Solution de base :"+str(SOLUTION.score))
        print("voisin :"+str(i)+" , score : "+str(v_score))
        
        vois.append(puzzle(v_matrice,v_score))
        
    return vois


    

def selectionSol(vois):


    print("fonction séléction :")
    scores = [ v.score for v in vois]
    print("list score voisin :")
    print(scores)
    indice_max =  scores.index(max(scores))
    print("meilleur score des voisins :"+str(scores[indice_max]))
    res_m = vois[indice_max].matrice
    res_score = vois[indice_max].score
    
    return puzzle(res_m,res_score)



#list tabou
#à chaque itéaration on va ajouter les modifications qui ont conduit les voisins à un moins bon score 
#Max : 




def recherche(arr,nb_voisins,iterations,h,l):
    list =[]
    list_sol =[]
    # pieces = solAleatoire(arr,h,l)
    # score = score_eval(arr,pieces,h,l)
    # print("first : "+ str(score))
    # solution = puzzle(pieces,score)
    solution = solAleatoire(arr,h,l)
    score = solution.score
    i = 0
    while(i < iterations ):
        i +=1
        voisins = voisinage(nb_voisins,solution,h,l)
        for v in voisins:
            list.append(v.score)
        new_solution = selectionSol(voisins)
        list_sol.append(new_solution.score)
        #list.append(new_solution.score)
        solution = puzzle( new_solution.matrice,new_solution.score)
        print("new solution  :"+ str(solution.score))
    list.append(solution.score)


    print("Random solution : score for :" + str(score))
    print("Solution for :" + str(solution.score))
    return list,list_sol,solution


nb_voisins = 10
# 16*16, i =100 , nb_voision   2=> 50 ,   10 => 70-80 , 15 => 70-82,   20 => 70-90 , 25 => 75-95 (converge peu) , 30 => 72 -83 (converge peu)  
# 16*16, i =150,  nb_voisin    4=> 40-79    ,  10 => 60-85  , 20 => 72-94 (converge peu)
# 16*16,  i = 200 , nb_voisin  4=> 70-87 
iterations = 2000

#TEST UNITAIRE

#list,list_sol,solution = recherche(arr,nb_voisins,iterations,h,l)
# plt.plot(list)
# plt.scatter([i for i in range(len(list))],list)
# plt.scatter([i for i in range(len(list)) if i % (nb_voisins+1) == 0], 
#             [list[i] for i in range(len(list)) if i % (nb_voisins+1) == 0],
#             color='red')
# plt.title("paramètres : nb_voisins : "+str(nb_voisins)+", nb_itérations : "+str(iterations)+", taille puzzle :"+str(h)+"*"+str(l))
# plt.xlabel("voisinage")
# plt.ylabel("Score")
# plt.show()


#Bench de test

def exponential_func(x, a, b, c):
    return a * np.exp(-b * x) + c


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

#analyse_nb_voisin(arr,2000,[4,10,20],h,l)

#Recherche V1.2
#On obersve une demi vie constante, montrant que la convergence est proportielement identique. Le nb de voisins n'influ pas sur la rapidité de la convergence 
#On observe un coef de variation qui diminue lorsque que le nb de voisins augmentent. Ce qui indique qu'augmenter le nb de voisin n'impacte pas la variation du score de la solution dans l'état actuel de l'algorithme. (Peut être quand rajoutant plus de mutation sur le voisin ceci peut jouer)
#Upadate Recherche V1.3 (avec changement au centre) le coef de variation remonte avec plus de voisin, form de cuve ( forte var petit voisin, forte var beaucoup de voisin)



#Réaliser des stats 
#Combiner des recherches locals différentes : ex( ajouté perturbation )


# test = [[1,1],[7, 2],[3 ,2],[6, 1],[8,0],[5,3],[0,0],[4,0],[2,3] ]


# s = score_eval(arr,test,3,3)

# print(s)


