import numpy as np
import pandas as pd
import random



class piece:
    def __init__(self,id,colors,rot):
        self.id = id
        self.colors = colors
        self.rot = rot

class solution:

    score  = 0
    def __init__(self,matrice):
        self.matrice =  matrice




#Read the file

def lecture(source):
    file = open(source, "r")
    contents = file.read()
    file.close()

    #Turn into dataframe
    rows = contents.split("\n")
    arr = [row.split(" ") for row in rows]
    arr = arr[1:-1]
    df =pd.DataFrame(arr)
    df["id" ] = [i for i in range(df[0].size)]

    #print(df)

    arr = df.values
    return arr
    #print(arr)
    #print(type(df[0][0])0)   #Str variables


arr = lecture("pieces_03x03.txt")



#Coder une première solution aléatoire

#corners of an n*n array :   top_left = 1 ; top_right = n ; bottom_left = n*n -n ; bottom_right = n*n
#Nb de bord avec coin = (2 x nombre de pièces en largeur – 2) + (2 x nombre de pièces en hauteur – 2) 
#Nb de bord = Nb de bord avec coin  - 4
#ex : 2*3-2+2*3-2 = 8

def solAleatoire(arr,n,m):

    
    sol = []
    for i in range(n):
        sol.append([])
        for j in range(m):
            sol[i].append( 0)

    

    #sol  = np.zeros((n,m))
    #angle = np.zeros((n,m))
    
    nB = 2*n-2 +2*m-2    #Nb de bords

    corners = [i for i in range(0,4)]

    
    borders = [i for i in range(4,nB)]
    
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
    
    sol[n-1][0] = piece(arr[c][4],[arr[c][0],arr[c][1],arr[c][2],arr[c][3]],3)
    


    c = random.choice(corners)
    corners.remove(c)
    sol[n-1][m-1] = piece(arr[c][4],[arr[c][0],arr[c][1],arr[c][2],arr[c][3]],0)
    
    
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
            print(str(value.id) +"-", end='')
        print("")
    print("---")
    

    list = []
    #Turn to solution
    for i in range(len(sol)):
        for j in range(len(sol[0])):
            list.append((sol[i][j].id,sol[i][j].rot))

    print(list)
    
    return list    #(id,angle)


#population
# p = 4
# for i in range(p):
    
#     solAleatoire(arr,3,3)
#     print("---------")



#Algo évaluation

ex = [(1.0, 1.0), (4.0, 2.0), (3.0, 2.0), (5.0, 1.0), (8.0, 0.0), (6.0, 3.0), (0.0, 3.0), (7.0, 0.0), (2.0, 0.0)]
test = solAleatoire(arr,3,3)

#Score_matrice = nb de match de face  (!= de pièces)   score max = 2*n*(n-1)    pour 16*16 = 480

def score_eval(sol):
    score = 0
    return score



#Score = nombre de match id     
#Eval 
def eval(real,sol):
    score = 0
    for i in range(len(real)):
        if(real[i][1] == sol[i][1] and real[i][0] == sol[i][0] ) :
            score +=1
    return score


print(eval(ex,test))















# top_left = array[0][0]
# top_right = array[0][-1]
# bottom_left = array[-1][0]
# bottom_right = array[-1][-1]
