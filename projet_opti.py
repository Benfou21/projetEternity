import numpy as np
import pandas as pd
import random

#Read the file
file = open("pieces_03x03.txt", "r")
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
#print(arr)
#print(type(df[0][0])0)   #Str variables





#Coder une première solution aléatoire



#corners of an n*n array :   top_left = 1 ; top_right = n ; bottom_left = n*n -n ; bottom_right = n*n
#Nb de bord avec coin = (2 x nombre de pièces en largeur – 2) + (2 x nombre de pièces en hauteur – 2) 
#Nb de bord = Nb de bord avec coin  - 4
#ex : 2*3-2+2*3-2 = 8

def solAleatoire(arr,n,m):

    sol  = np.zeros((n,m))
    angle = np.zeros((n,m))
    
    nB = 2*n-2 +2*m-2 -4    #Nb de bords sans coin
    corners = [i for i in range(0,4)]
    
    borders = [i for i in range(4,nB+4)]
    


    #fill the corners
    c = random.choice(corners)
    corners.remove(c)
    sol[0][0] = arr[c][4]
    angle[0][0] = 1
    c = random.choice(corners)
    
    corners.remove(c)
    sol[0][m-1] = arr[c][4]
    angle[0][m-1] = 2

    c = random.choice(corners)
    corners.remove(c)
    
    sol[n-1][0] = arr[c][4]
    angle[n-1][0] = 3


    c = random.choice(corners)
    corners.remove(c)
    
    sol[n-1][m-1] = arr[c][4]
    angle[n-1][m-1] = 0
    
    # fill top and bottom borders with specific value
    for i in range(1,m-1):
        j = random.choice(borders)
        borders.remove(j)
        sol[0][i] = arr[j][4]
        angle[0][i] = 2
        j = random.choice(borders)
        borders.remove(j)
        sol[n-1][i] = arr[j][4]
        angle[n-1][i] = 0
        
    # fill left and right borders with specific value
    for i in range(1,n-1):
        j = random.choice(borders)
        borders.remove(j)
        sol[i][0] = arr[j][4]
        angle[i][0] = 1
        j = random.choice(borders)
        borders.remove(j)
        sol[i][m-1] = arr[j][4]
        angle[i][m-1] = 3
        

    for i in range(1, n-1):
        for k in range(1, m-1):
            sol[i][k] = arr[j][4]
            j+=1

    for row in sol :
        print(row)
    print("---------")
    for row in angle :
        print(row)

    
    return sol,angle


#population
p = 4
for i in range(p):
    solAleatoire(arr,3,3)
    print("---")



#Algo évaluation












# top_left = array[0][0]
# top_right = array[0][-1]
# bottom_left = array[-1][0]
# bottom_right = array[-1][-1]
