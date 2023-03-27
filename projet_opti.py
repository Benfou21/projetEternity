import numpy as np
import pandas as pd
import random
import sys


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
    arr = arr[1:-1]
    
    df =pd.DataFrame(arr)
    df["id" ] = [i for i in range(df[0].size)]

    print(df)

    arr = df.values
    return arr
    
    #print(type(df[0][0])0)   #Str variables


arr = lecture("pieces_03x03.txt")
print(arr)


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





#Algo évaluation

#Score_matrice = nb de match de face  (!= de pièces)   score max = 2*n*(n-1)    pour 16*16 = 480

def score_eval(pieces,solution,h,l):
    score = 0
    
    print(h)
    print(l)

    print("Verif horizontal:")
    print("-----------------")
    for i in range(l*h-1):
        if i%l != l-1:
            print(i, ",", i+1, end='')
            f = pieces[solution[i][0]][(3-solution[i][1])%4]
            g =pieces[solution[i+1][0]][(5-solution[i+1][1])%4]
            if f == g:
                score += 1
                print(" -> ok")
            else:
                print(" -> not ok")

    

    print("Verif veritcal:")
    print("---------------")

    for i in range(l*(h-1)):
        print(i, ",", i+l, end='')
        a= pieces[solution[i][0]][(4-solution[i][1])%4]
        b =pieces[solution[i+l][0]][(6-solution[i+l][1])%4]
        if a == b:
            score += 1
            print(" -> ok")
        else:
            print(" -> not ok")
    
    
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
        pieces = solAleatoire(arr,h,l)
        score = score_eval(arr,pieces,h,l)
        p = puzzle(pieces,score)
        pop.append((p))
    return pop

pop = population(1,3,3)

print(pop[0].score )
print(pop[0].matrice)












