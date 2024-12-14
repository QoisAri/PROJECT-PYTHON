class Individu:
    def __init__(self, kromosom, score):
        self.kromosom = kromosom
        self.score = score


    def setScore(self, score):
        self.score = score

import random
import string

def InitialPopulation():
    global Populasi, array_target, jumlah_populasi

    for i in range(jumlah_populasi):
        kromosom = []
        for j in range(len(array_target)):
            kromosom.append(random.choice(string.printable))

        ind = Individu(kromosom, 0)
        Populasi.append(ind)

def FitnessFunction():
    global Populasi, array_target

    for ind in Populasi:
        cocok = 0
        for index, gen in enumerate(ind.kromosom):
            if gen == array_target[index]:
                cocok += 1

        score = cocok / len(array_target)
        ind.setScore(score)

import random

def Pickone(populasi):
    r = random.random()
    index = 0
    while (r>0 and index < len(populasi)):
        r = r - populasi[index].score
        index += 1

    return populasi[index-1]

import random
def Crossover(parentA,parentB):
    r = random.randint(0,len(parentA.kromosom)-1)
    childKromosom = parentA.kromosom[:r] + parentB.kromosom[r:]
    childScore = 0

    child = Individu(childKromosom, childScore)

    return child

def MutualSelection():
    global Populasi

    NewPopulasi=[]
    for ind in Populasi:
        parentA = Pickone(Populasi)
        parentB = Pickone(Populasi)

        child = Crossover(parentA,parentB)
        NewPopulasi.append(child)

    Populasi = NewPopulasi

def Mutation(Mutation_rate):
    global Populasi

    for ind in Populasi:
        for index, gen in enumerate(ind.kromosom):
            r = random.random()
            if r<= Mutation_rate:
                ind.kromosom[index]= random.choice(string.printable)

target = input("masukan kalimat target: ")
jumlah_populasi = int(input("masukan jumlah populasi: "))

array_target = list(target)
Populasi= []

InitialPopulation()

FitnessFunction()

maxloop = 2000
bestscore = 0
loop = 0

while (bestscore < 1.0 and loop< maxloop):

    pop = Populasi
    pop.sort(key=lambda x:x.score, reverse=True)
    bestscore = pop[0].score
    print("---*--- generansi: ", loop)
    for p in pop[:5]:
        print(p.kromosom, p.score)
    
    MutualSelection()
    Mutation(0.1)
    FitnessFunction()

    loop += 1