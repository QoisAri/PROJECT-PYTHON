import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def distance(p1, p2):
    return np.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)
#fungsi menghitung   total jarak tur
def tour_lenght(tour, cities):
    lenght = 0
    for i in range(len(tour)):
        lenght += distance(cities[tour[i - 1]], cities[tour[i]])
    return lenght

#fungsi untuk membuat sokusi awal secara acak
def initial_solution(num_cities):
    return np.random.permutation(num_cities)


#fungsi untuk membuat langkah perubhaan solusi
def move(tour):
    new_tour = tour.copy()
    i, j = np.random.randint(len(tour), size=2)
    new_tour[i], new_tour[j] = new_tour[j], new_tour[i]
    return new_tour

#fungsi hill climbing search
def hill_climbing_tsp(cities, max_iterations=1000):
    num_cities = len(cities)
    current_tour = initial_solution(num_cities)
    current_lenght = tour_lenght(current_tour, cities)
    iterations = 0
    lengths = [current_lenght]
    tours = [current_tour]

    while iterations < max_iterations:
        new_tour = move(current_tour)
        new_lenght = tour_lenght(new_tour, cities)
        if new_lenght < current_lenght:
            current_tour = new_tour
            current_lenght = new_lenght
            lengths.append(current_lenght)
            tours.append(current_tour)
        iterations += 1

    return current_tour, current_lenght, lengths, tours

#pengaturan random seed untuk hasil yang konsisten
np.random.seed(2)

#membuat 10 kota secara acak
num_cities = 10
cities = np.random.rand(num_cities, 2)
print(cities)

#menjalankan hiil climbing search
best_tour, best_length, lengths, tours = hill_climbing_tsp(cities)
print(best_tour)

#visualisasi hasil
fig, ax = plt.subplots()

def update(frame):
    ax.clear()
    ax.set_title("Iteration {}".format(frame))
    ax.scatter(cities[:, 0], cities[:, 1])
    ax.plot(cities[tours[frame-1],0], cities[tours[frame-1], 1])
    ax.set_xticks([])
    ax.set_yticks([])

ani= FuncAnimation(fig, update, frames=len(tours)+1, interval=500, repeat=False)

plt.show()