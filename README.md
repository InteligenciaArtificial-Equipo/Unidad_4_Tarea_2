## Unidad 4 - Tarea 1

## Preprocesamiento para un detector de emociones con Fer - 2013 con el modelo CNN

## Integrantes

### Chaparro Castillo Christopher
### Peñuelas López Luis Antonio

## Como Funciona?

## Preprocess.py
Código para llevar a cabo el preprocesamiento de imágenes para nuestro modelo CNN.

## Imports
Tenemos los siguientes **imports**, a la hora de ejecutar el programa:

- ### Cv2
  
```py
import cv2
```
Librería para procesar imágenes y videos. Se usa para leer imágenes (como cv2.imread()), detectar bordes (cv2.Canny()), redimensionarlas (cv2.resize()).

- ### Numpy

```py
import numpy as np
```
Librería para cálculos numéricos con arreglos. Sirve para manejar imágenes como matrices y realizar operaciones como normalizar o cambiar la forma de los datos.

- ### Blob_dog

```py
from skimage.feature import blob_dog
```
Parte de Scikit-Image, esta función detecta regiones circulares (blobs) en imágenes usando el método Difference of Gaussian. Se emplea para identificar características como ojos o boca en las caras.

- ### Path

```py
from pathlib import Path
```
Es un módulo de Python para trabajar con rutas de archivos y directorios de forma sencilla. Se usa para crear carpetas (como preprocessed_data) y buscar imágenes (Path.glob("*.jpg")).

- ### Random

```py
import random
```
Es una biblioteca para generar aleatoriedad. Sirve para mezclar las imágenes aleatoriamente antes de dividirlas en entrenamiento, prueba y validación, asegurando una distribución justa.

- ### Ruta base del dataset
  
```py
dataset_path = Path("Fer2013_Dataset")
output_path = Path("preprocessed_data")
```
Recibe una lista de listas (matriz) que inicializarán como el estado inicial.

- ### Comparación de Tableros:
  
```py
 def __eq__(self, other):
        return self.board == other.board
```
Compara si dos objetos Board son iguales, basándose en si sus tableros son idénticos.

- ### Generación de Hash:
  
```py
    def __hash__(self):
        return hash(tuple(map(tuple, self.board)))
```
Genera un valor hash del objeto Board, lo que permite usarlo en estructuras como diccionarios o conjuntos.

- ### Cálculo de la Distancia de Manhattan:
  
```py
 def manhattan(self):
        goal = [ (0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2), (2, 0), (2, 1), (2, 2) ]
        dist = 0
        for i in range(3):
            for j in range(3):            
                value = self.board[i][j] - 1
                if value == -1: continue
                (x, y) = goal[value]
                dist += (abs(x - i) + abs(y - j))
        return dist
```
Calcula y devuelve la distancia de Manhattan del tablero actual respecto al objetivo.

- ### Búsqueda de la Casilla Vacía:
  
```py
  def emptyTile(self):
        for i in range(3):
            for j in range(3):            
                if self.board[i][j] == 0: return (i, j)
```
Busca y devuelve las coordenadas (i, j) de la casilla vacía (0) en el tablero.

- ### Movimiento en el Tablero:
  
```py
 def move(self, r, c, x, y):
        newBoard = [row[:] for row in self.board]  # Crear una copia del tablero
        newBoard[r][c], newBoard[r + x][c + y] = newBoard[r + x][c + y], newBoard[r][c]
        return newBoard
```
Realiza un movimiento en el tablero, moviendo la casilla en (r, c) a la nueva posición (r + x, c + y) y devuelve el tablero resultante.

- ### Generación de Tableros Vecinos:
  
```py
def neighbors(self):
        moves = [ (-1, 0), (1, 0), (0, 1), (0, -1) ]
        (r, c) = self.emptyTile()
        def isValidMove(r, c):
            return r >= 0 and r < 3 and c >= 0 and c < 3

        boards = []

        for (x, y) in moves:    
            if isValidMove(r + x, c + y): 
                new_board = self.move(r, c, x, y)
                boards.append(Board(new_board))
        return boards
```
Genera y devuelve una lista de objetos Board con los tableros vecinos posibles, moviendo la casilla vacía en las cuatro direcciones (arriba, abajo, izquierda, derecha) dentro del tablero.

## Class Node

Contamos con la clase **Node**, la cual cuenta con los siguiente metodos:

- ### Inicializar Nodo
  
```py
def __init__(self, g, h, parent, board: Board):
        self.f = g + h
        self.g = g
        self.h = h
        self.parent = parent
        self.board = board
```
Inicializa un nodo para ser utilizado en un algoritmo de búsqueda (en este caso A*).

- ### Comparación de Nodos
  
```py
   def __lt__(self, other):
        return self.f < other.f 
```
Este método permite comparar dos nodos (Node) utilizando el operador < (menor que). Compara los valores f de los nodos para determinar cuál tiene un valor más bajo, lo que es útil para ordenar nodos en una cola de prioridad.

## Algoritmo A*

```py
 def a_star(initial):
    goal = Board([
        [1, 2, 3],[4, 5, 6],[7, 8, 0]
    ])
    open_set = []
    closed_set = set()
    
    start_node = Node(g = 0, h = initial.manhattan(), parent = None, board = initial)
    heapq.heappush(open_set, start_node)
    
    while open_set:
        current = heapq.heappop(open_set)
        
        if current.board == goal:
            return current
        
        closed_set.add(current.board)

        for neighbor in current.board.neighbors():
            if neighbor in closed_set:
                continue
            
            new_node = Node(g = current.g + 1, h = neighbor.manhattan(), parent = current, board = neighbor)
            heapq.heappush(open_set, new_node)

    return None
```
Este método implementa el algoritmo A* para encontrar la secuencia de movimientos que resuelve el 8-puzzle.

## Imprimir Secuencia

```py
def print_solution(node):
    path = []
    while node:
        path.append(node.board.board)
        node = node.parent
    for step in reversed(path):
        for row in step:
            print(row)
        print()
    return path
```
Este método imprime la secuencia de pasos para llegar desde el estado inicial hasta la solución.

## Entrada del tablero

```py
def get_input():
    while True:
        try:
            input_line = input("Initial Board: ")
            numbers = list(map(int, input_line.split()))
            if len(numbers) != 9 or any(n < 0 or n > 8 for n in numbers):
                print("The numbers of the board must be between 0 and 8")
            else:
                board = [numbers[i:i + 3] for i in range(0, 9, 3)]
                return board
        except ValueError:
            print("Invalid Input")
```
Este método solicita al usuario la entrada del tablero inicial para el rompecabezas 8-puzzle y valida que los valores ingresados sean correctos.

- ### Llamar input
  
```py
   current_board = Board(
    get_input()
)
```
Llama a la función get_input(), que solicita al usuario que ingrese el tablero inicial del 8-puzzle.

- ### Calcular tiempo
  
```py
start = time.perf_counter()
solution = a_star(current_board)
end = time.perf_counter()
)
```
Mide el tiempo que toma ejecutar el algoritmo A* para encontrar la solución del 8-puzzle. 

- ### Imprime la solución
  
```py
if solution:
    path = print_solution(solution)
    print(f"Solution found in : {end - start:.6f} seg")
    print(f"Solved in {len(path)} moves")
else:
    print("No solution found.")
```
Si encuentra solución, imprimirá el camino que se tomó, el tiempo que le tomó al algoritmo para resolverlo y el número de movimientos para ello. De caso contrario, se imprimirá que no sea encontrada la solución.

## Ejecuciones

### Ejecución 1

![Image](https://github.com/user-attachments/assets/49719420-74cc-4fa5-9beb-dac8e2c5a639)

### Ejecución 2

![Image](https://github.com/user-attachments/assets/fc526750-6620-4aa1-b952-76a0a3a26cb1)

![Image](https://github.com/user-attachments/assets/095425d2-48fc-47e7-ae25-5229722a5092)

