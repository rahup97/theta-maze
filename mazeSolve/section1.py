'''
Team Id: eYRC-NT#558
Author List: Chethan K P, Rahul Patil, Rahul Ramprasad, Rohit Goud
Filename: section1.py
Theme: Navigate a terrain
Functions:
    sine(angle)
    cosine(angle)
    readImage(filepath)
    findNeighbours(img, level, cellnum, size)
    colourCell(img, level, cellnum, size, colourVal)
    buildGraph(img, size)
    findStartPoint(img, size)
    findPath(graph, initial, final)
    next_num()
    numberMaze(graph, initial, new, l)
    shortestNumberedPath(l, initial, final, path)
    main(filePath, flag)
Global variables:
    x - list
    path - list
    new - Dictionary
    l - list

    *explanation of functions and variables used before function definition*
    '''
import numpy as np
import cv2
import math
import time

#Description for this class is given in the numberMaze() function
class Link():
    value = 0
    parent = 0
    def __init__(self, a, b):
        self.value = a
        self.parent = b
        
##  Returns sine of an angle
def sine(angle):
    return math.sin(math.radians(angle))

##  Returns cosine of an angle
def cosine(angle):
    return math.cos(math.radians(angle))

##  Reads an image from the specified filepath and converts it to Grayscale. Then applies binary thresholding
##  to the image.
def readImage(filePath):
    mazeImg = cv2.imread(filePath)
    grayImg = cv2.cvtColor(mazeImg, cv2.COLOR_BGR2GRAY)
    ret,binaryImage = cv2.threshold(grayImg,127,255,cv2.THRESH_BINARY)
    return binaryImage

##  This function accepts the img, level and cell number of a particular cell and the size of the maze as input
##  arguments and returns the list of cells which are traversable from the specified cell.
'''
    Function name: findNeighbours(img, level, cellnum, size)
    Input:
        img - binary image of the maze which is a numpy array
        size - 1 or 2 based on the number of levles in the image
        level - Level of the cell
        cellnum - cell number for the cell
    Output:
        Returns the traversable neighbours for a given cell (level, cellnum)
    Logic:
        Based on the level and cellnum of the cell, the distance and angle of the cell from the center pixel of the image is found.
        The circularly topmost( or topleft and topright) pixel for that cell is checked for its value. If the pixel is not 0(black) then the corresponding neighbour is appended to the list neighbours[].
        Similarly left right and bottom neighbours are also found.
        r - radius of the given level in terms of picel numbers with the center being the center pixel of the image
        angle - angle of the cell from the horizontal in clockwise direction
        r and angle depend on the level and cellnum for each cell
        x0 = Column value of the center pixel of the image
        y0 = Row value of the center pixel of the image
        top, topleft, left, right, bottom - pixel values in terms of indices of the top (or topleft and topright depending on level), left, right and bottom boundary pixels for the cell
        neighbours - list of neighbours for the given cell
    Example call: findNeighbours(img, level, cellnum, size)
'''
def findNeighbours(img, level, cellnum, size):
    neighbours = []
    ############################# Add your Code Here ################################
    r = 40 * (level) + 20
    if level == 0 or level == 1:
        angle = 60
    elif level == 2:
        angle = 30
    elif level == 3 or level == 4 or level == 5:
        angle = 15
    else:
        angle = 7.5
    x0 = len(img[0]) // 2
    y0 = len(img) // 2
    offset = angle / 2
    if level == 0:
        for cellnum in range(1, 7):
            top = [y0 + int((r + 20) * sine(cellnum * angle - offset)), x0 + int((r + 20) * cosine(cellnum * angle - offset))]
            if(img[top[0], top[1]] != 0):
                neighbours.append((level + 1, cellnum))
        return neighbours
    if level == 3 or level == 4:
        top = [y0 + int((r + 20) * sine(cellnum * angle - offset)), x0 + int((r + 20) * cosine(cellnum * angle - offset))]
        if size == 1:
            if(img[top[0], top[1]] != 0) and level != 4:
                neighbours.append((level + 1, cellnum))
        else:
            if(img[top[0], top[1]] != 0):
                neighbours.append((level + 1, cellnum))
    else:
        topleft = [y0 + int((r + 20) * sine(cellnum * angle - 1.5 * offset)), x0 + int((r + 20) * cosine(cellnum * angle - 1.5 * offset))]
        topright = [y0 + int((r + 20) * sine(cellnum * angle - 0.5 * offset)), x0 + int((r + 20) * cosine(cellnum * angle - 0.5 * offset))]
        if(img[topleft[0], topleft[1]] != 0) and level != 6:
            neighbours.append((level + 1, 2 * cellnum - 1))
        if(img[topright[0], topright[1]] != 0) and level != 6:
            neighbours.append((level + 1, 2 * cellnum))
    bottom = [y0 + int((r - 20) * sine(cellnum * angle - offset)), x0 + int((r - 20) * cosine(cellnum * angle - offset))]
    left = [y0 + int(r * sine(cellnum * angle - angle)), x0 + int(r * cosine(cellnum * angle - angle))]
    right = [y0 + int(r * sine(cellnum * angle)), x0 + int(r * cosine(cellnum * angle))]
    if(img[bottom[0], bottom[1]] != 0):
        if level == 5 or level == 4:
            neighbours.append((level - 1, cellnum))
        elif level == 1:
            neighbours.append((0, 0))
        else:
            neighbours.append((level - 1, (cellnum + 1) // 2))
    if(img[left[0], left[1]] != 0):
        if cellnum > 1:
            cell = cellnum - 1
        else:
            if level == 1:
                cell = 6
            elif level == 2:
                cell = 12
            elif level == 3 or level == 4 or level == 5:
                cell = 24
            else:
                cell = 48
        neighbours.append((level, cell))
    if(img[right[0], right[1]] != 0):
        if level == 1 and cellnum == 6:
            cell = 1
        elif level == 2 and cellnum == 12:
            cell = 1
        elif (level == 3 or level == 4 or level == 5) and cellnum == 24:
            cell = 1
        elif level == 6 and cellnum == 48:
            cell = 1
        else:
            cell = cellnum + 1
        neighbours.append((level, cell))

    #################################################################################
    return neighbours

'''
    colourCell function takes 5 arguments:-
        img - input image
        level - level of cell to be coloured
        cellnum - cell number of cell to be coloured
        size - size of maze
        colourVal - the intensity of the colour.
    Logic: colourCell basically highlights the given cell by painting it with the given colourVal. Care should be taken that
        the function doesn't paint over the black walls and only paints the empty spaces. This function returns the image
        with the painted cell.
        r - radius of the given level in terms of picel numbers with the center being the center pixel of the image
        angle - angle of the cell from the horizontal in clockwise direction
        r and angle depend on the level and cellnum for each cell
        x0 = Column value of the center pixel of the image
        y0 = Row value of the center pixel of the image
        j - iterator for angle in steps of 0.25 , 1
        The angle and radius for the cells are incremented in nested loops and in each pass, the pixel with the corresponding angle and radius is coloured.
        The pixel is coloured only if it is white, therefore cell walls are not coloured.
'''
def colourCell(img, level, cellnum, size, colourVal):
    ############################# Add your Code Here ################################
    r = 40 * (level) 
    if level == 0 or level == 1:
        angle = 60
    elif level == 2:
        angle = 30
    elif level == 3 or level == 4 or level == 5:
        angle = 15
    else:
        angle = 7.5
    x0 = len(img[0]) // 2
    y0 = len(img) // 2
    if level == 0:
        for i in range(40):
            for j in np.arange(0, 360, 1):
                x = x0 + int(i * cosine(j))
                y = y0 + int(i * sine(j))
                if img[y, x] == 255:
                    img[y, x] = colourVal
        return img
    for i in range(40):
        radius = r + i
        for j in np.arange(0, angle, 0.185):
            a = angle * (cellnum - 1) + j
            x = x0 + int(radius * cosine(a))
            y = y0 + int(radius * sine(a))
            if img[y, x] == 255:
                img[y, x] = colourVal

    #################################################################################
    return img

##  Function that accepts some arguments from user and returns the graph of the maze image.
'''
    Function name: buildGraph(img, size)
    Input:
        img - binary image of the maze
        size - 1 or 2 based on the number of levles in the image
    Output: Returns the graph of the image
    Logic: Each cell is traversed and the neighbours for the cell are obtained from the findNeighbours() function.
        A new entry is made in the dictionary with the current cell coordinates as its key and value returned from the findNeighbours function as its value.
        graph - Dictionary with key as cell and values as its corresponding neighbours
        l - Maximum level in the given maze
        c - Maximum cellnum in the current level
    Example call: bulidGraph(img, size)
'''
def buildGraph(img, size):   ## You can pass your own arguments in this space.
    graph = {}
    ############################# Add your Code Here ################################
    if size == 1:
        l = 4
    else:
        l = 6
    graph[(0, 0)] = findNeighbours(img, 0, 0, size)
    for i in range(l):
        level = i + 1
        if level == 1:
            c = 6
        elif level == 2:
            c = 12
        elif level == 3 or level == 4 or level == 5:
            c = 24
        else:
            c = 48
        for j in range(c):
            cellnum = j + 1
            graph[(level, cellnum)] = findNeighbours(img, level, cellnum, size)

    #################################################################################
    return graph

##  Function accepts some arguments and returns the Start coordinates of the maze.
'''
    Function name: findStartPoint(img, size)
    Input:
        img - binary image of the maze
        size - 1 or 2 based on the number of levles in the image
    Output: Returns the coordinates (level, cellnum) for the maze in the image
    Logic:
        The entire outer boundary of the maze is traversed and when the boundary is white, the corresponding cell (level, cellnum) is returned.
        The traversal for the outer boundary is done using conversion of polar to cartesian system using:
            x = r * cosine(angle)
            y = r * sine(angle)
        top = pixel value of the outer boundary for a given angle of the maze
        r - radius of the given level in terms of picel numbers with the center being the center pixel of the image
        angle - angle of the cell from the horizontal in clockwise direction
        r and angle depend on the level and cellnum for each cell
        x0 = Column value of the center pixel of the image
        y0 = Row value of the center pixel of the image
    Example call: findStartPoint(img, size)
'''
def findStartPoint(img, size):     ## You can pass your own arguments in this space.
    ############################# Add your Code Here ################################
    if size == 1:
        level = 4
    else:
        level = 6
    r = 40 * (level) + 20
    if level == 1:
        angle = 60
    elif level == 2:
        angle = 30
    elif level == 3 or level == 4 or level == 5:
        angle = 15
    else:
        angle = 7.5
    x0 = len(img[0]) // 2
    y0 = len(img) // 2
    offset = angle / 2
    for cellnum in range(1, 49):
        top = [y0 + int((r + 20) * sine(cellnum * angle - offset)), x0 + int((r + 20) * cosine(cellnum * angle - offset))]
        if(img[top[0], top[1]] != 0):
            start = (level, cellnum)
            break

    #################################################################################
    return start

'''
    Function name: next_num()
    Output:
        Returns the next number from a list having numbers 0 to 1000 whenever it is called. Returns 1 when it is called for the first time, 2 when called for the second time,
        3 for the third time and so on.
    Logic:
        global variable x is used.
        x is a list containing numbers from 0 to 1000.
        whenever next_num() is called it deletes the first element and returns the new first element.
    Example call: next_num()
'''
def next_num():
    del x[0]
    return x[0]

'''
    Function name: numberMaze(graph, initial, new, l)
    Input:
        graph - Obtained from buildGraph()
        initial - Cell coordinates of the initial cell
        new - Dictionary with cells as keys and a number assignes to the cells as their corresponding values. It is a global variable.
        l - list of objects of type Link class
        The list l is a global variable that stores objects of the type Link.
        Link is a class with two members:
            value - the number that is assigned to the cell
            parent - the number from which the current cell number has branched or originated
    Output:
        It takes the graph as the input and assigns numbers to the cells based on their neighbours.
    Logic:
        Algorithm used:
            Number 1 is assigned to the initial cell (0, 0). The neighbours of the cell are checked.
            (i) If it has only one neighbour, the same number is assigned to the neighbour cell and this function is called recursively on the neighbour.
            (ii) If it has more than one neighbours, the next number that is to be assigned to the cells (which is obtained from next_num()) is assigned to these neighbours and the
                function is called recursively on these neighbour cells.
            (iii) If it has no neighbours, then it returns.
            The shortest path is then found by going from the final cell to its parent numbered cells till it reaches the initial cell which is assigned from number 1 (done by shortestNumberedPath())
    Example call: numberMaze(graph, initial, new, l)
'''
def numberMaze(graph, initial, new, l):
    empty = 0
    lar = 0
    for k in graph[initial]:
        if new[tuple(k)] == -1:
            empty += 1
    if empty == 0:
        return
    elif empty == 1:
        for i in range(len(graph[initial])):
            if new[tuple(graph[initial][i])] == -1:
                new[tuple(graph[initial][i])] = new[initial] 
                numberMaze(graph, tuple(graph[initial][i]), new, l) 
    else:
        for i in range(len(graph[initial])):
            if new[tuple(graph[initial][i])] == -1:
                n = next_num()
                new[tuple(graph[initial][i])] = n
                l.append(Link(n, new[initial]))
                numberMaze(graph, tuple(graph[initial][i]), new, l)

'''
    Function name: shortestNumberedPath(l, initial, final, path)
    Input:
        l - list of objects of type Link class
        final - number assigned to the final cell by the algorithm
        initial - the number assigned to the initial cell by the algorithm
        path - global variable to store the path from initial to final cell in terms of the numbers that they are assigned
    Output:
        Returns the shortest path in terms of number assigned to the cells        
    Logic:
        The list l is a global variable that stores objects of the type Link.
        Link is a class with two members:
            value - the number that is assigned to the cell
            parent - the number that the current cell number was branched from
        The function begins by returning the parent number of the number assigned to the final cell and then recursively calls itself
        with the obtained parent number and final and this continues till the values of initial and final are equal. At this point, path
        contains the path going from the final to the initial cell in terms of the numbers assigned to these cells.
    Example call: shortestNumberedPath(l, initial, final, path)
'''
def shortestNumberedPath(l, initial, final, path):
    if final == initial:
        return
    else:
        for i in l:
            if i.value == final:
                parent = i.parent
                break
        path.append(parent)
        shortestNumberedPath(l, initial, parent, path)

##  Finds shortest path between two coordinates in the maze. Returns a set of coordinates from initial point
##  to final point.
'''
    Function name: findPath(graph, initial, final)
    Input:
        graph - Obtained from buildGraph()
        final - Coordinates of the final cell
        initial - Coordinates of the initial cell
    Output:
        Returns the shortest path fron initial to final as a list of cell coordinated of the cells to be traversed to reach the final.    
    Logic:
        path - Global variable that holds a list of all the cell coordinates that are along the path to be traversed, to go from initial to final cell.
        visited - List of all the cells in the maze that have already been visited.
        shortest - Shortest path from initial to final in terms of cell numbers
        new - Global variable that holds the dictionary containing a list of all the cell numbers along with their corresponding numbers assigned by the algorithm(see number path)
        l - Global variable that holds the list of objects of type Link containing a value adn a parent
        x - List used to generate successive numbers for successive calls of next_num
        In an infinite loop, the initial is the current cell initially. Checks for a neighbour with the same number. If found, it appends that neighbour to shortest and makes it current.
        If not found, it it checks for a neighbour with the value equal to the next number in list path(whose track is kept using path). When found, it appends that neighbour to shortest and makes it current.
        This is done till final and current are the same cells. Then, shortest is returned
    Example call: findPath(graph, initial, final)
'''
def findPath(graph, initial, final):      ## You can pass your own arguments in this space.
    ############################# Add your Code Here ################################
    shortest = [initial]
    visited = [initial]
    count = 1
    global path
    path = []
    global new
    new = {}
    global l
    l = []
    for k in graph.keys():
        new[k] = -1
    new[initial] = 1
    global x
    x = range(1, 1000)
    numberMaze(graph, initial, new, l)
    path.append(new[final])
    shortestNumberedPath(l, new[initial], new[final], path)
    path.reverse()
    current = initial
    while(True):
        if current == final:
            return shortest
        for j in graph[current]:
            if not tuple(j) in visited:
                if new[tuple(j)] == new[current]:
                    current = tuple(j)
                    shortest.append(current)
                    visited.append(current)
                    break
                else:
                    if new[tuple(j)] == path[count]:
                        count += 1
                        shortest.append(tuple(j))
                        current = tuple(j)
                        visited.append(current)
                        break

    #################################################################################
    return shortest

##  This is the main function where all other functions are called. It accepts filepath
##  of an image as input. You are not allowed to change any code in this function. You are
##  You are only allowed to change the parameters of the buildGraph, findStartPoint and findPath functions
def main(filePath, flag = 0):
    img = readImage(filePath)     ## Read image with specified filepath
    if len(img) == 440:           ## Dimensions of smaller maze image are 440x440
        size = 1
    else:
        size = 2
    maze_graph = buildGraph(img, size)   ## Build graph from maze image. Pass arguments as required
    start = findStartPoint(img, size)  ## Returns the coordinates of the start of the maze
    shortestPath = findPath(maze_graph, start, (0, 0))  ## Find shortest path. Pass arguments as required.
    print shortestPath
    string = str(shortestPath) + "\n"
    for i in shortestPath:               ## Loop to paint the solution path.
        img = colourCell(img, i[0], i[1], size, 230)
    if __name__ == '__main__':     ## Return value for main() function.
        return img
    else:
        if flag == 0:
            return string
        else:
            return graph
## The main() function is called here. Specify the filepath of image in the space given.
if __name__ == "__main__":
    filepath = "image_07.jpg"     ## File path for test image
    img = main(filepath)          ## Main function call
    cv2.imshow("image",img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
