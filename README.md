# thetaMaze
Maze solving of theta mazes using Image processing implemented using Python modules.

We have selected certain stock theta maze images of an NxN dimension, and using image processing techniques and shortest path algorithms we have solved these mazes.
The first and foremost step was converting these theta images into rectangular images. We didn't actually convert them so to speak, we used polar coordinates to access the respective pixels.

This is further divided into three different directories where we are trying to do three different things.

1) mazeSolve - Here we find the optimum path from the start coordinate of (0,0) to the final coordinate (n-1, n-1), and highlight the said path. (From the central cell, to the outermost level)

2) mazeSolve-checkpoints - Here our images contain certain checkpoints depicted with coloured boxes. Here our job is to find the paths through each checkpoint individually, and only highlight the one which is smaller of the bunch, rather the most optimum one.

We have used the OpenCV and numpy libraries in Python to find these solutions. The shortest path algorithm we have used is based on the BFS algorithm.

Credits - IIT-B, e-Yantra Labs
