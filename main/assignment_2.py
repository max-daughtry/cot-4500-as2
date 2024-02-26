import matplotlib.pyplot as plt
import numpy as np

def NevilleMethod():
    # Dataset
    x = [3.6, 3.8, 3.9]
    val = [1.675, 1.436, 1.318]

    # Target value
    w = 3.7

    # Empty table to store the calculated values
    neville = [[0 for i in range(len(x))] for j in range(len(x))]

    # Fill first column with known values
    for i in range(len(x)):
        neville[i][0] = val[i]
    
    # Calculate the table values
    for i in range(1, len(x)):
        for j in range(1, i+1):
            term1 = (w - x[i-j]) * neville[i][j-1]
            term2 = (w - x[i]) * neville[i-1][j-1]

            neville[i][j] = (term1 - term2) / (x[i] - x[i-j])

    # Print the calculated value
    print(neville[len(neville)-1][len(neville)-1])

def NewtonForward():
    # Dataset
    xi = [7.2, 7.4, 7.5, 7.6]
    fxi = [23.5492, 25.3913, 26.8224, 27.4589]

    # Number of datapoints
    lim = len(xi)
    
    # Table to hold the differences
    diffs = [[0 for i in range(lim)] for j in range(lim)]

    # Fill first column with known values
    for i in range(lim):
        diffs[i][0] = fxi[i]

    # Fill the rest of the table with the differences
    for i in range(1, lim):
        for j in range(1, i+1):
            diffs[i][j] = (diffs[i][j-1] - diffs[i-1][j-1]) / (xi[i] - xi[i-j])

    # Desired x-value
    x0=7.3

    # First, second, third degree polynomials
    P1 = lambda x: diffs[0][0] + diffs[1][1] * (x-xi[0])
    P2 = lambda x: P1(x) + diffs[2][2] * (x-xi[0]) * (x-xi[1])
    P3 = lambda x: P2(x) + diffs[3][3] * (x-xi[0]) * (x-xi[1]) * (x-xi[2])

    # Print the difference table
    for i in range(1,len(diffs)):
        print(diffs[i][i])
    print()

    # Print the approximation of target x-value
    print(P3(x0))

def HermiteMatrix():
    # Dataset
    xi = [3.6, 3.8, 3.9]
    fxi = [1.675, 1.436, 1.318]
    dfxi = [-1.195, -1.188, -1.182]

    # Differences table
    diff = [[0 for i in range(2*len(xi) + 1)] for j in range(2*len(xi))]

    # Fill known cells with data
    for i in range(0, len(diff), 2):
        diff[i][0] = xi[int(i/2)]
        diff[i+1][0] = xi[int(i/2)]

        diff[i][1] = fxi[int(i/2)]
        diff[i+1][1] = fxi[int(i/2)]

        diff[i+1][2] = dfxi[int(i/2)]


    # Calculate remaining cells
    for i in range(2, len(diff), 2):
        diff[i][2] = (diff[i][1] - diff[i-1][1]) / (diff[i][0] - diff[i-1][0])

    for i in range(2, len(diff)):
        for j in range(3, i+2):
            diff[i][j] = (diff[i][j-1] - diff[i-1][j-1]) / (diff[i][0] - diff[i-j+1][0])
    
    # Print the differences table
    for i in diff:
        for val in i:
            print('{: f}'.format(val), end=' ')
        print()

    # # Uncomment this to see the fit
        
    # plt.scatter(xi, fxi)
    # plt.scatter(xi, dfxi)
    # x0 = np.linspace(xi[0], xi[len(xi)-1], 100)
    # fx0 = HPoly(x0, diff)
    # plt.plot(x0, fx0)
    # plt.show()


def CubicSpline():
    # Dataset
    x = [2,5,8,10]
    a = [3,5,7,9]

    # Max index
    n=len(x)-1

    # Differences between x-values
    h = []

    for i in range(n):
        h.append(x[i+1]-x[i])
    
    # A matrix
    A = [[0 for i in range(len(x))] for j in range(len(x))]
    A[0][0] = 1
    A[len(x)-1][len(x)-1] = 1

    for i in range(1, len(A)-1):
        A[i][i] = 2 * (h[i-1] + h[i])
        A[i][i-1] = h[i-1]
        A[i][i+1] = h[i]
    
    # b vector
    b = []

    b.append(0)
    for i in range(2, len(x)):
        b.append((3/h[i-1]) * (a[i]-a[i-1]) - (3/h[i-2]) * (a[i-1] - a[i-2]))
    b.append(0)

    # Print A matrix
    for i in A:
        for val in i:
            print('{: .5f}'.format(val), end=' ')
        print()

    # Print b vector
    print(b)

    # Calculate and print the x vector
    x = np.linalg.solve(A, b)
    print(x)

if __name__ == "__main__":
    # Run all the methods/functions
    NevilleMethod()
    print()
    print()
    NewtonForward()
    print()
    print()
    HermiteMatrix()
    print()
    print()
    CubicSpline()
