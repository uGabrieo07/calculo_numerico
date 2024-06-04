import numpy as np


def Gauss_Jacobi(A, b, tol=1e-10, max_iterations=1000):
    '''Função que aproxima uma solução para um dado sistema linear do tipo Ax = b, utilizando o método
    iterativo de Gauss-Jacobi.'''
    n = len(b)
    x = np.zeros_like(b, dtype=np.double)
    x_novo = np.zeros_like(x, dtype=np.double)

    for iteration in range(max_iterations):
        for i in range(n):
            s = sum(A[i][j] * x[j] for j in range(n) if j != i)
            x_novo[i] = (b[i] - s) / A[i][i]

        if np.allclose(x, x_novo, atol=tol):
            break
        
        x = np.copy(x_novo)

    return x


def Gauss_Seidel(A, b, tol=1e-10, max_iterations=1000):
    '''Função que aproxima uma solução para um dado sistema linear do tipo Ax = b, utilizando o método
    iterativo de Gauss-Seidel.'''
    n = len(b)
    x = np.zeros_like(b, dtype=np.double)

    for iteration in range(max_iterations):
        x_novo = np.copy(x)
        
        for i in range(n):
            s1 = sum(A[i][j] *  x_novo[j] for j in range(i))
            s2 = sum(A[i][j] * x[j] for j in range(i + 1, n))
            x_novo[i] = (b[i] - s1 - s2) / A[i][i]

        if np.allclose(x, x_novo, atol=tol):
            break
        
        x = x_novo

    return x


if __name__ == '__main__':
  A = np.array([[10, 2, -1 ],
                [-3, -6, 2],
                [1, 1, 5]], dtype=float)
  b = np.array([27, -61.5, -21.5], dtype=float)

  print("\n")

  x = Gauss_Jacobi(A, b)
  print("Solução pelo método de Gauss-Jacobi:")
  print(f"{x}\n")

  y = Gauss_Seidel(A, b)
  print("Solução pelo método de Gauss-Seidel:")
  print(f"{y}\n")






'''def Gauss_Jacobi(A, b, tol=1e-10, iteracoesMax=1000):
    n = A.shape[0]
    x = np.zeros(n)
    x_ant = np.zeros(n)  # Inicializa x_ant

    for i in range(iteracoesMax):
        for j in range(n):
            soma = 0
            for k in range(n):
                if k != j:
                    soma += A[j, k] * x[k]
            x[j] = (b[j] - soma) / A[j, j]
        
        if np.linalg.norm(x - x_ant, np.inf) < tol:
            return x
        
        x_ant = x.copy()
    
    return x'''

'''def Gauss_Seidel(A, b, tol=1e-10, iteracoesMax = 1000):
    n = A.shape[0]
    x = np.zeros(n)
    x_ant = np.zeros(n)  # Inicializa x_ant
    
    for i in range(iteracoesMax):
        soma = 0
        for k in range(n):'''