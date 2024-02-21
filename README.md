# TDDFT_Davidson
a demonstration of Davidson eigensolver for TDA and TDDFT problem based on PySCF


I implemented the Davidson algorithm for TDDFT Casida equation. it is more reliable than the PySCF built-in solver.

See the comparsion, just run

```
python main.py
```
## Problem: PySCF TDDFT does not converge

PySCF solves the eigenpairs of 

$$
\left(
\begin{matrix}
A & B \\
-B & -A
\end{matrix}\right)
\left(\begin{matrix}
X \\
Y
\end{matrix}\right) =
\left(\begin{matrix}
X \\
Y
\end{matrix}\right)
\Omega$$


and diagonalize the response matrix $\left(\begin{matrix}A&B\\\\-B&-A\end{matrix}\right)$ using a generalized Davidson algorithm[1], and it leads to unconvergence problem.



## Solution

In stead, a more reliable way [2] is to solve 

$$\left(
\begin{matrix}
A & B \\\\
B & A
\end{matrix}\right)
\left(\begin{matrix}
X \\\\
Y
\end{matrix}\right) =
\left(\begin{matrix}
1 & 0 \\\\
0 & -1
\end{matrix}
\right)
\left(\begin{matrix}
X \\\\
Y
\end{matrix}\right)
\Omega
$$

In each iteration, both $\left(\begin{matrix} A & B \\\\ -B & -A \end{matrix}\right) $ and $ \left(\begin{matrix} 1 & 0 \\\\ 0 & -1 \end{matrix}\right) $ will be projected into the subspace, where we solve
$$
  \left(
  \begin{matrix}
  a & b \\\\
  b & a
  \end{matrix}\right)
  \left(\begin{matrix}
  x\\\\
  y
  \end{matrix}\right)
  = \left(\begin{matrix}
  \sigma & \pi \\\\
  -\pi & -\sigma
  \end{matrix}\right)
  \left(\begin{matrix}
  x\\\\
  y
  \end{matrix}\right)
  \Omega
$$

## reference

[1] Hirao, K. and Nakatsuji, H., 1982. A generalization of the Davidson's method to large nonsymmetric eigenvalue problems. Journal of Computational Physics, 45(2), pp.246-254.

[2] Olsen, J., Jensen, H.J.A. and Jørgensen, P., 1988. Solution of the large matrix equations which occur in response theory. Journal of Computational Physics, 74(2), pp.265-282.