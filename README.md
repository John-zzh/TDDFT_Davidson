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
\begin{equation*}
\left(
\begin{matrix}
A & B \\
-B & -A
\end{matrix}\right)
\left(\begin{matrix}
X \\
Y
\end{matrix}\right)
=
\left(\begin{matrix}
X \\
Y
\end{matrix}\right)
\Omega
\end{equation*}$$

and diagonalize the response matrix $ \left(\begin{matrix} A & B \\-B & -A \end{matrix}\right) $ using traditional Davidson algorithm as if this response matrix is symmatric. But it is actually not symmatric, and it leads to unconvergence problem.


## Solution

In stead, a more reliable way is to solve 

$$\begin{equation*}
\left(
\begin{matrix}
A & B \\
B & A
\end{matrix}\right)
\left(\begin{matrix}
X\\
Y
\end{matrix}\right)
=
\left(\begin{matrix}
1 & 0 \\
0 & -1
\end{matrix}
\right)
\left(\begin{matrix}
X\\
Y
\end{matrix}\right)
\Omega
\end{equation*}$$

In each iteration, both $ \left(\begin{matrix} A & B \\-B & -A \end{matrix}\right) $ and $ \left(\begin{matrix} 1 & 0 \\0 & -1 \end{matrix}\right) $ will be projected into the subspace, where we solve
$$
  \left(
  \begin{matrix}
  a & b \\
  b & a
  \end{matrix}\right)
  \left(\begin{matrix}
  x\\
  y
  \end{matrix}\right)
  =
  \left(\begin{matrix}
  \sigma & \pi \\
  -\pi & -\sigma
  \end{matrix}\right)
  \left(\begin{matrix}
  x\\
  y
  \end{matrix}\right)
  \Omega
$$