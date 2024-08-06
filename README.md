# PINN-for-Schrodinger-Equation-1D-2D-Harmonic-Oscillator-Solutions-
This repository contains Python codes implementing Physics-Informed Neural Networks (PINNs) using PyTorch to solve the Schrödinger equation for one-dimensional harmonic oscillator:

$\left[-\frac{\hbar^2}{2m} \frac{d^2}{dx^2} + \frac{1}{2}m\omega^2x^2\right]\psi(x) = E\psi(x)$\
$\psi(x)\longrightarrow 0 \hspace{4pt}as\hspace{4pt} x \longrightarrow \infty, -\infty$ 

and two-dimensional harmonic oscillator:

$\left[-\frac{\hbar^2}{2m} \left(\frac{\partial^2}{\partial x^2} + \frac{\partial^2}{\partial y^2}\right)+ \frac{1}{2}m\omega^2(x^2+y^2)\right]\psi(x,y) = E\psi(x,y)$\
$\psi(r)\longrightarrow 0 \hspace{4pt}as\hspace{4pt} r \longrightarrow \infty \hspace{4pt} where\hspace{4pt} r = \sqrt{x^2+y^2}$

This approch is totally unsupervised, meaning the network acts like a numerical solver of differential equations. The loss function is defined in the following form:

$L= \alpha L_{DE}+\beta L_{norm}+\gamma L_{ortho}$

While the codes are written specially for the harmonic oscillator problem, they can be modified for any potential or any other differential equation in the form of eigenvalue problem.

