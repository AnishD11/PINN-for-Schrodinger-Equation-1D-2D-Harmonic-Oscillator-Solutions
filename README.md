# PINNs to solve Schrödinger Equation for 1D and 2D Harmonic Oscillator
This repository contains Python codes implementing Physics-Informed Neural Networks (PINNs) using PyTorch to solve the Schrödinger equation for one-dimensional harmonic oscillator:

$\left[-\frac{\hbar^2}{2m} \frac{d^2}{dx^2} + \frac{1}{2}m\omega^2x^2\right]\psi(x) = E\psi(x)$\
$\psi(x)\longrightarrow 0 \hspace{4pt}as\hspace{4pt} x \longrightarrow \infty, -\infty$ 

and two-dimensional harmonic oscillator:

$\left[-\frac{\hbar^2}{2m} \left(\frac{\partial^2}{\partial x^2} + \frac{\partial^2}{\partial y^2}\right)+ \frac{1}{2}m\omega^2(x^2+y^2)\right]\psi(x,y) = E\psi(x,y)$\
$\psi(r)\longrightarrow 0 \hspace{4pt}as\hspace{4pt} r \longrightarrow \infty \hspace{4pt} where\hspace{4pt} r = \sqrt{x^2+y^2}$

This approch is totally unsupervised, meaning the network acts like a numerical solver of differential equations. The loss function is defined in the following form:

$L= \alpha L_{DE}+\beta L_{norm}+\gamma L_{ortho}$

The wavefunctions discovered by the networks are presented below:

![Predicted wave functions for the one-dimensional problem](https://github.com/AnishD11/PINN-for-Schrodinger-Equation-1D-2D-Harmonic-Oscillator-Solutions/blob/main/1dwf.png)
![Predicted wave functions for the two-dimensional problem](https://github.com/AnishD11/PINN-for-Schrodinger-Equation-1D-2D-Harmonic-Oscillator-Solutions/blob/main/2dwf.png)

While the codes are written specifically for the harmonic oscillator problem, they can be modified for any potential or any other differential equation in the form of eigenvalue problem.

The follwing dependenices are required:
<img src="https://upload.wikimedia.org/wikipedia/commons/thumb/1/10/PyTorch_logo_icon.svg/1200px-PyTorch_logo_icon.svg.png" width="20"/> [**PyTorch**](https://pytorch.org) - A deep learning framework.  
<img src="https://upload.wikimedia.org/wikipedia/commons/thumb/3/31/NumPy_logo_2020.svg/1280px-NumPy_logo_2020.svg.png" width="20"/> [**NumPy**](https://numpy.org) - A library for numerical computations.  
<img src="https://upload.wikimedia.org/wikipedia/commons/thumb/0/01/Created_with_Matplotlib-logo.svg/1024px-Created_with_Matplotlib-logo.svg.png" width="20"/> [**Matplotlib**](https://matplotlib.org) - A plotting library for Python.  
<img src="https://github.com/tqdm/tqdm/raw/master/images/tqdm.svg" width="20"/> [**TQDM**](https://tqdm.github.io/) - A fast, extensible progress bar for Python.  
<img src="https://upload.wikimedia.org/wikipedia/commons/2/28/Random_Lake_of_Spain.jpg" width="20"/> [**Random**](https://docs.python.org/3/library/random.html) - Python's built-in module for generating random numbers.  
<img src="https://upload.wikimedia.org/wikipedia/commons/6/66/Clock_simple.svg" width="20"/> [**Time**](https://docs.python.org/3/library/time.html) - Python's built-in module for time-related functions.



---
## Resources
- **Original PINN Paper:** [https://doi.org/10.48550/arXiv.2203.00451]
- **Official Software Implementation:** [https://github.com/henry1jin/quantumNN/tree/main]
- **Reference for 2D Problem:** [https://github.com/pmaczuga/pinn-notebooks]
