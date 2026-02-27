#### Homework #1, AA2024-2024: Frequency estimation and CRB

CHEN Hong  11024836





## Sinusoidal Signal Model

The given sinusoidal signal is defined as:
$$
x[n]=a_0\cos(\omega_0n+\phi_0)
$$
The signal is passed through a discrete-time filter with impulse response:
$$
h[n]\quad\longleftrightarrow\quad H(z)=\frac{2}{1+0.9z^{-1}}
$$


The filtered output is:
$$
y[n]=x[n]*h[n]+w[n]
$$


Using the $z$-transform representation, the output in the frequency domain is:
$$
Y(z)=H(z)X(z)+W(z)
$$
The noise$w[n]$ follows a zero-mean Gaussian distribution with covariance matrix:
$$
w\sim\mathcal{N}(0,C_{ww})
$$
where the $N×N$ covariance matrix is defined as:
$$
[C_{ww}]_{i,j}=\sigma_w^2\rho^{|i-j|}
$$

### Estimation and MSE Computation

Given a set of $L$ realizations of the noise sequence ${{w_k[n]}^{L}_{k=1}}$, the sample covariance matrix is estimated as:
$$
\hat{C}_{ww}=\frac{1}{L}\sum_{k=1}^Lw_kw_k^T
$$
The MSE between the estimated sample covariance and the theoretical covariance is defined as:
$$
MSE(N)=\frac{1}{N^2}\sum_{i=1}^N\sum_{j=1}^N\left([\hat{C}_{ww}]_{i,j}-[C_{ww}]_{i,j}\right)^2
$$
We analyze the behavior of MSE as a function of $N$ for different values of $\rho$ (e.g., $\rho=0,0.5,0.99$ ).

## 1.1 Noise generation

<img src="D:\Aa_Polimi\Polimi_Studying\Sem1\SP&L\homework\HW1\1_1_Noise_Generation_MSEvsN.jpg" style="zoom:80%;" />

## 1.2 Frequency estimation

The noise $w[n]$ is independent and identically distributed (i.i.d.) Gaussian noise, so the probability density function (PDF) of the observed signal $x[n]$ is:
$$
p(\mathbf{x};\omega)=\prod_{n=0}^{N-1}\frac{1}{\sqrt{2\pi\sigma_w^2}}\exp\left(-\frac{(x[n]-A\cos(\omega n+\phi))^2}{2\sigma_w^2}\right)
$$
The log-likelihood function is:
$$
\ln p(\mathbf{x};\omega)=-\frac{N}{2}\ln(2\pi\sigma_w^2)-\frac{1}{2\sigma_w^2}\sum_{n=0}^{N-1}(x[n]-A\cos(\omega n+\phi))^2
$$
The Fisher information $I(\omega)$ is key to the Cramér-Rao bound (CRB) and is defined as the negative expectation of the second derivative of the log-likelihood function:
$$
I(\omega)=-\mathbb{E}\left[\frac{\partial^2\ln p(\mathbf{x};\omega)}{\partial\omega^2}\right]
$$
First, compute the first-order derivative:
$$
\frac{\partial\ln p(\mathbf{x};\omega)}{\partial\omega}=\frac{A}{\sigma_w^2}\sum_{n=0}^{N-1}(x[n]-A\cos(\omega n+\phi))\cdot n\sin(\omega n+\phi)
$$
Next, compute the second-order derivative:
$$
\frac{\partial^2\ln p(\mathbf{x};\omega)}{\partial\omega^2}=-\frac{A}{\sigma_w^2}\sum_{n=0}^{N-1}\left[n^2A\cos(\omega n+\phi)\sin(\omega n+\phi)-n(x[n]-A\cos(\omega n+\phi))\cos(\omega n+\phi)]\right.
$$
Since the noise $w[n]$ has a mean of 0, we have:

​											$E[x[n]−Acos⁡(ωn+ϕ)]=0$

The cross term satisfies:
$$
E[w[n]cos⁡(ωn+ϕ)]=0
$$
Thus, we finally obtain:
$$
I(\omega)=\frac{A^2}{\sigma_w^2}\sum_{n=0}^{N-1}n^2\cos^2(\omega n+\phi)
$$
For large $N$, the summation can be approximated by an integral, leveraging trigonometric properties:
$$
\sum_{n=0}^{N-1}n^2\cos^2(\omega n+\phi)\approx\sum_{n=0}^{N-1}n^2\cdot\frac{1}{2}=\frac{1}{2}\sum_{n=0}^{N-1}n^2
$$
Using the summation formula:


$$
\sum_{n=0}^{N-1}n^2=\frac{(N-1)N(2N-1)}{6}
$$
Thus,
$$
I(\omega)\approx\frac{A^2}{\sigma_w^2}\cdot\frac{1}{2}\cdot\frac{(N-1)N(2N-1)}{6}=\frac{A^2(N-1)N(2N-1)}{12\sigma_w^2}
$$

## **Cramér-Rao Bound**

The Cramér-Rao Bound (CRB) is the inverse of the Fisher information:
$$
\mathrm{CRB}(\omega)=\frac{1}{I(\omega)}=\frac{12\sigma_w^2}{A^2(N-1)N(2N-1)}
$$
Substitute the Signal-to-Noise Ratio $\text{SNR} = \frac{A^2}{2 \sigma_w^2}$
$$
\mathrm{CRB}(\omega)=\frac{6}{\mathrm{SNR}\cdot N\cdot(N-1)\cdot(2N-1)}
$$


### 1.2.1 Filter $h[n] = δ[n]$

![](D:\Aa_Polimi\Polimi_Studying\Sem1\SP&L\homework\HW1\plot\1_2_1_1.jpg)

![](D:\Aa_Polimi\Polimi_Studying\Sem1\SP&L\homework\HW1\plot\1_2_1_2.jpg)

### 1.2.2 Filter **$h[n] \neq δ[n]$**

![](D:\Aa_Polimi\Polimi_Studying\Sem1\SP&L\homework\HW1\plot\1_2_2_1.jpg)

![](D:\Aa_Polimi\Polimi_Studying\Sem1\SP&L\homework\HW1\plot\1_2_2_2.jpg)



### 1.2.3 Covariance is correlated

①$h[n]=\delta[n]$

![](D:\Aa_Polimi\Polimi_Studying\Sem1\SP&L\homework\HW1\plot\1_2_3_1.jpg)

![](D:\Aa_Polimi\Polimi_Studying\Sem1\SP&L\homework\HW1\plot\1_2_3_2.jpg)

②$h[n]\neq\delta[n]$

![](D:\Aa_Polimi\Polimi_Studying\Sem1\SP&L\homework\HW1\plot\1_2_3_3.jpg)

![](D:\Aa_Polimi\Polimi_Studying\Sem1\SP&L\homework\HW1\plot\1_2_3_4.jpg)

### 1.3 Frequency modulation

![](D:\Aa_Polimi\Polimi_Studying\Sem1\SP&L\homework\HW1\plot\1_3_a_1.jpg) 