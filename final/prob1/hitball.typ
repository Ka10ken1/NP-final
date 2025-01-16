#import "@preview/tinyset:0.1.0": *
#import "@preview/codly:1.1.1": *
#import "@preview/codly-languages:0.1.1": *
#show: codly-init.with()
#show link: underline


#set page("us-letter")

#set page(numbering: "— 1 —")

#set math.mat(delim: "[")

#set math.mat(gap: 1em)

#set text(
  style: "italic",
  size: 10pt
)

#set math.equation(numbering: "(1)")

#align(center)[
  #line(length:100%)
  = Projectile Motion \
  Kopaliani Mate
  #line(length:100%)
]

= Introduction

#figure(
image("./imgshooting.png")
)

#qs[
  Projectile Motion give as Second order ODE
  $
  (d^2 y)/(d t^2) = -g
  $
  where
  - $g = 9.81$
  - $y(t)$ is the position of the ball 
  initial conditions $y(a) = a_0,space y(b) = b_0$
  
]
== Solution
#qs[
  we solve by shooting method. first transform the ODE into first order
  $
  (d y)/(d t) = v \
  (d v)/(d t) = - g
  $
  written in vector form
  $
  S(t) = mat(y(t); v(t)) \
  (d S(t))/(d t) = mat(0, -1; 0, -g\/v) S(t)
  $
  ```python
  def F(t, s):
    return np.dot(np.array([[0, 1], [0, -9.8 / s[1]]]), s)
  ```
  we use RK$4$ and Euler to find the optimal velocity in shooting method
   - formulation of RK$4$
  $
  S_(n+1) = S_n + h/6 (k_1 + 2 k_2 + 2 k_3 + k_4)
  $
  #text(size:9pt)[
  $
  &k_1 = F(t_n, S_n) \
  &k_2 = F(t_n + h/2, S_n + h/2 k_1) \
  &k_3 = F(t_n + h/2, S_n + h/2 k_2) \
  &k_4 = F(t_n + h, S_n + h k_3)
  $
  ]
  - formulation of Euler
  $
  S_(n+1) = S_(n) + h F(t_n, S_n)
  $
  where
  - $F(t,S)$ is the system of equations defined as $F(t,s) = (d S)/(d t) $
  - $h$ is a step size
  - $t_n$ is the current time
  - $S_n$ is the state vector at time $t_n$
]

== analysis and comparison of methods
#qs[
  - Rk4 output
  ```
  target coordinates: 618, 375
  last trajectory point: 618.0, 375.0000000000109
  target coordinates: 229, 387
  last trajectory point: 229.0, 387.00000000000307
  target coordinates: 487, 377
  last trajectory point: 487.0, 377.0000000000409
  target coordinates: 356, 379
  last trajectory point: 356.0, 379.0000000000023
  target coordinates: 91, 379
  last trajectory point: 91.0, 378.99999999999454
  ```
  - Euler output
  ```
  target coordinates: 618, 375
  last trajectory point: 618.0, 375.0000000000582
  target coordinates: 229, 387
  last trajectory point: 229.0, 386.99999999998545
  target coordinates: 487, 377
  last trajectory point: 487.0, 376.9999999998545
  target coordinates: 356, 379
  last trajectory point: 356.0, 378.9999999997599
  target coordinates: 91, 379
  last trajectory point: 91.0, 379.0000000000025
  ```
  - Rk4  method achieves extremely high accuracy, with errors on the order of $10^(-10)$ to $10^(-12)$.
    - The errors are consistently small across all target coordinates.
  - Euler method also reaches target coordinates with error on the order of $10^(-10)$ to $10^(-12)$ aswell.
    - The errors on the other hand result in less consistent form. (i.e $376.9999999998545$ vs $377.0000000000409$ of Rk4)
    - Euler on the other hand is computationally much cheaper and therefore faster than Rk4
]

== References
#qs[
  heplful article \
  #link("https://pythonnumericalmethods.studentorg.berkeley.edu/notebooks/chapter23.02-The-Shooting-Method.html")
]
