#import "@preview/tinyset:0.1.0": *

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
  = Sturm-Lioville Problem \
  Kopaliani Mate
  #line(length:100%)
]

= Introduction
#pad(y:20pt)[]

#qs[
  General form of Sturm-Liouville problem
  $
  d/(d x) [p(x) (d phi.alt)/(d x)] + q(x) phi.alt = - lambda omega (x) phi.alt quad "for" a <= x <= b
  $
  where 
  - $p(x), q(x)$ and $omega (x)$ are continuous.
  - $p(x), omega(x) > 0$ on $[a,b]$
  - $phi.alt(x)$ is an eigen function.
]

== problem 1

#qs[
  consider the following chebyshev's differential equation
  $
  (1 - x^2) phi.alt '' - x phi.alt ' = - lambda phi.alt quad "for" -1 < x < 1
  $
  with initial conditions $phi.alt (-1) = 0, space phi.alt (1) = 0$. \
  to rewrite in Sturm-Lioville form first we divide by $sqrt(1 - x^2)$ to get
  $
  underbrace(sqrt(1 - x^2) phi.alt '' - x/sqrt(1 - x^2) phi.alt', (sqrt(1 - x^2) phi.alt ')') = - lambda/sqrt(1 - x^2) phi.alt \
  $
  with $p(x) = sqrt(1 - x^2)$, $q(x) = 0$ and $omega(x) = 1/sqrt(1 - x^2)$ so we have
  $
   d/(d x)[sqrt(1 - x^2) space (d phi.alt)/(d x)] = - lambda 1/sqrt(1 - x^2) phi.alt quad "for" -1 < x < 1
  $

]
== Solution
#qs[
  we will use shooting method to solve chebyshev's differential equation. general approach to rewrite second order into first order ODE.
  $
  u(x) = phi.alt(x)
  $
  $
  v(x) = phi.alt(x)
  $
  $
  u'(x) = v(x)
  $
  $
  v'(x) = (p'(x) v  + q(x) u + lambda omega (x) u)/(p(x))
  $
  in vector form
  $
  mat(u'; v') = mat(0,1; (-q(x) + lambda w(x))/p(x), (p'(x))/p(x)) mat(u;v)
  $
  in our case  \
  for first initial value problem
  $
  u_1 '(x) = v_1 (x)
  $
  $
  v_1 '(x) &= (-x/sqrt(1 - x^2) v_1 (x) + lambda/sqrt(1 - x^2) u_1 (x))/sqrt(1 - x^2) \
  &= (- x v_1 (x) + lambda u_1 (x)) / (1 - x^2)
  $
  $
  u_1 (-1) = 1, v_1 (-1) = 0
  $
  for second initial value problem
  $
  u_2 '(x) = v_2 (x)
  $
  $
  v_2 '(x) &= (-x/sqrt(1 - x^2) v_2 (x) + lambda/sqrt(1 - x^2) u_2 (x))/sqrt(1 - x^2) \
  &= (- x v_2 (x) + lambda u_2 (x)) / (1 - x^2)
  $
  $
  u_2 (-1) = 0, v_2 (-1) = 1
  $
  we find appropriate $alpha_1, alpha_2, beta_1, beta_2$ for Robin boundary conditions
  
  $
  &alpha_1 phi.alt (-1) + beta_1 phi.alt'(-1) = 0 \
  &alpha_2 phi.alt(1) + beta_2 phi.alt'(1) = 0
  $

  determinant function
  $
  F(lambda) &= det mat(
    &alpha_1 u_1(-1) + beta_1 v_1 (-1), &alpha_1 u_2 (-1) + beta_1 v_2 (-1);
    &alpha_2 u_1 (1) + beta_2 v_1 (1), &alpha_2 u_2 (1) + beta_2 v_2 (1)
  ) 
  $
  then we calcualte $lambda_n$ using newton's iteration
  $
  lambda_(n+1) = lambda_n - (F (lambda))/ (F' (lambda))
  $
  and find eigenfunction as linear combination
  $
  phi.alt (x) = c_1 u_1 (x) + c_2 u_2 (x)
  $
  and since eigenfunction of chebyshev's differential is same as chebyshev's polynomial we get the following result

  #figure(
    image("./chebyshev.png")
  )

]
== problem 2
#qs[
  $
  -1/2 (cos^4 (x) (d^2 u)/(d x^2) + (cos^3 (x) cos(2x))/sin(x) (d u)/ (d x)) + ((m^2 cos^2(x))/(2 sin^2 (x)) - cos(x)/sin(x)) u = lambda u \
  u(0) = 0, quad u(pi/2) = 0
  $
]
== Solution
#qs[
  firstly, we rewrite it in first order form
  $
  v(x) = (d u)/(d x) \
  v'(x) = (d^2 u)/(d x^2)
  $
  thus we have
  $
    -1/2 (cos^4 (x) (d v)/(d x) + (cos^3 (x) cos(2x))/sin(x) v) + ((m^2 cos^2(x))/(2 sin^2 (x)) - cos(x)/sin(x)) u = lambda u \
  $
  $=>$
  $
  &(d u)/(d x) = v \
  &(d v)/(d x) = -2/(cos^4(x)) (lambda u + (cos^3(x) cos(2x))/(2sin(x)) v - ((m^2 cos^2(x))/(2 sin^2 (x)) - cos(x)/sin(x)) u)
  $

  then rewrite in matrix form like equation $(9)$
  $
  mat(u'; v') &=  mat(
    0, 1; 
    (-2)/(cos^4(x)) (lambda - (m^2 cos^2 (x))/(2 sin^2 (x)) + cos(x)/sin(x)),
    - cos(2x)/(cos(x)sin(x))

  ) mat(u;v)
  $
  replace $cos(x)$ and $sin(x)$ by its taylor function approximations to avoid boundary value errors.
  $
  cos(x) approx 1 - x^2/2 + x^4/24 \
  sin(x) approx x - x^3/6 + x^5/120
  $
  then we have

  $
    mat(u'; v') &=  mat(
    0, 1; 
    (-2)/(1 - x^2/2 + x^4/24 )^4 (lambda - (m^2 (1 - x^2/2 + x^4/24 )^2)/(2 (x - x^3/6 + x^5/120)^2) + (1 - x^2/2 + x^4/24 )/(x - x^3/6 + x^5/120)),
    - (1 - 2x^2 + (4x^4)/6)/((1 - x^2/2 + x^4/24)(x - x^3/6 + x^5/120))

  ) mat(u;v)
  $
  we use implicit Runge Kutta formulas with butcher table.
  $ 
  mat(c_1, a_11, a_12;
      c_2, a_21, a_22;
      "", b_1, b_2;
    augment: #(hline: 2, vline: 1)) quad "and" mat(1/2, 1/2;
      "", 1;
    augment: #(hline: 1, vline: 1))
  $
  where for first table coefficients are from 2-stage 4-th order Gauss-Legendre method.
  $
  &c_1 = 1/2 - sqrt(3)/6, quad c_2 = 1/2 + sqrt(3)/6 \
  &a_11 = 1/4, space a_12 = 1/4 - sqrt(3)/6, space a_21 = 1/4 + sqrt(3)/6, space a_22 = 1/4  \
  &b_1 = 1/2, quad b_2 = 1/2
  $
  and for second case 1-stage second order Gauss-Legendre method.
  \
  some basic notations
  $
  Y = mat(u;v) quad A(x) =  mat(
    0, 1; 
    (-2)/(1 - x^2/2 + x^4/24 )^4 (lambda - (m^2 (1 - x^2/2 + x^4/24 )^2)/(2 (x - x^3/6 + x^5/120)^2) + (1 - x^2/2 + x^4/24 )/(x - x^3/6 + x^5/120)),
    - (1 - 2x^2 + (4x^4)/6)/((1 - x^2/2 + x^4/24)(x - x^3/6 + x^5/120))

  )  
  $
  with our problem
  $
  Y' = A(x) Y
  $
  for each stage $K_1, K_2$ general formula is
  $
  K_j = A(x_n + c_j h) (Y_n + h sum_(k=1)^2 a_(j k) K_k)
  $
  We solve the resulting nonlinear system for $K = [K_1, K_2]$, minimize the residual
  $
  R_j (K) = K_j - A(x_n + c_j h) (Y_n + h sum_(k=1)^2 a_(j k) K_k) = 0
  $
  we use Newton's method for that
  $
  K^(n+1) = K^n - J^(-1) R(K^n)
  $
  where $J$ is jacobian of $R(k)$ and we update the value
  $
  Y_(n+1) = Y_n + h (b_1 K_1 + b_2 K_2)
  $
  after rk4 and rk2 calculations we follow same as in the first example from equations $(17) - (18)$. ad we get following results.
  #figure(
    image("./examplesturm.png")
  )
]
== drawbacks
#qs[
- initial guess of lambda being far from the root results in Newton's method not converging
]
