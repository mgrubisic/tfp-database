import numpy as np
import math, cmath

gapRatio 	= 0.0293
TmRatio 	= 4.45
T2Ratio 	= 1.25
zeta = 0.20

# gapRatio 	= 0.0247
# T2Ratio 	= 1.25
# zeta 		= 0.10
# Tm 			= 3.25

mu1 		= 0.02

g 		= 386.4
inch 	= 1.0
pi 		= math.pi
S1 		= 1.017

Ss 		= 2.281
Ts 		= S1/Ss
Tm 		= TmRatio*Ts

SaTm 	= S1/Tm

# from ASCE Ch. 17
zetaRef = [0.02, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50]
BmRef	= [0.8, 1.0, 1.2, 1.5, 1.7, 1.9, 2.0]

Bm 		= np.interp(zeta, zetaRef, BmRef)


moatGap = g*(SaTm/Bm)*(Tm**2)*gapRatio
T2 		= T2Ratio*Tm
R2 		= T2**2*g/(8*pi**2)

b 		= 1/(2*R2)

Dm 		= g*S1*Tm/(4*pi**2*Bm)

x 		= Dm

xy		= 0.01*inch

k0		= mu1/xy


kM 		= (2*pi/Tm)**2 * (1/g)
Wm 		= zeta*(2*pi*kM*x**2)

# def f(mu2):
# 	return (((mu2-mu1)/(x-2*R2*(kM*x - mu2)) - 1/(2*R2))**(-1/2)*cmath.sqrt(-Wm/4 + (kM - 1/(2*R2))*x**2 - (k0 - (mu2-mu1)/(x-2*R2*(kM*x - mu2)))*xy**2) - x + 2*R2*(kM*x - mu2))

# mu2Val = fsolve(f, 0.10)
# print(mu2Val)


import sympy as sy
sy.init_printing()

# mu1, mu2, R1, R2, x, x1, kM, Wm, xy, k0 = sy.symbols('mu1 mu2 R1 R2 x x1 kM Wm xy k0')
mu2, R1 = sy.symbols('mu2 R1')

# solset = sy.solve( ( (mu2 + 1/(2*R2)*(x - 2*R1*(mu2-mu1))) /x - kM,
# 	4*(mu2 - 1/(2*R2)*(2*R1*(mu2-mu1)))*x - 4*(1/(2*R1) - 1/(2*R2))*(2*R1*(mu2-mu1))**2 - 4*(k0 - 1/(2*R1))*xy**2 - Wm), [mu2, R1], force=True, manual=True, set=True)

solset = sy.nsolve( [ (mu2 + 1/(2*R2)*(x - 2*R1*(mu2-mu1))) /x - kM,
	4*(mu2 - 1/(2*R2)*(2*R1*(mu2-mu1)))*x - 4*(1/(2*R1) - 1/(2*R2))*(2*R1*(mu2-mu1))**2 - 4*(k0 - 1/(2*R1))*xy**2 - Wm], [mu2, R1], [0.20, 35])

npsol = np.array(solset).astype(np.float64)

mu2 	= npsol[0].item()
R1 		= npsol[1].item()

a 		= 1/(2*R1)

x1 		= (a-b)**(-1/2)*math.sqrt(-Wm/4 + (kM - b)*x**2 - (k0 - a)*xy**2)
mu1 	=  mu2 - x1/(2*R1)

print("x1, mu1, mu2, R1, R2: ", "%.3f" % x1, "%.3f" % mu1, 
	"%.3f" % mu2, "%.3f" % R1, "%.3f" % R2)


ke 		= (mu2.real + b*(x - x1.real))/x
We 		= (4*(mu2.real - b*x1.real)*x - 
4*(a-b)*x1.real**2 - 4*(k0 -a)*xy**2)
zetaE	= We/(2*pi*ke*x**2)
Te 		= 2*pi/(cmath.sqrt(ke/(1/g)))

muList 	= [mu1, mu2, mu2]