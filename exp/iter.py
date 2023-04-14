import iio    # image input output
import imgra  # image processing with graphs

iio.version

x = iio.read("gbarbsquare.png")   # input image
m = iio.read("mask.png")[:,:,0]   # input mask
iio.gallery([x, 255*m])           # display input image and mask

h,w,d = x.shape                  # extract data dimensions
x = x.flatten()                  # flatten array data into a vector

B = imgra.grid_incidence(h,w)    # gradient operator
L = -B.T @ B                     # laplacian operator (divergence of gradient)
I = 0*L; I.setdiag(1)            # identity operator
M = 0*L; M.setdiag(m.flatten())  # mask operator

f = M @ L @ x                    # laplacian inside the mask
g = (I - M) @ x                  # original data outside the mask

iio.gallery([g, 127 - 2*f])        # show input data

# heat equation iteration (jacobi is a particular case for τ=1/8)
def iter_heat(u, I, M, L, f, g, τ):
	return (I - M) @ g + M @ ( u + τ * (L @ u - f))


τ = 0.24    # time step
u0 = 0*x    # initialization

U = [u0]    # array of iterations
for i in range(200):
	u = U[-1]
	u = iter_heat(u, I, M, L, f, g, τ)
	U.append(u)

iio.gallery([U[0],U[1],U[2],U[3],U[4],U[100],U[-1]])

# solve the system using a fancy linear solver
from scipy.sparse.linalg import spsolve

A = M @ L + I - M        # matrix of the linear system
b = M @ f + (I - M) @ g  # independent term
u = spsolve(A, b)        # solve the system Ax = b

iio.gallery([x,u,127+100000*(x-u)])


