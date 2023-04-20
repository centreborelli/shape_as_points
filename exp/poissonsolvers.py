# # Poisson editing

# ## Auxiliary functions

# matrix of the gradient operator for a rectangular domain of size WxH
def discrete_gradient(h, w):
	from scipy.sparse import eye, kron, vstack
	x = eye(w-1, w, 1) - eye(w-1, w)    # path graph of length W
	y = eye(h-1, h, 1) - eye(h-1, h)    # path graph of length H
	p = kron(eye(h), x)                 # H horizontal paths
	q = kron(y, eye(w))                 # W vertical paths
	B = vstack([p, q])                  # union of all paths
	return B


def zoom_out_1d(n):
	from scipy.sparse import eye, kron
	P = kron(eye(n//2), [1,1]).tolil()  # identity made of [1,1] blocks
	k = n//2 + n%2                      # +1 if n is odd (no-op if even)
	P.resize(k,n)                       # new size (no-op if n is even)
	P[k-1,n-1] = 1                      # corner=1 (no-op if n is even)
	return P.tocsr()


# matrix of the un-normalized zoom-out operator for a rectangular domain WxH
# to normalize, multiply it by inv(diags(sum(P)))
def zoom_out(h, w):
	from scipy.sparse import kron
	p = zoom_out_1d(h)
	q = zoom_out_1d(w)
	P = kron(p, q)
	return P


def perform_zoom_out(x):
	P = zoom_out(*x.shape)
	y = P @ x.flatten()
	return y.reshape(x.shape[0]//2, x.shape[1]//2)  # TODO: fix odd case

#zoom_out(4,6).A.astype(int)


# find an image u such that
#         Δu = f  where m
#         u = g   where not m
# (solution by local discrete method)
def poisson_equation_local(
		f, # target laplacian
		g, # boundary condition
		m  # mask
		):
	from scipy.sparse import diags, eye
	from scipy.sparse.linalg import spsolve

	# flatten the images into vectors
	h,w = f.shape
	f = f.flatten()
	g = g.flatten()
	m = m.flatten()

	# state and solve the linear system
	B = discrete_gradient(h, w)  # gradient operator
	L = -B.T @ B                 # laplacian operator
	M = diags(m)                 # mask operator
	I = eye(h*w,h*w)             # identity operator
	A = (I - M)     - M @ L      # linear system: matrix
	b = (I - M) @ g - M @ f      # linear system: constant terms
	z = spsolve(A, b)            # linear system: solution
	u = z.reshape(h,w)           # recover a matrix from the solution vector
	return u

# run a few heat equation iterations
def heat_iterations(
		f,  # heat source (target laplacian)
		g,  # boundary condition
		m,  # mask
		i,  # initialization
		τ,  # time step
		n   # number of iterations
		):

	# flatten the images into vectors
	h,w = f.shape
	f = f.flatten()
	g = g.flatten()
	m = m.flatten()

	# build linear operators
	B = discrete_gradient(h,w)  # gradient operator
	L = -B.T @ B                # laplacian operator
	I = 0*L; I.setdiag(1)       # identity operator
	M = 0*L; M.setdiag(m)       # mask operator

	u = i.flatten()             # start with the given initalization
	for _ in range(n):          # iterate
		u = (I - M) @ g + M @ ( u + τ * (L @ u - f))

	# return result with correct shape
	return u.reshape(h,w)


# find a zero-mean image u such that Δu=f
# (solution by global frequency domain method)
def poisson_equation_global(f):
	from numpy.fft import fft2, ifft2, fftfreq
	from numpy import meshgrid
	from numpy import pi as π
	h,w = f.shape
	F = fft2(f)                             # laplacian, frequency domain
	p,q = meshgrid(fftfreq(w), fftfreq(h))  # frequency indices
	k = -(p**2 + q**2)**(-1)                # linear filter (inv. laplacian)
	k[0,0] = 0                              # set avg to zero
	U = k * F                               # apply linear filter
	u = ifft2(U).real                       # go back to spatial domain
	return u

# ## Experiments

# read input images
import iio
x = iio.read("gbarbsquare.png")[:,:,0]
m = iio.read("mask.png")[:,:,0]
iio.gallery([x, 255*m])


# produce data for poisson solver
def poisson_data(x, m):
	h,w = x.shape               # extract data dimensions
	x = x.flatten()             # flatten array into vector
	m = m.flatten()             # flatten array into vector

	B = discrete_gradient(h,w)  # gradient operator
	L = -B.T @ B                # laplacian operator
	I = 0*L; I.setdiag(1)       # identity operator
	M = 0*L; M.setdiag(m)       # mask operator

	f = M @ L @ x               # laplacian inside the mask
	g = (I - M) @ x             # original data outside the mask

	return f.reshape(h,w), g.reshape(h,w)

f,g = poisson_data(x,m)
iio.gallery([g, 127 - 4*f])

# solve poisson equation by various methods
u = poisson_equation_local(f, g, m)

u0 = heat_iterations(f, g, m, 127+0*f, 0.24, 0)
u1 = heat_iterations(f, g, m, 127+0*f, 0.24, 1)
u2 = heat_iterations(f, g, m, 127+0*f, 0.24, 10)
u3 = heat_iterations(f, g, m, 127+0*f, 0.24, 100)
u4 = heat_iterations(f, g, m, 127+0*f, 0.24, 300)


ug = poisson_equation_global(f)

iio.gallery([x,u0,u1,u2,u3,u4,127+0.01*ug])


# try zoom out
y = perform_zoom_out(x)
iio.gallery([x, y/4])

