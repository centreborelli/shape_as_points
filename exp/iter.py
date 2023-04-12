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

f = M @ L @ x      # laplacian inside the mask
g = (I - M) @ x    # original data outside the mask

iio.gallery([g, 127-2*f])
