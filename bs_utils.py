import math
from logsplines import array_lib as np
from logsplines import scipy_lib_linalg
from logsplines import scipy_lib_ndimage as nd
from logsplines import scipy_lib_sg as sg
from plt import plt


def rescale(xys, new_domain, old_domain=None):

    a, b = new_domain[:,0], new_domain[:,1]

    if old_domain is None:
        maxi = np.max(xys, axis=0)
        mini = np.min(xys, axis=0)
    else:
        mini, maxi = old_domain[:,0], old_domain[:,1]

    return a + (xys-mini)*(b-a)/(maxi-mini)


#def bs(eval_at_points, deg):
#    return sg.bspline(eval_at_points, deg) # bugs with integer points


def bs(eval, deg, positive=False):
    if positive:
        ax = eval
    else:
        ax = np.abs(eval)

    #h_sup = 0.5*(1+deg)
    if deg==0:
        return (ax<0.5)
    elif deg==1:
        return (ax<1.)*(1. - ax)
    elif deg==3:
        return (ax<2.)*(\
                         (ax<1)*(2./3. - ax**2 + 0.5*ax**3) \
                        +(ax>=1)*(1./6.)*((2. - ax)**3) \
                       )

    # Callables in piecewise not supported in cupy
    # elif deg==3:
    #     return np.piecewise(ax, \
    #                          [ax<1, ax>=1, ax>=2], \
    #                          [lambda x: 2./3. - x**2 + 0.5*x**3, \
    #                           lambda x: (1./6.)*((2. - x)**3), \
    #                           0])


def bs_to_filter(deg, scale=1, int_grid=True): # methods assume grid is regular so mayb no point
    half_domain = math.ceil((deg+1)/2)
    half_domain *= int(scale)
    if int_grid:
        half_domain -= 1 # edges are zero
    return bs(np.arange(-half_domain, half_domain+1, dtype=float)/scale, deg)


def bs_deriv_filter(deriv):
    pascal = [1.]
    for i in range(deriv):
        pascal.append(-1*pascal[i]*(deriv-i)/(i+1))
    return np.array(pascal)


def upsample_coeffs(coeffs, scale=1, bcs=None):
    up_size = (np.array(coeffs.shape)-1)*scale+1
    up_size = tuple(int(s) for s in up_size) # cupy compatibility
    up_coeffs = np.zeros(up_size) #autocast to tuple?
    #up_coeffs[::scale, ::scale] = coeffs
    sl = (np.s_[::scale],)*coeffs.ndim
    up_coeffs[sl] = coeffs
    return up_coeffs if bcs is None else np.pad(up_coeffs, tuple([(0,(scale-1)*(bc=="wrap")) for bc in bcs]))
    #np.pad(up_coeffs, tuple([(scale-1)*(bc=="wrap") for bc in bcs]))


def downsample_coeffs(coeffs, scale):
    #return coeffs[::scale, ::scale].copy()
    sl = (np.s_[::scale],)*coeffs.ndim
    return coeffs[sl].copy()


def convolve_coeffs(coeffs, bs_filters, bcs): # successive separable convolutions
    bs_conv = coeffs.copy()
    for i, bc in enumerate(bcs):
        bs_conv = nd.convolve1d(bs_conv, bs_filters[i], mode=bc, axis=i) #oaconvolve #look for convolves that changes fft or mult according to speed
    return bs_conv


def eval_bspline_on_grid(coeffs, deg, bcs, scale=1): # scale included just for some tricks
    bs_filters = np.array(coeffs.ndim*[bs_to_filter(deg, scale=scale)])
    return convolve_coeffs(coeffs, bs_filters, bcs)


def eval_bspline_on_subgrid(coeffs, deg, scale, bcs):
    bs_filters = np.array(coeffs.ndim*[bs_to_filter(deg, scale)])
    up_coeffs = upsample_coeffs(coeffs, scale, bcs=bcs)
    return convolve_coeffs(up_coeffs, bs_filters, bcs)


def bcs_to_value(bcs, periods=None, domain_shape=np.infty):
    if all(bc=="wrap" for bc in bcs) and (periods is None) and (domain_shape is not np.infty): return np.array(domain_shape)
    if hasattr(periods, "__iter__"): periods = list(periods)[::-1] # flip because pop later
    ct = 2*np.max(np.array(domain_shape)) # so that % does not affect the non-periodic dimensions; infinite=sys.maxsize-1 #np.infty (casts to float)
    return np.array([bc_to_value(bc, periods, ct) for bc in bcs])


def bc_to_value(bc, periods=1, ct=0):
    if bc == "wrap":
        return periods.pop() if hasattr(periods, "__iter__") else periods
    elif bc == "constant":
        return ct
    else:
        raise ValueError("This kind of boundary condition is not supported.") #ValueError or RuntimeError


def adjoint_conv_bcs(bcs):
    return [adjoint_conv_bc(bc) for bc in bcs]


def adjoint_conv_bc(bc):
    #if bc is not "wrap":
    #    bc = "constant"
    return bc


def spline_integral(coeffs, deg, scale, bcs, measure=1., out=False):
    values_on_grid = measure*eval_bspline_on_subgrid(coeffs, deg, scale, bcs)
    sum_of_values = np.sum(values_on_grid)/(scale**coeffs.ndim)
    return (sum_of_values, values_on_grid) if out else sum_of_values


def single_bspline_integral(support_dim, deg, power=1, scale=80):
    # compute for 1D bspline^power then power of support_dim because they are separable
    one_coeff = np.zeros(5); one_coeff[2]=1.
    _, v = spline_integral(one_coeff, deg, scale, ["wrap"], out=True)
    return (np.sum(v**power)/(scale**one_coeff.ndim))**support_dim


def eval_logspline_on_subgrid(coeffs, deg, scale, bcs, measure=1.):
    return measure * np.exp(
                    eval_bspline_on_subgrid(coeffs, deg, scale, bcs)
                    )


def overlapping_bs_filters_1d(deg, scale=1): #beta*beta filter
    supp = deg+1
    npoints = scale*supp+1-2
    #origin = int((npoints-1)/2)
    bf = bs_to_filter(deg, scale=scale)
    assert(npoints==len(bf))
    padin = npoints-1
    shifting_filter = scipy_lib_linalg.circulant(np.pad(bf, (0, padin))).T[:npoints,:npoints]
    
    return shifting_filter * bf#, origin