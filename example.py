from __init__ import array_lib as np
from plt import plt
from mas_splines.bs_Hessian import bs_Hessian
from mas_splines.HessianSchattenNorm import LpSchattenNorm, HessianSchattenNorm
from mas_splines.descent_algorithms import APGD
from mas_splines.bs_utils as bu


# Some class with call and gradient (i.e. can do nlsl.jacobianT() and nlsl() )
nlsl = bu.neg_logspline_likelihood(scaled_coords, coeffs_shape, deg, integration_scale, bcs, measure=measure, 
        updater={"approx":False, "u_multi":1., "use_circles":True})



# Some class with prox (i.e. can do xx.prox() and xx() )

reg = .5
prox_iters = 100

deriv_order = 2 # Hessian
bs_filters = [bu.bs_deriv_filter(i) for i in range(deriv_order+1)]
bs_filters[0] = bu.bs_to_filter(deg, int_grid=True)
bs_filters[1]= np.array([1., 0 ,-1.])/2.

Hessian = bs_Hessian(coeffs_shape, bs_filters, ["wrap", "wrap"])
Schatt_Norm = LpSchattenNorm(Hessian.hess_shape, p=1, hermitian=True, flat=True, flat_input=False)
Hessian_Schatt_Norm = HessianSchattenNorm(Hessian, Schatt_Norm, prox_iters, lam=reg)

#APGD
out_coeffs = APGD(nlsl, Hessian_Schatt_Norm, niter=50, verbose=1, update_ilip=True, restart_method="RESTART")
