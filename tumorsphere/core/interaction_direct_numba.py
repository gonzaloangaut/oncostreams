"""EXPERIMENTAL fast Grosmann interaction path, keeping Culture API intact.

No per-Cell temporary neighbor dictionaries. Compile Gaussian overlaps in
canonical (first-visited/lower-index) direction to preserve old symmetric
cache semantics. Preserve force summation order: earlier cached neighbors
first (ascending), then newly discovered later neighbors in bucket iteration
order. This is intentionally a full alternative to Culture.interaction,
NOT a production patch. Only supports fixed-N planar Numba dynamics; validation is required before production use.
"""
from __future__ import annotations
import numpy as np
from numba import njit
from tumorsphere.core.forces_numba import grosmann_force_torque_kernel, XI_POWER_MAX


@njit(cache=False, fastmath=False)
def overlap_canonical(i, inds, di, ei, pi, qix, qiy,
                      dj_array, ej_array, phij, qjx, qjy, relative, area):
    """Evaluate normalizations in the same reference direction as the old cache.

    Under the ascending active-cell iteration of fixed-N planar cultures,
    each pair's first overlap is calculated with min(i,j) as the reference.
    """
    n=len(inds)
    result=np.empty(n, dtype=np.float64)
    for k in range(n):
        if int(inds[k]) >= i:
            d0=di; d1=dj_array[k]
            e0=ei; e1=ej_array[k]
            p0=pi; p1=phij[k]
            x0=qix; y0=qiy
            x1=qjx[k]; y1=qjy[k]
            rx=relative[k,0]; ry=relative[k,1]; rz=relative[k,2]
        else:
            d0=dj_array[k]; d1=di
            e0=ej_array[k]; e1=ei
            p0=phij[k]; p1=pi
            x0=qjx[k]; y0=qjy[k]
            x1=qix; y1=qiy
            rx=-relative[k,0]; ry=-relative[k,1]; rz=-relative[k,2]
        ds=d0+d1
        delta=p0-p1
        cos_delta=np.cos(delta)
        beta=(ds ** 2 - (d0 * e0 - d1 * e1) ** 2
              - 4.0 * d0 * d1 * e0 * e1 * (cos_delta ** 2))
        w0=d0*e0
        w1=d1*e1
        mxx=(w0*x0+w1*x1)/ds
        mxy=(w0*y0+w1*y1)/ds
        quad=((1.0-mxx)*rx**2+(1.0+mxx)*ry**2
              - 2.0*mxy*rx*ry+rz**2)
        # The legacy kernel evaluates imax before norm, although imax does not
        # affect norm; reproduce it to minimize numerical divergence.
        imax=4.0*area**2/(np.pi*np.sqrt(beta))
        norm=np.exp(-(ds/beta)*quad)
        result[k]=norm
    return result


def direct_interaction(self, cell_index, delta_t):
    """Drop-in Culture.interaction for fixed-N Grosmann/Numba experiments."""
    from tumorsphere.core.forces import Anisotropic_Grosmann
    if not isinstance(self.force, Anisotropic_Grosmann) or self.force.force_backend != 'numba':
        raise RuntimeError('Only Anisotropic_Grosmann force_backend=numba')
    if not self.movement or self.reproduction:
        raise RuntimeError('Requires movement=True, reproduction=False')
    cell=self.cells[cell_index]
    candidates=[int(j) for j in self.grid.find_neighbors(position=self.cell_positions[cell_index])
                if int(j)!=cell_index]
    # The old symmetric cache populated earlier-cell neighbors before the
    # current cell's bucket enumeration. Force summation order must match.
    earlier=sorted(j for j in candidates if j < cell_index)
    later=[j for j in candidates if j > cell_index]
    indices=np.asarray(earlier+later,dtype=np.intp)
    ni=len(indices)
    di=float(cell.squared_diagonal)
    ei=float(cell.anisotropy)
    pi=float(self.cell_phies[cell_index])
    qi=self.nematic_tensors[cell_index]
    if ni:
        neighbors=[self.cells[int(j)] for j in indices]
        dj=np.array([c.squared_diagonal for c in neighbors],dtype=np.float64)
        ej=np.array([c.anisotropy for c in neighbors],dtype=np.float64)
        pj=np.asarray(self.cell_phies[indices],dtype=np.float64)
        qj=self.nematic_tensors[indices]
        qjx=np.ascontiguousarray(qj[:,0,0])
        qjy=np.ascontiguousarray(qj[:,0,1])
        relative=self.calculate_relative_positions(self.cell_positions[cell_index],
                                                    self.cell_positions[indices])
        xis=overlap_canonical(cell_index,indices,di,ei,pi,float(qi[0,0]),float(qi[0,1]),
                              dj,ej,pj,qjx,qjy,relative,float(self.cell_area))
        if getattr(self.output,'collect_overlap_statistics',True):
            self.max_normalized_overlap_interval=max(self.max_normalized_overlap_interval,
                                                     float(np.max(xis)))
            self.normalized_overlap_sum_interval+=float(np.sum(xis))
            self.normalized_overlap_count_interval+=int(xis.size)
        select=xis>self.overlap_threshold_ratio
        ns=int(np.count_nonzero(select))
    else:
        ns=0
    mP,mS,mR=self.force.calculate_mobilities(cell,self.cell_area)
    if ns:
        force,torque=grosmann_force_torque_kernel(
            di,ei,pi,float(qi[0,0]),float(qi[0,1]),
            np.ascontiguousarray(dj[select]),np.ascontiguousarray(ej[select]),
            np.ascontiguousarray(pj[select]),np.ascontiguousarray(qjx[select]),
            np.ascontiguousarray(qjy[select]),
            np.ascontiguousarray(relative[select]),np.ascontiguousarray(xis[select]),
            float(self.force.kRep),float(self.force.bExp),float(self.force.lambda_core),XI_POWER_MAX)
    else:
        force=np.zeros(3,dtype=np.float64)
        torque=0.0
    identity=np.identity(3)
    dif_velocity=np.matmul(((mP+mS)/2)*identity+((mP-mS)/2)*qi,force)
    dif_position=(cell.velocity()+dif_velocity)*delta_t
    dif_phi=mR*torque*delta_t
    if self.force.translational_noise_enabled or self.force.rotational_noise_enabled:
        translational_noise,rotational_noise=self.force.calculate_noise(
            self.cells,self.cell_phies,cell_index,self.cell_area,delta_t,mP,mS,mR)
        dif_position+=translational_noise
        dif_phi+=rotational_noise
    if self.force.shrinking is True:
        self.force.check_shrink_condition(cell,dif_velocity)
    return dif_position,dif_phi
