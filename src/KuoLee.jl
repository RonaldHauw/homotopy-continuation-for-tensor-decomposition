
"""
# KuoLee

This module contains two methods.
    1. `kl_cpd`
    2. `kl_lr_cpd`

The first implements the cpd algorithm for a third order tensor as described in

_Y.-C. Kuo and T.-L. Lee. Computing the unique candecomp/parafac decomposition of
unbalanced tensors by homotopie method.
Linear Algebra and its applications, 556:238–264, 2018._


The second implements a similar algorithm but optimised towards lower ranks. This method
only works if r < n₂ and r < n₃.

In both cases it is required that r < n₁.
"""
module KuoLee
using DynamicPolynomials
using Random
using LinearAlgebra: norm, \, *, svd, rank, Diagonal, nullspace, dot, qr, cond
using HomotopyContinuation

include("KuoLeeUtils.jl")
using .KuoLeeUtils

export kl_cpd, kl_lr_cpd


"""
# `kl_cpd`

Calculate the cpd of an order 3 tensor via the method described in Kuo, Lee.

_Y.-C. Kuo and T.-L. Lee. Computing the unique candecomp/parafac decomposition of
unbalanced tensors by homotopie method.
Linear Algebra and its applications, 556:238–264, 2018._

## Inputs

- `T`: flattened real or complex tensor, type `Array{d_type, 2}`
- `r`: rank of the tensor, type `Int64`
- `dims`: tuple of the dimensions, first dimension should be larger than r, type `Tuple{Int64, Int64, Int64}`.

## Outputs

The method outputs the calculated factor matrices
```math
    T \\approx (Z_{\\text{calc}} \\odot Y_{\\text{calc}})X_{\\text{calc}}^H
```

- `Xcalc`
- `Ycalc`
- `Zcalc`

## Example
```julia
dims = (43, 8, 7);
r = 17;
X, Y, Z, T = cpd_gen(dims, r);

Xcalc, Ycalc, Zcalc = kl_cpd(T, r, dims);

T̃ = cpd_gen(Xcalc, Ycalc, Zcalc);
ϵ = norm(T-T̃)/norm(T);
```
"""
function kl_cpd(
    T::Array{d_type, 2},
    r::Int64,
    dims::Tuple{Int64, Int64, Int64};
    relative_residual_check = 1e-4,
    kwargs...
    ) where d_type

    # to be extended for higher orders
    @assert length(dims) == 3 "This function only works for tensors of order 3."

    # unpack dimensions
    n₁, n₂, n₃ = dims

    # check if Kuo Lee applies for this tensor
    if n₁ != maximum(dims)
        println("Warning: The first dimension is not the largest dimension. Consider swapping the dimensions.")
    end
    @assert r <= n₁  "The largest dimension is smaller than the rank. Decrease the rank or increase the largest dimension."
    @assert r <= n₂*n₃ "The product of the second and third dimension is larger than the rank."

    # rank factorization
    E, F = rankfact(T, r)

    # orthogonal basis of null space of E
    N = nullspace(Array{d_type}(E'));

    # define variables for polynomial
    @polyvar 𝐲[1:n₂] 𝐳[1:n₃];

    # set polynomial system
    𝐩 = constrpoly(N, dims, r, (𝐲, 𝐳))

    # solve polynomial system using Homotopy Continuation. System is 2-homogeneous.
    res = HomotopyContinuation.solve(𝐩; variable_groups=[𝐲, 𝐳], kwargs...);

    # filter solutions such that they are unique (overkill; could potentially be removed)
    𝛇 = solutions(res);
    𝛇 = HomotopyContinuation.unique_points(𝛇)

    # check number of solutions
    @assert length(𝛇) >= r "Homotopy continuation to solve the square system didn`t find enough solutions.
    $(length(𝛇)) solutions found while the rank r = $(r)."

    # check solution
    maxresidual = abs(maximum([norm([p([𝐲; 𝐳]=>𝛇[l]) for p in 𝐩]) for l ∈ 1:length(𝛇)]))
    maxsolution = abs(maximum([norm(𝛇[l]) for l ∈ 1:length(𝛇)]))
    @assert maxresidual/maxsolution < relative_residual_check "Residual after homotopy continuation on the square system too high.
    Maximum residual for an equation = $(maxresidual). Maximum solution norm = $(maxsolution). Is the system well conditioned?"

    # filter solutions to be orthogonal to U and not only to U[:, 1:I+J-2]
    Ycalc, Zcalc = filtsols(𝛇, N, dims, r)

    # check number of solutions
    @assert length(Ycalc) == r "Number of solutions of full system too small.
    $(length(Ycalc)) found while the rank r = $(r)."

    # check if solutions are perpendicular to Null(Eᵀ)
    maxresidual = abs(maximum([norm([dot(kron(Zcalc[i], Ycalc[i]), N[:, j]) for i ∈ 1:r]) for j ∈ 1:n₂*n₃-r]))
    maxsolution = abs(maximum([norm([Ycalc[i]; Zcalc[i]]) for i ∈ 1:r]))
    @assert maxresidual/maxsolution < relative_residual_check "Residual for remaining soltuions on full system too high.
    Maximum residual equals $(maxresidual). Maximum norm of a solution equals $(maxsolution)."

    # re-compose matrices
    Ycalc = reduce(hcat, Ycalc);
    Zcalc = reduce(hcat, Zcalc);

    # normbalance
    Ycalc, Zcalc = normbalance(Ycalc, Zcalc)

    # compute the rotation between the given basis E and the calculated rank one basis Z ⊙ Y
    E_ = reduce(hcat, [kron(Zcalc[:, i], Ycalc[:, i]) for i ∈ 1:r]);
    W = E'*E_;

    # compute X by solving a linear system
    Xcalc = (W \ F')';
    Xcalc = Array{d_type, 2}(Xcalc);

    # normbalance
    Xcalc, Ycalc, Zcalc = normbalance(Xcalc, Ycalc, Zcalc)

    # recompose flattened tensor
    Tcalc = reduce(hcat, [kron(Zcalc[:, i], Ycalc[:, i]) for i ∈ 1:r])*Xcalc';

    # calculate backward error
    bwerr = maximum(abs.(Tcalc-T))/maximum(abs.(T))
    @assert bwerr < relative_residual_check "Relative backward error in ∞ norm of $(bwerr) too high."

    return (Xcalc, Ycalc, Zcalc)

end # klcpd
# overload for different order of required variables
function kl_cpd(
    T::Array{d_type, 2},
    dims::Tuple{Int64, Int64, Int64},
    r::Int64;
    kwargs...
    ) where d_type
    return kl_cpd(T, r, dims, kwargs...)
end # klcpd



"""
# `kl_lr_cpd`

Calculate the cpd of an order 3 unbalanced flattened tensor via the method described in Kuo, Lee.

_Y.-C. Kuo and T.-L. Lee. Computing the unique candecomp/parafac decomposition of
unbalanced tensors by homotopie method.
Linear Algebra and its applications, 556:238–264, 2018._

This method is optimised towards lower ranks: r < n₂ and r < n₃. The flattened tensor has the following decomposition:
```math
    T = (Z \\odot Y)X^H = EF^H.
```
As described in the thesis, this method exploits the information that the linear subspaces 𝐘 and 𝐙 are r dimensional and known.
Therefor we search a basis Y for 𝐘 and a basis Z for 𝐙 s.t. Z ⊙ Y is a basis for E.

## Inputs

- `T`: flattened real or complex tensor, type `Array{d_type, 2}`
- `r`: rank of the tensor, type `Int64`
- `dims`: tuple of the dimensions, first dimension should be larger than r, type `Tuple{Int64, Int64, Int64}`.

## Outputs

The method outputs the calculated factor matrices
```math
    T \\approx (Z_{\\text{calc}} \\odot Y_{\\text{calc}})X_{\\text{calc}}^H
```

- `Xcalc`
- `Ycalc`
- `Zcalc`

"""
function kl_lr_cpd(
    T::Array{d_type, 2},
    r::Int64,
    dims::Tuple{Int64, Int64, Int64};
    kwargs...
    ) where d_type

    # to be extended for higher orders
    @assert length(dims) == 3 "This function only works for tensors of order 3."

    # unpack dimensions
    n₁, n₂, n₃ = dims

    # check if Kuo Lee applies for this tensor
    if n₁ != maximum(dims)
        println("Warning: The first dimension is not the largest dimension. Consider swapping the dimensions.")
    end
    @assert r <= n₁  "The largest dimension is smaller than the rank. Decrease the rank or increase the largest dimension."
    @assert r <= n₂*n₃ "The product of the second and third dimension is larger than the rank."
    @assert r < n₂ && r < n₃ "This method requires r < min(n₂, n₃)."

    # rank factorization
    E, F = rankfact(T, r)

    Ŷ, Ẑ = KuoLeeUtils.solution_space(E, dims, r)

    # orthogonal basis of null space of E
    N = nullspace(Array{d_type}(E'));

    # define variables for polynomial
    @polyvar 𝐪₁[1:r] 𝐪₂[1:r];

    # set polynomial system
    𝐩 = constrpoly_lowrank(Ŷ, Ẑ, N, dims, r, (𝐪₁, 𝐪₂))

    # solve polynomial system using Homotopy Continuation. System is 2-homogeneous.
    res = HomotopyContinuation.solve(𝐩; variable_groups=[𝐪₁, 𝐪₂], kwargs...);

    # filter solutions such that they are unique (overkill; could potentially be removed)
    𝛇 = solutions(res);
    𝛇 = HomotopyContinuation.unique_points(𝛇)

    # check number of solutions
    @assert length(𝛇) >= r "Homotopy continuation to solve the square system didn`t find enough solutions.
    $(length(𝛇)) solutions found while the rank r = $(r)."

    # check solution
    maxresidual = abs(maximum([norm([p([𝐪₁; 𝐪₂]=>𝛇[l]) for p in 𝐩]) for l ∈ 1:length(𝛇)]))
    maxsolution = abs(maximum([norm(𝛇[l]) for l ∈ 1:length(𝛇)]))
    @assert maxresidual/maxsolution < 1e-4 "Residual after homotopy continuation on the square system too high.
    Maximum residual for an equation = $(maxresidual). Maximum solution norm = $(maxsolution). Is the system well conditioned?"

    # filter solutions to be orthogonal to U and not only to U[:, 1:I+J-2]
    Ycalc, Zcalc = filtsols_lowrank(𝛇, Ŷ, Ẑ, N, dims, r)

    # check number of solutions
    @assert length(Ycalc) == r "Number of solutions of full system too small.
    $(length(Ycalc)) found while the rank r = $(r)."

    # check if solutions are perpendicular to Null(Eᵀ)
    maxresidual = abs(maximum([norm([dot(kron(Zcalc[i], Ycalc[i]), N[:, j]) for i ∈ 1:r]) for j ∈ 1:n₂*n₃-r]))
    maxsolution = abs(maximum([norm([Ycalc[i]; Zcalc[i]]) for i ∈ 1:r]))
    @assert maxresidual/maxsolution < 1e-4 "Residual for remaining soltuions on full system too high.
    Maximum residual equals $(maxresidual). Maximum norm of a solution equals $(maxsolution)."

    # re-compose matrices
    Ycalc = reduce(hcat, Ycalc);
    Zcalc = reduce(hcat, Zcalc);

    # normbalance
    Ycalc, Zcalc = normbalance(Ycalc, Zcalc)

    # compute the rotation between the given basis E and the calculated rank one basis Z ⊙ Y
    E_ = reduce(hcat, [kron(Zcalc[:, i], Ycalc[:, i]) for i ∈ 1:r]);
    W = E'*E_;

    # compute X by solving a linear system
    Xcalc = (W \ F')';
    Xcalc = Array{d_type, 2}(Xcalc);

    # normbalance
    Xcalc, Ycalc, Zcalc = normbalance(Xcalc, Ycalc, Zcalc)

    # recompose flattened tensor
    Tcalc = reduce(hcat, [kron(Zcalc[:, i], Ycalc[:, i]) for i ∈ 1:r])*Xcalc';

    # calculate backward error
    bwerr = maximum(abs.(Tcalc-T))/maximum(abs.(T))
    @assert bwerr < 1e-4 "Relative backward error in ∞ norm of $(bwerr) too high."

    return (Xcalc, Ycalc, Zcalc)

end # klcpd
function kl_lr_cpd(
    T::Array{d_type, 2},
    dims::Tuple{Int64, Int64, Int64},
    r::Int64;
    kwargs...
    ) where d_type
    return kl_lr_cpd(T, r, dims, kwargs...)
end # klcpd



end # module KuoLee
