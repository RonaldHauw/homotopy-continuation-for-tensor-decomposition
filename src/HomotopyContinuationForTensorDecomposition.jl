module HomotopyContinuationForTensorDecomposition

import LinearAlgebra
import HomotopyContinuation
import DynamicPolynomials
import Random
import Markdown

include("Utils.jl")
using .Utils

#include("Asymptote.jl")
#using .AsyPlots

include("KuoLeeUtils.jl")
using .KuoLeeUtils

include("KuoLee.jl")
using .KuoLee

include("Dual.jl")
using .Dual


#include("../benchmarks/Benchmarks.jl")
#using .BenchMarks


using DynamicPolynomials
using HomotopyContinuation
using Random
using LinearAlgebra
using Markdown

export kl_cpd, vfunc, cpd_gen, kld_fd_cpd, kld_pr_cpd, cpd, kl_lr_cpd
export normbalance, column_align, signnorm, condtdp


introduction_text = md"""
Dear user,

The HomotopyContinuationForTensorDecomposition package should be installing now.

This software package accompagnies my master's thesis: `Homotopy Continuation for Tensor Decompositions`.
Documentation can be accessed in 3 ways:

        1. In the appendices of the thesis text;
        2. As a web-page in `docs/src/index.html`;
        3. By prompting `?method_to_look_up` in the Julia REPL.

The 5 implemented methods are:

        1. kl_cpd (cpd via the method described by Kuo, Lee);
        2. kld_fd_cpd (cpd via the dual problem by reducing the size of the square system);
        3. kld_pr_cpd (cpd via the dual problem by a parameter homotopy);
        4. kl_lr_cpd (Kuo and Lee's method optimised for low ranks);
        5. cpd (wrapper which selects the optimal method from those above).

Don't hestitate to reach out should there be an issue during installation or if you have any other question regarding the usage of this package.

Best,

Ronald

May 14th 2020



"""

# print a welcome message
display(introduction_text)

# execute common case such that this code precompiles
# these are also tests since @assert statements on backward error are included
dimensions = (4, 3, 3);
r = 2;
X, Y, Z, T = cpd_gen(dimensions, r);
Xcalc, Ycalc, Zcalc = kl_cpd(T, dimensions, r);
println("\n\n   - compiled and tested kl_cpd")
Xcalc, Ycalc, Zcalc = kld_fd_cpd(T, dimensions, r);
println("   - compiled and tested kld_fd_cpd")
Xcalc, Ycalc, Zcalc = kld_pr_cpd(T, dimensions, r);
println("   - compiled and tested kld_pr_cpd")
Xcalc, Ycalc, Zcalc = kl_lr_cpd(T, dimensions, r);
println("   - compiled and tested kl_lr_cpd")


"""
# `cpd(T, dims, r)`

Chooses and calls the most efficiënt algorithm from

1. `kl_cpd`
2. `kld_fd_cpd`
3. `kld_pr_cpd`
4. `kl_lr_cpd`

by taking into acount the dimensions and the rank.

- if r < n₂ and r < n₃: `kl_lr_cpd`
- if r < n₂ or r < n₃: `kld_fd_cpd`
- if r <= n₂+n₃-2: `kld_pr_cpd`
- if r < n₁: `kl_cpd`

If none of the above conditions are met, `cpd` throws an assertion error.
"""
function cpd(T::Array{d_type, 2}, dims::Tuple{Int64, Int64, Int64}, r::Int64) where d_type
    n₁, n₂, n₃ = dims
    if r < n₂ && r < n₃
        return kl_lr_cpd(T, dims, r)
    elseif r < n₂
        return kl_fd_cpd(T, dims, r; strategy="x_last_xy")
    elseif r < n₃
        return kl_fd_cpd(T, dims, r; strategy="y_last_yx")
    elseif r <= n₂ + n₃ - 2
        return kl_pr_cpd(T, dims, r)
    elseif r < n₁
        return kl_cpd(T, dims, r)
    else
        @assert false "Implemented algorithms not suited for this problem."
    end
end

end # module
