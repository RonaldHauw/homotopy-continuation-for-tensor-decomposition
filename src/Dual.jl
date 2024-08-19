module Dual
using DynamicPolynomials
using Random
using LinearAlgebra
using HomotopyContinuation
include("DualUtils.jl")
include("KuoLeeUtils.jl")
using .KuoLeeUtils
using .DualUtils

export kld_fd_cpd, kld_pr_cpd

"""
# `kld_fd_cpd`

Calculate the cpd of a flattened order 3 tensor via the dual problem. The algoritm numerically
reduces the size of the square system used for homotopy continuation by setting some variables to a random value.
This method solves the dual problem by calling `dual_solve` and than translates the
dual problem to the primal via `translate_dual_prim`.
The documentation of these methods elaborates more on the techniques used.

## Inputs

- `T`: flattened real or complex tensor
- `r`: rank of the cpd
- `dims`: tuple of the dimensions, first dimension should be larger than r.

## Outputs
The method outputs the calculated factor matrices
```math
    T \\approx (Z_{\\text{calc}} \\odot Y_{\\text{calc}})X_{\\text{calc}}^H
```

- `Xcalc`
- `Ycalc`
- `Zcalc`

## Optional arguments
- `maxit_basis`: (`Int64`) maximum number of iterations used in `solve_dual`.
- `maxit_ortho`: (`Int64`) maximum number of iterations used in `translate_dual_prim`.
- `verbose`: (`Bool`) give some information about the number of iterations / the type of problems solved.
- `strategy`: (`String`) tell `solve_dual` which methods it should use to grow the basis. See `solve_dual` for the possible values.

## Example
```julia
dims = (43, 8, 7);
r = 6;
X, Y, Z, T = cpd_gen(dims, r);

Xcalc, Ycalc, Zcalc = kld_fd_cpd(T, r, dims);

T̃ = cpd_gen(Xcalc, Ycalc, Zcalc);
ϵ = norm(T-T̃)/norm(T);
```

"""
function kld_fd_cpd(
    T::Array{d_type, 2},
    r::Int64,
    dims::Tuple{Int64, Int64, Int64};
    verbose=false,
    dual_solve_maxit =sum(dims)*4,
    translate_dual_prim_maxit = prod(dims)*4,
    measure_condition = false,
    strategy="optimal",
    relative_orthogonality_check = 1e-4,
    tranlsate_dual_prim_sigma_threshold = 1e-3,
    dual_solve_filtsols_independent_threshold = 1e-5,
    dual_solve_filtsols_residual_threshold = 1e-10,
    dual_solve_filtsols_triviality_threshold = 1e-4,
    nonzero_threshold = 1e-10,
    ) where d_type

    # to be extended for higher orders
    @assert length(dims) == 3 "This function only works for tensors of order 3."

    # unpack
    n₁, n₂, n₃ = dims;

    # check if Kuo Lee applies for this tensor
    if n₁ != maximum(dims)
        println("Warning: The first dimension is not the largest dimension. Consider swapping the dimensions.")
    end
    @assert r <= n₁  "The largest dimension is smaller than the rank. Decrease the rank or increase the largest dimension."
    @assert r <= n₂*n₃ "The product of the second and third dimension is larger than the rank."
    @assert r <= n₂+n₃-2 "This tensor cannot be decomposed with the dual algorithm since r > n₂+n₃-2."

    # full rank factorization
    E, F, N = rankfact_nullspace(T, r);


    # calculate the basis U ⊙ V for N
    if measure_condition
        N_r1, UV, lin_cond, poly_cond = dual_solve(E, dims, r;
            maxit = dual_solve_maxit,
            verbose = verbose,
            strategy = "last_xy",
            full=false,
            measure_condition=true
        );
        success = 1;
    else
        success, N_r1, UV = dual_solve(E, dims, r;
            maxit = dual_solve_maxit,
            verbose = verbose,
            strategy =strategy,
            full=false,
            filtsols_independent_threshold = dual_solve_filtsols_independent_threshold,
            filtsols_residual_threshold = dual_solve_filtsols_residual_threshold,
            filtsols_triviality_threshold = dual_solve_filtsols_triviality_threshold,
            nonzero_threshold = nonzero_threshold,
        );
    end

    @assert success == 1 "Failed to find a full basis U ⊙ V for N."

    # unpack the datastructure of UV to the matrices U and V
    V = reduce(hcat, [UV[i][2] for i ∈ 1:length(UV)])
    U = reduce(hcat, [UV[i][1] for i ∈ 1:length(UV)])

    N_ = reduce(hcat, [kron(U[:, i], V[:, i]) for i ∈ 1:n₂*n₃-r]);
    @assert maximum(abs.(E'*N_))/maximum(abs.(N_)) < relative_orthogonality_check "Calculated basis U ⊙ V not orthogonal to E."


    # provide output if asked
    if verbose; println("\n"); end

    # calculate a basis orthogonal to U ⊙ V
    success, 𝛇 = translate_dual_prim(V, U, dims, r;
        maxit = translate_dual_prim_maxit,
        sigma_threshold = tranlsate_dual_prim_sigma_threshold,
        nonzero_threshold=nonzero_threshold,
        verbose = verbose
    );

    @assert success == 1 "Failed to find a full basis Z ⊙ Y orthogonal to U ⊙ V."

    # unpack the datastructure 𝛇 to the matrices Ycalc and Zcalc
    Ycalc = reduce(hcat, [𝛇[i][1] for i ∈ 1:length(𝛇)]);
    Zcalc = reduce(hcat, [𝛇[i][2] for i ∈ 1:length(𝛇)]);

    # normbalance
    Ycalc, Zcalc = KuoLeeUtils.normbalance(Ycalc, Zcalc)

    # E_ = Z ⊙ Y
    E_ = reduce(hcat, [kron(Zcalc[:, i], Ycalc[:, i]) for i ∈ 1:r]);

    @assert maximum(abs.(E_'*N))/maximum(abs.(E_)) < relative_orthogonality_check "Calculated basis Z ⊙ Y not orthogonal to N."

    # calculate the rotation between the given base E and it's rank-one basis Z ⊙ Y
    W = E'*E_;

    if measure_condition
        w_cond = cond(W);
    end;

    # calculate X
    Xcalc = (W \ F')';
    Xcalc = Array{ComplexF64, 2}(Xcalc);

    # normbalance
    Xcalc, Ycalc, Zcalc = KuoLeeUtils.normbalance(Xcalc, Ycalc, Zcalc);

    # recompose tensor
    Tcalc = reduce(hcat, [kron(Zcalc[:, i], Ycalc[:, i]) for i ∈ 1:r])*Xcalc';

    bwerr = maximum(abs.(Tcalc-T))/maximum(abs.(T))
    @assert bwerr < relative_orthogonality_check "Relative backward error of $(bwerr) too high."

    if measure_condition
        return Xcalc, Ycalc, Zcalc, lin_cond, poly_cond, w_cond
    else
        return Xcalc, Ycalc, Zcalc
    end
end
# overload for different order of variables
function kld_fd_cpd(
    T::Array{d_type, 2},
    dims::Tuple{Int64, Int64, Int64},
    r::Int64;
    verbose=false,
    kwargs...
    ) where d_type
    return kld_fd_cpd(T, r, dims;
        kwargs...
    )
end


"""
# `kld_pr_cpd_refine`

Calculate the cpd of an unbalanced flattened order 3 tensor via the dual problem.
 The method solves the dual problem by paramter homotopy and requires an initial decomposition.
    See `kl_pr_cpd`for a wrapper which also constructs the initial decomposition.


## Inputs

- `T₀`: start flattened tensor
- `T₁`: target flattened tensor
- `Y₀`: factor matrix of start tensor
- `Z₀`: factor matrix of start tensor
- `dims`:  dimensions `(n₁, n₂, n₃)` of start and target tensor.
    `Y₀` should have dimensions n₂×r, `Z₀` should have dimensions n₃×r.
- `r`: rank of the tensor


## Outputs
The method outputs the calculated factor matrices

```math
    T \\approx (Z_{\\text{calc}} \\odot Y_{\\text{calc}})X_{\\text{calc}}^H
```

- `Xcalc`
- `Ycalc`
- `Zcalc`
"""
function kld_pr_cpd_refine(T₀, T₁, Y₀, Z₀, dims, r;
        translate_prim_dual_sigma_threshold = 1e-4,
        translate_dual_prim_sigma_threshold = 1e-6,
        translate_prim_dual_maxit = 5000,
        translate_dual_prim_maxit = 5000,
        independent_threshold = 1e-5,
        verbose = false,
        maxit_orth = 100,
        residual_check_threshold = 1e-4,
        maxit = 10,
        )


    n₁, n₂, n₃ = dims


    # full rank factorization
    E₁, F₁, N₁ = rankfact_nullspace(T₁, r);
    E₀, F₀, N₀ = rankfact_nullspace(T₀, r);

    # construct a basis U₀ ⊙ V₀ for N₀
    success, UV₀ = DualUtils.translate_prim_dual(Y₀, Z₀, dims, r;
        sigma_threshold=translate_prim_dual_sigma_threshold,
        maxit=translate_prim_dual_maxit
        )

    V₀ = reduce(hcat, [conj(UV₀[i][1]) for i ∈ 1:length(UV₀)])
    U₀ = reduce(hcat, [conj(UV₀[i][2]) for i ∈ 1:length(UV₀)])

    N₀_ = reduce(hcat, [kron(U₀[:, i], V₀[:, i]) for i ∈ 1:length(UV₀)]);


    @assert success == 1 "Failed to construct a basis U₀ ⊙ V₀ from Z₀ ⊙ Y₀."

    # loop condition
    no_base_found = true

    # loop counter
    it = 0

    # solution containers should be in outer scope to access them after the loop
    Ycalc = nothing
    Zcalc = nothing
    E_ = nothing
    UV₁ = Array{Array{ComplexF64, 1}}[]
    YZ₁ = nothing
    E₁_ = nothing

    lastrank_N₁_ = 0;

    while no_base_found


        success, UV = DualUtils.dual_track_single(UV₀, E₀, E₁, dims, r)

        @assert success == 1 "Path tracking failed."

        for 𝐮𝐯 ∈ UV
            push!(UV₁, 𝐮𝐯)
            V₁ = reduce(hcat, [conj(UV₁[i][1]) for i ∈ 1:length(UV₁)])
            U₁ = reduce(hcat, [conj(UV₁[i][2]) for i ∈ 1:length(UV₁)])
            N₁_ = reduce(hcat, [kron(U₁[:, i], V₁[:, i]) for i ∈ 1:length(UV₁)]);
            N₁_svd = svd(N₁_);
            if N₁_svd.S[end]/N₁_svd.S[1] < independent_threshold
                pop!(UV₁)
            end
        end

        V₁ = reduce(hcat, [conj(UV₁[i][1]) for i ∈ 1:length(UV₁)])
        U₁ = reduce(hcat, [conj(UV₁[i][2]) for i ∈ 1:length(UV₁)])
        N₁_ = reduce(hcat, [kron(U₁[:, i], V₁[:, i]) for i ∈ 1:length(UV₁)]);

        if verbose
            println("   - Rank current base of N: $(rank(N₁_))")
        end

        @assert maximum(abs.(N₁_'*E₁))/maximum(abs.(N₁_)) < residual_check_threshold "U₁ ⊙ V₁ not perpendicular to E₁."

        # note that this search may be unsucssesfull
        success, YZ₁ = DualUtils.translate_dual_prim(V₁, U₁, dims, r;
            soft=true,
            maxit = translate_dual_prim_maxit,
            sigma_threshold=translate_dual_prim_sigma_threshold,
            verbose = false
            );

        if verbose
                println("   - Rank corresponding base of E: $(length(YZ₁))")
        end
        if success == 1 || length(YZ₁) >= r # full base and not positive dimensional (success flag seems to be too strict: to adapt code)
            Ycalc = reduce(hcat, [YZ₁[i][1] for i ∈ 1:length(YZ₁)]);
            Zcalc = reduce(hcat, [YZ₁[i][2] for i ∈ 1:length(YZ₁)]);
            Ycalc, Zcalc = KuoLeeUtils.normbalance(Ycalc, Zcalc);
            E₁_ = reduce(hcat, [kron(Zcalc[:, i], Ycalc[:, i]) for i ∈ 1:r]);
            if maximum(abs.(E₁_'*N₁))/maximum(abs.(E₁_)) < residual_check_threshold # the base we have is the right one
                no_base_found = false
            end
        end

        it+=1

        @assert it < maxit "Reached maxit."
    end



    W = E₁'*E₁_;
    Xcalc = (W \ F₁')';
    Xcalc = Array{ComplexF64, 2}(Xcalc);
    Xcalc, Ycalc, Zcalc = KuoLeeUtils.normbalance(Xcalc, Ycalc, Zcalc);
    Tcalc = reduce(hcat, [kron(Zcalc[:, i], Ycalc[:, i]) for i ∈ 1:r])*Xcalc';

    @assert maximum(abs.(Tcalc-T₁))/maximum(abs.(T₁)) < residual_check_threshold "Relative backward error too high."

    return Xcalc, Ycalc, Zcalc
end

"""
# `kld_pr_cpd`

Calculate the cpd of an order 3 unbalanced flattened tensor. This method uses a parameter homotopy to solve
the dual problem. This function constructs an initial decomposition and then calls `kld_pr_cpd_refine`.

## Inputs
- `T₁`: flattened tensor
- `r`: rank of `T₁`
- `dims`: tuple with dimensions `(n₁, n₂, n₃)` of the tensor  such that
    `T₁` has dimensions n₂n₃×n₁

## Outputs
The method outputs the calculated factor matrices
```math
    T \\approx (Z_{\\text{calc}} \\odot Y_{\\text{calc}})X_{\\text{calc}}^H
```

- `Xcalc`
- `Ycalc`
- `Zcalc`

"""
function kld_pr_cpd(T₁::Array{d_type, 2}, r::Int64, dims::Tuple{Int64, Int64, Int64}; verbose=false, kwargs...) where d_type
    n₁, n₂, n₃ = dims
    X₀ = randn(d_type, n₁, r);
    Y₀ = randn(d_type, n₂, r);
    Z₀ = randn(d_type, n₃, r);
    T₀ = reduce(hcat, [kron(Z₀[:, i], Y₀[:, i]) for i ∈ 1:r])*X₀';
    return kld_pr_cpd_refine(T₀, T₁, Y₀, Z₀, dims, r; verbose=verbose, kwargs...)
end
# overload for different order of required arguments
function kld_pr_cpd(T₁::Array{d_type, 2}, dims::Tuple{Int64, Int64, Int64}, r::Int64; verbose=false, kwargs...) where d_type
    return kld_pr_cpd(T₁, r, dims; verbose=verbose, kwargs...)
end











end # Dual
