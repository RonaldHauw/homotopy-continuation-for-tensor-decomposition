
"""
# KuoLeeUtils

Utilities to run the klcpd in 'KuoLee.jl'.

## Contains

- rankfact: rank factorization of a matrix based on the SVD,
- constrpoly: constructs the target polynomial system,
- normbalance: balance the factor matrices in norm,
- filtsols: filters solutions to fulfill all equations.

"""
module KuoLeeUtils

# Imports
using DynamicPolynomials
using Random
using LinearAlgebra: norm, \, *, svd, rank, Diagonal, nullspace, dot, opnorm, qr, cond
using HomotopyContinuation

# Exports
export rankfact, constrpoly, normbalance, filtsols, solution_space
export filtsols_lowrank, constrpoly_lowrank


"""
# Rankfact

Factorize a matrix in rank one matrices via the singular value decomposition.
```math
    T = EF^T, \\quad E \\in \\mathbb{F}^{n_2n_3 \\times r}, \\quad F\\in \\mathbb{F}^{n_1\\times R}

```

```math
    T = U \\Sigma V^T, \\quad E = U, \\quad F^T = \\Sigma V^T.
```
## Inputs

- T: two-dimensonal array to be factorized
- r: number of rank one terms

## Output

- E: E and F are the factor matrices of the factorization
- F: E anf F are the factor matrices of the factorization
"""
function rankfact(T::Array{<:Number, 2}, r::Int64)

    T_svd = svd(T)

    Efull = T_svd.U
    Ftransfull = Diagonal(T_svd.S) * T_svd.V'

    Ffull = Ftransfull'

    E = Efull[:, 1:r]
    F = Ffull[:, 1:r]

    return E, F
end



"""
# constrpoly

Construct the polynomial system for the algorithm `kl_cpd`.

## Inputs

- `N`: space to which orthogonal rank-one vectors are sought
- `dims`: tensor dimensions
- `vars`: polynomial variables

## Outputs

- `𝐩`: array of polynomial equations

"""
function constrpoly(
        N::Array{d_type, 2},
        dims::Tuple{Int64, Int64, Int64},
        r::Int64,
        vars::Tuple{Array{v_type,1},Array{v_type,1}} where v_type
    ) where d_type

    # unpack
    n₁, n₂, n₃ = dims
    𝐲, 𝐳 = vars

    # initialise
    𝐩 = zeros(Polynomial{true, d_type}, n₂+n₃)

    # random selections of columns from N
    Q, _ = qr(randn(d_type, n₂*n₃-r, n₂*n₃-r));

    # orthogonality conditions
    for j ∈ 1:n₂+n₃-2
        𝐩[j] = dot(kron(𝐳, 𝐲), N*Q[:, j])
    end

    # dehomogenisation
    𝐜₁ = randn(d_type, n₂)
    𝐜₂ = randn(d_type, n₃)

    β₁ = 5.0
    β₂ = 5.0

    𝐩[n₂+n₃-1] = 𝐜₁'𝐲 - β₁
    𝐩[n₂+n₃] = 𝐜₂'𝐳 - β₂

    return 𝐩
end



function constrpoly_lowrank(
        Ŷ::Array{d_type, 2},
        Ẑ::Array{d_type, 2},
        N::Array{d_type, 2},
        dims::Tuple{Int64, Int64, Int64},
        r::Int64,
        vars::Tuple{Array{v_type,1},Array{v_type,1}} where v_type
    ) where d_type

    # unpack
    n₁, n₂, n₃ = dims
    𝐪₁, 𝐪₂ = vars

    nb_vars = length(𝐪₁) + length(𝐪₂)

    # initialise
    𝐩 = zeros(Polynomial{true, d_type}, nb_vars)

    # random selections of columns from N
    Q, _ = qr(randn(d_type, n₂*n₃-r, n₂*n₃-r));

    # orthogonality conditions
    for j ∈ 1:nb_vars-2
        𝐩[j] = dot(kron(Ẑ*𝐪₂, Ŷ*𝐪₁), N*Q[:, j])
    end

    # dehomogenisation
    𝐜₁ = randn(d_type, length(𝐪₁))
    𝐜₂ = randn(d_type, length(𝐪₂))

    β₁ = 5.0
    β₂ = 5.0

    𝐩[nb_vars-1] = 𝐜₁'𝐪₁ - β₁
    𝐩[nb_vars] = 𝐜₂'𝐪₂ - β₂

    return 𝐩
end




function solution_space(E::Array{d_type, 2}, dims, r) where d_type

    n₁, n₂, n₃ = dims

    Q, _ = qr(randn(d_type, r, r));
    E_mix = E*Q;

    e_col_1 = E_mix[:, 1];
    e_col_1_mat = reshape(e_col_1, n₂, n₃)
    e_col_1_mat_svd = svd(e_col_1_mat)

    Ŷ = e_col_1_mat_svd.U[:, 1:r]
    Ẑ = e_col_1_mat_svd.V[:, 1:r]
    Ẑ = conj(Ẑ)

    return Ŷ, Ẑ

end


"""
# filtsols

Filter solutions which are orthogonal to every vector in N.

## Inputs

- 𝛇: array of solution vectors
- N: basis of space to which orthogonal vectors are sought
- dims: dimensions of the tensor
- r: rank of the tensor

## Outputs

- Y_filt: array of vectors 𝐲ᵢ (1≦i≦r) of length r such that residual of (𝐳ᵢ ⊙ 𝐲ᵢ)ᴴ𝐧ⱼ is smallest.
- Z_filt: array of vectors 𝐳ᵢ (1≦i≦r) of length r such that residual of (𝐳ᵢ ⊙ 𝐲ᵢ)ᴴ𝐧ⱼ is smallest.
"""
function filtsols(
    𝛇::Array{Array{d_type, 1}, 1},
    N::Array{d_type, 2},
    dims::Tuple{Int64, Int64, Int64},
    r::Int64,
    ) where d_type

    n₁, n₂, n₃ = dims
    Y = [];
    Z = [];
    minresidues = []
    for 𝜁 ∈ 𝛇
        𝐲 = 𝜁[1:n₂];
        𝐳 = 𝜁[n₂+1:end];
        𝐲 = conj(𝐲);
        𝐳 = conj(𝐳);
        push!(Y, 𝐲);
        push!(Z, 𝐳)
        residues = []
        for j ∈ 1:size(N)[2]
            𝐧 = N[:, j];
            residue = abs(dot(kron(𝐳, 𝐲), 𝐧))/(norm(kron(𝐳, 𝐲))*norm(𝐧))
            push!(residues, residue)
        end
        push!(minresidues, maximum(residues))
    end
    𝔍 = sortperm(minresidues)
    Y_filt = Y[𝔍[1: r]];
    Z_filt = Z[𝔍[1: r]]
    @assert length(Y_filt) == length(Z_filt)
    return Y_filt, Z_filt
end



function filtsols_lowrank(
    𝛇::Array{Array{d_type, 1}, 1},
    Ŷ::Array{d_type, 2},
    Ẑ::Array{d_type, 2},
    N::Array{d_type, 2},
    dims::Tuple{Int64, Int64, Int64},
    r::Int64,
    ) where d_type

    n₁, n₂, n₃ = dims
    Y = [];
    Z = [];
    minresidues = []
    for 𝜁 ∈ 𝛇
        𝐪₁ = 𝜁[1:r];
        𝐪₂ = 𝜁[r+1:end];
        𝐲 = Ŷ*conj(𝐪₁)
        𝐳 = Ẑ*conj(𝐪₂)
        𝐲 = 𝐲
        𝐳 = 𝐳
        push!(Y, 𝐲);
        push!(Z, 𝐳);
        residues = []
        for j ∈ 1:size(N)[2]
            𝐧 = N[:, j];
            residue = abs(dot(kron(𝐳, 𝐲), 𝐧))/(norm(kron(𝐳, 𝐲))*norm(𝐧))
            push!(residues, residue)
        end
        push!(minresidues, maximum(residues))
    end
    𝔍 = sortperm(minresidues)
    Y_filt = Y[𝔍[1: r]];
    Z_filt = Z[𝔍[1: r]]
    @assert length(Y_filt) == length(Z_filt)
    return Y_filt, Z_filt
end




"""
# Normbalance

Distribute the norm equally over all factor vectors. Put all angle in X.

## Inputs

- X
- Y
- Z

## Outputs

- X: factor matrix X s.t. ||𝐱ᵢ|| = ||𝐲ᵢ|| = ||𝐳ᵢ||  ∀ i,
- Y: factor matrix Y s.t. ||𝐱ᵢ|| = ||𝐲ᵢ|| = ||𝐳ᵢ|| ∠yᵢ₁ = 0. ∀ i,
- Z: factor matrix Z s.t. ||𝐱ᵢ|| = ||𝐲ᵢ|| = ||𝐳ᵢ|| ∠zᵢ₁ = 0. ∀ i
"""
function normbalance(X::Array{<:Number, 2}, Y::Array{<:Number, 2}, Z::Array{<:Number, 2})
    r = size(X)[2]
    for i ∈ 1:r
        nx = norm(X[:, i])
        ny = norm(Y[:, i])
        nz = norm(Z[:, i])
        sy = sign(Y[1, i])
        sz = sign(Z[1, i])
        α = nx*ny*nz
        α₃ = α^(1.0/3)
        X[:, i] = X[:, i] / nx * α₃ .* conj(sy) .* conj(sz)
        Y[:, i] = Y[:, i] / ny * α₃ ./ sy
        Z[:, i] = Z[:, i] / nz * α₃ ./ sz
    end
    return X, Y, Z
end
function normbalance(X::Array{<:Number, 2}, Y::Array{<:Number, 2})
    r = size(X)[2]
    for i ∈ 1:r
        nx = norm(X[:, i])
        ny = norm(Y[:, i])
        α = nx*ny
        α₃ = α^(1.0/2)
        X[:, i] = X[:, i] / nx * α₃
        Y[:, i] = Y[:, i] / ny * α₃
    end
    return X, Y
end






end
