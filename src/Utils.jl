module Utils
using LinearAlgebra
include("KuoLeeUtils.jl")
using .KuoLeeUtils

export cpd_gen, column_align, signnorm, condtdp

"""
# `cpd_gen(dims, r)`

Generate a flattened tensor with dimensions n₂n₃×n₁ of rank r. The tensor is
generated from random factor matrices as

```math

    T = (Z \\odot Y)X^H.

```

## Inputs

- `dims`: tuple with the dimensions `(n₁, n₂, n₃)`.
- `r`: the desired tensor rank

## Output

- `X`: factor matrix of dimension n₁×r
- `Y`: factor matrix of dimension n₂×r
- `Z`: factor matrix of dimension n₃×r
- `T`: flattened tensor n₂n₃×n₁

"""
function cpd_gen(dims::Tuple{Int64, Int64, Int64}, r::Int64; d_type=ComplexF64)
    n₁, n₂, n₃ = dims;

    X = randn(ComplexF64, n₁, r);
    Y = randn(ComplexF64, n₂, r);
    Z = randn(ComplexF64, n₃, r);

    X, Y, Z = normbalance(X, Y, Z)

    T = reduce(hcat, [kron(Z[:, i], Y[:, i]) for i ∈ 1:r])*X'
    return (X, Y, Z, T)
end

"""
# `cpd_gen(X, Y, Z)`

Generate a flattened tensor from the given factor matrices such that
```math

    T = (Z \\odot Y)X^H.

```
"""
function cpd_gen(X::Array{d_type, 2}, Y::Array{d_type, 2}, Z::Array{d_type, 2}) where d_type
    n₁ = size(X)[1];
    n₂ = size(Y)[1];
    n₃ = size(Z)[1];
    dims = (n₁, n₂, n₃);

    @assert size(X)[2] == size(Y)[2] "Factor matrices don't have the same number of columns."
    @assert size(Y)[2] == size(Z)[2] "Factor matrices don't have the same number of columns."

    r = size(X)[2];
    @assert r <= maximum(dims) "Rank not smaller than maximum dimension."
    @assert r <= n₁ "Rank not smaller than first dimension. Consider changing the dimensions."

    T = reduce(hcat, [kron(Z[:, i], Y[:, i]) for i ∈ 1:r])*X'
    return T
end



function perturbation_matrix(A, Ã; soft=false)
    P = abs.(A \ Ã);
    out = zeros(size(P))
    for col ∈ 1:size(P)[2];
        out[argmax(P[:, col]), col] = 1.
    end
    @assert rank(out) == size(P)[1] || soft "Perturbation matrix not of full rank"
    return out
end


function column_align((X, Y, Z), (X̃, Ỹ, Z̃); soft=false)

    Py = perturbation_matrix(Y, Ỹ; soft=soft);
    Pz = perturbation_matrix(Z, Z̃; soft=soft);

    X = X*Py;
    Y = Y*Py;
    Z = Z*Pz;

    return X, Y, Z
end


"""
# signnorm

Rescale matrix columns such that the first row has positive sign. In the case
of complex entries, the elements on the first row have zero complex angle.

## Examples

```julia
    A
    # 2×2 Array{Complex{Float64},2}:
    #    1.30736-1.34255im    0.183123-0.504201im
    #    -0.00153446+0.0305201im  0.222609-0.0412371im
    signnorm(A)
    # 2×2 Array{Complex{Float64},2}:
    #    1.87393+1.5908e-16im  0.536426-2.60882e-17im
    #    -0.0229362+0.0201932im   0.114753+0.195159im
```

"""
function signnorm(A::Array{<:Number})
    A = reduce(hcat, [A[:, i]/sign(A[1, i]) for i ∈ 1:size(A)[2]])
end


"""
# `condtdp`

Calculate the condition number of the tensor decomposition problem based
on Terracini's matrix.

## Inputs

- `X`
- `Y`
- `Z`
- `dims`
- `R`

"""
function condtdp(X, Y, Z, dims, R)
    Σ = sum(di-1 for di ∈ dims);
    N = R*(Σ +1);
    Π = prod(dims);
    M = R*Π;
    Tlist = []
    for i ∈ 1:R
        a¹ = X[:, i]/norm(X[:, i])
        a² = Y[:, i]/norm(Y[:, i])
        a³ = Z[:, i]/norm(Z[:, i])
        I¹ = eye(length(a¹))
        I² = eye(length(a²))
        I³ = eye(length(a³))
        Tᵢ = [kron(I¹, a², a³) kron(a¹, I², a³) kron(a¹, a², I³) ]
        push!(Tlist, Tᵢ)
    end
    Tp = reduce(hcat, Tlist);
    Tpsvd = svd(Tp);
    return 1.0/Tpsvd.S[N]
end

function eye(n::Int64)
    return Diagonal(ones(n))
end


function ⊙(A::Array{d_type, 2}, B::Array{d_type, 2}) where d_type
    @assert size(A, 2) == size(B, 2)
    return reduce(hcat, [kron(A[:, i], B[:, i]) for i ∈ 1:size(A, 2)])
end

end #Utils
