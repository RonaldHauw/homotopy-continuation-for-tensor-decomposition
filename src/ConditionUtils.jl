"""
# KuoLeeCondUtils

Utilities for calculating the condition number of the TDP posed as in
    Kuo Lee's paper

## Contains

- machzero: set zero elements to e-17
- signnorm: normalise vector such that first element has angle zero.
- normangle: norm of a vector with angle of first element
- mingap: smallest distance between element and other element in the array.
- dist: heuristic minimum vectorized L2 norm between two set of factor matrices
which are rescaled such that the rank 1 tensors remain the same.
- eye: construct a unit matrix.
- condtdp: calculate the condition number of the tensor decomposition problem
- acute: calculate the acute angles between two sets of vectors.
"""
module KuoLeeCondUtils



# Imports
using DynamicPolynomials
using Random
using Logging
using LinearAlgebra: norm, \, *, svd, rank, Diagonal, nullspace, dot, opnorm, qr, cond
using HomotopyContinuation
using DataFrames
using CSV
using Plots
using Documenter
include("3_Kuo_Lee_Utils.jl")


# Exports
export machzero, signnorm, normangle, mingap, dist, eye, condtdp, acute, rangematch
export domain

"""
# Machzero

## Method  machzero(::Float64)
Introduce an error of ~10^(-17), the machine precision of a Float64 type.
Main use is to set exactly zero elements to an order of 10^-17.

## Method  machzero(x::Array{<:Number})
Apply machzero(x<:Number) elementwise.

## Examples:

    >> machzero(Float64(0.)) # 2.77e-17
    >> machzero(Float64(1.)) # 1.0
"""
function machzero(x::T) where T<:Number
    if isapprox(x, 0.0; atol=eps(T), rtol=0)
        x = eps(T);
    end
    return x
end;
function machzero(x::Array{T}) where T
    x = machzero.(x)
    return x
end;

"""
# signnorm

Rescale matrix columns such that the first row has positive sign. In the case
of complex entries, the elements on the first row have zero complex angle.

## Examples

    >> A
    >> 2×2 Array{Complex{Float64},2}:
        1.30736-1.34255im    0.183123-0.504201im
        -0.00153446+0.0305201im  0.222609-0.0412371im
    >> signnorm(A)
    >> 2×2 Array{Complex{Float64},2}:
        1.87393+1.5908e-16im  0.536426-2.60882e-17im
        -0.0229362+0.0201932im   0.114753+0.195159im

"""
function signnorm(A::Array{<:Number})
    A = reduce(hcat, [A[:, i]/sign(A[1, i]) for i ∈ 1:size(A)[2]])
end;



"""
# colnorm

Normalises the column of a matrix to have L2 norm equal to one.
"""
function colnorm(X)
    for i ∈ 1:size(X)[2]
        X[:, i] /= norm(X[:, i]);
    end
    return X
end

"""
# normangle

Returns the a complex number with as absolute value the L2 norm of x and with
the angle of the first element in x.

## Examples

    >> x
    >> 4-element Array{Complex{Float64},1}:
        0.0 + 0.1im
        1.0 + 0.0im
        2.0 + 0.0im
        3.0 + 0.0im
    >> norms(x)
    >> 0.0 + 3.743im

"""
function normangle(x::Array{<:Number, 1})
    return norm(x)*sign(x[1])
end;

function norms(x)
    return normangle(x)
end


"""
# mingap

Returns the minimum gap between each element and another elemnt in an array.

## Examples

    >> x = [1, 0.1, 1.1, -0.1, 2, 5];
    >> mingap(x)
    >> 6-element Array{Float64,1}:
        0.10000000000000009
        0.2
        0.10000000000000009
        0.2
        0.8999999999999999
        3.0

"""
function mingap(eigs::Array{<:Number, 1})
    dist = ones(length(eigs), length(eigs)) * 2 * (maximum(abs.(eigs))-minimum(abs.(eigs)))
    out = zeros(length(eigs))
    for i ∈ 1:length(eigs)
        for j ∈ 1:length(eigs)
            if j != i
                dist[i, j] = abs(eigs[i]-eigs[j])
            end
        end
    end
    out = [minimum(dist[i, :]) for i ∈ 1:length(eigs)]
    return out
    end;

"""
# acute(U, Ũ)

Return array of acute angles between vectors in U and Ũ. Calculated in a
numerically stable way. The acute angle is defined as

``θ = cos⁻¹|uᵢᵀũᵢ|.``

## Examples

    >> X
    # 2×2 Array{Complex{Float64},2}:
    # 0.0+0.1im  2.0+0.0im
    # 0.0+0.2im  0.0+2.0im
    >> X̃
    # 2×2 Array{Complex{Float64},2}:
    # 0.1+0.0im  0.0+2.0im
    # 0.2+0.0im  2.0+0.0im
    >> acute(X, X̃)
    # 2-element Array{Float64,1}:
    # 1.5707963267948966
    # 1.5707963267948966

"""
function acute(U::Array{<:Number, 2}, Ũ::Array{<:Number, 2})
    # neem geen acos meer!
    c = [norm(U[:, i] - Ũ[:, i]) for i ∈ 1:size(U)[2]];
    a = [norm(U[:, i]) for i ∈ 1:size(U)[2]];
    b = [norm(Ũ[:, i]) for i ∈ 1:size(U)[2]];

    angles = Float64[]

    for i ∈ 1:size(U)[2]
        if b[i] >= c[i]
            mu = c[i]-(a[i]-b[i])
        elseif c[i] > b[i]
           mu = b[i]-(a[i]-c[i])
        else
            println("WARNING acute: no condition found.")
        end
        push!(angles, 2*atan(sqrt(  (  ( (a[i]-b[i])+c[i] )*mu  )/( (a[i]+(b[i]+c[i]) )*((a[i]-c[i])+b[i])  )  )))
    end

    return angles;
end;


"""
# dist

Heuristically calculate the minimal L2 distance of the vectorized factor matrices
such that they represent the same set of rank 1 tensors.

## Pseudo code

The algorithm exhibits two big steps:

A. Match the order of the factor matrices such that the corresponding rank 1
tensors are closest
B. Match the norms of the different factor matrices such that the norm
of the difference is expected to be low.

Given two sets of factor matrices (X, Y, Z) and (X̃, Ỹ, Z̃)

1. Build the vectorized rank 1 tensors and stack them in a matrices A and Ã
2. Match the vectorized rank 1 tensors by calculating an almost permutation
matrix P: AP̃ = Ã
3. Apply the exact permuation matrix P (derived from P̃) to the factor matrices
X̃, Ỹ, Z̃
4. For each column in the factor matrices match the two biggest norms. For
example, if norm(X[:, i])>norm(Y[:, i])>norm(Z[:, i]), match the norms of
X̃[:, i] and Ỹ[:, i] and adjust the norm of Z̃[:, i] such that the rank 1 tensor remains
the same. In addition set the angle of the first element to zero
5. Stack the norm matched factor matrices vertically.
6. Vectorize the stacked, norm matched factor matrices and calculate the L2 norm.

"""
function dist((X, Y, Z), (X̃, Ỹ, Z̃), dims, R)
    Π = prod(dims)

    A = reduce(hcat, [kron(X[:, i], Y[:, i], conj(Z[:, i])) for i ∈ 1:R]);
    Ã = reduce(hcat, [kron(X̃[:, i], Ỹ[:, i], conj(Z̃[:, i])) for i ∈ 1:R]);
    P = perturbation_matrix(Ã, A)
    #P̄ = Ã \ A;
    # P = zeros(Int8, size(P̄))
    # for i ∈ 1:R
    #     k = argmax([p.re for p ∈ P̄[:, i]])
    #     P[k, i] = 1
    # end
    @assert rank(P) == R "P not of full rank, rank(P) = $(rank(P))"
    X̃m, Ỹm, Z̃m = X̃*P, Ỹ*P, Z̃*P;
    plist = []
    p̂list = []
    for i ∈ 1:R
        x, x̃ = X[:, i], X̃m[:, i]
        y, ỹ = Y[:, i], Ỹm[:, i]
        z, z̃ = conj(Z[:, i]), conj(Z̃m[:, i])
        pi, p̃i = [x;y;z], [x̃;ỹ;z̃]
        λᵖ̃ = norms(x̃)*norms(ỹ)*norms(z̃)
        Λ = zeros(ComplexF64, 3)
        if norm(x) > norm(y) && norm(x) > norm(z)
            Λ[1] = norms(x)
            if norm(y) > norm(z)
                Λ[2] = norms(y)
                Λ[3] = λᵖ̃/(Λ[1]*Λ[2])
            else
                Λ[3] = norms(z)
                Λ[2] = λᵖ̃/(Λ[1]*Λ[3])
            end
        elseif norm(y) > norm(x) && norm(y) > norm(z)
            Λ[2] = norms(y)
            if norm(x) > norm(z)
                Λ[1] = norms(x)
                Λ[3] = λᵖ̃/(Λ[1]*Λ[2])
            else
                Λ[3] = norms(z)
                Λ[1] = λᵖ̃/(Λ[2]*Λ[3])
            end
        else
            Λ[3] = norms(z)
            if norm(x) > norm(y)
                Λ[1] = norms(x)
                Λ[2] = λᵖ̃/(Λ[1]*Λ[3])
            else
                Λ[2] = norms(y)
                Λ[1] = λᵖ̃/(Λ[2]*Λ[3])
            end
        end
        @assert abs(prod(Λ) - λᵖ̃) < 1e-10
        x̂ = x̃*Λ[1]/norms(x̃)
        ŷ = ỹ*Λ[2]/norms(ỹ)
        ẑ = z̃*Λ[3]/norms(z̃)
        @assert norm(kron(x̂, ŷ, ẑ) - kron(x̃, ỹ, z̃)) < 1e-15*Π
        p̂i = [x̂; ŷ; ẑ]
        push!(plist, pi)
        push!(p̂list, p̂i)
    end
    p = reduce(vcat, plist);
    p̂ = reduce(vcat, p̂list);
    return norm(p-p̂)
end;

"""
# eye(n)

Construct a unit matrix of dimension n × n. Introduced since I from the
LinearAlgebra package is overwritten.
"""
function eye(n)
    id = zeros(n, n)
    for i ∈ 1:n
        id[i, i] = 1
    end
    return id
end;

"""
# condtdp

Calculate the condition number of the tensor decomposition problem based
on Terracini's matrix.

TODO finish documentation with
- Reference to paper
- Specifics regarding in which norms the condition number is measured.
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
    end;

"""
# rangematch(U, Ũ)

Match every vector in U with ũ such that
ũ = argmin ||ũ-u||_2, ũ ∈ Range(Ũ). All ũ are forced to have L2 norm 1.
"""
function rangematch(U::Array{T, 2}, Ũ::Array{T, 2}) where T
    Ũmatch = zeros(T, size(U));
    for i ∈ 1:size(U)[2]
        u = U[:, i]; # u ∈ Range(U)
        x = Ũ \ u; # x = argmin ||Ũx-u||_2
        ũ = Ũ*x; # ũ = argmin ||ũ-u||_2, ũ ∈ Range(Ũ)
        for j ∈ 1:i-1 # uncomment to orthoganalise as well
            ũ_ = Ũmatch[:, j]
            ũ = ũ - dot(ũ, ũ_) .* ũ_
        end
        ũ = ũ/norm(ũ) # ||ũ||_2 = 1
        Ũmatch[:, i] = ũ
    end
    return Ũmatch
end

"""
# domain(x, l, h)

Bound x to have minimum value l and maximum value h. If x < l return l, if
x > h return h else return x.
"""
function domain(x::Number, l::Union{Float64, String}, h::Union{Float64, String}; warning::Bool = false, kwargs...)
    out = x
    if l != "-∞" && x < l
        out = l
        if warning
            println("WARNING: domain, lower value $(l) exeeded with $(x-l)")
        end
    end
    if h != "∞" && x > h
        out = h
        if warning
            println("WARNING: domain, upper value $(h) exeeded with $(x-h)")
        end
    end
    return out
end
function domain(x::Array{<:Number}, l::Union{Float64, String}, h::Union{Float64, String}; warning::Bool = false, kwargs...)
    for i ∈ 1:length(x)
        x[i] = domain(x[i], l, h; warning=warning)
    end
    return x;
end


"""
# canonical(U, Ũ)

Calculates the canonical angles between Range(U) and Range(Ũ).
"""
function canonical(u, ũ)
    svduũ = svd(u'*ũ);
    return acos.(svduũ.S)
end

"""
# sincanonical(U, Ũ)

Calculates the sine of the canonical angles between Range(U) and Range(Ũ)
"""
function sincanonical(u::Array{T, 2}, ũ::Array{T, 2}) where T<:Number
    svduũ = svd(u'*ũ);
    return sqrt.(T(1.0) .- svduũ.S .^ 2)
end

"""
# checkswitch(dims, R)

Prints if factor matrices can be switched in Kuo Lee's algorithm.
"""
function checkswitch(dims, R; verbose = false)
    I, J, K = dims;
    if R <= I*K && R <= J
        println("Last dimension J = $J possible for R = $R: ($I, $K, $J) = (I, K, J) or ($K, $I, $J) = (K, I, J)")
    elseif verbose
        println("Last dimension J = $J impossible for R = $R and other dimensions ($I, $K, $J) = (I, K, J) because:")
        if  R>I*K
            println("R > IK, $R > $(I*K)")
        end;
        if  R > J
            println("R > J, $R > $J")
        end
    end
    if R <= I*J && R <= K
        println("Last dimension K = $K possible for R = $R: ($I, $J, $K) = (I, J, K) or ($J, $I, $K) = (J, I, K)")
    elseif verbose
        println("Last dimension K = $K impossible for R = $R and other dimensions ($I, $J, $K) = (I, J, K) because:")
        if  R>I*J
            println("R > IJ, $R > $(I*J)")
        end;
        if  R > K
            println("R > K, $R > $K")
        end
    end
    if R <= J*K && R <= I
        println("Last dimension I = $I possible for R = $R: ($J, $K, $I) = (J, K, I) or ($K, $J, $I) = (K, J, I)")

    elseif verbose
        println("Last dimension I = $I impossible for R = $R and other dimensions ($J, $K, $I) = (J, K, I) because:")

        if  R>I*J
            println("R > JK, $R > $(K*J)")
        end;
        if  R > I
            println("R > I, $R > $I")
        end
    end
end





"""
    init_tensor

Initialises a tensor for experimentation purposes.
"""
function init_tensor(dims₀, R, γ, η, setting = ("random", "none", "false"), verbose = true)

    init_type, pert_type, switch = setting
    I₀, J₀, K₀ = dims₀;

    # set init type
    if init_type == "identity"
        X₀, Y₀, Z₀ = zeros(ComplexF64, (I₀, R)), zeros(ComplexF64, (J₀, R)), zeros(ComplexF64, (K₀, R));
        for i ∈ 1:R
            Z₀[i, i] = 1.0;
            X₀[mod(i-1, I₀)+1, i] = 1.0;
            Y₀[mod(i-1, J₀)+1, i] = 1.0;
        end
    elseif init_type == "orthogonal"
        X₀, Y₀, Z₀ = zeros(ComplexF64, (I₀, R)), zeros(ComplexF64, (J₀, R)), zeros(ComplexF64, (K₀, R));
        @assert R <= I₀ "Set lower rank"; @assert R <= J₀ "Set lower rank";
        QI, _ = qr(randn(I₀, I₀)); QJ, _ = qr(randn(J₀, J₀));
        for i ∈ 1:R
            Z₀[i, i] = 1.0;
            X₀[:, i] = QI[:, i];
            Y₀[:, i] = QJ[:, i];
        end
    elseif init_type == "random"
        X₀, Y₀, Z₀ = randn(ComplexF64, (I₀, R)), randn(ComplexF64, (J₀, R)), randn(ComplexF64, (K₀, R));
        colnorm(X₀);
        colnorm(Y₀);
        colnorm(Z₀);
    else
        prinltn("init_type could not be understood: identity, orthogonal or random")
    end

    # set perturbation
    if pert_type == "copy"
        Y₀[:, end] = Y₀[:, end-1] + γ*Y₀[:, end];
    elseif pert_type == "linear_combination"
        Y₀[:, end] = Y₀[:, 1:end-1]*randn(size(Y₀)[2]-1) + γ*Y₀[:, end];
    elseif pert_type == "none"
        println("")
    else
        println("Perturbation type could not be understood: copy, linear_combination or none")
    end

    # set switch
    if switch == "true"
        I, J, K = I₀, K₀, J₀;
        dims = (I, J, K);
        X, Y, Z = X₀, Z₀, Y₀;
    elseif switch == "false"
        I, K, J = I₀, K₀, J₀;
        dims = (I, J, K);
        X, Z, Y = X₀, Z₀, Y₀;
    else
        println("Switch settign could not be understood: true or false")
    end

    if verbose
        T = reduce(hcat, [kron(Y[:, i], X[:, i]) for i ∈ 1:R])*Z';
        E = rand(I*J, K);
        E = η/norm(E) .* E;
        T̃ = T + E;
        svdT, svdT̃ = svd(T; full=true), svd(T̃; full=true);
        V, Ṽ, S, S̃, U, Ũ = svdT.V, svdT̃.V, svdT.S, svdT̃.S, svdT.U, svdT̃.U;
        rank1term(i) = kron(X[:, i], Y[:, i], Z[:, i]);
        #println("orthogonality check: ", dot(rank1term(1), rank1term(2)));
        println("rank check: ", rank(T)); println("singular values: ", S[1:R])
        println("condition number: $(Main.KuoLeeCondUtils.condtdp(X,Y,Z, dims, R))\n")
        Main.KuoLeeCondUtils.checkswitch(dims, R; verbose=true)
        println("\ncond(X) = $(cond(X))")
        println("cond(Y) = $(cond(Y))")
        println("cond(Z) = $(cond(Z))")
        Σ = sum(di-1 for di ∈ dims₀); N = R*(Σ +1); Π = prod(dims₀); M = R*Π;
        println("nb variables = $(Σ*R + R)      nb equations = $(Π)");
        plt = scatter();
        scatter!(plt, 1:length(S̃), Main.KuoLeeCondUtils.machzero(S̃), label = "T~");
        scatter!(plt, 1:length(S), Main.KuoLeeCondUtils.machzero(S), yscale = :log10, title = "Singuliere waarden", label = "T", xlabel = "i", ylabel = "sigma_i" );
        scatter!(plt, [R], [S[R]], label = "singuliere waarde R van T", color=:purple);
        display(plt);
    end

    return X, Y, Z, dims;

end

function weylbound(D, S, V, N, dims, R; E = nothing, cxcy = nothing, D̃S̃ṼÑ = nothing, verbose = true)

    I, J, K = dims

    cx, cy = nothing, nothing
    D̃, S̃, Ṽ, Ñ = nothing, nothing, nothing, nothing

    if E == nothing || cxcy == nothing || D̃S̃ṼÑ == nothing
        if verbose
            println("Nothing to compare with, cannot be verbose.")
            verbose = false
        end
        E = zeros(I*J, K)
        cx = zeros(I)
        cy = zeros(J)
        D̃  = D
        S̃  = S
        Ṽ  = V
        Ñ  = N
    else
        cx, cy = cxcy
        D̃, S̃, Ṽ, Ñ = D̃S̃ṼÑ
    end


    # left singular space
    push!(S, 0.0);
    δ = S[R]-S[R+1];
    sinΘ = Main.KuoLeeCondUtils.sincanonical(D, D̃);
    canonerrorU = norm(sinΘ);
    boundcanU = sqrt(2 * norm(vec(E))^2)/δ;
    condcanU = sqrt(2)/δ;
    if verbose;
        println("Left singular space, dependent on norm(E) and δ")
        println("   Gap between singular value R and R+1:                δ = $δ");
        println("   Bound on canonical angle of left singular space: bound = $(boundcanU)")
        println("   Canonical angle of left singular space:    ||sin Φ||_F = $(canonerrorU)")
        println("   Condition number for left singular space:            κ = $(condcanU)")
    end

    # nullspace of left singular space
    sinΦ = Main.KuoLeeCondUtils.sincanonical(N, Ñ);
    canonerrorN = norm(sinΦ);
    boundcanN = boundcanU
    condcanN = 1.
    if verbose;
        println("\nNullspace of left singular space, dependent on previous bound.")
        println("   Bound on the canonical angle of the nullspace:   bound = $(boundcanN)")
        println("   Canonical angle of nullspace:              ||sin Φ||_F = $(canonerrorN)")
        println("   Condition number for nullspace:                      κ = 1")
    end

    # L2 bound on N
    Ñmatch = Main.KuoLeeCondUtils.rangematch(N, Ñ)
    boundnormN = 2 * sin( asin(boundcanN) / 2 );
    boundnormN_1s = 2 * sin( asin(maximum(canonerrorN)) /2 );
    normerrorN = [norm(N[:, i]- Ñmatch[:, i]) for i ∈ 1:I*J-R];
    condnormN = 1.

    if verbose;
        println("\nVectors spanning nullspace, dependent on previous bound. ")
        println("   L2 norm bound on each vector:                    bound = $(boundnormN)")
        println("   One-step bound:                                  bound = $(boundnormN_1s)")
        println("   Maximum L2 norm error:                   max ||n-ñ||_2 = $(maximum(normerrorN))")
        println("   Condition number for vector norm nullspace:          κ = 1")
    end

    # Weyl bound on polynomial system
    binomcoeff = factorial(2)/(factorial(1)*factorial(1));

    boundweylfi = boundnormN ./ sqrt(binomcoeff);
    boundweylf = (I*J-R-2)*boundweylfi;

    boundweylfi_1s =  maximum(normerrorN) ./ sqrt(binomcoeff);
    boundweylf_1s = (I*J-R-2)*boundweylfi_1s;

    weylerrorfi = normerrorN ./ sqrt(binomcoeff);
    weylerrorf = sum(weylerrorfi);
    weylnormf = sqrt(sum(norm(N[:, i])^2 for i ∈ 1:I*J-R) ./ binomcoeff + norm(cx)^2 + norm(cy)^2);

    condnormf = (I*J-R-2) / sqrt(binomcoeff);

    if verbose;
        println("\nWeyl norm of polynomial system, dependent on previous bound. ")
        println("   Weyl norm bound on full system:                  bound = $(boundweylf)")
        println("   One-step bound:                                  bound = $(boundweylf_1s)")
        println("   Measured error:                              ||f-f̃||_W = $(weylerrorf)")
        println("   Condition number weyl norm:                          κ = $(condnormf)")

    end

    return boundweylf, condnormf*condnormN*condcanN*condcanU

end

function exact_solutions(P, X, Y, x, cx, cy, α; threshold = 1e-12)

    ζ = Array{ComplexF64, 1}[]
    for i ∈ 1:size(X)[2]
        sol = conj([ α .* X[:, i] ./ dot(conj(cx), X[:, i]); α .* Y[:, i] ./ dot(conj(cy), Y[:, i])]);
        push!(ζ, sol);
    end

    @assert maximum(abs.(P[i](x=>ζ[j])) for i ∈ 1:length(P) for j ∈ 1:length(ζ)) / maximum(norm.(ζ)) < threshold

    return conj.(ζ)

end

function exact_solutions_reduced(P, X, Y, x_f, y_f, x, y; threshold = 1e-12)

    ζ = Array{ComplexF64, 1}[]
    x_set = x_f[1];
    y_set = y_f[1];

    for i ∈ 1:size(X)[2]
        x_sol = X[:, i] ./ X[1, i] .* x_set;
        y_sol = Y[:, i] ./ Y[1, i] .* y_set;
        sol = Array{ComplexF64,1}([x_sol[2:end]; y_sol[2:end]]);
        push!(ζ, conj(sol));
    end

    @assert maximum(abs.(P[i]([x; y]=>ζ[j])) for i ∈ 1:length(P) for j ∈ 1:length(ζ)) / maximum(norm.(ζ)) < threshold

    return conj.(ζ)

end

function calculate_solutions(P, dims, R, x, y,  N)

    res = HomotopyContinuation.solve(P; variable_groups=[x, y]);
    sols = solutions(res);
    sols = HomotopyContinuation.unique_points(sols)

    # check solution
    residualsum = abs(sum([sum([p([x; y]=>sols[i]) for p in P]) for i ∈ 1:length(sols)]))
    @assert residualsum < 1e-4 "Polynomial system ill-solved, residual = $(residualsum)"

    solsx, solsy = KuoLeeUtils.filtsols(sols, N, dims, R)

    ζ = [ [solsx[i]; solsy[i]] for i ∈ 1:R ];

    return ζ

end

function polynomial_condition(P, ζ, x; verbose = true)

    # degrees
    degrees = Int64[]
    for p ∈ P; push!(degrees, maxdegree(p.x)); end;

    ## differential
    dP = differentiate(P, x);

    #dpζ = [[dPi(x=> ζ[i]/norm(ζ[i])) for dPi ∈ dP] for i ∈ 1:length(ζ)];
    dpζ = [[dPi(x=>conj(ζ[i])) for dPi ∈ dP] for i ∈ 1:length(ζ)];
    D = [Diagonal([sqrt(d)*norm(ζ[i])^(d-1) for d ∈ degrees]) for i ∈ 1:length(ζ)];

    # POLYNOMIAL CONDITION NUMBER
    κ = abs.([norm( dpζ[i] \ D[i] ) for i ∈ 1:length(ζ)]) ;

    if verbose;
        display([["Inverse of Jacobian"; norm.(inv.(dpζ))] ["Condition"; κ]])
    end;

    return κ
end

function process_solutions(ζ, D, F, R, dims; ζ̃D̃F̃ = nothing, verbose = true, only_condition = true)
    I, J, K = dims;

    Xcalc  = reduce(hcat, [  ζ[i][1:I]                      for i ∈ 1:R]);
    Ycalc  = reduce(hcat, [  ζ[i][I+1:end]                  for i ∈ 1:R]);
    YXcalc = reduce(hcat, [  kron(Ycalc[:, i], Xcalc[:, i]) for i ∈ 1:R]);
    W      = D'*YXcalc;
    Zcalc  = (W \ F')'
    Zcalc  = Array{ComplexF64, 2}(Zcalc);
    Xcalc, Ycalc, Zcalc = KuoLeeUtils.normbalance(Xcalc, Ycalc, Zcalc);
    if verbose
        println("κʷ = $(cond(W))")
    end
    if ζ̃D̃F̃ != nothing
        ζ̃, D̃, F̃ = ζ̃D̃F̃;
        X̃calc  = reduce(hcat, [  ζ̃[i][1:I]                      for i ∈ 1:R]);
        Ỹcalc  = reduce(hcat, [  ζ̃[i][I+1:end]                  for i ∈ 1:R]);
        ỸX̃calc = reduce(hcat, [  kron(Ỹcalc[:, i], X̃calc[:, i]) for i ∈ 1:R]);
        W̃      = D̃'*ỸX̃calc;
        Z̃calc  = (W̃ \ F̃')'
        Z̃calc  = Array{ComplexF64, 2}(Z̃calc);
        X̃calc, Ỹcalc, Z̃calc = KuoLeeUtils.normbalance(X̃calc, Ỹcalc, Z̃calc);
        if verbose
            P = perturbation_matrix(Xcalc, X̃calc)
            Xcalc = Xcalc*P;
            Ycalc = Ycalc*P;
            Zcalc = Zcalc*P;
            YXcalc = YXcalc*P;
            println("- max ||xᵢ-x̃ᵢ||_2/||xᵢ||_2             = $(maximum([norm(Xcalc[:, i]-X̃calc[:, i])/norm(Xcalc[:, i]) for i ∈ 1:R]))")
            println("- max ||yᵢ-ỹᵢ||_2/||yᵢ||_2             = $(maximum([norm(Ycalc[:, i]-Ỹcalc[:, i])/norm(Ycalc[:, i]) for i ∈ 1:R]))")
            println("- max ||zᵢ-z̃ᵢ||_2/||zᵢ||_2             = $(maximum([norm(Zcalc[:, i]-Z̃calc[:, i])/norm(Zcalc[:, i]) for i ∈ 1:R]))")
            println("- max ||(y⊙x)ᵢ-(ỹ⊙x̃)ᵢ||_2/||(y⊙x)ᵢ||_2 = $(maximum([norm(YXcalc[:, i]-ỸX̃calc[:, i])/norm(YXcalc[:, i]) for i ∈ 1:R]))")

        end
    end

    if only_condition
        return cond(W);
    else;
        return Xcalc, Ycalc, Zcalc;
    end;
end

# function perturbation_matrix(A, Ã; threshold = 1e-3)
#     P = A \ Ã;
#     P = abs.(P);
#     P = (abs.(P .- 1.) .<  threshold) .* ones(size(P))
#     @assert rank(P) == size(P)[1] "P not of full rank, are the inputs norm balanced?"
#     return P
# end


function perturbation_matrix(A, Ã; threshold = 1e-3)
    P = abs.(A \ Ã);
    out = zeros(size(P))
    for col ∈ 1:size(P)[2];
        out[argmax(P[:, col]), col] = 1.
    end
    #display(out)
    @assert rank(out) == size(P)[1] "P not of full rank, are the inputs norm balanced? $P,\n\n\n $out \n\n\n"
    return out
end


function column_align((X, Y, Z), (X̃, Ỹ, Z̃))
    Px = perturbation_matrix(X, X̃);
    Py = perturbation_matrix(Y, Ỹ);
    #Pz = perturbation_matrix(Z, Z̃);
    X = X*Px;
    Y = Y*Py;
    Z = Z*Px;

    return X, Y, Z


end

end
