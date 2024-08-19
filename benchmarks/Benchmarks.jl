#!/Applications/JuliaPro-1.1.1.1.app/Contents/Resources/julia/Contents/Resources/julia/bin/julia
#push!(LOAD_PATH,"../src/")

"""
# Benchmarks

Module which contains scripts to test the performance in both time and accuracy of
HomotopyContinuationForTensorDecomposition compared to Tensorlab2016.
"""
module Benchmarks

using HomotopyContinuationForTensorDecomposition
import HomotopyContinuationForTensorDecomposition.cpd_gen
import Base.display
using Random
using LinearAlgebra
using DataFrames
using CSV
using TensorToolbox
using ProgressMeter
using HomotopyContinuation

using MATLAB

export run_benchmark, Benchmarkinstance

function rand_string(len = 6)
    chars = collect("abcdefghijklmnopqrstuvwxyzABSCEFGHIJKLMNOPQRSTUVWXYZ1234567890")
    return String(chars[rand(1:length(chars), len)])
end

function Input(prompt)
    print(prompt)
    return readline()
end


"""
# `Benchmarkinstance`

Struct which represents an experiment to perform.

## Contains

- `n₁`: first dimension of tensor.
- `n₂`: second dimension of tensor.
- `n₃`: third dimension of tensor.
- `r`: rank of the tensor.
- `η`: parameter to controll noise. The noise is applied to the tensor to approximate the empirical condition number.
- `model`: the model used to construct the tensor. The models are described in [3]
- `s`: parameter used in model 1 and model 2
- `c`: parameter used in model 1
- `sample_fm`: can only be false if the tensor format is perfect. In that case a random tensor is generated directly. If true the tensor is generated form random factor matrices.


[3] _Breiding and Vannieuwenhoven, A Riemannian trust region method for the canonical tensor rank approximation problem_
"""
struct Benchmarkinstance
    n₁::Int64
    n₂::Int64
    n₃::Int64
    r::Int64
    γ::Float64
    s::Float64
    c::Float64
    η::Float64
    model::Int
    sample_fm::Bool

    # construct with ill_condtioning
    function Benchmarkinstance(n₁::Int64, n₂::Int64, n₃::Int64, r::Int64;
            γ::Float64 = 0.,
            s::Float64 = 0.,
            c::Float64 = 0.,
            η::Float64 = 0.,
            model::Int = 0,
            sample_fm = true
            )
        new(n₁, n₂, n₃, r, γ, s, c, η, model, sample_fm)
    end
    function Benchmarkinstance(dims::Tuple{Int64, Int64, Int64}, r::Int64;
            γ::Float64 = 0.,
            s::Float64 = 0.,
            c::Float64 = 0.,
            η::Float64 = 0.,
            model::Int = 0,
            sample_fm = true
            )
            n₁, n₂, n₃ = dims
        new(n₁, n₂, n₃, r, γ, s, c, η, model, sample_fm)
    end
end



function isstructured(bmi)
    return bmi.s != 0. || bmi.c !=0 || bmi.γ != 0
end

function display(bmi::Benchmarkinstance)
    println("   - ($(bmi.n₁), $(bmi.n₂), $(bmi.n₃)) of rank $(bmi.r), s: $(bmi.s), c: $(bmi.c), η: $(bmi.η), model: $(bmi.model).")
end

function cpd_gen(dims::Tuple{Int64, Int64, Int64}, r::Int64, model::Int64; c = 0., s = 0.)
    if model == 0
        return cpd_gen(dims, r)
    elseif model == 1
        return cpd_gen_m1(dims, r, c, s);
    elseif model == 2
        return cpd_gen_m2(dims, r, s)
    else
        @assert false "model not recognised"
    end
end


function cpd_gen_m1(dims, r, c, s)

    n₁, n₂, n₃ = dims

    C = c .* ones(r, r) + (1. - c) .* Diagonal(ones(r));
    Rc = Array{Float64, 2}(cholesky(C).U);

    𝐝 = Float64[];
    for i ∈ 1:r
        push!(𝐝, 10^((s*i)/(3*r)));
    end

    X = randn(n₁, r)*Rc*Diagonal(𝐝);
    Y = randn(n₂, r)*Rc*Diagonal(𝐝);
    Z = randn(n₃, r)*Rc*Diagonal(𝐝);
    X = Array{ComplexF64, 2}(X);
    Y = Array{ComplexF64, 2}(Y);
    Z = Array{ComplexF64, 2}(Z);

    X, Y, Z = normbalance(X, Y, Z);

    T = cpd_gen(X, Y, Z);
    return X, Y, Z, T;
end

function cpd_gen_m2(dims, r, s)

    n₁, n₂, n₃ = dims

    𝐝₂ = Float64[];
    for i ∈ 1:r
        push!(𝐝₂, 5^((i-1)/(r-1)));
    end

    𝐝₁ = ones(r) .* 10^((2-s)/2);


    X = randn(n₁, r)*(Diagonal(𝐝₁)+(randn(r, 5)*randn(r, 5)'))*Diagonal(𝐝₂)
    Y = randn(n₂, r)*(Diagonal(𝐝₁)+(randn(r, 5)*randn(r, 5)'))*Diagonal(𝐝₂)
    Z = randn(n₃, r)*(Diagonal(𝐝₁)+(randn(r, 5)*randn(r, 5)'))*Diagonal(𝐝₂)

    X = Array{ComplexF64, 2}(X);
    Y = Array{ComplexF64, 2}(Y);
    Z = Array{ComplexF64, 2}(Z);

    X, Y, Z = normbalance(X, Y, Z);

    T = cpd_gen(X, Y, Z);
    return X, Y, Z, T;
end




set_43_8_7 = false
set_20_12_7 = false
set_20_x_7 = false

set_7_7_7 = false
set_15_8_7 = false

var_set = false

debug = false


function setup()

    benchmark_set = Benchmarkinstance[]

    if Input("  - Debug? [y/n] (default n) ") == "y"
        debug = true
    else
        debug = false
    end
    if Input("  - 43 × 8 × 7 rank r (used in Kuo, Lee., made for timings.) set? [y/n] ") == "y"
        set_43_8_7 = true
    else
         set_43_8_7 = false
    end
    if Input("  - 20 × 12 × 7 rank r (good to illustrate dual algorithm timings.) set? [y/n] ") == "y"
        set_20_12_7 = true
    else
         set_20_12_7 = false
    end
    if Input("  - 20 × n₂ × 7 rank 6 (show that Kuo and Lee's algorithm's timing is sensitive to the dimension.) set? [y/n] ") == "y"
        set_20_x_7 = true
    else
         set_20_x_7 = false
    end
    if Input("  - 7 × 7 × 7 rank r (difficult problem with model 1. Shows slow convergence of NLS and ALS.) set? [y/n] ") == "y"
        set_7_7_7 = true
    else
        set_7_7_7 = false
    end
    if Input("  - 15 × 8 × 7 rank r (difficult problem with model 2. Shows slow convergnece of NLS and ALS.) set? [y/n] ") == "y"
        set_15_8_7 = true
    else
        set_15_8_7 = false
    end

    while Input("  - Specify new set? [y/n] ") == "y"
        var_set = true
        n₁ = parse(Int64, Input("        - n₁ = "))
        n₂_min = parse(Int64, Input("        - minimum n₂ value = "))
        n₂_max = parse(Int64, Input("        - maximum n₂ value = "))
        n₃_min = parse(Int64, Input("        - minimum n₃ value = "))
        n₃_max = parse(Int64, Input("        - maximum n₃ value = "))
        r_min = parse(Int64, Input("        - minimum value for r = "))
        r_max = parse(Int64, Input("        - maximum value for r = "))
        for n₂ ∈ n₂_min:n₂_max
            for n₃ ∈ n₃_min:n₃_max
                for r ∈ r_min:r_max
                    push!(benchmark_set, Benchmarkinstance(n₁, n₂, n₃, r));
                end
            end
        end
    end
    while Input("  - Manually add bmi? [y/n] ") == "y"
        n₁ = parse(Int64, Input("        - n₁ = "))
        n₂ = parse(Int64, Input("        - n₂ = "))
        n₃ = parse(Int64, Input("        - n₃ = "))
        r = parse(Int64, Input("        - r = "))
        η = parse(Float64, Input("        - η = "))
        dims = (n₁, n₂, n₃)
        sample_fm = true
        if isperfect(dims, r)
            if Input("  - Tensor is perfect. Sample from 𝒯? [y/n] ") == "y"
                sample_fm = false
            else
                sample_fm = true
            end
        end
        push!(benchmark_set, Benchmarkinstance(n₁, n₂, n₃, r, sample_fm=sample_fm, η = η))
    end


    if set_43_8_7
        for i ∈ 2:42
            push!(benchmark_set, Benchmarkinstance(43, 8, 7, i));
        end
    end

    if set_20_12_7
        for i ∈ 2:17
            push!(benchmark_set, Benchmarkinstance(20, 12, 7, i));
        end
    end

    if set_20_x_7
        for i ∈ 2:14
            push!(benchmark_set, Benchmarkinstance(20, i, 7, 6));
        end
    end

    if set_7_7_7
        r = 6
        for (c, s) ∈ [(0.95, 4.), (0.75, 2.)]
            push!(benchmark_set, Benchmarkinstance(7, 7, 7, r, c=c, s=s, model=1, η=1e-7));
        end

    end

    if set_15_8_7
        for (s, r) ∈ [(4., 9), (2., 7)]
            push!(benchmark_set, Benchmarkinstance(15, 8, 7, r, model=2, s=s,  η=1e-7));
        end
    end

    if debug==true
        benchmark_set = benchmark_set[1:1];
    end

    return benchmark_set
end # setup


function isperfect(dims, r)
    n₁, n₂, n₃ = dims
    return (n₁*n₂*n₃)/(r*(n₁-1+n₂-1+n₃)) % 1 == 0
end




"""
# `run_benchmark`

Runs different experiments. A command-line user interface allows the user to set the different parameters.

Optionally, a list of `Benchmarkinstance` types can be provided.

"""
function run_benchmark(benchmark_set::Array{Benchmarkinstance, 1} = Benchmarkinstance[])

    kl = true
    kl_lr = true
    kld_pr = true
    kld_fd = true
    als_rnd = true
    als_gevd = true
    nls = true

    if Input("  - Run kl_cpd? [y/n] ") == "y"
        kl = true
        kl_lr = true
    else
        kl = false
        kl_lr = false
    end
    if Input("  - Run kld_pr_cpd? [y/n] (default y) ") == "y"
        kld_pr = true
    else
        kld_pr = false
    end
    if Input("  - Run kld_fd_cpd? [y/n] (default y) ") == "y"
        kld_fd = true
    else
        kld_fd = false
    end
    if Input("  - Run als_rnd? [y/n] (default y) ") == "y"
        als_rnd = true
    else
        als_rnd = false
    end
    if Input("  - Run als_gevd? [y/n] (default n) ") == "y"
        als_gevd = true
    else
        als_gevd = false
    end
    if Input("  - Run nls? [y/n] (default y) ") == "y"
        nls = true
    else
        nls = false
    end

    if length(benchmark_set) == 0
        benchmark_set = setup()
    end


    rstr = rand_string();
    output = "../data/benchmark_$(rstr).csv";


    nb_warmup = parse(Int64, Input("  - number of warmups? (default = 2) "));
    nb_samples = parse(Int64, Input("  - number of samples? (default = 5) "));
    nb_runs = parse(Int64, Input("  - number of runs? (default = 1, for average condition set high) "));
    write_count = parse(Int64, Input("  - after how much bmi's should the results be saved? (default = 1) "));
    max_trials = parse(Int64, Input("  - maximum trials for iterative methods? (default = 20) "));
    maxit_als_rnd = parse(Int64, Input("  - maximum iterations for als_rnd? (default = 20000, difficult cases: 100000) "));
    maxit_nls = parse(Int64, Input("  - maximum iterations for nls? (default = 10000, difficult cases: 20000) "));

    succesfull_bwer = 1e-14;


    println("\n ============================ \n")

    println("Settings for benchmark:")
    println("   - Saving file as: $(output).")
    println("   - # Maximum iterative trials: $(max_trials). Method is succesfull if rel bwerr < $(succesfull_bwer)")
    println("   - # Maximum iterations als_rnd: $(maxit_als_rnd)")
    println("   - # Maximum iterations als_nls: $(maxit_nls)")
    println("   - # Warmups: $(nb_warmup).")
    println("   - # Samples: $(nb_samples).")
    println("   - # Runs: $(nb_runs).")

    if kl == false
        println("   - kl_cpd will not run.")
        println("   - kl_lr_cpd will not run.")
    end
    if kld_fd == false
        println("   - kld_fd_cpd will not run.")
    end
    if kld_pr == false
        println("   - kld_pr will not run.")
    end
    if als_rnd == false
        println("   - als_rnd will not run.")
    end
    if als_gevd == false
        println("   - als_gevd will not run.")
    end
    if nls == false
        println("   - nls will not run.")
    end


    dummy = Input("OK? ")
    println("Formats to test: ")
    for bmi ∈ benchmark_set
        display(bmi)
    end
    dummy = Input("OK? ")

    println("Starting matlab and loading Tensorlab2016.")

    # use tensorlab
    mat"""
        addpath("Tensorlab2016")
    """

    dummy = Input("Start benchmark? ")


    df = DataFrame(
            function_name = String[],
            bwerr = Float64[],
            fwerr = Float64[],
            time = Float64[],
            bytes = Float64[],
            gctime = Float64[],
            rank=Int64[],
            n1=Int64[],
            n2=Int64[],
            n3=Int64[],
            eta=Float64[],
            gamma=Float64[],
            s=Float64[],
            c=Float64[],
            model=Int[],
            cond_tdp = Float64[],
            cond_emp = Float64[],
            fwerrn = Float64[],
            bwerrn = Float64[],
    )



    println("\n ============================ \n")

    write_index = 1

    for bmi ∈ benchmark_set

        # unpack bmi
        n₁ = bmi.n₁
        n₂ = bmi.n₂
        n₃ = bmi.n₃
        r = bmi.r
        η = bmi.η
        γ = bmi.γ
        c = bmi.c
        s = bmi.s
        model = bmi.model
        dims = (n₁, n₂, n₃)

        noise_active = false
        Tn = nothing
        if bmi.η > 0.0
            noise_active = true
        end


        if bmi.sample_fm == false
            X₀, Y₀, Z₀ = randn(ComplexF64, n₁-1, r), randn(ComplexF64, n₂-1, r), randn(ComplexF64, n₃, r);
            @polyvar Xv[1:n₁-1, 1:r] Yv[1:n₂-1, 1:r] Zv[1:n₃, 1:r];
            @polyvar Tp[1:n₂*n₃, 1:n₁];

            vars = [vec(Xv); vec(Yv); vec(Zv)];
            initsols = [vec(X₀); vec(Y₀); vec(Z₀)];
            params = vec(Tp)

            X₀, Y₀, Z₀ = [ones(1, r); X₀], [ones(1, r); Y₀], Z₀;
            Xv,  Yv,  Zv  = [ones(1, r); Xv],  [ones(1, r); Yv], Zv;

            T₀ = reduce(hcat, [kron(Z₀[:, i], Y₀[:, i]) for i ∈ 1:r])*X₀';
            Tv  = reduce(hcat, [kron(Zv[:, i],  Yv[:, i] ) for i ∈ 1:r])*Xv' ;

            Spoly = Tv - Tp;
        end

        @showprogress 5 for run ∈ 1:nb_runs

            println("")

            write_index += 1

            display(bmi)

            # set to this scope
            X, Y, Z, T = nothing, nothing, nothing, nothing

            if bmi.sample_fm == false

                T = ones(ComplexF64, n₂*n₃, n₁)
                Tc = zeros(ComplexF64, n₂*n₃, n₁)
                paramtrial = 0

                while norm(Tc-T)/norm(T) > 1e-14
                    paramtrial += 1

                    T = randn(ComplexF64, n₂*n₃, n₁)
                    T = T ./ norm(T)

                    res = HomotopyContinuation.solve(
                        vec(Spoly), [initsols];
                        parameters=params,
                        start_parameters=vec(T₀),
                        target_parameters =vec(T),
                        start_gamma = randn(ComplexF64),
                        target_gamma = randn(ComplexF64)
                    )

                    sols = solutions(res);
                    sol = sols[1]
                    lx = length(vec(Xv))-r
                    ly = length(vec(Yv))-r
                    vecx, vecy, vecz = conj(sol[1:lx]), sol[lx+1:lx+ly], sol[lx+ly+1:end]
                    X, Y, Z = reshape(vecx, n₁-1, r), reshape(vecy, n₂-1, r), reshape(vecz, n₃, r)
                    X, Y, Z = [ones(1, r); X], [ones(1, r); Y], Z

                    X, Y, Z = normbalance(X, Y, Z);

                    # check solutions
                    Tc = cpd_gen(X, Y, Z)

                    if paramtrial > 1
                        println("   - Warning:  didn't find an accurate decomposition while sampling from 𝒯, trial $(paramtrial).")
                    end
                end

                @assert norm(Tc-T)/norm(T) < 1e-14 "Sampling from T resulted in high backward error: $(norm(Tc-T)/norm(T))."
                T = copy(Tc);

            elseif isstructured(bmi)
                X, Y, Z, T = cpd_gen(dims, r, model, c=c, s=s)
            else
                X, Y, Z, T = cpd_gen(dims, r)
            end

            if noise_active
                Tn = randn(ComplexF64, n₂*n₃, n₁)
                Tn = Tn ./ norm(Tn) .* η
                Tn = Tn + T
            end




            condition_tdp = condtdp(X, Y, Z, dims, r);
            fwerrn = NaN
            bwerrn = NaN
            cond_emp = NaN
            bwerr = NaN
            fwerr = NaN
            t = NaN
            bts = NaN
            mem = NaN
            gct = NaN


            if r <= n₁ && kl == true
                try
                    for i ∈ 1:nb_warmup
                        (Xc, Yc, Zc), t, bts, gct, mem = @timed kl_cpd(T, r, dims)
                    end
                    for i ∈ 1:nb_samples
                        (Xc, Yc, Zc), t, bts, gct, mem = @timed kl_cpd(T, r, dims)

                        Tc = cpd_gen(Xc, Yc, Zc)

                        bwerr = norm(T-Tc)/norm(T)

                        Xc, Yc, Zc = column_align((Xc, Yc, Zc), (X, Y, Z); soft=true);

                        fwerr = norm(vcat(X-Xc, Y-Yc, Z-Zc))/norm(vcat(X, Y, Z))

                        if noise_active
                            Xcn, Ycn, Zcn =  kl_cpd(Tn, r, dims)

                            Tcn = cpd_gen(Xcn, Ycn, Zcn)

                            bwerrn = norm(T-Tcn)/norm(T)

                            Xcn, Ycn, Zcn = column_align((Xcn, Ycn, Zcn), (X, Y, Z); soft=true);

                            fwerrn = norm(vcat(X-Xcn, Y-Ycn, Z-Zcn))/norm(vcat(X, Y, Z))

                            relnoise = norm(T-Tn)/norm(T)

                            cond_emp = fwerrn/relnoise
                        end

                        observation = (
                            "kl_cpd", #function_name = String[],
                            bwerr, #bwerr = Float64[],
                            fwerr, #fwerr = Float64[],
                            t, #time = Float64[],
                            bts, #bytes = Float64[],
                            gct, #gctime = Float64[],
                            r, #rank=Int64[],
                            n₁, #n1=Int64[],
                            n₂,#n2=Int64[],
                            n₃, #n3=Int64[],
                            η, #eta=Float64[],
                            γ,#gamma=Float64[]
                            s,
                            c,
                            model,
                            condition_tdp,
                            cond_emp,
                            fwerrn,
                            bwerrn
                        )

                        push!(df, observation);

                        println("       - kl_cpd: t = $(t), fwerr = $(fwerr), bwerr = $(bwerr), fwerrn = $(fwerrn), bwerrn = $(bwerrn)")

                    end # sample loop
                catch e
                    println("       - kl_cpd throws the error: $(e)")
                end
            else
                println("       - kl_cpd not applicable.")
            end


            fwerrn = NaN
            bwerrn = NaN
            cond_emp = NaN


            if r <= n₂+n₃-2 && kld_fd == true
                try
                    for i ∈ 1:nb_warmup
                        (Xc, Yc, Zc), t, bts, gct, mem = @timed kld_fd_cpd(T, r, dims)
                    end
                    for i ∈ 1:nb_samples
                        (Xc, Yc, Zc), t, bts, gct, mem = @timed kld_fd_cpd(T, r, dims)

                        Tc = cpd_gen(Xc, Yc, Zc)

                        bwerr = norm(T-Tc)/norm(T)

                        Xc, Yc, Zc = column_align((Xc, Yc, Zc), (X, Y, Z));

                        fwerr = norm(vcat(X-Xc, Y-Yc, Z-Zc))/norm(vcat(X, Y, Z))

                        if noise_active
                            Xcn, Ycn, Zcn =  kld_fd_cpd(Tn, r, dims)

                            Tcn = cpd_gen(Xcn, Ycn, Zcn)

                            bwerrn = norm(T-Tcn)/norm(T)

                            Xcn, Ycn, Zcn = column_align((Xcn, Ycn, Zcn), (X, Y, Z));

                            fwerrn = norm(vcat(X-Xcn, Y-Ycn, Z-Zcn))/norm(vcat(X, Y, Z))

                            relnoise = norm(T-Tn)/norm(T)

                            cond_emp = fwerrn/relnoise
                        end


                        observation = (
                            "kld_fd_cpd", #function_name = String[],
                            bwerr, #bwerr = Float64[],
                            fwerr, #fwerr = Float64[],
                            t, #time = Float64[],
                            bts, #bytes = Float64[],
                            gct, #gctime = Float64[],
                            r, #rank=Int64[],
                            n₁, #n1=Int64[],
                            n₂,#n2=Int64[],
                            n₃, #n3=Int64[],
                            η, #eta=Float64[],
                            γ,#gamma=Float64[]
                            s,
                            c,
                            model,
                            condition_tdp,
                            cond_emp,
                            fwerrn,
                            bwerrn
                        )

                        push!(df, observation);

                        println("       - kld_fd_cpd: t = $(t), fwerr = $(fwerr), bwerr = $(bwerr), fwerrn = $(fwerrn), bwerrn = $(bwerrn)")

                    end # sample loop
                catch e
                    println("       - kld_fd_cpd throws the error: $(e)")
                end
            else
                println("       - kld_fd_cpd not applicable.")
            end


            fwerrn = NaN
            bwerrn = NaN
            cond_emp = NaN
            bwerr = NaN
            fwerr = NaN



            if r <= n₂+n₃-2 && kld_pr == true
                try
                    for i ∈ 1:nb_warmup
                        (Xc, Yc, Zc), t, bts, gct, mem = @timed kld_pr_cpd(T, dims, r)
                    end
                    for i ∈ 1:nb_samples
                        (Xc, Yc, Zc), t, bts, gct, mem = @timed kld_pr_cpd(T, dims, r)

                        Tc = cpd_gen(Xc, Yc, Zc)

                        bwerr = norm(T-Tc)/norm(T)

                        Xc, Yc, Zc = column_align((Xc, Yc, Zc), (X, Y, Z));

                        fwerr = norm(vcat(X-Xc, Y-Yc, Z-Zc))/norm(vcat(X, Y, Z))

                        if noise_active
                            Xcn, Ycn, Zcn =  kld_pr_cpd(Tn, r, dims)

                            Tcn = cpd_gen(Xcn, Ycn, Zcn)

                            bwerrn = norm(T-Tcn)/norm(T)

                            Xcn, Ycn, Zcn = column_align((Xcn, Ycn, Zcn), (X, Y, Z));

                            fwerrn = norm(vcat(X-Xcn, Y-Ycn, Z-Zcn))/norm(vcat(X, Y, Z))

                            relnoise = norm(T-Tn)/norm(T)

                            cond_emp = fwerrn/relnoise
                        end


                        observation = (
                            "kld_pr_cpd", #function_name = String[],
                            bwerr, #bwerr = Float64[],
                            fwerr, #fwerr = Float64[],
                            t, #time = Float64[],
                            bts, #bytes = Float64[],
                            gct, #gctime = Float64[],
                            r, #rank=Int64[],
                            n₁, #n1=Int64[],
                            n₂,#n2=Int64[],
                            n₃, #n3=Int64[],
                            η, #eta=Float64[],
                            γ,#gamma=Float64[]
                            s,
                            c,
                            model,
                            condition_tdp,
                            cond_emp,
                            fwerrn,
                            bwerrn
                        )

                        push!(df, observation);

                        println("       - kld_pr_cpd: t = $(t), fwerr = $(fwerr), bwerr = $(bwerr), fwerrn = $(fwerrn), bwerrn = $(bwerrn)")

                    end # sample loop
                catch e
                    println("       - kld_pr_cpd throws the error: $(e)")
                end
            else
                println("       - kld_pr_cpd not applicable.")
            end

            fwerrn = NaN
            bwerrn = NaN
            cond_emp = NaN
            bwerr = NaN
            fwerr = NaN


            if r < n₂ && r < n₃ && kl_lr
                try
                    for i ∈ 1:nb_warmup
                        (Xc, Yc, Zc), t, bts, gct, mem = @timed kl_lr_cpd(T, r, dims)
                    end
                    for i ∈ 1:nb_samples
                        (Xc, Yc, Zc), t, bts, gct, mem = @timed kl_lr_cpd(T, r, dims)

                        Tc = cpd_gen(Xc, Yc, Zc)

                        bwerr = norm(T-Tc)/norm(T)

                        Xc, Yc, Zc = column_align((Xc, Yc, Zc), (X, Y, Z));

                        fwerr = norm(vcat(X-Xc, Y-Yc, Z-Zc))/norm(vcat(X, Y, Z))

                        if noise_active
                            Xcn, Ycn, Zcn =  kl_lr_cpd(Tn, r, dims)

                            Tcn = cpd_gen(Xcn, Ycn, Zcn)

                            bwerrn = norm(T-Tcn)/norm(T)

                            Xcn, Ycn, Zcn = column_align((Xcn, Ycn, Zcn), (X, Y, Z));

                            fwerrn = norm(vcat(X-Xcn, Y-Ycn, Z-Zcn))/norm(vcat(X, Y, Z))

                            relnoise = norm(T-Tn)/norm(T)

                            cond_emp = fwerrn/relnoise
                        end


                        observation = (
                            "kl_lr_cpd", #function_name = String[],
                            bwerr, #bwerr = Float64[],
                            fwerr, #fwerr = Float64[],
                            t, #time = Float64[],
                            bts, #bytes = Float64[],
                            gct, #gctime = Float64[],
                            r, #rank=Int64[],
                            n₁, #n1=Int64[],
                            n₂,#n2=Int64[],
                            n₃, #n3=Int64[],
                            η, #eta=Float64[],
                            γ,#gamma=Float64[]
                            s,
                            c,
                            model,
                            condition_tdp,
                            cond_emp,
                            fwerrn,
                            bwerrn
                        )

                        push!(df, observation);

                        println("       - kl_lr_cpd: t = $(t), fwerr = $(fwerr), bwerr = $(bwerr), fwerrn = $(fwerrn), bwerrn = $(bwerrn)")

                    end # sample loop
                catch e
                    println("       - kl_lr_cpd throws the error: $(e)")
                end
            else
                println("       - kl_lr_cpd not applicable.")
            end



            fwerrn = NaN
            bwerrn = NaN
            cond_emp = NaN
            bwerr = NaN
            fwerr = NaN


            # communicate data to matlab
            dims_mat = collect(dims)
            T_mat = matten(T,[2,3],[1],collect(dims));
            @mput T_mat r dims_mat maxit_als_rnd maxit_nls;


            # als_rnd
            if als_rnd
                try
                    # initialization of matlab
                    mat"""
                        r_mat = cast(r, 'double');
                        options = struct;
                        options.LineSearch = false;
                        options.PlaneSearch = false;
                        options.MaxIter = maxit_als_rnd; % for ill condition test
                        options.TolFun = 1e-32;
                        options.TolX = 1e-16;
                        options.Delta = 1;
                        options.Display = false;
                        option.Complex = true;
                        options.LargeScale = true;
                        U0{1} = randn(dims_mat(1),r_mat);
                        U0{2} = randn(dims_mat(2),r_mat);
                        U0{3} = randn(dims_mat(3),r_mat);

                    """
                    for i ∈ 1:nb_warmup
                        mat"""
                        U0{1} = randn(dims_mat(1),r_mat);
                        U0{2} = randn(dims_mat(2),r_mat);
                        U0{3} = randn(dims_mat(3),r_mat);
                        """
                         eval_string("[U, output] = cpd_als(T_mat, U0, options);")
                    end
                    for i ∈ 1:nb_samples

                        total_t = 0.

                        not_converged = true

                        trials = 0

                        while not_converged && trials < max_trials

                            trials += 1

                            mat"""
                            U0{1} = randn(dims_mat(1),r_mat);
                            U0{2} = randn(dims_mat(2),r_mat);
                            U0{3} = randn(dims_mat(3),r_mat);
                            """

                            res, t, bts, gct, mem = @timed eval_string("[U, output_mat] = cpd_als(T_mat, U0, options);")

                            total_t += t

                            eval_string("T_mat_calc = cpdgen(U);")

                            @mget U output_mat T_mat_calc

                            Xc, Yc, Zc= U[1], U[2], U[3]

                            bwerr = norm(T_mat_calc-T_mat)/norm(T_mat);

                            if bwerr < succesfull_bwer
                                not_converged = false
                            else
                                println("       - trial nb $(trials), bwerr = $(bwerr)")
                            end

                        end

                        t = total_t

                        Xc, Yc, Zc = normbalance(Xc, Yc, Zc);

                        Xc, Yc, Zc = column_align((Xc, Yc, Zc), (X, Y, Z); soft=true);

                        Xc = signnorm(conj(Xc));
                        Xs = signnorm(copy(X));

                        fwerr = norm(vcat(Xs-Xc, Y-Yc, Z-Zc))/norm(vcat(X, Y, Z))

                        if noise_active

                            Tn_mat = matten(Tn,[2,3],[1],collect(dims));

                            @mput Tn_mat

                            trials = 0

                            not_converged = true

                            while not_converged && trials < max_trials

                                trials = trials + 1

                                mat"""
                                U0{1} = randn(dims_mat(1),r_mat);
                                U0{2} = randn(dims_mat(2),r_mat);
                                U0{3} = randn(dims_mat(3),r_mat);
                                """

                                eval_string("[U, output_mat] = cpd_als(Tn_mat, U0, options);")

                                eval_string("T_mat_calc = cpdgen(U);")

                                @mget U output_mat T_mat_calc

                                Xcn, Ycn, Zcn = U[1], U[2], U[3]

                                bwerrn = norm(T_mat_calc-T_mat)/norm(T_mat);

                                bwerr = norm(T_mat_calc-Tn_mat)/norm(T_mat);

                                if bwerr < succesfull_bwer
                                    not_converged = false
                                    println("       - trial: $(trials) for noise. bwerr = $(bwerr)")
                                end

                            end

                            Xcn, Ycn, Zcn = normbalance(Xcn, Ycn, Zcn);

                            Xcn, Ycn, Zcn = column_align((Xcn, Ycn, Zcn), (X, Y, Z); soft=true);

                            Xcn = signnorm(conj(Xcn));
                            Xs = signnorm(copy(X));

                            fwerrn = norm(vcat(Xs-Xcn, Y-Ycn, Z-Zcn))/norm(vcat(X, Y, Z))

                            relnoise = norm(T-Tn)/norm(T)

                            cond_emp = fwerrn/relnoise
                        end


                        observation = (
                            "tlab_als_rnd", #function_name = String[],
                            bwerr, #bwerr = Float64[],
                            fwerr, #fwerr = Float64[],
                            t, #time = Float64[],
                            bts, #bytes = Float64[],
                            gct, #gctime = Float64[],
                            r, #rank=Int64[],
                            n₁, #n1=Int64[],
                            n₂,#n2=Int64[],
                            n₃, #n3=Int64[],
                            η, #eta=Float64[],
                            γ,#gamma=Float64[]
                            s,
                            c,
                            model,
                            condition_tdp,
                            cond_emp,
                            fwerrn,
                            bwerrn
                        )

                        push!(df, observation);

                        println("       - tlab_als_rnd: t = $(t), fwerr = $(fwerr), bwerr = $(bwerr), fwerrn = $(fwerrn), bwerrn = $(bwerrn)")
                    end
                catch e
                    println("       - tlab_als_rnd throws an error: $(e).")
                end
            end


            fwerrn = NaN
            bwerrn = NaN
            cond_emp = NaN


            # als_gevd
            if ((n₁ >= n₂ && n₂ >= n₃ && r <= n₂) || (n₁ >= n₃ && n₂ <= n₃ && r <= n₃)) && als_gevd
                try
                    # initialization of matlab
                    mat"""
                    r_mat = cast(r, 'double');
                    options = struct;
                    [U0, output_mat] = cpd_gevd(T_mat, r_mat);
                    options = struct;
                    options.LineSearch = false;
                    options.PlaneSearch = false;
                    options.MaxIter = maxit_nls;
                    options.TolFun = 1e-32;
                    options.TolX = 1e-16;
                    options.Delta = 1;
                    options.Display = false;
                    option.Complex = true;
                    """
                    for i ∈ 1:nb_warmup
                         eval_string("[U, output] = cpd_als(T_mat, U0, options);")
                    end
                    for i ∈ 1:nb_samples
                        res, t, bts, gct, mem = @timed eval_string("[U, output_mat] = cpd_als(T_mat, U0, options);")

                        eval_string("T_mat_calc = cpdgen(U);")

                        @mget U output_mat T_mat_calc

                        Xc, Yc, Zc= U[1], U[2], U[3]

                        bwerr = norm(T_mat_calc-T_mat)/norm(T_mat);

                        Xc, Yc, Zc = normbalance(Xc, Yc, Zc);

                        Xc, Yc, Zc = column_align((Xc, Yc, Zc), (X, Y, Z); soft=true);

                        Xc = signnorm(conj(Xc));
                        Xs = signnorm(copy(X));

                        fwerr = norm(vcat(Xs-Xc, Y-Yc, Z-Zc))/norm(vcat(X, Y, Z))

                        if noise_active

                            Tn_mat = matten(Tn,[2,3],[1],collect(dims));

                            @mput Tn_mat

                            eval_string("[U, output_mat] = cpd_als(Tn_mat, U0, options);")

                            eval_string("T_mat_calc = cpdgen(U);")

                            @mget U output_mat T_mat_calc

                            Xcn, Ycn, Zcn = U[1], U[2], U[3]

                            bwerrn = norm(T_mat_calc-T_mat)/norm(T_mat);

                            Xcn, Ycn, Zcn = normbalance(Xcn, Ycn, Zcn);

                            Xcn, Ycn, Zcn = column_align((Xcn, Ycn, Zcn), (X, Y, Z); soft=true);

                            Xcn = signnorm(conj(Xcn));
                            Xs = signnorm(copy(X));

                            fwerrn = norm(vcat(Xs-Xcn, Y-Ycn, Z-Zcn))/norm(vcat(X, Y, Z))

                            relnoise = norm(T-Tn)/norm(T)

                            cond_emp = fwerrn/relnoise
                        end

                        observation = (
                            "tlab_als_gevd", #function_name = String[],
                            bwerr, #bwerr = Float64[],
                            fwerr, #fwerr = Float64[],
                            t, #time = Float64[],
                            bts, #bytes = Float64[],
                            gct, #gctime = Float64[],
                            r, #rank=Int64[],
                            n₁, #n1=Int64[],
                            n₂,#n2=Int64[],
                            n₃, #n3=Int64[],
                            η, #eta=Float64[],
                            γ,#gamma=Float64[]
                            s,
                            c,
                            model,
                            condition_tdp,
                            cond_emp,
                            fwerrn,
                            bwerrn
                        )

                        push!(df, observation);

                        println("       - tlab_als_gevd: t = $(t), fwerr = $(fwerr), bwerr = $(bwerr), fwerrn = $(fwerrn), bwerrn = $(bwerrn)")
                    end
                catch e
                    println("       - tlab_als_gevd throws an error: $(e).")
                end
            else
                println("       - tlab_als_gevd not applicable")
            end


            fwerrn = NaN
            bwerrn = NaN
            cond_emp = NaN



            # nls
            if nls
                try
                    # initialization of matlab
                    mat"""
                        r_mat = cast(r, 'double');
                        options = struct;
                        options.Display = false;
                        options.MaxIter = 20000; % for ill condition tests
                        option.Complex = true;
                        options.LineSearch = false;
                        options.PlaneSearch = false;
                        options.TolFun = 1e-32;
                        options.TolX = 1e-16;
                        options.Delta = 1;
                        options.LargeScale = true;
                        U0{1} = randn(dims_mat(1),r_mat);
                        U0{2} = randn(dims_mat(2),r_mat);
                        U0{3} = randn(dims_mat(3),r_mat);
                    """
                    for i ∈ 1:nb_warmup
                        mat"""
                            U0{1} = randn(dims_mat(1),r_mat);
                            U0{2} = randn(dims_mat(2),r_mat);
                            U0{3} = randn(dims_mat(3),r_mat);
                        """
                         eval_string("[U, output] = cpd_nls(T_mat, U0, options);")
                    end
                    for i ∈ 1:nb_samples

                        trials = 0

                        not_converged = true

                        total_t = 0.

                        while not_converged && trials < max_trials

                            trials = trials + 1

                            mat"""
                                U0{1} = randn(dims_mat(1),r_mat);
                                U0{2} = randn(dims_mat(2),r_mat);
                                U0{3} = randn(dims_mat(3),r_mat);
                            """
                            res, t, bts, gct, mem = @timed eval_string("[U, output_mat] = cpd_nls(T_mat, U0, options);")

                            eval_string("T_mat_calc = cpdgen(U);")

                            @mget U output_mat T_mat_calc

                            Xc, Yc, Zc= U[1], U[2], U[3]

                            bwerr = norm(T_mat_calc-T_mat)/norm(T_mat);

                            if bwerr < succesfull_bwer
                                not_converged = false
                            else
                                println("       - trial nb $(trials), bwerr = $(bwerr)")
                            end

                            total_t += t
                        end

                        t = total_t

                        Xc, Yc, Zc = normbalance(Xc, Yc, Zc);

                        Xc, Yc, Zc = column_align((Xc, Yc, Zc), (X, Y, Z); soft=true);

                        Xc = signnorm(conj(Xc));
                        Xs = signnorm(copy(X));

                        fwerr = norm(vcat(Xs-Xc, Y-Yc, Z-Zc))/norm(vcat(X, Y, Z))

                        if noise_active

                            Tn_mat = matten(Tn,[2,3],[1],collect(dims));

                            @mput Tn_mat

                            not_converged = true

                            trials = 0

                            while not_converged && trials < max_trials

                                trials = trials + 1

                                eval_string("[U, output_mat] = cpd_nls(Tn_mat, U0, options);")

                                eval_string("T_mat_calc = cpdgen(U);")

                                @mget U output_mat T_mat_calc

                                Xcn, Ycn, Zcn = U[1], U[2], U[3]

                                bwerrn = norm(T_mat_calc-T_mat)/norm(T_mat);

                                bwerr =  norm(T_mat_calc-Tn_mat)/norm(T_mat);

                                if bwerr < succesfull_bwer
                                    not_converged = false
                                    println("       - trial: $(trials) for noise. bwerr = $(bwerr)")
                                end

                            end

                            Xcn, Ycn, Zcn = normbalance(Xcn, Ycn, Zcn);

                            Xcn, Ycn, Zcn = column_align((Xcn, Ycn, Zcn), (X, Y, Z); soft=true);

                            Xcn = signnorm(conj(Xcn));
                            Xs = signnorm(copy(X));

                            fwerrn = norm(vcat(Xs-Xcn, Y-Ycn, Z-Zcn))/norm(vcat(X, Y, Z))

                            relnoise = norm(T-Tn)/norm(T)

                            cond_emp = fwerrn/relnoise
                        end

                        observation = (
                            "tlab_nls", #function_name = String[],
                            bwerr, #bwerr = Float64[],
                            fwerr, #fwerr = Float64[],
                            t, #time = Float64[],
                            bts, #bytes = Float64[],
                            gct, #gctime = Float64[],
                            r, #rank=Int64[],
                            n₁, #n1=Int64[],
                            n₂,#n2=Int64[],
                            n₃, #n3=Int64[],
                            η, #eta=Float64[],
                            γ,#gamma=Float64[]
                            s,
                            c,
                            model,
                            condition_tdp,
                            cond_emp,
                            fwerrn,
                            bwerrn
                        )

                        push!(df, observation);

                        println("       - tlab_nls: t = $(t), fwerr = $(fwerr), bwerr = $(bwerr), fwerrn = $(fwerrn), bwerrn = $(bwerrn)")
                    end
                catch e
                    println("       - tlab_nls throws an error: $(e).")
                end
            end



            # save results
            if write_index >= write_count
                CSV.write(output, df);
                write_index = 0;
            end


        end # benchmarkinstance loop
    end # run loop
    println("--> $(output)")
end # run_benchmark












end # module
