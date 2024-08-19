module DualUtils
using DynamicPolynomials
using Random
using LinearAlgebra: norm, \, *, svd, rank, Diagonal, nullspace, dot, opnorm, qr, cond
using HomotopyContinuation

export dual_solve, translate_dual_prim, rankfact_nullspace, translate_prim_dual

"""
# `rankfact_nullspace`

Calculate the full rank factorisation of T s.t. T = EF' and N = null(E').

## Inputs

- `T`: matrix to factorize
- `r`: rank of t

## Outputs

- `E`: matrix such that T = EF'
- `F`: matrix such that T = EF'
- `N`: matrix such that N'E = 0.

"""
function rankfact_nullspace(T::Array{d_type, 2}, r::Int64) where d_type
    T_svd = svd(T; full=true);
    V, S, U  = T_svd.V, T_svd.S, T_svd.U;

    F = Diagonal(S) * V';
    F = F';
    F = F[:, 1:r];

    E = U[:, 1:r]; N = U[:, r+1:end];
    return E, F, N
end


"""
# `dual_solve`

Calculates a rank-one basis for Null(E') by intersecting the positive dimensional variety with hyperplanes.
The dimensions of E are n₂n₃ × r. Requires that r < n₂ + n₃ - 2.

All solutions (𝐮ⱼ, 𝐯ⱼ) satisfy

```math
    (\\mathbf{u}_j \\odot \\mathbf{v}_j)^H\\mathbf{e}_i = 0, \\quad \\forall i=1,\\dots, r,\\quad \\forall j=1,\\dots, n_2n_3-r.
```

This method is a wrapper for constructing random systems, solving them and afterwards parsing and checking the solutions.
Three different solvers are used:

1. `Linpolysolve()`: if r < n₂ or r < n₃ it is possible to contruct linear systems in r variables. If this is done depends on the given strategy.
2. Full degree solver: calculates all solutions of a nonlinear system using `HomotopyContinuation`.
3. `dual_paramsolve`: calculates a set of solutions using, then tracks these solutions while the parameters of the system (those which define the hyperplanes) are changed.

Using the default strategy 'x_last_xy' (old naming convention) the method will start by
dehomogenizing the system such that it is linear in 𝐯 if r < n₂. When no new independent solutions are found,
it will use r variables in 𝐯 and one in 𝐮.
In the case that r > n₂, all variables in 𝐯 and r-n₂ variables in 𝐳 are used.

## Inputs

- `E` basis for the space 𝐄 to which we seek orthogonal rank-one vectors.
- `dims` dimensions of the tensor.
- `r` rank of the tensor.

## Outputs

- `success`: integer flag which equals 1 if all went well.
- `N`: the rank-one basis for 𝐍 = Nul(𝐄)
- `UV`: container with the vectors 𝐮 and 𝐯 such that 𝐮ⱼ ⊙ 𝐯ⱼ = 𝐧ⱼ. The vector 𝐮ⱼ can be accessed as `UV[j][2]` and the vector 𝐯ⱼ as `UV[j][1]`.

## Keyword arguments

- `verbose`: print some info to the command line
- `strategy`: indicate a strategy to numerically eliminate variables.

## Strategies


The strategy can be set via the keyword argument `strategy`. Possible values are

!!! note The strategy names come from an old naming convention.

- `"x_last_xy"`: if r < n₂ eliminate all variables in 𝐮 such that the dual problem is linear in 𝐯. Once this doesn't result in new solutions or r >= n₂, solve non-linear systems with one variable in 𝐯 and r variables in 𝐮 with the full degree solver.
- `"y_last_yx"`: if r < n₃ eliminate all variables in 𝐯 such that the dual problem is linear in 𝐮. Once this doesn't result in new solutions or r >= n₃, solve non-linear systems with one variable in 𝐮 and r variables in 𝐯 with the full degree solver.
- `"optimal"`: choose automatically between `"x_last_xy"` and `"y_last_yx"` such that linear systems are used when r < n₂ or r < n₃.
- `"x_last_xypar"`: (not properly tested) similar to `"x_last_xy"` but `dual_paramsolve` is used to solve the nonlinear systems.
- `"y_last_yxpar"`: (not properly tested) similar to `"y_last_yx"` but `dual_paramsolve` is used to solve the nonlinear systems.


## Possible improvements:

1. Optimise memory allocation. Most memory is still allocated 'on the fly'.
2. Uniform naming conventions in all helper functions.

"""
function dual_solve(E::Array{d_type, 2}, dims::Tuple{Int64, Int64, Int64}, r::Int64;
        maxit = 50,
        verbose = true,
        strategy = "optimal",
        full = false,
        measure_condition = false,
        filtsols_independent_threshold = 1e-5,
        filtsols_residual_threshold = 1e-10,
        filtsols_triviality_threshold = 1e-4,
        nonzero_threshold = 1e-10,
    ) where d_type

    # unpack
    n₁, n₂, n₃ = dims;

    @assert size(E)[2] == r "dual_solve: given rank does not match dimensions of matrix E. "
    @assert size(E)[1] == n₂*n₃ "dual_solve: given dimensions don't match dimensions of matrix E."


    # get strategy (how to decide which variables to keep in the reduced system)
    setuv, index_rep, strategy = dual_strategyinit(strategy, dims, r);

    # combinatorics to generate well chosen choices of variables
    V_combos = combis(collect(1:n₂), r, maxit)
    U_combos = combis(collect(1:n₃), r, maxit)

    # init loop index
    Vcomb_index = 1; Ucomb_index = 1;

    # container to store condition numbers if requested
    lin_cond = Float64[];
    poly_cond = Float64[];

    # container to store results
    N = Array{ComplexF64, 1}[];
    UV = Array{Array{ComplexF64, 1}, 1}[];

    # counters
    cur_rank = 0;
    it = 0;
    countv = 0;
    countu = 0;
    countvu = 0;
    countuv = 0;
    countrem = 0;

    # random mixing of basis vectors;
    Q, _ = qr(randn(ComplexF64, r, r));

    # container for polynomial system
    𝐩₁ = zeros(Polynomial{true, d_type}, n₂+n₃);

    # grow basis untill it is of full rank (or we keep searching until we reach maxit)
    while (cur_rank < n₂*n₃-r && it < maxit) || (full && it < maxit)

        it+= 1;

        # reset indices for random combinations if necessary
        if r <= n₂ || r <= n₃
            Vcomb_index = Int(mod.(it, index_rep) + floor.(it/index_rep))
            while Vcomb_index > length(V_combos); Vcomb_index = Vcomb_index - length(V_combos); end;
            Ucomb_index = Int(mod.(it, index_rep) + floor.(it/index_rep))
            while Ucomb_index > length(U_combos); Ucomb_index = Ucomb_index - length(U_combos); end;
        end

        ##### 1. CONSTRUCT #####################################################

        # set some variables in 𝐲 and 𝐳 to a fixed random value
        𝐯, 𝐮, 𝐭, add_eqs, rm_eqs, setuv, vars = dual_constructvars(n₂, n₃, r,
            verbose,
            setuv,
            V_combos,
            U_combos,
            Vcomb_index,
            Ucomb_index,
            it
        )

        # construct the polynomial system
        𝐩, 𝛄 = dual_constructpolysys(𝐩₁, Q, E, 𝐯, 𝐮, r, dims, setuv, add_eqs, rm_eqs);

        ##### 2. SOLVE #########################################################

        # solve the polynomial system
        𝛇 = dual_solvepolysys(𝐩, 𝐯, 𝐮, 𝐭, 𝛄, vars, setuv, r, dims, measure_condition, poly_cond, lin_cond, cur_rank)

        ##### 3. PROCESS #######################################################

        # build the resulting 𝐮 ⊙ 𝐯 = 𝐧 vectors from the solution of the polynomial system
        𝐧_list, 𝐯_list, 𝐮_list = dual_deparsevars(n₂, n₃, r, 𝐯, 𝐮, 𝛇, setuv, V_combos, U_combos, Vcomb_index, Ucomb_index, it, verbose)

        # update counters
        if setuv == "x"; countv += 1;
        elseif setuv == "y"; countu += 1;
        elseif setuv == "xy"; countvu += 1;
        elseif setuv == "yx"; countuv += 1;
        else countrem += 1 end;

        # update setyz, which decides which variables are chosen to be set
        setuv = dual_strategyupdate(strategy, setuv, it, length(N));

        # filter rank 1 terms which are trivial or already contained in the range of the basis
        dual_filtsols(E, r, 𝐧_list, 𝐯_list, 𝐮_list, N, UV, full;
            independent_threshold = filtsols_independent_threshold,
            residual_threshold = filtsols_residual_threshold,
            triviality_threshold = filtsols_triviality_threshold
        )

        if length(N) > 0; cur_rank = rank(reduce(hcat, N)); else; cur_rank = 0; end
    end

    # provide output if asked
    if verbose
        println("search: $(length(N)) of $(n₂*n₃-r) vectors after $it iterations (only 𝐯: $countv, only 𝐮: $countu, 𝐯 and one u: $countvu, one v and 𝐮: $countuv).");
        println("corresponds to solving: ")
        println(" - $(countvu) poly systems in $(r+1) variables")
        println(" - $(countuv) poly systems in $(r+1) variables")
        println(" - $(countv + countu) linear systems of dimension $r.")
        println(" - $(countrem) poly systems in $(min(r, n₂+n₃+1)) variables")
    end

    # did we succeed in finding a full basis?
    if length(N) == n₂*n₃-r; success = 1
    else; success = 0
    end

    # return condition numbers as well if asked to
    if measure_condition
        return N, UV, lin_cond, poly_cond
    else
        return success, N, UV
    end
end

# helper function
function nonzero(x); return x != 0; end;




"""
# `translate_dual_prim`

Calculate a rank-one basis Z ⊙ Y orthogonal to U ⊙ V. The dimensions are

- `U`: n₃ × n₂n₃-r;
- `V`: n₂ × n₂n₃-r;
- `Z`: n₃ × r;
- `Y`: n₂ × r.

This solutions (𝐲ᵢ, 𝐳ᵢ) satisfy

```math
    (\\mathbf{u}_j \\odot \\mathbf{v}_j)^H(\\mathbf{z}_i \\odot \\mathbf{y}_i) = 0, \\quad \\forall i=1,\\dots, r,\\quad \\forall j=1,\\dots, n_2n_3-r.
```

Moreover the unless `soft` is activated the method asserts that the returned solutions are zero-dimensional.

## Inputs

- `V`: see above.
- `U`: see above.
- `dims`: see above.
- `r`: see above.

## Keyword arguments

- `verbose`: (`true`/`false`) provide some output on the number of iterations.
- `maxit`: (int) maximum number of iterations.
- `sigma_threshold`: (float) decides when a new vector is considered to be part N. If σ[end]/σ[1] < sigma_threshold the solution is rejected due to linear dependency.
- `soft`: (`true`/`false`) silences assertions for zero-dimensional solution components. Thus, activating `soft` will not interrupt the software if a positive dimensional solution component is encountered.

## Outputs

- `success`: flag which equals the interger one if the method succeeded.
- `YZ`: container for 𝐲ᵢ and 𝐳ᵢ which can be accessed as `YZ[i][1]` and `YZ[i][2]`.

## Possible improvements

- reduce the number of calls to `nullspace`.


"""
function translate_dual_prim(V, U, dims, r;
    verbose = false,
    maxit = 5000,
    sigma_threshold = 1e-3,
    nonzero_threshold = 1e-10,
    soft = false
    )

    # unpack
    n₁, n₂, n₃ = dims

    @assert size(V)[2] == size(U)[2]  "Matrices V and U don't have the same size."

    nb_vecs = size(V)[2];
    if nb_vecs != n₂*n₃-r && soft==false;
        println("Warning: translating from dual to primal problem with only a partial basis for N.")
    end


    # loop counter
    it = 0

    # container for 𝐲ᵢ and 𝐳ᵢ
    YZ = Array{Array{ComplexF64, 1}, 1}[]

    # flag
    success = 1


    while length(YZ) < r && it < maxit

        it += 1;

        # settings can be set to a random int to increase variability (see translate_prim_dual)
        # in this case they serve to prevent an infinite loop
        max_v_it = nb_vecs*5;
        max_u_it = nb_vecs*5;

        # start from V

        # set of indices in y to set to zero
        𝔍_v = [rand(1:nb_vecs)];

        # loop counter
        v_it = 0

        # search untill size(Null(V[:, 𝔍_v]))[2] = 1;
        while size(nullspace(V[:, 𝔍_v]'))[2] > 1 && v_it < max_v_it
            push!(𝔍_v, rand(1:nb_vecs))
            v_it += 1
        end

        v_null = nullspace(V[:, 𝔍_v]')

        @assert size(v_null)[2] == 1 || soft "Found a positive dimensional solution component. Dim(Span(V))<n₂-1 or reached maximum number of iterations"
        if size(v_null)[2] != 1; success = 0; end;

        # this changes the datatype to an Array{d_type, 1} instead of Array{d_type, 2}.
        v_null = v_null[:, 1]


        𝔍_u = filter(nonzero, (abs.(V'*v_null)/norm(v_null) .> nonzero_threshold)[:, 1] .* collect(1:nb_vecs))

        @assert length(𝔍_u) != 0 || soft "Found a positive dimensional solution component. Dim(Span(V))<n₂"
        @assert length(𝔍_u) >= n₃-1 || soft "Found a positive dimensional solution component. Unsufficient number of columns in U^act."
        if length(𝔍_u) == 0 || length(𝔍_u) < n₃-1; success = 0; end;

        u_null = nullspace(U[:, 𝔍_u]')

        @assert size(u_null)[2] <= 1 || soft "Found a positive dimensional solution component.
        Columns in U^act don't span a space of dimension n₃-1."
        if size(u_null)[2] > 1; success = 0; end;

        if size(u_null)[2] == 1
            # add solution to solution container
            𝐲𝐳 = Array{ComplexF64, 1}[];
            push!(𝐲𝐳, Array{ComplexF64, 1}(v_null[:, 1]))
            push!(𝐲𝐳, Array{ComplexF64, 1}(u_null[:, 1]))
            push!(YZ, 𝐲𝐳);

            # test for linear independence
            E_ = reduce(hcat, [kron(YZ[i][2], YZ[i][1]) for i ∈ 1:length(YZ)]);
            E_svd = svd(E_)
            if E_svd.S[end]/E_svd.S[1] < sigma_threshold
                pop!(YZ)
            end
        end



        # start from U

        # initialise the indices list
        𝔍_u = [rand(1:nb_vecs)];

        # loop counter
        y_it = 0

        # add indices to 𝔍_u such that null(U[:, 𝔍_u]) is one dimensional
        while size(nullspace(U[:, 𝔍_u]'))[2] > 1 && y_it < max_u_it
            push!(𝔍_u, rand(1:nb_vecs))
            y_it += 1
        end

        u_null = nullspace(U[:, 𝔍_u]')

        @assert size(u_null)[2] == 1 || soft "Found a positive dimensional solution component. Dim(Span(U))<n₃-1 or reached maximum number of iterations"
        if size(u_null)[2] != 1; success = 0; end;

        # change data type from 2D array to 1D array
        u_null = u_null[:, 1]

        𝔍_v = filter(nonzero, (abs.(U'*u_null)/norm(u_null) .> nonzero_threshold)[:, 1] .* collect(1:nb_vecs))

        @assert length(𝔍_v) != 0 || soft "Found a positive dimensional solution component. Dim(Span(U))< n₃"
        @assert length(𝔍_v) >= n₂-1 || soft "Found a positive dimensional solution component. Unsufficient number of columns in V^act."
        if length(𝔍_v) == 0 || length(𝔍_v) < n₂-1; success = 0; end;

        v_null = nullspace(V[:, 𝔍_v]')

        @assert size(v_null)[2] <= 1 || soft "Found a positive dimensional solution component.
        Columns in V^act don't span a space of dimension n₂-1."
        if size(v_null)[2] > 1; success = 0; end;

        if size(v_null)[2] == 1
            # add solution to solution container
            𝐲𝐳 = Array{ComplexF64, 1}[];
            push!(𝐲𝐳, Array{ComplexF64, 1}(v_null[:, 1]))
            push!(𝐲𝐳, Array{ComplexF64, 1}(u_null[:, 1]))
            push!(YZ, 𝐲𝐳);

            # test for linear independence
            E_ = reduce(hcat, [kron(YZ[i][2], YZ[i][1]) for i ∈ 1:length(YZ)]);
            E_svd = svd(E_)
            if E_svd.S[end]/E_svd.S[1] < sigma_threshold
                pop!(YZ)
            end
        end
        if success == 0
            return success, YZ
        end
    end
    # provide output if requested
    if verbose;
        println("search: found $(length(YZ)) of $r vectors in $it iterations");
    end

    if length(YZ) < r
        success = 0
    end

    return success, YZ

end


"""
# `dual_constructvars`

Helper function for the `dual_solve()` method. The function constructs two arrays of variables 𝐯 and 𝐮
    which have a random set of elements set to a random value.

!!! note Internally this method uses a different naming convention.


## Inputs

- `n₂`: (int) dimension of 𝐯 (corresponding tensor has dimensions n₁×n₂×n₃).
- `n₃`: (int) dimension of 𝐮.
- `r`: (int) rank of the corresponding tensor.
- `verbose`: print some information about the proceedings to stdout.
- `setuv`: (string) defines which variables should be set.
- `X_combos`: (old naming convention, `V_combos` would be more appropriate) list of different combinations of r numbers from the range 1:n₂.
- `Y_combos`: (old naming convention, `U_combos` would be more appropriate) list of different combinations of r numbers from the range 1:n₃:
- `Xcomb_index`: (old naming convention) tells the method which 'combo' of `X_combos` should be used.
- `Ycomb_index`: (old naming convention) similar as above.
- `it`: global iteration counter. Used to only print a message during the first iteration if `verbose` equals `true`.

## Outputs

- `x`: (old naming convention, `𝐯` would be more appropriate) array consisting of `PolyVar` types and / or random values.
- `y`: (old naming convention, `𝐮` would be more appropriate) similar as above.
- `t`: if the polynomial system is solved with `dual_paramsolve`, then `t` represents the `PolyVar`s in `x` and `y` which would otherwise be set to a random value.
- `add_eqs`: if there are more than r variables (`PolyVar`), then `add_eqs` communicates this further.
- `rm_eqs`: depreciated
- `setuv`: (string) communicates which variables are set.
- `vars`: array of all variables in the polynomial system (only 'true' variables, so no parameters such as `t`).



"""
function dual_constructvars(n₂, n₃, r, verbose, setuv, X_combos, Y_combos, Xcomb_index, Ycomb_index, it)

    # rename variables to old naming convention
    I = n₂;
    J = n₃;
    R = r;
    setxy = setuv;


    add_eqs = 0; # communicate to construct polysys that additional linear equations are need
    rm_eqs = 0;  # communicate to construct polysys that equations have to be removed
    t = nothing; # variable parameters

    # if R > I && R > J && R < J+I; # use a full degree homotopy
    #     setxy = "fill"
    #     @polyvar x[1:I] y_var[1:R-I+1]
    #     y = [y_var; randn(ComplexF64, J-(R-I+1))];
    #     add_eqs = 1;
    #     vars = [x; y_var]
    if  R > I && R > J && R < J+I
        setxy = "fillpar"
        @polyvar x[1:I] y_var[1:R-I+1] t[1:J-(R-I+1)];
        y = [y_var; t];
        add_eqs = 1;
        if verbose && it == 1; println("Fill mode: #𝐲 = $I = n₂,  #𝐳 = $(R-I+1) = r-n₂+1."); end
        vars = [x; y_var]

    elseif R >= I + J; # overdetermined system, use only subset of equations
        setxy = "over"
        @polyvar x[1:I] y[1:J]
        add_eqs = 1;
        rm_eqs = R-I-J+2;
        if verbose && it == 1; println("Overdetermined mode: #𝐲 = $I, #𝐳 = $J."); end
        vars = [x; y]

    elseif (R < J && setxy == "y") || (setxy == "x" && (R >= I && R < J)); # linear poly in y possible
        setxy = "y"
        @polyvar y_var[1:R]; y = Array{Term{true, ComplexF64}, 1}(randn(ComplexF64, J));
        y[Y_combos[Ycomb_index]] = y_var;
        x = randn(I);
        if verbose && it == 1; println("Linear in z: #𝐲 = 0, #𝐳 = $R = r."); end
        vars = y_var;


    elseif (R < I && setxy == "x") || (setxy == "y" && (R >= J && R < I)); # linear polynomial system possible in x
        setxy = "x"
        @polyvar x_var[1:R];
        x = Array{Term{true, ComplexF64}, 1}(randn(ComplexF64, I));
        x[X_combos[Xcomb_index]] = x_var;
        y = randn(J);
        if verbose && it == 1; println("Linear in y: #𝐲 = $R = r, #𝐳 = 0."); end
        vars = x_var


    elseif (setxy == "xy" && R<=I) || (setxy == "yx" && R >= J && R <= I) # required to set both x and I
        setxy = "xy"
        @polyvar x_var[1:R] y_var
        y = Array{Term{true, ComplexF64}, 1}(randn(ComplexF64, J));
        x = Array{Term{true, ComplexF64}, 1}(randn(ComplexF64, I));
        x[X_combos[Xcomb_index]] = x_var;
        y[1] = y_var;
        add_eqs = 1;
        if verbose && it == 1; println("yz mode: #𝐲 = $R, #𝐳 = 1."); end
        vars = [x_var; y_var]

    elseif (setxy == "yx" && R<=J) || (setxy == "xy" && R>=I && R<=J); # required to set both x and I
        setxy = "yx"
        @polyvar y_var[1:R] x_var
        y = Array{Term{true, ComplexF64}, 1}(randn(ComplexF64, J));
        x = Array{Term{true, ComplexF64}, 1}(randn(ComplexF64, I));
        y[Y_combos[Ycomb_index]] = y_var;
        x[1] = x_var;
        add_eqs = 1;
        if verbose && it == 1; println("zy mode: #𝐲 = 1, #𝐳 = $R = r."); end
        vars = [y_var; x_var]

    elseif (setxy == "xypar" && R<=I) || (setxy == "yxpar" && R >= J && R <= I); # required to set both x and I
        setxy = "xypar"
        @polyvar x_var[1:R] y_var t[1:I-R+J-1]
        x = [x_var; t[1:I-R]]
        y = [y_var; t[I-R+1:end]]
        add_eqs = 1;
        vars = [x_var; y_var]
        if verbose && it == 1; println("xy parameter: nb_x = $R, nb_y = 1."); end


    elseif (setxy == "yxpar" && R<=J) || (setxy == "xypar" && R>=I && R<=J); # required to set both x and I
        setxy = "yxpar"
        @polyvar y_var[1:R] x_var t[1:J-R+I-1]
        y = [y_var; t[1:J-R]]
        x = [x_var; t[J-R+1:end]]
        add_eqs = 1;
        if verbose && it == 1; println("yx parameter: nb_x = 1, nb_y = $R = R"); end
        vars = [y_var; x_var]

    elseif R == I && (setxy == "xypar" || setxy == "yxpar")
        setxy = "xypar"
        @polyvar x_var[1:R] y_var t[1:I-R+J-1]
        x = x_var
        y = [y_var; t]
        add_eqs = 1;
        vars = [x_var; y_var]
        if verbose && it == 1; println("r == n₂: #𝐲 = $R, #𝐳 = 1."); end

    elseif R == J && (setxy == "xypar" || setxy == "yxpar")
        setxy = "yxpar"
        @polyvar y_var[1:R] x_var t[1:J-R+I-1]
        y = y_var
        x = [x_var; t]
        add_eqs = 1;
        if verbose && it == 1; println("R == n₃: #𝐲 = 1, #𝐳 = $R = r."); end
        vars = [y_var; x_var]

    elseif R == I # set nonlinear poly such that there is only one y variable
        setxy = "xy";
        @polyvar x_var[1:R] y_var
        y = Array{Term{true, ComplexF64}, 1}(randn(ComplexF64, J));
        x = Array{Term{true, ComplexF64}, 1}(randn(ComplexF64, I));
        x[X_combos[Xcomb_index]] = x_var; y[1] = y_var;
        add_eqs = 1;
        if verbose && it == 1; println("r == n₂: #𝐲 = $R, #𝐳 = 1."); end
        vars = [x_var; y_var]

    elseif R == J # set nonlinear poly such that here is only one x variable
        setxy = "yx";
        @polyvar  y_var[1:R] x_var
        y = Array{Term{true, ComplexF64}, 1}(randn(ComplexF64, J));
        x = Array{Term{true, ComplexF64}, 1}(randn(ComplexF64, I));
        y[Y_combos[Ycomb_index]] = y_var; x[1] = x_var;
        add_eqs = 1;
        if verbose && it == 1; println("r == n₃: #𝐲 = 1, #𝐳 = $R."); end
        vars = [y_var; x_var]

    else
        @assert false "NO STRATEGY FOR POLYSYS DIMENSIONS."
    end

    return x, y, t, add_eqs, rm_eqs, setxy, vars
end

"""
# `dual_deparsevars`

Put the solutions 𝛇 of the polynomial system at the right place in 𝐮 and 𝐯. Can be interpreted as the inverse function of `dual_constructvars`.

!!! note Still uses an old naming convention internally.

"""
function dual_deparsevars(n₂, n₃, r, 𝐯, 𝐮, 𝛇, setyz, X_combos, Y_combos, Xcomb_index, Ycomb_index, it, verbose)

    # rename to old naming convention
    I = n₂;
    J = n₃;
    R = r;
    x = 𝐯;
    y = 𝐮;
    xy_nrank1 = 𝛇
    setxy = setyz


    # solutioin lists
    x₁_list = Array{ComplexF64, 1}[];
    y₁_list = Array{ComplexF64, 1}[];
    nrank1 = Array{ComplexF64, 1}[];

    for i ∈ 1:length(xy_nrank1)

        xy = conj(xy_nrank1[i]);

        if R > I && R > J && R < I+J;
            push!(y₁_list, xy[I+1:end])
            push!(x₁_list, xy[1:I])
            if verbose && it == 1 && i == 1; println("Parse case: both y and z necessary."); end

        elseif R >= I + J
            push!(y₁_list, xy[I+1:end])
            push!(x₁_list, xy[1:I])
            if verbose && it == 1 && i == 1; println("Parse case: square or overdetermined system"); end

        elseif setxy == "x";
            push!(y₁_list , y)
            x[X_combos[Xcomb_index]] = xy;
            push!(x₁_list, x);
            if verbose && it == 1 && i == 1; println("Parse case: y setting"); end

        elseif setxy == "y";
            y[Y_combos[Ycomb_index]] = xy;
            push!(y₁_list, y);
            push!(x₁_list, x)
            if verbose && it == 1 && i == 1; println("Parse case: z setting"); end

        elseif setxy == "xy"
            push!(y₁_list,  [xy[R+1] ; y[2:end]])
            x[X_combos[Xcomb_index]] = xy[1:R];
            push!(x₁_list, x);
            if verbose && it == 1 && i == 1; println("Parse case: yz setting"); end

        elseif setxy == "yx"
            push!(x₁_list,  [xy[R+1] ; x[2:end]])
            y[Y_combos[Ycomb_index]] = xy[1:R];
            push!(y₁_list, y);
            if verbose && it == 1 && i == 1; println("Parse case: zy setting"); end

        elseif setxy == "xypar" || setxy == "yxpar";
            push!(y₁_list, xy[I+1:end])
            push!(x₁_list, xy[1:I])
            if verbose && it == 1 && i == 1; println("Parse case: both y and z necessary."); end
        end

        push!(nrank1, kron( y₁_list[i] , x₁_list[i] ))
    end
    return nrank1, y₁_list, x₁_list
end

"""
# `dual_filtsols`

Filter all solutions in 𝐧_list such that

1. their residual is low enough
2. they are not trivial
3. if added to the basis N, the vectors are independent enough.
"""
function dual_filtsols(E, r, 𝐧_list, 𝐯_list, 𝐮_list, N, UV, full;
        independent_threshold = 1e-5,
        residual_threshold = 1e-10,
        triviality_threshold = 1e-4
    )

    # iterate over solutions
    for i ∈ 1:length(𝐧_list)
        𝐧 = 𝐧_list[i];
        𝐯, 𝐮 = 𝐯_list[i], 𝐮_list[i];

        # proposed solution
        𝐯𝐮_ = Array{ComplexF64, 1}[];
        push!(𝐯𝐮_, 𝐯);
        push!(𝐯𝐮_, 𝐮);

        if maximum(abs(𝐧'*E[:, j]/norm(𝐧)) for j ∈ 1:r) <= residual_threshold # residual test
            if norm(𝐧) > triviality_threshold # triviality test
                push!(N, 𝐧)
                push!(UV, 𝐯𝐮_)
                N_mat = reduce(hcat, N);
                if full == false && svd(N_mat).S[length(N)] < independent_threshold  # independent test
                    pop!(N);
                    pop!(UV);
                end
            end
        end
    end
end





function dual_strategyinit(strategy, dims, r)


    # unpack
    n₁, n₂, n₃ = dims;

    if strategy == "always_xy"; setxy = "xy"; index_rep = 1;
    elseif strategy == "always_yx"; setxy = "yx"; index_rep = 1;
    elseif strategy == "mixed"; setxy = "x"; index_rep = 3;
    elseif strategy == "mixed_last_xy"; setxy = "x"; index_rep = 2;
    elseif strategy == "x_last_xy"; setxy = "x"; index_rep = 1;
    elseif strategy == "y_last_xy"; setxy = "y"; index_rep = 1;
    elseif strategy == "mixed_last_yx"; setxy = "x"; index_rep = 2;
    elseif strategy == "x_last_yx"; setxy = "x"; index_rep = 1;
    elseif strategy == "x_last_xypar"; setxy = "x"; index_rep = 1;
    elseif strategy == "y_last_yxpar"; setxy = "y"; index_rep = 1;
    elseif strategy == "y_last_yx"; setxy = "y"; index_rep = 1;
    elseif strategy == "optimal";
        if (r < n₂) || (r < n₃)
            if n₂ >= n₃
                strategy = "x_last_xy"; setxy = "x"; index_rep = 1;
            else
                strategy = "y_last_yx"; setxy = "y"; index_rep = 1;
            end
        else
            strategy = "x_last_xy"; setxy = "x"; index_rep = 1;
        end
    else
        @assert false "strategy could not be parsed"
        setxy = "x"; index_rep = 1;
    end

    return setxy, index_rep, strategy

end

function dual_strategyupdate(strategy, setxy, it, nb_found)
    if strategy == "mixed";
        if setxy == "x"; setxy = "y";
        elseif setxy == "y"; setxy = "xy";
        elseif setxy == "xy"; setxy = "y";
        end
    elseif strategy == "mixed_last_xy";
        if setxy == "x"; setxy = "y";
        elseif setxy == "y"; setxy = "x";
        end
        if it-nb_found > 1; setxy = "xy"; end
    elseif strategy == "x_last_xy" || strategy == "y_last_xy" ;
        if it-nb_found > 1; setxy = "xy"; end
    elseif strategy == "x_last_xypar" || strategy == "y_last_xypar" ;
        if it-nb_found > 1; setxy = "xypar"; end
    elseif strategy == "mixed_last_yx";
        if setxy == "x"; setxy = "y";
        elseif setxy == "y"; setxy = "x";
        end
        if it-nb_found > 1; setxy = "yx"; end
    elseif strategy == "x_last_yx" || strategy == "y_last_yx" ;
        if it-nb_found > 1; setxy = "yx"; end
    elseif strategy == "x_last_yxpar" || strategy == "y_last_yxpar" ;
        if it-nb_found > 1; setxy = "yxpar"; end
    end

    return setxy
end

function dual_constructpolysys(P, Q, E::Array{d_type, 2}, 𝐯, 𝐮, r, dims, setxy, add_eqs, rm_eqs) where d_type

    # unpack
    n₁, n₂, n₃ = dims

    # rename
    x = 𝐯
    y = 𝐮
    I = n₂;
    J = n₃;
    A = E;
    R = r;

    # set u in outer scope
    u = nothing

    # set size of polynomial system
    P = P[1:R+add_eqs]

    # bias for linear equations (𝐜'𝐮 + randn(ComplexF64)+ β)
    β = 5.;

    if setxy != "full"

        for i ∈ 1:R-rm_eqs;
            P[i] = dot(kron(y, x), A*Q[:,i]);
        end

        if setxy == "fillpar"
            @polyvar u[1:add_eqs*I]
            for i ∈ R+1:R+add_eqs;
                P[i] = dot(x, u[I*(i-1-R)+1:I*(i-R)]) - randn(ComplexF64) + β;
            end
        elseif setxy == "fill"
            for i ∈ R+1:R+add_eqs;
                P[i] = dot(x, randn(I)) - randn(ComplexF64) + β;
            end
        elseif setxy == "yx" || setxy == "x"
            for i ∈ R+1:R+add_eqs;
                P[i] = dot(y, randn(J)) - randn(ComplexF64) + β;
            end
        elseif setxy == "xy" || setxy == "y"
            for i ∈ R+1:R+add_eqs;
                P[i] = dot(x, randn(I)) - randn(ComplexF64) + β;
            end

        elseif setxy == "yxpar"
            @polyvar u[1:add_eqs*J]
            for i ∈ R+1:R+add_eqs;
                P[i] = dot(y, u[J*(i-1-R)+1:J*(i-R)]) - randn(ComplexF64) + β;
            end
        elseif setxy == "xypar"
            @polyvar u[1:add_eqs*I]
            for i ∈ R+1:R+add_eqs;
                P[i] = dot(x, u[I*(i-1-R)+1:I*(i-R)]) - randn(ComplexF64) + β;
            end
        else
            @assert false "parse error"
        end


    else
        P = zeros(Polynomial{true, ComplexF64}, I + J);
        for i ∈ 1:I+J-2
            P[i] = dot(kron(y, x), A*Q[:,i]);
        end
        P[I+J-1] = dot(y, randn(J)) - randn() + 5.;
        P[I+J] = dot(x, randn(I)) - randn() + 5.;
    end


    return P, u
end

"""
# `dual_paramsolve`

The polynomial system is parametrised in t and u. This method first calculates
a solution set for a random t_0 and u_0 after which it tracks those solutions to
    random t_1 and u_1, then it tracks those to t_2 and u_2 and so on.
"""
function dual_paramsolve(P, x, y, t, u, dims, R, cur_rank, setxy;
        max_sols = 500, # number of solutins to track
        nb_loc = 10
    )

    # unpack but new convention
    K, I, J = dims

    # random start and target parameters
    #   - t represents the variables which are eliminated
    #   - u represents the affine equations
    t₀ = randn(ComplexF64, length(t))
    u₀ = randn(ComplexF64, length(u))
    t₁ = randn(ComplexF64, length(t))
    u₁ = randn(ComplexF64, length(u))

    # startsystem
    startsys = subs(P, [t; u]=>[t₀; u₀])

    # container
    sols = Array{ComplexF64, 1}[]

    # easy system with solutions, we only need a few so therefor we dive deeper in HomotopyContinuation.jl
    tracker, init_sols = HomotopyContinuation.pathtracker_startsolutions(startsys)

    init_sols = HomotopyContinuation.collect_startsolutions(init_sols);

    start_sol = zeros(length(x)+length(y))

    # container
    start_sols = Array{Array{ComplexF64, 1}, 1}[];

    it = 1;

    # get some solutions for startsys by tracking from the 'easy' system to startsys
    while it < max_sols && it < length(init_sols)

        # initial solution of the easy system
        init_sol = init_sols[it]

        # track this solution to the startsystem
        start_sol = [solution(HomotopyContinuation.track(tracker, init_sol))]

        # if the tracking was succesful, save thes solutions in sols
        if PathResult(tracker, init_sol).return_code==:success
            push!(start_sols, start_sol)
            if setxy == "fillpar"
                push!(sols, [start_sol[1]; t₀]);
            elseif setxy == "xypar"
                push!(sols, [start_sol[1][1:R]; t₀[1:I-R]; start_sol[1][R+1:end]; t₀[I-R+1:end]])
            elseif setxy == "yxpar"
                push!(sols, [start_sol[1][1]; t₀[1:I-1]; start_sol[1][2:end]; t₀[I:end]])
            else
                @assert false "parse error"
            end
            break;
        end
        it += 1;
    end

    @polyvar θ
    startsys = subs(P, [t; u]=>[t₀; u₀])
    targsys = subs(P, [t; u]=>[t₁; u₁])
    parsys = θ .* targsys + (1 - θ) .* startsys

    # path tracker for the parametrised system

    # activate to reduce the need to instantiate a new path tracker every outer loop
    #tracker = pathtracker(P, parameters=vcat(t, u), p₀=vcat(t₁, u₁), p₁=vcat(t₀, u₀))

    # activate to reduce the number of parameters but a new pathtracker has to be instanciated every iteration
    tracker = pathtracker(parsys, parameters=[θ], p₀=[1.], p₁=[0.])

    # set scope
    targ_sol = nothing

    # track all start solutions to different location in parameter space
    for i ∈ 1:nb_loc # iterate over the locations
        for j ∈ 1:length(start_sols) # iterate over the start solutions

            # select start solution
            start_sol = start_sols[j]

            # activate to reduce the need to instantiate a new path tracker every outer loop
            #res = HomotopyContinuation.track(tracker, start_sol[1] , start_parameters = vcat(t₀, u₀), target_parameters = vcat(t₁, u₁))

            # activate to reduce the number of parameters but a new pathtracker has to be instanciated every iteration
            res = HomotopyContinuation.track(tracker, start_sol[1] , start_parameters = [0.], target_parameters = [1.])

            # store the solution
            targ_sol = [solution(res)]
            if setxy == "fillpar"
                push!(sols, [start_sol[1]; t₀]);
            elseif setxy == "xypar"
                push!(sols, [start_sol[1][1:R]; t₁[1:I-R]; start_sol[1][R+1:end]; t₁[I-R+1:end]])
            elseif setxy == "yxpar"
                push!(sols, [start_sol[1][1]; t₁[1:I-1]; start_sol[1][2:end]; t₁[I:end]])
            else
                @assert false "parse error"
            end
            start_sols[j] = targ_sol;

        end

        # set a new location in paremeter space
        t₀ = t₁
        u₀ = u₁
        t₁ = randn(ComplexF64, length(t))
        u₁ = randn(ComplexF64, length(u))
        startsys = subs(P, [t; u]=>[t₀; u₀])
        targsys = subs(P, [t; u]=>[t₁; u₁])
        parsys = θ .* targsys + (1 - θ) .* startsys
        tracker = pathtracker(parsys, parameters=[θ], p₀=[1.], p₁=[0.])

    end

    return sols

end


"""
# `dual_solvepolysys`

Wrapper which based on given settings selects the right solver. It can call four different methods

- `dual_paramsolve`: track a set of solutions to different locations in parameter space.
- `HomotopyContinuation.solve()`: full degree homotopy.
- `HomotopyContinuation.solve(…, variable_groups= …)`: full degree homotopy with m-homogeneous system.
- `linpolysolve`: solves a linear polynomial system.

"""
function dual_solvepolysys(P, x, y, t, u, vars, setxy, R, dims, measure_condition, poly_cond, lin_cond, cur_rank)

    # unpack under new naming convention
    K, I, J = dims

    # parameter homotopy
    if setxy == "fillpar" || setxy == "xypar" || setxy == "yxpar"
        xy_nrank1 = dual_paramsolve(P, x, y, t, u, dims, R, cur_rank, setxy);

    elseif setxy == "over"
        xy_nrank1 = solutions(HomotopyContinuation.solve(P, variable_groups = [x, y], show_progress=true));

    elseif setxy == "fill"
        xy_nrank1 = solutions(HomotopyContinuation.solve(P, show_progress=true));


    elseif R > I || setxy == "xy" || setxy == "yx" || R > J;
        xy_nrank1 = solutions(HomotopyContinuation.solve(P, show_progress=true));
        if measure_condition
            κmax = maximum(KuoLeeCondUtils.polynomial_condition(P, xy_nrank1, vars; verbose = true))
            push!(poly_cond, κmax)
        end

    elseif setxy == "x" || setxy == "y";
        if measure_condition
            xy_nrank1_, κ = linpolysolve(P, R; measure_condition = true)
            xy_nrank1 = [xy_nrank1_];
            push!(lin_cond, κ);
        else
            xy_nrank1 = [linpolysolve(P, R)];
        end
    else
        @assert false "polysys setting could not be parsed"
    end

    return xy_nrank1
end

"""
# linpolysolve

Solves a system of degree one polynomials.
"""
function linpolysolve(P, nb_vars; measure_condition = false)

    A = zeros(ComplexF64, length(P), nb_vars);
    b = zeros(ComplexF64, length(P));

    for i ∈ 1:length(P)
        p = P[i]
        @assert length(p.a) == nb_vars + 1;
            A[i, :] =  p.a[1:end-1];
            b[i] = -p.a[end];
    end

    x = A \ b
    if norm(A*x-b)/norm(b) > 1e-10
        println("ERROR, system ill-solved")
    end
    if measure_condition
        return x, cond(A);
    else
        return x
    end
end








"""
# combis

Experimental function to generate some combinatorics. Creates different combinations of
`select_nb` elements out of `select_list`. The method stops if it can find no new samples or if the number of combinations reaches `nb_samples`.
"""
function combis(select_list, select_nb, nb_samples)

    space = 1;
    list_length = length(select_list)
    samples = Array{Int64}[]

    if select_nb > list_length
        return [select_list]
    end

    while space < list_length/2 + 2 && length(samples) < nb_samples
        i = 1;
        while i <= list_length
            index_list = Int64[]
            for index ∈ collect(i:space:i-1+select_nb*space)
                while index > list_length
                    index = index-list_length
                end
                push!(index_list, index)
            end
            if maximum(count(el .== index_list) for el ∈ index_list) == 1
                push!(samples, select_list[index_list])
            end
            i += 1
        end
        space += 1;
    end

    return samples

end


function translate_prim_dual(Y₀::Array{d_type, 2}, Z₀::Array{d_type, 2}, dims, r;
        verbose = false,
        maxit = 5000,
        sigma_threshold = 1e-3,
        non_zero_threshold = 1e-10,
        ) where d_type

    # unpack
    n₁, n₂, n₃ = dims

    # container for the solutions
    UV₀ = Array{Array{ComplexF64, 1}, 1}[]

    @assert r == size(Y₀)[2] "Dimensions of Y₀ wrong."
    @assert r == size(Z₀)[2] "Dimensions of Z₀ wrong."

    nb_vecs = r

    # loop counter
    it = 0

    while length(UV₀) < n₂*n₃-r && it < maxit

        it += 1;

         max_y_it = rand(0:n₂)
         max_z_it = rand(0:n₃)

        # start from y
        y_it = 0

        # set 𝔍_y such that null(Y₀[:, 𝔍_y]) is positive dimensional
        𝔍_y = [rand(1:nb_vecs)];
        while size(nullspace(Array{ComplexF64, 2}(Y₀[:, 𝔍_y])'))[2] > 1 && y_it < max_y_it
            push!(𝔍_y, rand(1:nb_vecs))
            y_it += 1
        end

        y_null = nullspace(Y₀[:, 𝔍_y]')

        # select random vector in nullspace
        y_null = y_null[:, rand(1:size(y_null)[2])]

        𝔍_z = filter( nonzero, (abs.(Y₀'*y_null) .> non_zero_threshold)[:, 1] .* collect(1:nb_vecs))

        # if we have no conditions left just select a random vector
        if length(𝔍_z) == 0
            z_null = randn(ComplexF64, n₃, 1);
        # otherwise select a random vector in the corresponding nullspace if possible
        else
            z_null = nullspace(Z₀[:, 𝔍_z]')
            if size(z_null)[2] > 0
                z_null = z_null[:, [rand(1:size(z_null)[2])]]
            end
        end;

        # if we found a combo (𝐯, 𝐮) try to add it to the existing base of solutions
        if size(z_null)[2] > 0
            𝐮𝐯 = Array{ComplexF64, 1}[];
            push!(𝐮𝐯, Array{ComplexF64, 1}(y_null[:, 1]))
            push!(𝐮𝐯, Array{ComplexF64, 1}(z_null[:, 1]))
            push!(UV₀, 𝐮𝐯);
            V = reduce(hcat, [UV₀[i][1] for i ∈ 1:length(UV₀)]);
            U = reduce(hcat, [UV₀[i][2] for i ∈ 1:length(UV₀)]);
            N_ = reduce(hcat, [kron(UV₀[i][2], UV₀[i][1]) for i ∈ 1:length(UV₀)]);
            N_svd = svd(N_)
            if N_svd.S[end]/N_svd.S[1] < sigma_threshold
                pop!(UV₀)
            end
        end

        # start from z
        z_it = 0

        𝔍_z = [rand(1:nb_vecs)];

        while size(nullspace(Array{ComplexF64, 2}(Z₀[:, 𝔍_z])'))[2] > 1 && z_it < max_z_it
            push!(𝔍_z, rand(1:nb_vecs))
            z_it += 1
        end

        z_null = nullspace(Z₀[:, 𝔍_z]')

        z_null = z_null[:, rand(1:size(z_null)[2])]

        𝔍_y = filter( nonzero, (abs.(Z₀'*z_null) .> non_zero_threshold)[:, 1] .* collect(1:nb_vecs))


        if length(𝔍_y) == 0
            y_null = randn(ComplexF64, n₂, 1);
        else
            y_null = nullspace(Y₀[:, 𝔍_y]')
            if size(y_null)[2] > 0
                y_null = y_null[:, [rand(1:size(y_null)[2])]]
            end
        end;


        if size(y_null)[2] > 0
            𝐮𝐯 = Array{ComplexF64, 1}[];
            push!(𝐮𝐯, Array{ComplexF64, 1}(y_null[:, 1]))
            push!(𝐮𝐯, Array{ComplexF64, 1}(z_null[:, 1]))
            push!(UV₀, 𝐮𝐯);
            V = reduce(hcat, [UV₀[i][1] for i ∈ 1:length(UV₀)]);
            U = reduce(hcat, [UV₀[i][2] for i ∈ 1:length(UV₀)]);
            N_ = reduce(hcat, [kron(UV₀[i][2], UV₀[i][1]) for i ∈ 1:length(UV₀)]);
            N_svd = svd(N_)
            if N_svd.S[end]/N_svd.S[1] < sigma_threshold
                pop!(UV₀)
            end
        end

    end
    if verbose
        println("search: found $(length(UV₀)) of $(n₂*n₃-r) vectors in $it iterations")
    end

    if length(UV₀) >= n₂*n₃-r;
        UV₀ = UV₀[1:n₂*n₃-r]
        success = 1;
    else
        success = 0;
    end
    return success, UV₀
end



"""
    # dual_track_single

Track all solutions of the dual start problem to the target problem:

- start condition: U₀ ⊙ V₀ ⟂ E₀,
- target condition: U₁ ⊙ V₁ ⟂ E₁.

## Inputs

- `UV₀`: container for the start solutions.
- `E₀`: as explained above.
- `E₁`: as explained above.
- `dims`: (n₁, n₂, n₃) dimensions of the tensor. E₀ and E₁ have dimension
    n₂n₃×r.
- `r`: rank of the tensor.

## Outputs

- `success`: flag
- `UV₁`: container for target solutions

"""
function dual_track_single(UV₀, E₀::Array{d_type, 2}, E₁::Array{d_type, 2}, dims, r) where d_type

    # unpack
    n₁, n₂, n₃ = dims

    # number of basis vectors for N₀
    nb_vecs = length(UV₀);

    # number of variables
    nb_vars = n₂ + n₃;

    # number of equations
    nb_eqs = r;

    # number of dehomogenising equations needed
    nb_lin_eqs = nb_vars - nb_eqs;

    # create variables
    @polyvar 𝐯[1:n₂] 𝐮[1:n₃] t

    # matrix and vector to store linear equations
    C = zeros(d_type, nb_lin_eqs, n₂+n₃)
    𝛂 = zeros(d_type, nb_lin_eqs)

    defect_sols = Int[]

    # container for solutions
    UV₁ = Array{Array{ComplexF64, 1}, 1}[]

    # set tracker to outer scope
    tracker = nothing

    # introduce randomness in the basis
    Q, _ = qr(randn(ComplexF64, r, r))

    # polynomial system
    𝐩 = zeros(Polynomial{true, d_type}, r+nb_lin_eqs);

    γ₁ = exp(1im*randn());
    γ₀ = exp(1im*randn());

    for j ∈ 1:r;
        𝐩[j] = dot(kron(𝐮, 𝐯), (t .* (γ₀ .* E₀) + (1. - t) .* (γ₁ .* E₁) )*Q[:,j]);
    end

    # track all solution paths
    for i ∈ 1:n₂*n₃-r

        # unpack start solution
        𝐯₀ = UV₀[i][1];
        𝐮₀ = UV₀[i][2];

        # make sure linear equations match this start solution
        for j ∈ r+1:r+nb_lin_eqs
            C[j-r, :] = randn(ComplexF64, n₂+n₃);
            𝛂[j-r] = dot(conj(C[j-r,:]), conj(vcat(𝐯₀, 𝐮₀)))
            𝐩[j] = dot(conj(C[j-r, :]), vcat(𝐯, 𝐮)) - 𝛂[j-r]
        end

        𝐩_start =  [subs(𝐩[i], vcat(𝐯, 𝐮, t) => vcat(conj(𝐯₀), conj(𝐮₀), 1.)) for i ∈ 1:length(𝐩)]
        𝐩_start_res = norm(𝐩_start)
        @assert 𝐩_start_res/norm([𝐯₀; 𝐮₀]) < 1e-4 "Residual of start system too high."

        # initialise tracker
        if i == 1
            pars = [t]
            pars₀ = [1.]
            pars₁ = [0.]
            tracker = pathtracker(𝐩, parameters=pars, p₀=pars₁, p₁=pars₀)
        end

        pars₀ = [1.]
        pars₁ = [0.]
        res = HomotopyContinuation.track(tracker, [conj(𝐯₀); conj(𝐮₀)] , start_parameters = pars₀, target_parameters = pars₁)

        𝐮𝐯 = solution(res)
        𝐮𝐯₁ = Array{d_type, 1}[];
        push!(𝐮𝐯₁, Array{d_type, 1}(𝐮𝐯[1:n₂]))
        push!(𝐮𝐯₁, Array{d_type, 1}(𝐮𝐯[n₂+1:end]))
        push!(UV₁, 𝐮𝐯₁);

        if res.return_code != :success
            push!(defect_sols, i)
        end

    end

    success = 1;

    if length(defect_sols) > 0;
        success = 0;
        println("Warning: during path tracking some solutions encountered an error. $(defect_sols)")
    end
    return success, UV₁
end


end # DualUtils






# function dual_paramsolve_old(P, x, y, t, u, dims, R, cur_rank, setxy;
#     max_sols = 100
#     )
#
#     # unpack with new naming convention
#     K, I, J = dims
#
#     t₀ = randn(ComplexF64, length(t))
#     u₀ = randn(ComplexF64, length(u))
#     t₁ = randn(ComplexF64, length(t))
#     u₁ = randn(ComplexF64, length(u))
#
#     startsys = subs(P, [t; u]=>[t₀; u₀])
#
#     sols = Array{ComplexF64, 1}[]
#
#     tracker, init_sols = HomotopyContinuation.pathtracker_startsolutions(startsys)
#
#     start_sol = zeros(length(x)+length(y))
#
#     it = 1;
#
#     while it < max_sols && it < length(init_sols)
#         init_sol = init_sols[it]
#         start_sol = [solution(HomotopyContinuation.track(tracker, init_sol))]
#         if PathResult(tracker, init_sol).return_code==:success
#             if setxy == "fillpar"
#                 push!(sols, [start_sol[1]; t₀]);
#             elseif setxy == "xypar"
#                 push!(sols, [start_sol[1][1:R]; t₀[1:I-R]; start_sol[1][R+1:end]; t₀[I-R+1:end]])
#             elseif setxy == "yxpar"
#                 push!(sols, [start_sol[1][1]; t₀[1:I-1]; start_sol[1][2:end]; t₀[I:end]])
#             else
#                 @assert false "parse error"
#             end
#             break;
#         end
#         it += 1;
#     end
#
#
#     tracker = pathtracker(P, parameters=vcat(t, u), p₁=vcat(t₁, u₁), p₀=vcat(t₀, u₀))
#
#     for i ∈ 1:(I*J-R-cur_rank)*20
#
#         res = HomotopyContinuation.track(tracker, start_sol[1] , start_parameters = vcat(t₀, u₀), target_parameters = vcat(t₁, u₁))
#         targ_sol = [solution(res)]
#         if setxy == "fillpar"
#             push!(sols, [start_sol[1]; t₀]);
#         elseif setxy == "xypar"
#             push!(sols, [start_sol[1][1:R]; t₁[1:I-R]; start_sol[1][R+1:end]; t₁[I-R+1:end]])
#         elseif setxy == "yxpar"
#             push!(sols, [start_sol[1][1]; t₁[1:I-1]; start_sol[1][2:end]; t₁[I:end]])
#         else
#             @assert false "parse error"
#         end
#         t₀ = t₁
#         u₀ = u₁
#         t₁ = randn(ComplexF64, length(t))
#         u₁ = randn(ComplexF64, length(u))
#         start_sol = targ_sol;
#     end
#
#     return sols
#
# end
