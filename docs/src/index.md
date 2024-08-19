# Homotopy Continuation For Tensor Decomposition
This is a short and rudimentary user guide for Homotopy Continuation For Tensor Decomposition.
This package comes with my master's thesis and detailed motivations for the algorithms available in this package can be found in the thesis text.

## Getting Started

### Installing the package
Since this is not an official package, it has to be installed manually.

First, load all the dependencies via Julia's package manager.

```julia
using Pkg

Pkg.activate("path/to/HomotopyContinuationForTensorDecomposition")

Pkg.instantiate()
```


Secondly, we should install HomotopyContinuationForTensorDecomposition. There are two ways to do this.

- Add ```HomotopyContinuationForTensorDecomposition.jl``` to Julia's `LOAD_PATH`.
```julia
push!(LOAD_PATH,"path/to/src/HomotopyContinuationForTensorDecomposition.jl")

using HomotopyContinuationForTensorDecomposition

```

- Use `include()`.
```julia
include("path/to/src/HomotopyContinuationForTensorDecomposition.jl")

using .HomotopyContinuationForTensorDecomposition

```


### A first example
The example below shows how to interact with the implemented algorithms.
```@example
using HomotopyContinuationForTensorDecomposition # hide
using LinearAlgebra: norm

dims = (43, 8, 7); # dimensions of the tensor
r = 5; # tensor rank

X, Y, Z, T = cpd_gen(dims, r); # generate a random tensor

Xcalc, Ycalc, Zcalc = cpd(T, dims, r);

Tcalc = cpd_gen(Xcalc, Ycalc, Zcalc);

norm(Tcalc - T)/norm(T) # backward relative error   

```

The implemented algorithms only work for third order tensors with dimensions (n₁, n₂, n₃) for which it holds that r < min(n₁, n₂n₃) and n₁ ≧ n₂ ≧ n₃.
The implemented methods are:

1. `kl_cpd` (cpd via the method described by Kuo, Lee [1]);
2. `kld_fd_cpd` (cpd via the dual problem by reducing the size of the square system);
3. `kld_pr_cpd` (cpd via the dual problem by a parameter homotopy);
4. `kl_lr_cpd` (Kuo and Lee's method optimised for low ranks);
5. `cpd` (wrapper which selects the optimal method from those above).

[1] _Y.-C. Kuo and T.-L. Lee. Computing the unique candecomp/parafac decomposition of
unbalanced tensors by homotopie method.
Linear Algebra and its applications, 556:238–264, 2018._

## Benchmarking
The file `benchmarks/Benchmark.jl` contains a small module to test the performance of the different algorithms.
Additionally, an interface with `Tensorlab2016`[2] is provided via the `MATLAB` package. The following four steps explain how to run some tests.



-  Set Julia's working directory to `../benchmarks`.

```julia
cd("path/to/HomotopyContinuationForTensorDecomposition/benchmarks")
```

-  Put a copy of Tensorlab2016 in `../benchmarks`.

-  Load the `Benchmarks` module.

```julia
include("Benchmark.jl")

using .Benchmarks
```

-  Call the function `run_benchmark()`.

```julia
run_benchmark().
```

This starts a command line interface through which it is possible to provide all the experiment settings. Should the options provided there be insufficient, experiments can be created manually
and passed as an argument to `run_benchmark`.

```julia

?Benchmarkinstance # show documentation for the benchmarkinstance

# every experiment is represented by a Benchmarkinstance
bmis = Benchmarkinstance[]

dims = (43, 8, 7);
r = 15;
model = 1;
s = 2.0;
c = 1.0;

push!(bmis, Benchmarkinstance(dims, r; model=model, s=s, c=c));

run_benchmark(bmis); # run the experiments in `bmis`
# ...

```

The experiment results will be saved in a `.csv` file in the folder `../data`. Before and
after the experiment `run_benchmark()` prints the exact file name. Additionally, the measurements are printed
during the experiment.

[2] _Vervliet N., Debals O., Sorber L., Van Barel M. and De Lathauwer L. Tensorlab 3.0, Available online, Mar. 2016. URL:_ https://www.tensorlab.net/


## CPD methods

```@autodocs
Modules = [HomotopyContinuationForTensorDecomposition, KuoLee, Dual]
Order   = [:function, :type]
```

## Benchmarks

```@autodocs
Modules = [Benchmarks]
Order   = [:function, :type]
```

## Helper functions

```@autodocs
Modules = [DualUtils, KuoLeeUtils, Utils]
Order   = [:function, :type]
```
