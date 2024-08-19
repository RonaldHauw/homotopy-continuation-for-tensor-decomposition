push!(LOAD_PATH,"../src/")
push!(LOAD_PATH,"../benchmarks/")

using Documenter
using HomotopyContinuationForTensorDecomposition
using KuoLee
using Dual
using Utils
using KuoLeeUtils
using DualUtils
using Benchmarks

makedocs(sitename="HomotopyContinuationForTensorDecomposition")
