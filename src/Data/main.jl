"""
`AbstractData` contains the observations on which the model is based.

Should implement `getindex`, `length`, `nalt`, `nobs` and `explanatoryLength`
"""
abstract type AbstractData end
#honestly, not sure if we will ever develop the code for the panel case
#abstract type AbstractPanelData <: AbstractData end

include("LinedObs.jl")
#MatrixObs not fully tested yet.
#include("MatrixObs.jl")
