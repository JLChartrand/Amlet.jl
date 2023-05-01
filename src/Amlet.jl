"""
Another Machine Learning Estimation Tool (`AMLET`).

A few steps to folow:
- create data structure and utility function
- create LogitModel
Then, you will have access to the functions: 
"""
module Amlet

using ForwardDiff, LinearAlgebra, RDST, Random, NLPModels, Distributions

import Base.getindex
import Base.iterate
import Base.length
import Base.copy
import Base.*
import Base.display

include("Observation/main.jl")
include("Data/main.jl")
include("Utilities/main.jl")
include("Model/main.jl")
end # module
