@enum isLinear Linear NotLinear

#utilities should define gradient, and hessian functions. It is best behavior to warn the user if the hessian of
#a linear utility is called 
#the parametric type isl in AbstractUtility{isl} define if the utility is linear or not.
abstract type AbstractUtility{isl} end


include("LogitUtility/main.jl")
include("MixedLogitUtility/main.jl")