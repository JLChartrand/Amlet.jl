abstract type AmletModel{D <: AbstractData} <: AbstractNLPModel{Float64, Vector{Float64}} end
function nobs(mo::AmletModel)
    return nobs(mo.data)
end
include("Logit/main.jl")
#include("MixedLogit/main.jl")


function getchoice(mo::LogitModel{D, L, UTI}, beta::Vector; sample = 1:length(mo.data)) where {U, D, L, UTI}
    choices = [argmax(computeUtilities(UTI, mo.data[i], beta)) for i in sample]
end
function ratiorightchoice(mo::LogitModel{D, L, UTI}, beta::Vector; sample = 1:length(mo.data)) where {U, D, L, UTI}
    choicesmodel = getchoice(mo, beta; sample = sample)
    truechoice = choice.(mo.data)
    return count(iszero, choicesmodel - truechoice)
end