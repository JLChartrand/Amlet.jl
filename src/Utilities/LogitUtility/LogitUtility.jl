
abstract type AbstractLogitUtility{L} <: AbstractUtility{L} end

struct StandardLogitUtility <: AbstractLogitUtility{Linear} end
#return the utility of the i-th alternatives of observation obs with tastes beta
function u(::Type{StandardLogitUtility}, obs::AbstractObs, beta::AbstractVector, i::Integer)
    return dot(explanatory(obs, i), beta)
end
#return the gradient of the utility of the i-th alternatives of observation obs with tastes beta
function gradient(::Type{StandardLogitUtility}, obs::AbstractObs, beta::AbstractVector, i::Integer)
    return explanatory(obs, i)
end
#return the hessian utility of the i-th alternatives of observation obs with tastes beta
function hessian(::Type{StandardLogitUtility}, obs::AbstractObs, beta::AbstractVector{T}, i::Integer) where T
    @warn "Hessian of linear utility called"
    lb = length(beta)
    return zeros(T, lb, lb)
end
#return the hessian utility of the i-th alternatives of observation obs with tastes beta multiplied by
#a vector v
function hessiandotv(::Type{StandardLogitUtility}, obs::AbstractObs, beta::AbstractVector{T}, i::Integer, v::Vector) where T
    @warn "Hessian of linear utility called"
    lb = length(beta)
    return zeros(T, lb)
end


"""
    computeUtilities(::Type{UTI}, obs::ObsAsVector, beta::AbstractArray{T})

Computes utility value for every alternative in a Logit context -> returns an array.
"""
function computeUtilities(::Type{UTI}, obs::AbstractObs, beta::AbstractArray{T}) where {T, UTI <: AbstractLogitUtility}
    n = nalt(obs)
    ar = Array{T, 1}(undef, n)
    #for some reason, faster than [u(UTI, obs, beta, i) for i in 1:nalt(obs)]???
    for i in 1:n
        ar[i] = dot(explanatory(obs, i), beta)
    end
    return ar
end
function dimension(::Type{StandardLogitUtility}, s::AbstractData)
    n = explanatoryLength(s)
    return n 
end
