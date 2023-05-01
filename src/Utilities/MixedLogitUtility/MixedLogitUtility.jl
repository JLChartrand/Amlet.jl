#all mixed logit utilities are subtypes of AbstractMixedLogitUtility{L, GT}
#L is either Linear or NotLinear. GT is the type of gamma. 
#default ways to compute the gradients and hessian of the utility are given using ForwardDiff
abstract type AbstractMixedLogitUtility{L, GT} <: AbstractUtility{L} end
#compute the utilities of associated with all alternatives assuming tastes theta and gamma
function computeUtilities(::Type{TYPEU}, obs::AbstractObs, theta::AbstractVector, gamma::GT) where {L, GT, TYPEU <: AbstractMixedLogitUtility{L, GT}}
    return [u(TYPEU, obs, theta, gamma, i) for i in 1:nalt(obs)]
end
function gradient(::Type{TYPEU}, obs::AbstractObs, theta::AbstractVector, gamma::GT, i::Integer) where {L, GT, TYPEU <: AbstractMixedLogitUtility{L, GT}}
    return ForwardDiff.gradient(t -> u(TYPEU, obs, t, gamma, i), theta)
end
function hessian(::Type{TYPEU}, obs::AbstractObs, theta::AbstractVector, gamma::GT, i::Integer) where {L, GT, TYPEU <: AbstractMixedLogitUtility{L, GT}}
    @warn "Hessian of linear utility called"
    lt = length(theta)
    return zeros(Float64, lt, lt)
end
function hessian(::Type{TYPEU}, obs::AbstractObs, theta::AbstractVector, gamma::GT, i::Integer) where {GT, TYPEU <: AbstractMixedLogitUtility{NotLinear, GT}}
    return ForwardDiff.hessian(t -> u(TYPEU, obs, t, gamma, i), theta)
end

#if beta has the following form beta = f(theta, gamma), we call this type of utility parametric
#Distro is the distribution from which to draw gamma vectors. 
#f is the function that create beta from theta and gamma
#GT is the type of the random variable gamma returned from rand(Distro). Usually Vector{Float64}
abstract type AbstractLinearParametricMixedLogitUtility{Distro , f, GT} <: AbstractMixedLogitUtility{Linear, GT} end
const ALPMLU{Distro, f, GT} = AbstractLinearParametricMixedLogitUtility{Distro, f, GT}
#linear mixed logit utility. 
function linearUtilityInMixedLogit(obs::AbstractObs, beta::AbstractVector, i::Integer)
    #get the explanatory variables associated with obs and the i-th alternative.
    xi = explanatory(obs, i)
    return dot(xi, beta)
end
#return the utility of the i-th alternative of observation obs based on tastes distribution and gamma.
function u(::Type{TYPEU}, obs::AbstractObs, theta::AbstractVector, gamma::AbstractVector, 
        i::Integer) where {Distro, f, GT, TYPEU <: ALPMLU{Distro, f, GT}}
    beta = f(theta, gamma)
    return linearUtilityInMixedLogit(obs, beta, i)
end
function hessian(::Type{TYPEU}, obs::AbstractObs, theta::GT, gamma::GT, i::Integer) where {Distro, f, T, GT, TYPEU <: ALPMLU}
    @warn "Hessian of linear utility called"
    lt = length(theta)
    return zeros(Float64, lt, lt)
end
function computeUtilities(::Type{TYPEU}, obs::AbstractObs, theta::GT, gamma::GT) where {Distro, f, GT, TYPEU <: ALPMLU{Distro, f, GT}}
    beta = f(theta, gamma::GT)
    return [linearUtilityInMixedLogit(obs, beta, i) for i in 1:nalt(obs)]
end
function getGamma(::Type{TYPEU}, rng::AbstractRNG, n::Integer)::GT where {Distro, f, GT, TYPEU <: ALPMLU{Distro, f, GT}}
    d = Distro(n)
    return rand(rng, d)
end



