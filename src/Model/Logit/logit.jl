@doc raw"""
    computePrecomputedVal(beta::AbstractVector{T}, obs::AbstractObs, Type{UTI})

Returns an array containing the probabilities associated to every alternative.
"""
function computePrecomputedVal(::Type{UTI}, obs::AbstractObs, beta::AbstractVector{T})::Array{T, 1} where {T, UTI}
    # compute utility value for all alternative
    uti = computeUtilities(UTI, obs, beta)
    uti .-= maximum(uti)
    # compute exp(utilities)
    map!(exp, uti , uti)
    s = sum(uti)
    return uti / s

end

@doc raw"""
    logit(beta::AbstractVector{T}, obs::AbstractObs, Type{UTI}, precomputedValues::AbstractVector{T})

Returns the log-likelihood of the model on this observation.
"""
function logit(::Type{UTI}, obs::AbstractObs, beta::AbstractVector{T}; 
        precomputedValues::Vector{T} = computePrecomputedVal(UTI, obs, beta))::T where {T, UTI}
    return log(precomputedValues[choice(obs)])  
end

"""
    gradlogit(beta::AbstractVector{T}, obs::AbstractObs, Type{UTI}, precomputedValues::AbstractVector{T})


"""
function gradientLogit(::Type{UTI}, obs::AbstractObs, beta::AbstractVector{T};
        precomputedValues::Vector{T} = computePrecomputedVal(UTI, obs, beta))::Vector{T} where {T, UTI}
    n = length(beta)
    g = Array{T, 1}(undef, n)
    g[:] = gradient(UTI, obs, beta, choice(obs))
    for k in 1:length(precomputedValues)
        g[:] -= precomputedValues[k] * gradient(UTI, obs, beta, k)
    end
    return g
end
#Case with linear utility
function hessianLogit(::Type{UTI}, obs::AbstractObs, beta::AbstractVector{T};
        precomputedValues::AbstractVector{T} = computePrecomputedVal(UTI, obs, beta))::Matrix{T} where {T, UTI <: AbstractLogitUtility{Linear}}
    dim = length(beta)
    numberOfAlternatives = nalt(obs)
    H = zeros(T, dim, dim)
    gradS = sum(precomputedValues[k] * gradient(UTI, obs, beta, k) for k in 1:numberOfAlternatives)
    H[:, :]  += gradS * gradS'
    for k in 1:numberOfAlternatives
        H[:, :] -= precomputedValues[k] * gradient(UTI, obs, beta, k) * gradient(UTI, obs, beta, k)'
    end
    return H
end
#Case with non-linear utility
function hessianLogit(::Type{UTI}, obs::AbstractObs, beta::AbstractVector{T};
        precomputedValues::AbstractVector{T} = computePrecomputedVal(UTI, obs, beta))::Matrix{T} where {T, UTI <: AbstractLogitUtility{NotLinear}}
    dim = length(beta)
    numberOfAlternatives = nalt(obs)
    H = zeros(T, dim, dim)
    H[:, :] = hessian(UTI, obs, beta, choice(obs))

    gradS = sum(precomputedValues[k] * gradient(UTI, obs, beta, k) for k in 1:numberOfAlternatives)
    H[:, :]  += gradS * gradS'
    for k in 1:numberOfAlternatives
        H[:, :] -= precomputedValues[k] * gradient(UTI, obs, beta, k) * gradient(UTI, obs, beta, k)'
        H[:, :] -= precomputedValues[k] * hessian(UTI, obs, beta, k)
    end
    return H
end
#case with linear utility
function hessianLogitdotv(::Type{UTI}, obs::AbstractObs, beta::AbstractVector{T}, v::Vector; 
        precomputedValues::Vector{T} = computePrecomputedVal(UTI, obs, beta))::Vector{T} where {T, L, UTI <: AbstractLogitUtility{Linear}}
    dim = length(beta)
    numberOfAlternatives = nalt(obs)
    Hv = zeros(T, dim)
    gradS = sum(precomputedValues[k] * gradient(UTI, obs, beta, k) for k in 1:numberOfAlternatives)
    Hv[:]  += gradS * dot(gradS, v)
    for k in 1:numberOfAlternatives
        Hv[:] -= precomputedValues[k] * gradient(UTI, obs, beta, k) * dot(gradient(UTI, obs, beta, k), v)
    end
    return Hv
end
#case with nonlinearutility
function hessianLogitdotv(::Type{UTI}, obs::AbstractObs, beta::AbstractVector{T}, v::Vector; 
        precomputedValues::Vector{T} = computePrecomputedVal(UTI, obs, beta))::Vector{T} where {T, L, UTI <: AbstractLogitUtility{NotLinear}}
    dim = length(beta)
    numberOfAlternatives = nalt(obs)
    Hv = zeros(T, dim)
    H[:] = hessdotv(UTI, obs, beta, choice(obs), v)
    gradS = sum(precomputedValues[k] * gradient(UTI, obs, beta, k) for k in 1:numberOfAlternatives)
    Hv[:]  += gradS * dot(gradS, v)
    for k in 1:numberOfAlternatives
        Hv[:] -= precomputedValues[k] * gradient(UTI, obs, beta, k) * dot(gradient(UTI, obs, beta, k), v)
        Hv[:] -= precomputedValues[k] * hessdotv(UTI, obs, beta, k, v)
    end
    return Hv
end
