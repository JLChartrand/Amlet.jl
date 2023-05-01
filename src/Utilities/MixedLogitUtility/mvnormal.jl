
module ModuleNormalTasteUtility
using LinearAlgebra, Distributions, Random

const GT = Vector{Float64}
function fdiag(theta::Vector{Float64}, gamma::GT)
    n = length(gamma)
    mu = @view theta[1:n]
    sig = @view theta[n+1:end]
    return mu + (sig .* gamma)
end
function utilitygraddiagnormal(xi::V, theta::Vector{Float64}, gamma::GT) where {V <: AbstractVector{Float64}}
    return [xi; xi.*gamma]
end
function fuppertriangular(theta::Vector{Float64}, gamma::GT)
    n = length(gamma)
    #this is a copy, SHOULD NOT be a view!!!
    mu = theta[1:n]
    result = mu
    sig = @view theta[n+1:end]
    current = 1
    for i in 1:n
        v1 = @view sig[current:current+n-i] 
        v2 = @view gamma[i:end]
        result[i] += dot(v1, v2)
        current += n-i+1
    end
    return result
end
function utilitygraduppertrignormal(xi::V, theta::Vector{Float64}, gamma::GT) where {V <: AbstractVector{Float64}}
    g = similar(theta)
    n = length(gamma)
    g[1:n] = xi
    current = n+1
    currentlength = n
    for i in 1:n
        v = @view gamma[i:end]
        g[current:current + currentlength - 1] = xi[i] * v
        current += currentlength
        currentlength -= 1
    end
    return g
end

function mvnormal(n::Integer)
    return MvNormal(zeros(n), Diagonal(ones(n)))
end
function getgammanormal(rng::AbstractRNG, n::Integer)::GT
    return rand(rng, mvnormal(n))
end

end #end module

"""
    NormalDiagUtility

Parametric mixed logit utility where the gamma are folowing multivatiate normal distributions.
"""
const NormalDiagUtility = ALPMLU{MvNormal, ModuleNormalTasteUtility.fdiag, ModuleNormalTasteUtility.GT}
const NDU = NormalDiagUtility


function gradient(::Type{NDU}, obs::AbstractObs, theta::Vector, gamma::ModuleNormalTasteUtility.GT, i::Integer)
    lb = div(length(theta), 2)
    #a view is slower here.
    xi = explanatory(obs, i)
    return ModuleNormalTasteUtility.utilitygraddiagnormal(xi, theta, gamma)
end
function hessian(::Type{NDU}, obs::AbstractObs, theta::Vector, gamma::ModuleNormalTasteUtility.GT, i::Integer)
    @warn "Hessian of linear utility called"
    lt = length(theta)
    return zeros(Float64, lt, lt)
end


const NormalUpperTriangularUtility = ALPMLU{MvNormal, ModuleNormalTasteUtility.fuppertriangular}
const NUTU = NormalUpperTriangularUtility
function gradient(::Type{NUTU}, obs::AbstractObs, theta::Vector, gamma::ModuleNormalTasteUtility.GT, i::Integer)
    m = length(theta)
    n = div(round(Integer, -3 + sqrt(9 + 8*m)), 2)
    xi = explanatory(obs, i)
    return ModuleNormalTasteUtility.utilityGradientUpperTriangularNormal(xi, theta, gamma)
end


function getGamma(::Type{T}, rng::AbstractRNG, n::Integer) where {T <: Union{NDU, NUTU}}
    return ModuleNormalTasteUtility.getGammaNormal(rng, n)
end

function gammaDim(::Type{T}, obs::AbstractObs) where {T <: Union{NDU, NUTU}}
    return explanatoryLength(obs)
end

function dim(::Type{NUTU}, s::AbstractData)::Integer
    n = explanatoryLength(s)
    return n + div(n*(n-1), 2)
end
function dim(::Type{NDU}, s::AbstractData)::Integer
    n = explanatoryLength(s)
    return 2 * n
end

