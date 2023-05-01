function NLPModels.obj(mo::LogitModel, beta::Vector;
        sample = 1:nobs(mo), eachObs::Bool = false)
    return eachObs ? Fs(mo, beta, sample = sample) : F(mo, beta, sample = sample)
end
function F(mo::LogitModel{D, L, UTI}, beta::Vector{T};
        sample = 1:nobs(mo)) where {T, D, L, UTI}
    ac = zero(T)
    nind = 0
    for i in sample
        ns = nsim(mo.data[i])
        ac += ns*logit(UTI, mo.data[i], beta)
        nind += ns
    end
    return -ac/nind
end
function Fs(mo::LogitModel{D, L, UTI}, beta::Vector{T};
        sample = 1:nobs(mo)) where {T, D, L, UTI}
    ac = zeros(T, length(sample))
    for i in sample
        ac[i] = -logit(UTI, mo.data[i], beta)
    end
    return ac
end
function NLPModels.grad!(mo::LogitModel{D, L, UTI}, beta::Vector{T}, ac::Array{T, 1};
        sample = 1:length(mo.data), eachObs::Bool = false) where {T, D, L, UTI}
    @assert !eachObs "to compute the gradient of each observation individually, you need to pass a matrix as argument"
    ac[:] .= zero(T)
    nind = 0
    for i in sample
        ns = nsim(mo.data[i])
        ac[:] += ns*gradientLogit(UTI, mo.data[i], beta)
        nind += ns
    end
    ac[:] ./= -nind
    return ac
end
function NLPModels.grad!(mo::LogitModel{D, L, UTI}, beta::Vector{T}, ac::Array{T, 2};
        sample = 1:length(mo.data), eachObs::Bool = true) where {T, D, L, UTI}
    @assert eachObs "to compute the gradient, you need to pass a vector as argument"
    ac[:] .= zero(T)
    nind = 0
    for (index, i) in sample
        ac[:, index] = -gradientLogit(UTI, mo.data[i], beta)
    end
    return ac
end
function NLPModels.hess(mo::LogitModel{D, L, UTI}, beta::Vector{T};
        sample = 1:length(mo.data), obj_weight::Float64 = 1.0, BHHH::Bool = false) where {T, D, L, UTI}
    n = length(beta)
    result = zeros(n, n)
    BHHH ? bhhhApprox!(mo, beta, result, sample = sample) : hessian!(mo, beta, result, sample = sample)
    return 
end
function hessian!(mo::LogitModel{D, L, UTI}, beta::Vector{T}, ac::Array{T, 2};
        sample = 1:length(mo.data), obj_weight::Float64 = 1.0) where {T, D, L, UTI}
    ac[:, :] .= zero(T)
    nind = 0
    for i in sample
        ns = nsim(mo.data[i])
        ac[:, :] += ns*hessianLogit(UTI, mo.data[i], beta)
        nind += ns
    end
    ac[:, :] ./= -nind
    return ac
end
function bhhhApprox!(mo::LogitModel{D, L, UTI}, beta::Vector{T}, ac::Array{T, 2};
        sample = 1:length(mo.data)) where {T, D, L, UTI}
    ac[:, :] .= zero(T)
    nind = 0
    for i in sample
        ns = nsim(mo.data[i])
        gll = gradientLogit(UTI, mo.data[i], beta)
        ac[:, :] += ns*gll * gll'
        nind += ns
    end
    ac[:, :] ./= nind
    return ac
end
function NLPModels.hprod!(mo::LogitModel{D, L, UTI}, beta::AbstractVector{T}, v::AbstractVector, ac::Array{T, 1};
        sample = 1:length(mo.data), obj_weight::Float64 = 1.0, BHHH::Bool = false) where {T, D, L, UTI}
    return BHHH ? bhhhApproxProd!(mo, beta, v, ac, sample = sample) : hessianProd!(mo, beta, v, ac, sample = sample)
end
function NLPModels.hprod(mo::LogitModel{D, L, UTI}, beta::AbstractVector{T}, v::AbstractVector;
        sample = 1:length(mo.data), obj_weight::Float64 = 1.0, BHHH::Bool = false) where {T, D, L, UTI}
    ac = zeros(length(beta))
    BHHH ? bhhhApproxProd!(mo, beta, v, ac, sample = sample) : hessianProd!(mo, beta, v, ac, sample = sample)
    return ac
end
function hessianProd!(mo::LogitModel{D, L, UTI}, beta::Vector{T}, v::Vector{T}, ac::Array{T, 1};
        sample = 1:length(mo.data), obj_weight::Float64 = 1.0) where {T, D, L, UTI}
    ac[:, :] .= zero(T)
    nind = 0
    for i in sample
        ns = nsim(mo.data[i])
        ac[:] += ns*hessianLogitdotv(UTI, mo.data[i], beta, v)
        nind += ns
    end
    ac[:] ./= -nind
    return ac
end
function bhhhApproxProd!(mo::LogitModel{D, L, UTI}, beta::Vector{T}, v::Vector{T}, ac::Array{T, 1};
        sample = 1:length(mo.data)) where {T, D, L, UTI}
    ac[:, :] .= zero(T)
    nind = 0
    for i in sample
        ns = nsim(mo.data[i])
        gll = gradientLogit(UTI, mo.data[i], beta)
        ac[:] += ns*dot(gll, v) * gll
        nind += ns
    end
    ac[:] ./= nind
    return ac
end
