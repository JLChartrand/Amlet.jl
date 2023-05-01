"""
     LogitModel
"""
mutable struct LogitModel{D <: AbstractData, L, UTI <: AbstractLogitUtility{L}} <: AmletModel{D}
    data::D
    meta::NLPModelMeta{Float64, Vector{Float64}}
    counters::Counters
    nobs::Int
    function LogitModel{D, L, UTI}(data::D, ::Type{T} = Float64) where {T, D <: AbstractData, L, UTI <: AbstractLogitUtility{L}}
        model = new{D, L, UTI}(data)
        n = dimension(UTI, data)
        model.meta = NLPModelMeta(n)
        model.counters = Counters()
        model.nobs = nobs(data)
        return model
    end
end
function LogitModel{UTI}(data::D, ::Type{T} = Float64) where {T, D <: AbstractData, L, UTI <: AbstractLogitUtility{L}}
    model = LogitModel{D, L, UTI}(data)
    n = dimension(UTI, data)
    model.meta = NLPModelMeta(n)
    model.counters = Counters()
    model.nobs = nobs(data)
    return model
end
function LogitModel(data::D, ::Type{T} = Float64) where {T, D <: AbstractData}
    UTI = StandardLogitUtility
    model = LogitModel{D, Linear, UTI}(data)
    n = dimension(UTI, data)
    model.meta = NLPModelMeta(n)
    model.counters = Counters()
    model.nobs = nobs(data)
    return model
end
function dimension(lm::LogitModel{D, L, UTI}) where {D, L, UTI}
    return dimension(UTI, lm.data)
end

