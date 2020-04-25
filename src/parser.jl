
import Base.~

using MacroTools
using MacroTools: postwalk

import Distributions: logpdf # since macro is defined here


function ~(left, right)
    return logpdf.(right, left) |> sum
end

function ~(left::AbstractFloat, right)
    return logpdf(right, left)
end

function build_model(ex)
    postwalk(ex) do x
        @capture(x, X_ ~ Y_) || return x
        return  :(target += $X ~ $Y)
    end
end

function build_model_inline(ex)
    postwalk(ex) do x
        @capture(x, X_ ~ Dist_) || return x
        return  :(target += logpdf.($Dist, $X) |> sum)
    end
end

abstract type Constraints end;

struct Positive{T} <: Constraints
    data::T
end

Base.length(x::Constraints) = length(x.data)


struct ParamInfo{T}
    name::Symbol
    data::T
    left::Int
    right::Int
end

function make_assign(info::ParamInfo{T}) where T <: AbstractFloat
    return :($(info.name) = p[$(info.left)])
end

function make_assign(info::ParamInfo{T}) where T <: AbstractVector
    return :($(info.name) = p[$(info.left) : $(info.right)])
end

function make_assign(info::ParamInfo{T}) where T <: Constraints
    return make_assign(ParamInfo(info.name, info.data.data, info.left, info.right))
end

function build_assign(info_list)
    head = :block
    args = Any[]
    
    for info in info_list
        push!(args, make_assign(info))
    end
    
    return Expr(head, args...)
end

function get_param_info_list(params)
    info_list = ParamInfo[]
    left = 1
    for (key, value) in zip(keys(params), params)
        info = ParamInfo(key, value, left, left+length(value)-1)
        push!(info_list, info)
        left += length(value)
    end
    return info_list
end

function make_transform(info::ParamInfo)
    return :()
end

function make_transform(info::ParamInfo{T}) where T <: Positive
    return :($(info.name) = exp($(info.name)))
end

function build_transform(info_list)
    head = :block
    args = Any[]
    
    for info in info_list
        push!(args, make_transform(info))
    end
    
    return Expr(head, args...)
end

function make_adjustment(info::ParamInfo)
    return :()
end

function make_adjustment(info::ParamInfo{Positive{T}}) where T <: AbstractFloat
    return :(target += p[$(info.left)])
end

function make_adjustment(info::ParamInfo{Positive{T}}) where T <: AbstractVector
    return :(target += sum(p[$(info.left) : $(info.right)]))
end


function build_adjustment(info_list)
    head = :block
    args = Any[]
    
    for info in info_list
        push!(args, make_adjustment(info))
    end
    
    return Expr(head, args...)
end

function build_data(data_keys::Tuple)
    head = :block
    args = Any[]

    for key in data_keys
        ex = Expr(:., :data, QuoteNode(key))
        ex2 = :($key = $ex)
        push!(args, ex2)
    end
    
    return Expr(head, args...)
end


function build_likeli_quote(ex::Expr, params::NamedTuple, data_keys::Tuple)

    info_list = get_param_info_list(params)

    return quote
        function(data)
            function(p)
                $(build_data(data_keys))

                $(build_assign(info_list))
                $(build_transform(info_list))
        
                target = 0.0
        
                $(build_adjustment(info_list))
        
                $(build_model(ex))
                
                return target
            end
        end
    end
end

function build_likeli_quote_inline(ex, params, data_keys)

    info_list = get_param_info_list(params)

    return quote
        function(data)
            function(p)
                $(build_data(data_keys))

                $(build_assign(info_list))
                $(build_transform(info_list))
        
                target = 0.0
        
                $(build_adjustment(info_list))
        
                $(build_model_inline(ex))
                
                return target
            end
        end
    end
end

macro build_likeli(ex, params, data)
    quote
        quote_ex = build_likeli_quote($(esc(ex)), $(esc(params)), keys($(esc(data))))
        eval_ex = $(esc(:eval))
        eval_ex(quote_ex)($(esc(data)))
    end
end

macro build_likeli_inline(ex, params, data)
    quote
        quote_ex = build_likeli_quote_inline($(esc(ex)), $(esc(params)), keys($(esc(data))))
        eval_ex = $(esc(:eval))
        eval_ex(quote_ex)($(esc(data)))
    end
end