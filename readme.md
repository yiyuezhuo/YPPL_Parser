![YPPL_Parser](https://github.com/yiyuezhuo/YPPL_Parser/workflows/YPPL_Parser/badge.svg)

# YPPL_Parser

Build a probability function (Stan's target) using macro.

## Example

Yet another [eight school non-central model](https://mc-stan.org/users/documentation/case-studies/divergences_and_bias.html):

```julia
using YPPL_Parser: Positive, @build_likeli
using Distributions: Normal, Cauchy

const schools_dat = (
    J = 8,
    y = [28.,  8, -3,  7, -1,  1, 18, 12],
    sigma = [15., 10, 16, 11,  9, 11, 10, 18]
)

ex = quote
    theta = mu .+ tau * theta_tilde
    
    mu ~ Normal(0, 5)
    tau ~ Cauchy(0, 5)
    theta_tilde ~ Normal(0, 1)
    y ~ Normal.(theta, sigma)
end

p = (
    mu = 1.,
    tau = Positive(1.),
    theta_tilde = ones(8)
)

likeli = @build_likeli(ex, p, schools_dat)
```

Which is equivalence to following code:

```julia
using Distributions

const schools_dat = (
    J = 8,
    y = [28.,  8, -3,  7, -1,  1, 18, 12],
    sigma = [15., 10, 16, 11,  9, 11, 10, 18]
)

function likeli(p)
    mu = p[1]
    tau = exp(p[2])
    theta_tilde = p[3:10]
    
    theta = mu .+ tau * theta_tilde
    
    target = 0.
    
    target += logpdf(Normal(0, 5), mu)
    target += logpdf(Cauchy(0, 5), tau) + p[2]
    target += logpdf.(Normal(0, 1), theta_tilde) |> sum
    target += logpdf.(Normal.(theta, schools_dat.sigma), schools_dat.y) |> sum
    
    return target
end
```