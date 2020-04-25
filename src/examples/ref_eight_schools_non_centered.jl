"""
Reference:
https://mc-stan.org/users/documentation/case-studies/divergences_and_bias.html
"""
module ref_eight_schools_non_centered

using Distributions
#using DistributionsAD

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
    target += logpdf(Cauchy(0, 5), tau) + p[2] # p[2] jacobian correction, so we posit prior on non-constrait parameter instead of tau
    target += logpdf.(Normal(0, 1), theta_tilde) |> sum
    target += logpdf.(Normal.(theta, schools_dat.sigma), schools_dat.y) |> sum
    
    return target
end

function recover_lp(mu, theta_t, tau)
    
    theta = mu .+ theta_t * tau
    
    target = 0
    target += logpdf(Normal(0, 5), mu)
    target += logpdf(Cauchy(0, 5), tau) + log(tau) + log(2)
    target += logpdf.(Normal(0, 1), theta_t) |> sum
    target += logpdf.(Normal.(theta, schools_dat.sigma), schools_dat.y) |> sum
    
    return target
end

const reference_mean = [4.41, 3.51, 6.2, 4.81, 3.97, 4.76, 3.63, 4.08, 6.22, 4.82]

function decode(posterior)
    decoded = copy(posterior)
    decoded[:, :, 2] = exp.(decoded[:, :, 2])
    decoded[:, :, 3:end] = decoded[:, :, 3:end] .* decoded[:, :, 2] .+ decoded[:, :, 1]
    return decoded
end

const size_p = 10

end