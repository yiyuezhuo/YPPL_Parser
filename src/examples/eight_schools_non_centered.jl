module eight_schools_non_centered

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

end