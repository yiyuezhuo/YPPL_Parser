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

const reference_mean = [4.40, 3.60, 6.23, 4.94, 3.92, 4.76, 3.61, 4.04, 6.30, 4.86]
const reference_std =  [3.32, 3.22, 5.60, 4.67, 5.26, 4.76, 4.66, 4.83, 5.09, 5.29]

function decode(posterior)
    decoded = copy(posterior)
    decoded[:, :, 2] = exp.(decoded[:, :, 2])
    decoded[:, :, 3:end] = decoded[:, :, 3:end] .* decoded[:, :, 2] .+ decoded[:, :, 1]
    return decoded
end

const mean_ref = [ 4.40429047,  0.80017951,  0.31605359,  0.09696118, -0.08594352,
        0.06018543, -0.16091642, -0.07310546,  0.35473399,  0.07750239]
const std_ref = [3.3244735 , 1.17338364, 0.98828502, 0.93565339, 0.96596205,
       0.94260066, 0.92815457, 0.94551266, 0.96143081, 0.97496407]

const cov_ref = hcat([[ 1.10521517e+01, -2.80851352e-01, -2.19597706e-01,
       -3.11016454e-01, -1.24914133e-01, -2.71591366e-01,
       -3.33788942e-01, -2.37886583e-01, -3.65265587e-01,
       -1.33818429e-01],
      [-2.80851352e-01,  1.37683261e+00,  2.27774680e-01,
        6.70428051e-02, -6.06694159e-02,  4.80922872e-02,
       -9.26606018e-02, -4.10373749e-02,  2.33662775e-01,
        5.79792359e-02],
      [-2.19597706e-01,  2.27774680e-01,  9.76709731e-01,
        1.91325842e-02, -1.24709186e-02,  1.22898391e-02,
       -6.19435940e-03, -3.50778768e-03,  4.74222589e-02,
        1.57880875e-02],
      [-3.11016454e-01,  6.70428051e-02,  1.91325842e-02,
        8.75449458e-01, -1.11964549e-03,  9.18374429e-03,
        6.60559137e-03,  6.15104960e-03,  2.18373425e-02,
        5.69380107e-03],
      [-1.24914133e-01, -6.06694159e-02, -1.24709186e-02,
       -1.11964549e-03,  9.33085018e-01,  1.12539718e-03,
        1.17104408e-02,  9.92101808e-03, -5.63502089e-03,
       -1.18892914e-03],
      [-2.71591366e-01,  4.80922872e-02,  1.22898391e-02,
        9.18374429e-03,  1.12539718e-03,  8.88498224e-01,
        8.72741761e-03,  8.00330826e-03,  2.03860315e-02,
        7.21304699e-03],
      [-3.33788942e-01, -9.26606018e-02, -6.19435940e-03,
        6.60559137e-03,  1.17104408e-02,  8.72741761e-03,
        8.61473067e-01,  1.73604102e-02, -5.64011103e-03,
        2.38392448e-03],
      [-2.37886583e-01, -4.10373749e-02, -3.50778768e-03,
        6.15104960e-03,  9.92101808e-03,  8.00330826e-03,
        1.73604102e-02,  8.93996433e-01,  5.33180255e-03,
       -4.77479851e-04],
      [-3.65265587e-01,  2.33662775e-01,  4.74222589e-02,
        2.18373425e-02, -5.63502089e-03,  2.03860315e-02,
       -5.64011103e-03,  5.33180255e-03,  9.24351509e-01,
        1.52028886e-02],
      [-1.33818429e-01,  5.79792359e-02,  1.57880875e-02,
        5.69380107e-03, -1.18892914e-03,  7.21304699e-03,
        2.38392448e-03, -4.77479851e-04,  1.52028886e-02,
        9.50557308e-01]]...)

const size_p = 10

end