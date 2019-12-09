using Random

function generate_random_data(
    n::Int64,
    d::Int64,
    p::Int64,
    K::Int64;
    σ_noise::Float64=1.0,
    σ_group::Float64=10.0,
    seed::Int64=1
)::Tuple{Array{Float64, 2}, Array{Int64, 1}, Array{Int64, 1}}
    rng = MersenneTwister(seed)
    w_true = shuffle(rng, vcat(ones(p), zeros(d-p)))
    μ = σ_group*randn(rng, d, K) .* w_true
    true_assignments = rand(rng, 1:K, n)
    X = σ_noise * randn(rng, n, d)
    for i=1:n
        X[i,:] += μ[:, true_assignments[i]]
    end
    return X, true_assignments, findall(w_true .== 1)
end
