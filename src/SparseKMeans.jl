using Clustering, Random, JuMP, Gurobi
GUROBI_ENV = Gurobi.Env();


function solve_inner_problem(
    X::Array{Float64, 2},
    w::Array{Float64, 1},
    K::Int64,
    num_restarts_per_cut::Int64
)::Array{Float64, 2}
    n = size(X, 1)
    cluster_model = kmeans(X[:, w .== 1]', K)
    min_cost = cluster_model.totalcost
    for _=1:num_restarts_per_cut-1
        new_cluster_model = kmeans(X[:, w .== 1]', K)
        new_cost = new_cluster_model.totalcost
        if new_cost < min_cost
            cluster_model = new_cluster_model
            min_cost = new_cost
        end
    end
    cluster_assignments = assignments(cluster_model)
    z = zeros(n, K)
    for i=1:n
        for k=1:K
            if cluster_assignments[i] == k
                z[i,k] = 1
            end
        end
    end
    return z
end


function calculate_gradient(
    X::Array{Float64, 2},
    z::Array{Float64, 2},
    K::Int64
)::Array{Float64, 1}
    n, d = size(X)
    ∇f = zeros(d)
    for j=1:d
        x_bar = sum(X[:,j]) / n
        ∇f[j] += sum((X[:,j] .- x_bar).^2)
        for k=1:K
            μ = sum(X[z[:,k] .== 1, j]) / sum(z[:,k] .== 1)
            ∇f[j] -= sum((X[z[:,k] .== 1, j] .- μ).^2)
        end
    end
    return ∇f
end


function get_values_for_cut(
    X::Array{Float64, 2},
    w0::Array{Float64, 1},
    K::Int64,
    num_restarts_per_cut::Int64
)::Tuple{Array{Float64, 2}, Float64, Array{Float64, 1}}
    z0  = solve_inner_problem(X, w0, K, num_restarts_per_cut)
    ∇f0 = calculate_gradient(X, z0, K)
    f0 = sum(∇f0 .* w0)
    return z0, f0, ∇f0
end


function solve_outer_problem(
    X::Array{Float64,2},
    K::Int64,
    p::Int64,
    num_restarts_per_cut::Int64;
    seed::Int64 = 1,
    tol::Float64=1e-5
) #::Array{Int64, 1}

    n, d = size(X)
    model = Model(solver=GurobiSolver(OutputFlag=0, TimeLimit=60, GUROBI_ENV))

    # Variables
    @variable(model, w[1:d], Bin)
    @variable(model, t >= 0)

    # Objective
    @objective(model, Max, t)

    # Initial guess
    rng = MersenneTwister(seed)
    w0 = zeros(d)
    w0[1:p] .= 1
    w0 = shuffle(rng, w0)
    z0, f0, ∇f0 = get_values_for_cut(X, w0, K, num_restarts_per_cut)

    # Initial constraints
    @constraint(model, sum(w) == p)
    @constraint(model, t .<= f0 + ∇f0'*(w .- w0))

    # Lazy constriants
    function outer_approximation(callback)
        w0 = getvalue(w)
        z0, f0, ∇f0 = get_values_for_cut(X, w0, K, num_restarts_per_cut)
        @constraint(model, t .>= f0 + ∇f0'*(w .- w0))
    end
    addlazycallback(model, outer_approximation)

    # Solve
    solve(model)
    w_opt = float.(getvalue(w) .> tol)
    z_opt = solve_inner_problem(X, w_opt, K, num_restarts_per_cut)
    obj_value = sum(w_opt .* calculate_gradient(X, z_opt, K))
    return w_opt, z_opt, obj_value

end


function sparse_kmeans(
    X::Array{Float64,2},
    K::Int64,
    p::Int64;
    num_restarts_per_solve::Int64 = 10,
    num_restarts_per_cut::Int64 = 1
)::Tuple{Array{Int64, 1}, Array{Int64, 1}}
    w_opt, z_opt, max_obj_value = solve_outer_problem(X, K, p, num_restarts_per_cut)
    for seed=1:num_restarts_per_solve-1
        w_new, z_new, new_obj_value = solve_outer_problem(X, K, p, num_restarts_per_cut, seed=seed)
        if new_obj_value > max_obj_value
            w_opt, z_opt, max_obj_value = w_new, z_new, new_obj_value
        end
    end
    selected_features = findall(round.(w_opt) .== 1)
    cluster_assignments = [k for (i, k) in sort(Tuple.(findall(z_opt .== 1)))]
    return selected_features, cluster_assignments
end
