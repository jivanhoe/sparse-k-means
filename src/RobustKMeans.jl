using Clustering, Random, JuMP, Gurobi

function solve_inner_problem_for_robust_kmeans(
    X::Array{Float64, 2},
    w::Array{Float64, 1},
    K::Int64
)::Array{Float64, 2}
    n, d = size(X)
    X_weighted = deepcopy(X)
    for j=1:d
        X_weighted[:,j] = sqrt(w[j]) * X_weighted[:,j]
    end
    cluster_model = kmeans(X_weighted', K)
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


function calculate_cost_by_feature(
    X::Array{Float64, 2},
    z::Array{Float64, 2},
    K::Int64
)::Array{Float64, 1}
    n, d = size(X)
    cost = zeros(d)
    for j=1:d
        x_bar = sum(X[:,j]) / n
        cost[j] += sum((X[:,j] .- x_bar).^2)
        for k=1:K
            μ = sum(X[z[:,k] .== 1, j]) / sum(z[:,k] .== 1)
            cost[j] -= sum((X[z[:,k] .== 1, j] .- μ).^2)
        end
    end
    return cost
end


function solve_outer_problem_for_robust_kmeans(
    X::Array{Float64,2},
    z::Array{Float64,2},
    K::Int64,
    λ::Float64
) #::Array{Int64, 1}

    n, d = size(X)
    cost = calculate_cost_by_feature(X, z, K)
    model = Model(solver=GurobiSolver(OutputFlag=0, TimeLimit=60, GUROBI_ENV))

    # Variables
    @variable(model, w[1:d] >= 0)
    @variable(model, t >= 0)

    # Objective
    @objective(model, Max, t)

    # Constraints
    @constraint(model, w'*w <= 1)
    @constraint(model, ones(d)'*w <= λ)
    @constraint(model, t == cost'*w)

    # Solve
    solve(model)
    return max.(getvalue(w), 0), getobjectivevalue(model)

end


function robust_kmeans(
    X::Array{Float64,2},
    K::Int64;
    λ::Float64 = 1.0,
    tol::Float64 = 1e-3,
    max_iter::Int64 = 1000
)::Tuple{Array{Float64, 1}, Array{Int64, 1}}
    n, d = size(X)
    w = ones(d) / sqrt(d)
    z = zeros(n)
    prev_obj_val = -Inf
    for _=1:max_iter
        z = solve_inner_problem_for_robust_kmeans(X, w, K)
        w, obj_val = solve_outer_problem_for_robust_kmeans(X, z, K, λ)
        if abs(obj_val - prev_obj_val) < tol
            break
        else
            prev_obj_val = obj_val
        end
    end
    z = solve_inner_problem_for_robust_kmeans(X, w, K)
    cluster_assignments = [k for (i, k) in sort(Tuple.(findall(z .== 1)))]
    return w, cluster_assignments
end
