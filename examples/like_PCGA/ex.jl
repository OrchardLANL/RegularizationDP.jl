import DPFEHM
import GaussianRandomFields
import Optim
import PyPlot
import Random
import RegularizationDP
import Zygote

Random.seed!(0)
function getobsnodes(coords, obslocs)
	obsnodes = Array{Int}(undef, length(obslocs))
	for i = 1:length(obslocs)
		obsnodes[i] = findmin(map(j->sum((obslocs[i] .- coords[:, j]) .^ 2), 1:size(coords, 2)))[2]
	end
	return obsnodes
end
mins = [0, 0]#meters
maxs = [100, 100]#meters
ns = [101, 101]
num_eigenvectors = 200
x_true = randn(num_eigenvectors)
x0 = zeros(num_eigenvectors)
sqrtnumobs = 16
obslocs_x = range(mins[1], maxs[1]; length=sqrtnumobs + 2)[2:end - 1]
obslocs_y = range(mins[2], maxs[2]; length=sqrtnumobs + 2)[2:end - 1]
obslocs = collect(Iterators.product(obslocs_x, obslocs_y))[:]
observations = Array{Float64}(undef, length(obslocs))
dist(x, y) = sum((x .- y) .^ 2)

coords, neighbors, areasoverlengths, _ = DPFEHM.regulargrid2d(mins, maxs, ns, 1.0)
Qs = zeros(size(coords, 2))
boundaryhead(x, y) = 5 * (x - maxs[1]) / (mins[1] - maxs[1])
dirichletnodes = Int[]
dirichleths = zeros(size(coords, 2))
for i = 1:size(coords, 2)
	if coords[1, i] == mins[1] || coords[1, i] == maxs[1] || coords[2, i] == mins[2] || coords[2, i] == maxs[2]
		push!(dirichletnodes, i)
		dirichleths[i] = boundaryhead(coords[1:2, i]...)
	end
	if coords[1, i] == 50 && coords[2, i] == 50
		Qs[i] = 1.0
	end
end

lambda = 50.0#meters -- correlation length of log-conductivity
sigma = 1.0#standard deviation of log-conductivity
mu = 0.0#mean of log-permeability can be arbitrary because this is steady-state and there are no fluid sources

cov = GaussianRandomFields.CovarianceFunction(2, GaussianRandomFields.Matern(lambda, 1; σ=sigma))
x_pts = range(mins[1], maxs[1]; length=ns[1])
y_pts = range(mins[2], maxs[2]; length=ns[2])
grf = GaussianRandomFields.GaussianRandomField(cov, GaussianRandomFields.KarhunenLoeve(num_eigenvectors), x_pts, y_pts)
logKs = GaussianRandomFields.sample(grf)
parameterization = copy(grf.data.eigenfunc)
sigmas = copy(grf.data.eigenval)

logKs2Ks_neighbors(Ks) = exp.(0.5 * (Ks[map(p->p[1], neighbors)] .+ Ks[map(p->p[2], neighbors)]))
function solveforh(logKs, dirichleths)
	@assert length(logKs) == length(Qs)
	if maximum(logKs) - minimum(logKs) > 25
		return fill(NaN, length(Qs))#this is needed to prevent the solver from blowing up if the line search takes us somewhere crazy
	else
		Ks_neighbors = logKs2Ks_neighbors(logKs)
		return DPFEHM.groundwater_steadystate(Ks_neighbors, neighbors, areasoverlengths, dirichletnodes, dirichleths, Qs)
	end
end
solveforheigs(x) = solveforh(x2logKs(x), dirichleths)
x2logKs(x) = reshape(parameterization * (sigmas .* x), ns...)
logKs_true = x2logKs(x_true)
h_true = solveforheigs(x_true)

obsnodes = getobsnodes(coords, obslocs)
obssigma = 1e-3
observations .= h_true[obsnodes]#set up the observations
f(logKs) = sum((solveforh(logKs, dirichleths)[obsnodes] - observations) .^ 2 ./ obssigma ^ 2)

#do the optimization
options = Optim.Options(iterations=250, extended_trace=false, store_trace=true, show_trace=true, x_tol=1e-6)
logKs_est, opt = RegularizationDP.optimize(f, x2logKs, x->sum(x .^ 2), zeros(num_eigenvectors), options)
#plot results from the inverse analysis
fig, axs = PyPlot.subplots(2, 3, figsize=(12, 8))
ims = axs[1].imshow(logKs_true)
axs[1].title.set_text("True Conductivity")
fig.colorbar(ims, ax=axs[1])
ims = axs[2].imshow(logKs_est, vmin=minimum(logKs_true), vmax=maximum(logKs_true))
axs[2].title.set_text("Estimated Conductivity")
fig.colorbar(ims, ax=axs[2])
ims = axs[3].imshow(reshape(solveforh(logKs_true, dirichleths), ns...))
axs[3].title.set_text("True Head")
fig.colorbar(ims, ax=axs[3])
ims = axs[4].imshow(reshape(solveforh(logKs_est, dirichleths), ns...))
axs[4].title.set_text("Estimated Head")
fig.colorbar(ims, ax=axs[4])
axs[5].plot([0, 5], [0, 5], "k", alpha=0.5)
axs[5].plot(observations, solveforh(logKs_est, dirichleths)[obsnodes], "k.")
axs[5].set_xlabel("Observed Head")
axs[5].set_ylabel("Predicted Head")
axs[6].semilogy(map(i->opt.trace[i].value, 1:length(opt.trace)))
axs[6].set_xlabel("Iteration")
axs[6].set_ylabel("Loss")
fig.tight_layout()
display(fig)
println()
PyPlot.close(fig)
