##
using Logging: global_logger
using TerminalLoggers: TerminalLogger
global_logger(TerminalLogger())

using DifferentialEquations
using Random
using Distributions
using LinearAlgebra
using DataFrames
using CSV
using Dates
using Distributed
using SpecialPolynomials

##
function orderParameterGet(phaseVector)
    return abs(sum(exp.(phaseVector * (0 + 1im))))
end

function kernel(r)
    scale = 10
    amplitudeMultiple = 2

    r = r / scale
    p = basis(Hermite, 4)
    return amplitudeMultiple * 0.25 * p(r / sqrt(2)) * pdf(Normal(), r)
end

function indexToCoord(i)
    i = i - 1
    nOneDimension = Int(sqrt(nOsc))
    x = mod(i, nOneDimension)
    y = Int(floor(i / nOneDimension))
    return [x, y]
end

function distance(x0, x1)
    coord0 = indexToCoord(x0)
    coord1 = indexToCoord(x1)
    return norm(coord0 - coord1)
end

function distanceMatrixGet(nOsc)
    distanceMatrix = zeros(Float64, nOsc, nOsc)
    for i in 1:nOsc
        for j in i:nOsc
            distanceMatrix[i,j] = distance(i, j)
            distanceMatrix[j,i] = distanceMatrix[i,j]
        end
    end
    return distanceMatrix
end

function kernelMatrixGet(distanceMatrix)
    nOsc = size(distanceMatrix)[1]
    kernelMatrix = zeros(nOsc, nOsc)
    for i in 1:nOsc
        for j in i:nOsc
            r = distanceMatrix[i,j]
            kernelMatrix[i, j] = kernel(r)
            kernelMatrix[j, i] = kernelMatrix[i, j]
        end
    end
    return kernelMatrix
end

function jac!(J, u, p, t)

    kernelMatrix, W = p
    cosDiff = cos.(u .- u')

    J .= zeros(nOsc, nOsc)

    for i in 1:nOsc
        for j in 1:nOsc
            J[i,j] = kernelMatrix[i,j] * mean(cosDiff[:,j]) * (-1)^(i == j)
        end
    end
end

function kuramoto2d!(du, u, p, t)

    coupleMatrix, W = p

    nOsc = length(W)
    for i in 1:nOsc
        du[i] = W[i]
        for j in 1:nOsc
            du[i] = du[i] + (coupleMatrix[i,j]) * sin(u[j] - u[i])
        end
    end
end

function kuramotoPMAP(parameterList)

    K, theta0, W, description, directory, kernelMatrix = parameterList

    nOsc = length(theta0)
    upperTimeBound = 100
    saveat = upperTimeBound/1000.
    method = AutoTsit5(Rosenbrock23()) 

    coupleMatrix = (K/nOsc) * kernelMatrix

    descriptionString = "K_" * string(K) * "_" * description
    println("Solving: " * descriptionString)

    prob = ODEProblem(kuramoto2d!, theta0, (0, upperTimeBound), [coupleMatrix, W]);
    sol = solve(prob, method = method, reltol = 1e-8, abstol = 1e-8, saveat = saveat, progress=true, progress_steps=50);

    # WRITE TO FILE
    df = DataFrame(sol)
    try
        mkdir("../data/" * directory)
    catch e
    end
    CSV.write("../data/" * directory * "/kuramoto_" * descriptionString * ".csv" , df, writeheader=false)

    println("Finished!")
end
