##

#=
using Logging: global_logger
using TerminalLoggers: TerminalLogger
global_logger(TerminalLogger())
=#

using DifferentialEquations
#using LSODA
using Random
using Distributions
#using SpecialPolynomials
using LinearAlgebra
#using BenchmarkTools
using DataFrames
using CSV
using Dates
#using PlotlyJS
using Distributed

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

    kernelMatrix, W = p

    nOsc = length(W)
    for i in 1:nOsc
        du[i] = W[i]
        for j in 1:nOsc
            du[i] = du[i] + (kernelMatrix[i,j]) * sin(u[j] - u[i])
        end
    end
end

function kuramotoPMAP(parameterList)

    K, theta0, W, description, directory = parameterList

    nOsc = length(theta0)
    upperTimeBound = 100
    saveat = upperTimeBound/1000.
    method = AutoTsit5(Rosenbrock23()) 

    kernelMatrix = (K/nOsc) * ones((nOsc, nOsc))

    descriptionString = "nOsc_" * string(nOsc) * "_K_" * string(K) * "_" * description
    println("Solving: " * descriptionString)

    prob = ODEProblem(kuramoto2d!, theta0, (0, upperTimeBound), [kernelMatrix, W]);
    sol = solve(prob, method = method, reltol = 1e-8, abstol = 1e-8, saveat = saveat);

    # WRITE TO FILE
    df = DataFrame(sol)
    try
        mkdir("../data/" * directory)
    catch e
    end
    CSV.write("../data/" * directory * "/kuramoto_" * descriptionString * ".csv" , df, writeheader=false)

    println("Finished!")
end

function pmapListGet(distributionSwitch)
    
    Random.seed!(1234)

    nOsc = 1024

    kList = collect(1.:.2:2.)
    theta0List = [2 * pi * rand(Float64, nOsc)]

    if distributionSwitch == 0
        probabilityDistribution = MixtureModel([Exponential(2.5), Normal(5., .5), Normal(7., .5)], [.9,.05,.05])
        directory = "Mixed"
        description = "Mixed"
    elseif distributionSwitch == 1
        probabilityDistribution = Exponential(2.5)
        directory = "Exponential_2.5"
        description = "Exponential_2.5"
    else
        println("Distribution error!")
        return
    end

    wList = [rand(probabilityDistribution, nOsc)]
    descriptionList = [description]
    directoryList = [directory]

    thisProduct = collect(Base.product(kList, theta0List, wList, descriptionList, directoryList))

    return reshape(thisProduct, length(thisProduct))
end
