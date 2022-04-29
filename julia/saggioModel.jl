using DifferentialEquations, Plots, LinearAlgebra, Polynomials
plotlyjs()

function parametersMake(z0)

    A = [.3448, .0228, .2014]
    B = [.3351, .0746, .2053]
    c = .001
    dstar = .3

    R = norm(A)
    e = A/R
    crossProduct = cross(cross(A,B), A)
    f = crossProduct/norm(crossProduct)

    #=
    muTwo, minusMuOne, nu = R * ((e * cos(z0)) + (f * sin(z0)))
    allRoots = roots(Polynomial([minusMuOne, -muTwo, 0, 1]))
    xs = real(allRoots[1])
    =#
    xs = .55

    [e, f, R, c, dstar, xs]
end


function saggio!(du, u, p, t)

    e, f, R, c, dstar, xs = p
    muTwo, minusMuOne, nu = R * ((e * cos(u[3])) + (f * sin(u[3])))

    #allRoots = roots(Polynomial([minusMuOne, -muTwo, 0, 1]))
    #xs = real(allRoots[1])

    du[1] = -u[2]
    du[2] = u[1]^3 - (muTwo * u[1]) + minusMuOne - (u[2] * (nu + u[1] + u[1]^2))
    du[3] = -c * (sqrt( (u[1] - xs)^2 + u[2]^2) - dstar)
end

y0 = 0.0
z0 = 0.0
p = parametersMake(z0)
x0 = p[6]

u0 = [x0, y0, z0]
tspan = (0., 2000.)

prob = ODEProblem(saggio!, u0, tspan, p)
sol = solve(prob)

xt = plot(sol, vars=(1))
zt = plot(sol, vars=(3))
plot(xt,zt, layout = (2,1))

#=
xyzt = plot(sol, plotdensity=10000,lw=1.5, vars=(1))
xy = plot(sol, plotdensity=10000, vars=(1,2))
xz = plot(sol, plotdensity=10000, vars=(1,3))
yz = plot(sol, plotdensity=10000, vars=(2,3))
xyz = plot(sol, plotdensity=10000, vars=(1,2,3))
plot(plot(xyzt,xyz),plot(xy, xz, yz, layout=(1,3),w=1), layout=(2,1))
=#
