module TwoBlockBEM

using Bem2d, LinearAlgebra, TOML

export build_model, solve_equilibrium, readcsv, writecsv, interface_values,
       contact_update, operators, make_boundary, patch_check

function readcsv(path)
    lines = readlines(path)
    header = split(first(lines), ',')
    data = reduce(vcat, [permutedims(parse.(Float64, split(s, ','))) for s in lines[2:end]])
    return header, data
end

function writecsv(path, header, data)
    mkpath(dirname(path))
    open(path, "w") do io
        println(io, join(header, ','))
        for row in eachrow(data)
            println(io, join(row, ','))
        end
    end
end

"""Counterclockwise polygon; edge labels are attached before discretization."""
function make_boundary(vertices, labels, h)
    els = Elements(10000)
    for i in eachindex(vertices)
        a, b = vertices[i], vertices[mod1(i + 1, length(vertices))]
        n = max(1, ceil(Int, hypot(b[1] - a[1], b[2] - a[2]) / h - 1e-10))
        addelsez!(els, discretizedline(a..., b..., n)..., labels[i])
    end
    return els
end

"""Use the library's kernels, vectorized over boundary observation centers.

Each domain gets its own operator. q is a fictitious displacement
discontinuity; physical boundary displacement is U*q, never q itself.
"""
function operators(els, mu, nu)
    n = els.endidx
    U, H = zeros(2n, 2n), zeros(2n, 2n)
    x, y = els.xcenter[1:n], els.ycenter[1:n]
    nx, ny = els.xnormal[1:n], els.ynormal[1:n]
    for j in 1:n, component in 1:2
        d, s = constdispstress(slip2dispstress, x, y, els, [j],
                              [component == 1 ? 1.0 : 0.0],
                              [component == 2 ? 1.0 : 0.0], mu, nu)
        k = 2j - 2 + component
        U[1:2:end, k], U[2:2:end, k] = d[:, 1], d[:, 2]
        H[1:2:end, k] = s[:, 1] .* nx + s[:, 3] .* ny
        H[2:2:end, k] = s[:, 3] .* nx + s[:, 2] .* ny
    end
    return U, H
end

function interp(x, y, z)
    i = clamp(searchsortedlast(x, z), 1, length(x) - 1)
    t = (z - x[i]) / (x[i+1] - x[i])
    return (1-t)*y[i] + t*y[i+1]
end

function build_model(casepath, restartpath; h=10.0)
    c = TOML.parsefile(casepath)
    g, s, f = c["geometry"], c["solver"], c["friction"]
    get(s, "dimension", 2) == 2 || error("Only 2D plane strain is implemented")
    s["friction_law"] == "slip-weakening" || error("Unsupported friction law")
    s["normal_loading_mode"] == "stress" || error("This run needs stress normal loading")
    s["shear_loading_mode"] == "spring-displacement" || error("Unsupported shear loading")
    get(s, "shear_spring_rigid_face", false) || error("This implementation needs a rigid loading face")
    get(s, "shear_loading_face_gap", 0.0) == 0 || error("Nonzero platen gaps are not implemented")
    E, nu = c["material"]["young_modulus"], c["material"]["poisson_ratio"]
    mu = E / (2*(1+nu))
    mx, my = g["moving"]["origin"]; mw, ml = g["moving"]["dimensions"]
    sx, sy = g["stationary"]["origin"]; sw, sl = g["stationary"]["dimensions"]
    L = s["moving_leading_chamfer_along_fault"]
    depth = s["moving_leading_chamfer_perpendicular"]
    @assert abs(sx - mx - mw) < 1e-9 && sy == my
    @assert L > 0 && depth > 0 && sl >= ml
    contact_end = my + ml - L
    moving = make_boundary([(mx,my), (mx+mw,my), (mx+mw,contact_end),
                            (mx+mw-depth,my+ml), (mx,my+ml)],
                           ["platen", "contact", "chamfer", "free", "normal"], h)
    stationary = make_boundary([(sx,sy), (sx+sw,sy), (sx+sw,sy+sl),
                                (sx,sy+sl), (sx,contact_end)],
                               ["free", "support_x", "support_y", "free", "contact"], h)
    Um, Hm = operators(moving, mu, nu)
    Us, Hs = operators(stationary, mu, nu)
    nm, ns = moving.endidx, stationary.endidx
    nd = 2*(nm+ns)
    U, H = zeros(nd+1, nd+1), zeros(nd+1, nd+1)
    U[1:2nm,1:2nm], H[1:2nm,1:2nm] = Um, Hm
    U[2nm+1:nd,2nm+1:nd], H[2nm+1:nd,2nm+1:nd] = Us, Hs
    im = sort(findall(==("contact"), moving.name[1:nm]); by=i->moving.ycenter[i])
    js = sort(findall(==("contact"), stationary.name[1:ns]); by=i->stationary.ycenter[i])
    @assert length(im) == length(js)
    @assert maximum(abs.(moving.ycenter[im] - stationary.ycenter[js])) < 1e-8
    is = js .+ nm
    Bn = U[2im.-1,:] - U[2is.-1,:]
    Bt = U[2im,:] - U[2is,:]
    A, b = copy(H), zeros(nd+1)
    scale = mu / h
    for i in 1:nm
        label = moving.name[i]
        if label == "normal"
            b[2i-1] = s["normal_stress_override"]
        elseif label == "platen"
            A[2i,:] = scale * U[2i,:]
            A[2i,end] = -scale
        end
    end
    for j in 1:ns
        i = nm + j
        label = stationary.name[j]
        if label == "support_x"
            A[2i-1,:] = scale * U[2i-1,:]
        elseif label == "support_y"
            A[2i,:] = scale * U[2i,:]
        end
    end
    for (i,j) in zip(im,is), d in 0:1
        A[2j-d,:] = H[2i-d,:] + H[2j-d,:]
    end
    platen = findall(==("platen"), moving.name[1:nm])
    weights = moving.length[platen]
    kL = s["shear_loading_stiffness"] * sum(weights)
    # Dividing by platen width leaves the resultant equation in MPa.
    A[end,:] = vec(sum(H[2platen,:] .* weights; dims=1)) / sum(weights)
    A[end,end] = s["shear_loading_stiffness"]
    _, ref = readcsv(restartpath)
    yy = moving.ycenter[im]
    z0 = [interp(ref[:,1],ref[:,4],y) for y in yy]
    c0 = [interp(ref[:,1],ref[:,5],y) for y in yy]
    return (; c, moving, stationary, nm, ns, nd, U, H, A, b, im, is, Bn, Bt,
              y=yy, weights=moving.length[im], z0, c0,
              kn=s["normal_penalty"], kt=s["tangential_penalty"],
              mus=f["mu_s"], muk=f["mu_k"], dc=f["d_c"], kL, mu, nu, h)
end

"""Implicit local penalty return map with end-of-increment LSW strength.

Tatva uses the old cumulative slip in each explicit microstep. This local
implicit update has the same continuous law but a different time discretization.
All trial evaluations use immutable old state; commit only after convergence.
"""
function contact_update(dn, ds, zold, cold, kn, kt, mus, muk, dc; frozen=false)
    p = kn * max(dn, 0)
    if p == 0
        return (; p, tau=0.0, z=zold, cumulative=cold, dp=0.0, ts=0.0, tp=0.0)
    end
    trial = kt*(ds-zold)
    if frozen
        return (; p, tau=trial, z=zold, cumulative=cold, dp=kn, ts=kt, tp=0.0)
    end
    muold = mus - (mus-muk)*min(cold/dc,1)
    if abs(trial) <= p*muold
        return (; p, tau=trial, z=zold, cumulative=cold, dp=kn, ts=kt, tp=0.0)
    end
    direction = sign(trial)
    slope = cold < dc ? (mus-muk)/dc : 0.0
    denominator = kt-p*slope
    denominator > 0 || error("Local return map loses uniqueness: refine the loading increment/formulation")
    increment = (abs(trial)-p*muold)/denominator
    if cold+increment < dc
        munew = mus-(mus-muk)*(cold+increment)/dc
        tau = direction*p*munew
        ts = -kt*p*slope/denominator
        tp = direction*kt*munew/denominator
    else
        tau = direction*p*muk
        increment = (abs(trial)-p*muk)/kt
        ts, tp = 0.0, direction*muk
    end
    return (; p, tau, z=zold+direction*increment, cumulative=cold+increment,
              dp=kn, ts, tp)
end

function residual(m, x, D, zold, cold; frozen=false, jacobian=true)
    r = m.A*x-m.b
    r[end] -= m.c["solver"]["shear_loading_stiffness"]*D
    J = jacobian ? copy(m.A) : nothing
    dn, ds = m.Bn*x, m.Bt*x
    state = [contact_update(dn[k], ds[k], zold[k], cold[k], m.kn, m.kt,
                            m.mus, m.muk, m.dc; frozen) for k in eachindex(dn)]
    for (k,i) in enumerate(m.im)
        st = state[k]
        r[2i-1] += st.p
        r[2i] += st.tau
        if jacobian
            J[2i-1,:] += st.dp*m.Bn[k,:]
            J[2i,:] += st.ts*m.Bt[k,:] + st.tp*st.dp*m.Bn[k,:]
        end
    end
    return r, J, state
end

function solve_equilibrium(m, D, zold, cold; guess=nothing, frozen=false,
                           tolerance=1e-7, maxiter=60)
    x = isnothing(guess) ? zeros(m.nd+1) : copy(guess)
    # Start with closed, elastic contact to avoid a singular freely translating
    # moving block at an exactly zero-penetration initial iterate.
    if isnothing(guess)
        J = copy(m.A); b=copy(m.b)
        b[end] += m.c["solver"]["shear_loading_stiffness"]*D
        for (k,i) in enumerate(m.im)
            J[2i-1,:] += m.kn*m.Bn[k,:]
            J[2i,:] += m.kt*m.Bt[k,:]
            b[2i] += m.kt*zold[k]
        end
        x = J\b
    end
    for iteration in 1:maxiter
        r,J,state = residual(m,x,D,zold,cold; frozen)
        err = norm(r,Inf)
        if err < tolerance
            return (; success=true, x, state, residual=err, iterations=iteration)
        end
        step = -(J\r)
        alpha = 1.0
        accepted = false
        for attempt in 1:22
            trial = x+alpha*step
            rr,_,_ = residual(m,trial,D,zold,cold; frozen,jacobian=false)
            if norm(rr,Inf) < (1-1e-4*alpha)*err || norm(rr,Inf) < tolerance
                x = trial; accepted=true; break
            end
            alpha *= 0.5
        end
        accepted || return (; success=false, x, state, residual=err, iterations=iteration)
    end
    r,_,state = residual(m,x,D,zold,cold; frozen,jacobian=false)
    return (; success=false,x,state,residual=norm(r,Inf),iterations=maxiter)
end

function interface_values(m, result)
    u=m.U*result.x
    p=[s.p for s in result.state]; tau=[s.tau for s in result.state]
    z=[s.z for s in result.state]; c=[s.cumulative for s in result.state]
    mu=m.mus .- (m.mus-m.muk).*min.(c./m.dc,1)
    return hcat(m.y,p,tau,z,c,mu,u[2m.im.-1],u[2m.im],u[2m.is.-1],u[2m.is])
end

"""Independent uniform-compression patch check using a mixed-boundary square."""
function patch_check(; h=5.0)
    E, nu, p = 7662.0, 0.2, 16.0
    els=make_boundary([(0.,0.),(100.,0.),(100.,100.),(0.,100.)],
                      ["bottom","right","top","left"],h)
    mu=E/(2*(1+nu)); U,H=operators(els,mu,nu)
    A,b=copy(H),zeros(2els.endidx)
    for i in 1:els.endidx
        if els.name[i]=="left"
            A[2i-1,:]=mu/h*U[2i-1,:]
        elseif els.name[i]=="bottom"
            A[2i,:]=mu/h*U[2i,:]
        elseif els.name[i]=="right"
            b[2i-1]=-p
        end
    end
    q=A\b
    xy=[25. 25.; 50. 50.; 75. 75.]
    d,s=constdispstress(slip2dispstress,xy[:,1],xy[:,2],els,1:els.endidx,
                        q[1:2:end],q[2:2:end],mu,nu)
    expected=hcat(-(1-nu^2)*p/E*xy[:,1],nu*(1+nu)*p/E*xy[:,2])
    return (; displacement_error=norm(d-expected)/norm(expected),
              stress_error=norm(s-hcat(fill(-p,3),zeros(3,2)))/(sqrt(3)*p),
              residual=norm(A*q-b,Inf))
end

end
