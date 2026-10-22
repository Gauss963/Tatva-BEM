using LinearAlgebra, TOML, Printf
BLAS.set_num_threads(4)
include(joinpath(@__DIR__, "src", "TwoBlockBEM.jl"))
using .TwoBlockBEM

function option(name, default)
    i=findfirst(==(name),ARGS)
    isnothing(i) && return default
    i < length(ARGS) || error("Missing value for $name")
    return ARGS[i+1]
end

const HERE=@__DIR__
const PROFILE=["y","p","tau","plastic_slip","cumulative_slip","mu","umx","umy","usx","usy"]

function output_path(name)
    # Deliberately no output-directory argument: every output stays in this folder.
    p=joinpath(HERE,"output",name)
    mkpath(dirname(p))
    return p
end

function save_report(name, values)
    open(output_path(name),"w") do io
        TOML.print(io,values)
    end
end

weighted_mean(v,w)=sum(v.*w)/sum(w)
relative_error(a,b,w)=sqrt(sum(w.*(a-b).^2)/max(sum(w.*b.^2),eps()))

function replay(hs)
    meta=TOML.parsefile(joinpath(HERE,"input/reference.toml"))
    header,history=readcsv(joinpath(HERE,"reference/fem_history.csv"))
    col=Dict(String(k)=>i for (i,k) in enumerate(header))
    comparisons=Vector{Float64}[]
    for h in hs
        println("Building two independent BEM domains, maximum panel length = $h mm")
        m=build_model(joinpath(HERE,"input/case.toml"),joinpath(HERE,"input/restart_interface.csv");h)
        println("  $(m.nm+m.ns) panels; $(length(m.y)) paired contact panels")
        guess=nothing
        for frame in meta["frames"]
            _,ref=readcsv(joinpath(HERE,"reference",@sprintf("frame_%05d.csv",frame)))
            z=[TwoBlockBEM.interp(ref[:,1],ref[:,4],y) for y in m.y]
            cumulative=[TwoBlockBEM.interp(ref[:,1],ref[:,5],y) for y in m.y]
            D=history[frame+1,col["applied_shear_displacement"]]
            # Reconstruct static elasticity with the measured FEM plastic slip.
            # This does not predict the FEM rupture or integrate its dynamics.
            result=solve_equilibrium(m,D,z,cumulative;guess,frozen=true)
            result.success || error("Frozen-plastic equilibrium failed at frame $frame, h=$h")
            guess=result.x
            v=interface_values(m,result)
            stem=@sprintf("h%g/frame_%05d",h,frame)
            writecsv(output_path(stem*".csv"),PROFILE,v)
            pref=[TwoBlockBEM.interp(ref[:,1],ref[:,2],y) for y in m.y]
            tref=[TwoBlockBEM.interp(ref[:,1],ref[:,3],y) for y in m.y]
            w=m.weights
            trimmed=(m.y .>= minimum(ref[:,1])+20) .& (m.y .<= maximum(ref[:,1])-20)
            pmean=weighted_mean(v[:,2],w); tmean=weighted_mean(v[:,3],w)
            load=m.c["solver"]["shear_loading_stiffness"]*(D-result.x[end])
            normal_force=m.c["solver"]["normal_stress_override"]*m.c["geometry"]["moving"]["dimensions"][2]
            shear_force=load*m.c["geometry"]["moving"]["dimensions"][1]
            capacity=v[:,6].*v[:,2]
            utilization=maximum(abs.(v[:,3])./max.(capacity,eps()))
            row=[h,frame,history[frame+1,col["time"]],D,pmean,
                 history[frame+1,col["avg_sigma_n"]],tmean,history[frame+1,col["avg_tau"]],
                 load,history[frame+1,col["applied_shear"]],result.x[end],
                 history[frame+1,col["loading_face_displacement"]],
                 relative_error(v[:,2],pref,w),relative_error(v[:,3],tref,w),
                 relative_error(v[trimmed,2],pref[trimmed],w[trimmed]),
                 relative_error(v[trimmed,3],tref[trimmed],w[trimmed]),
                 abs(sum(v[:,2].*w)-normal_force)/normal_force,
                 abs(sum(v[:,3].*w)-shear_force)/max(abs(shear_force),eps()),
                 utilization,result.residual,result.iterations]
            push!(comparisons,row)
            @printf("  frame %4d: p̄=%.5f MPa, τ̄=%.5f MPa, residual %.2e, max strength ratio %.3f\n",
                    frame,pmean,tmean,result.residual,utilization)
        end
        GC.gc()
    end
    names=["h_mm","frame","time_s","actuator_mm","bem_mean_p_MPa","fem_mean_p_MPa",
           "bem_mean_tau_MPa","fem_mean_tau_MPa","bem_applied_shear_MPa","fem_applied_shear_MPa",
           "bem_platen_mm","fem_platen_mm","p_profile_rel_l2_all","tau_profile_rel_l2_all",
           "p_profile_rel_l2_trim20","tau_profile_rel_l2_trim20","normal_force_relative_error",
           "shear_force_relative_error","max_strength_ratio","scaled_residual_MPa","iterations"]
    writecsv(output_path("comparison.csv"),names,reduce(vcat,permutedims.(comparisons)))
    save_report("replay_status.toml",Dict(
        "status"=>"completed_frozen_plastic_static_reconstruction",
        "full_dynamic_reproduction"=>false,"source_run"=>"local-mus08-muk06-gap0-v2-ev16",
        "h_mm"=>hs,"frames"=>meta["frames"],
        "description"=>"Uses measured plastic slip and actuator displacement as inputs; outputs are static BEM equilibria. Strength exceedance is reported, not silently clipped.",
        "trimmed_metrics"=>"20 mm removed at each contact end; full-profile errors also reported.",
        "normal_contact"=>"unilateral penalty, p=kn*max(umx-usx,0)",
        "tangential_contact"=>"frozen FEM plastic slip, tau=kt*(umy-usy-z); no yield enforcement in replay",
        "pressure_resolution_note"=>"FEM interface values reconstructed from float32 displacement/plastic-slip dumps; corner tractions are mesh dependent."))
end

function evolve(h)
    meta=TOML.parsefile(joinpath(HERE,"input/reference.toml"))
    m=build_model(joinpath(HERE,"input/case.toml"),joinpath(HERE,"input/restart_interface.csv");h)
    D0=meta["restart_actuator_exact"]
    seed=solve_equilibrium(m,D0,m.z0,m.c0;frozen=true)
    seed.success || error("Restart equilibrium failed")
    v=interface_values(m,seed)
    writecsv(output_path("evolve_restart.csv"),PROFILE,v)
    ratio=maximum(abs.(v[:,3])./max.(v[:,2].*v[:,6],eps()))
    allow="--allow-restart-relaxation" in ARGS
    if ratio>1+1e-5 && !allow
        save_report("evolve_status.toml",Dict("status"=>"incompatible_restart_projection",
            "full_dynamic_reproduction"=>false,"max_strength_ratio"=>ratio,
            "reason"=>"Interpolated FEM plastic slip is not friction-admissible on this BEM mesh. No autonomous event history was generated.",
            "optional_diagnostic"=>"--allow-restart-relaxation explicitly permits a different re-equilibrated initial state."))
        println("Stopped: mapped restart exceeds friction strength (ratio=$ratio). See evolve_status.toml.")
        return false
    end
    initial=solve_equilibrium(m,D0,m.z0,m.c0;guess=seed.x)
    initial.success || error("Nonlinear restart relaxation failed; no event history is valid")
    increment=parse(Float64,option("--increment-mm","0.002"))
    advance=parse(Float64,option("--advance-mm","0.8"))
    increment>0 && advance>0 || error("Positive increments and advance are required")
    state=initial; z=[s.z for s in state.state]; cum=[s.cumulative for s in state.state]
    rows=Vector{Float64}[]
    reason="requested_actuator_advance_reached"
    yielded=false
    for step in 0:ceil(Int,advance/increment)
        D=D0+min(step*increment,advance)
        if step>0
            trial=solve_equilibrium(m,D,z,cum;guess=state.x)
            if !trial.success
                reason="nonlinear_equilibrium_failed"; break
            end
            state=trial;z=[s.z for s in state.state];cum=[s.cumulative for s in state.state]
        end
        p=[s.p for s in state.state]; tau=[s.tau for s in state.state]
        push!(rows,[step,D,weighted_mean(p,m.weights),weighted_mean(tau,m.weights),
                    state.x[end],maximum(cum),state.residual])
        yielded=maximum(cum)>maximum(m.c0)+1e-7
        if yielded && !("--continue-sliding" in ARGS)
            reason="first_yield_or_restart_relaxation"; break
        end
    end
    writecsv(output_path("evolve_history.csv"),
             ["step","actuator_mm","mean_p_MPa","mean_tau_MPa","platen_mm","max_cumulative_slip_mm","residual_MPa"],
             reduce(vcat,permutedims.(rows)))
    writecsv(output_path("evolve_final.csv"),PROFILE,interface_values(m,state))
    save_report("evolve_status.toml",Dict("status"=>reason,"full_dynamic_reproduction"=>false,
        "restart_relaxation_allowed"=>allow,"initial_max_strength_ratio"=>ratio,
        "restart_added_slip_mm"=>maximum(s.cumulative for s in initial.state)-maximum(m.c0),
        "warning"=>"Quasistatic branch only. No inertia, wave propagation, or stability/branch-selection criterion. Do not interpret static post-yield solutions as rupture trajectories."))
    println("Quasistatic continuation stopped: $reason")
    return true
end

mode=option("--mode","replay")
h=parse(Float64,option("--h","5"))
isfinite(h) && h>0 || error("h must be finite and positive")
if mode=="replay"
    hs="--convergence" in ARGS ? [20.0,10.0,5.0,2.0] : [h]
    replay(hs)
elseif mode=="evolve"
    evolve(h) || exit(2)
else
    error("Use --mode replay or --mode evolve")
end
