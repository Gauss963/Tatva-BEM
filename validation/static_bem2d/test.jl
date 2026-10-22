using Test, LinearAlgebra
BLAS.set_num_threads(4)
include(joinpath(@__DIR__,"src/TwoBlockBEM.jl"))
using .TwoBlockBEM

@testset "Plane-strain elastic patch and refinement" begin
    coarse=patch_check(h=10.0)
    fine=patch_check(h=2.0)
    @test fine.residual<1e-8
    @test fine.displacement_error<coarse.displacement_error
    @test fine.stress_error<coarse.stress_error
    @test fine.displacement_error<0.025
    @test fine.stress_error<0.025
    println("Uniform-compression patch: ",fine)
end

const KN=76620.0
const KT=7662.0
const MUS=0.8
const MUK=0.6
const DC=0.0125959
law(dn,ds,z=0.0,c=0.0)=contact_update(dn,ds,z,c,KN,KT,MUS,MUK,DC)

@testset "Unilateral contact, LSW consistency, reversal, tangents" begin
    open=law(-1e-3,0.2,0.1,0.03)
    @test open.p==0 && open.tau==0
    @test open.z==0.1 && open.cumulative==0.03
    stick=law(16/KN,0.0005)
    @test stick.z==0 && stick.cumulative==0
    @test stick.tau≈KT*0.0005
    soft=law(16/KN,0.003)
    @test 0<soft.cumulative<DC
    @test soft.tau≈soft.p*(MUS-(MUS-MUK)*soft.cumulative/DC)
    kinetic=law(16/KN,0.05)
    @test kinetic.cumulative>DC
    @test kinetic.tau≈16*MUK
    reverse=law(16/KN,-0.004,0.001,0.002)
    @test reverse.tau<0 && reverse.z<0.001
    @test reverse.cumulative≈0.002+abs(reverse.z-0.001)
    for ds in [0.0005,0.003,0.05,-0.003]
        dn=16/KN; s=law(dn,ds)
        dtau_ds=(law(dn,ds+1e-8).tau-law(dn,ds-1e-8).tau)/2e-8
        dtau_dn=(law(dn+1e-9,ds).tau-law(dn-1e-9,ds).tau)/2e-9
        @test isapprox(dtau_ds,s.ts;atol=1e-5,rtol=1e-6)
        @test isapprox(dtau_dn,s.tp*s.dp;atol=1e-4,rtol=1e-6)
    end
    samples=Vector{Float64}[]
    for c in [0.0,0.2DC,DC,2DC], ds in [-0.05,-0.003,0.0,0.0005,0.003,0.05]
        s=law(16/KN,ds,0.0,c)
        mu=MUS-(MUS-MUK)*min(s.cumulative/DC,1)
        @test abs(s.tau)<=s.p*mu+1e-10
        @test s.cumulative>=c
        @test isapprox(s.tau,KT*(ds-s.z);atol=1e-10)
        push!(samples,[ds,c,s.p,s.tau,s.z,s.cumulative,mu])
    end
    writecsv(joinpath(@__DIR__,"output/law_samples.csv"),
             ["relative_slip","old_cumulative","p","tau","plastic_slip","new_cumulative","mu"],
             reduce(vcat,permutedims.(samples)))
end

@testset "Two-domain boundary conditions and contact equilibrium" begin
    m=build_model(joinpath(@__DIR__,"input/case.toml"),
                  joinpath(@__DIR__,"input/restart_interface.csv");h=10.0)
    r=solve_equilibrium(m,10.806226665707777,m.z0,m.c0;frozen=true)
    @test r.success && r.residual<1e-7
    u=m.U*r.x;t=m.H*r.x
    @test maximum(abs.(t[2m.im.-1]+t[2m.is.-1]))<1e-7
    @test maximum(abs.(t[2m.im]+t[2m.is]))<1e-7
    for i in 1:m.nm
        if m.moving.name[i]=="normal"
            @test t[2i-1]≈16 atol=1e-7
        elseif m.moving.name[i]=="platen"
            @test u[2i]≈r.x[end] atol=1e-9
        end
    end
    for j in 1:m.ns
        i=m.nm+j
        if m.stationary.name[j]=="support_x"
            @test abs(u[2i-1])<1e-9
        elseif m.stationary.name[j]=="support_y"
            @test abs(u[2i])<1e-9
        end
    end
end
