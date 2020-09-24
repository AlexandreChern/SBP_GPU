using SBP_GPU
using CUDA
using Test
using SafeTestsets

@testset "SBP_GPU.jl" begin
    # Write your tests here.
    @test π ≈ 3.14 atol=0.01
    Nx = Ny = 10
    y_in = randn(Nx*Ny)
    y_in_GPU = CuArray(y_in)
    y_out_GPU = similar(y_in_GPU)
    y_out_GPU = D2x_GPU(y_in_GPU,y_out_GPU,Nx,Ny)
    @test Array(y_out_GPU) ≈ D2x(y_in,Nx,Ny,1/Nx)
    @test Array(Dx_GPU(y_in_GPU,y_out_GPU,Nx,Ny)) ≈ Dx(y_in,Nx,Ny,1/Nx)

    Nx = Ny = 100
    y_in = randn(Nx*Ny)
    y_in_GPU = CuArray(y_in)
    y_out_GPU = similar(y_in_GPU)
    y_out_GPU = D2x_GPU(y_in_GPU,y_out_GPU,Nx,Ny)
    @test Array(y_out_GPU) ≈ D2x(y_in,Nx,Ny,1/Nx)
    @test Array(Dx_GPU(y_in_GPU,y_out_GPU,Nx,Ny)) ≈ Dx(y_in,Nx,Ny,1/Nx)

    Nx = Ny = 1000
    y_in = randn(Nx*Ny)
    y_in_GPU = CuArray(y_in)
    y_out_GPU = similar(y_in_GPU)
    y_out_GPU = D2x_GPU(y_in_GPU,y_out_GPU,Nx,Ny)
    @test Array(y_out_GPU) ≈ D2x(y_in,Nx,Ny,1/Nx)
    @test Array(Dx_GPU(y_in_GPU,y_out_GPU,Nx,Ny)) ≈ Dx(y_in,Nx,Ny,1/Nx)
end
