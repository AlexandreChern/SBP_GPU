using CUDA
using SparseArrays
using LinearMaps
#using IterativeSolvers
using Parameters
using BenchmarkTools
using LinearAlgebra
using SBP_GPU
using Random

include("deriv_ops_beta.jl")
include("deriv_ops_GPU.jl")


struct variables
    Nx::Int64
    Ny::Int64
    N::Int64
    hx::Float64
    hy::Float64
    x
    y
    alpha1::Float64
    alpha2::Float64
    alpha3::Float64
    alpha4::Float64
    beta::Float64
end

Nx = 1001
Ny = 1001
N = Nx * Ny

variable = variables(Nx,Ny,N,(1/(Nx-1)),(1/(Ny-1)),0:1/(Nx-1):1,0:1/(Ny-1):1,-1.0,-1.0,-13/(1/(Nx-1)),-13/(1/(Ny-1)),1.0)



# intermediate results
mutable struct intermediates
    Nx::Int64
    Ny::Int64
    N::Int64
    Vol2Face::Array{Array{Float64,1},1}
    du_x::Array{Float64,1}
    du_y::Array{Float64,1}
    du_ops::Array{Float64,1}
    du1::Array{Float64,1}
    du2::Array{Float64,1}
    du3::Array{Float64,1}
    du4::Array{Float64,1}
    du5::Array{Float64,1}
    du6::Array{Float64,1}
    du7::Array{Float64,1}
    du8::Array{Float64,1}
    du9::Array{Float64,1}
    du10::Array{Float64,1}
    du11::Array{Float64,1}
    du12::Array{Float64,1}
    du13::Array{Float64,1}
    du14::Array{Float64,1}
    du15::Array{Float64,1}
    du16::Array{Float64,1}
    du17::Array{Float64,1}
    du0::Array{Float64,1}
    du::Array{Float64,1}
end

intermediate = intermediates(Nx,Ny,N,[zeros(N),zeros(N),zeros(N),zeros(N)],zeros(N),zeros(N),zeros(N),zeros(N),zeros(N),zeros(N),zeros(N),zeros(N),zeros(N),zeros(N),zeros(N),zeros(N),zeros(N),zeros(N),zeros(N),zeros(N),zeros(N),zeros(N),zeros(N),zeros(N),zeros(N),zeros(N))


mutable struct intermediates_GPU_mutable
    Nx::Int64
    Ny::Int64
    N::Int64
    # du_ops::CuArray{Float64,1}
    du_x::CuArray{Float64,1}
    du_y::CuArray{Float64,1}
    du_ops::CuArray{Float64,1}
    du1::CuArray{Float64,1}
    du2::CuArray{Float64,1}
    du3::CuArray{Float64,1}
    du4::CuArray{Float64,1}
    du5::CuArray{Float64,1}
    du6::CuArray{Float64,1}
    du7::CuArray{Float64,1}
    du8::CuArray{Float64,1}
    du9::CuArray{Float64,1}
    du10::CuArray{Float64,1}
    du11::CuArray{Float64,1}
    du12::CuArray{Float64,1}
    du13::CuArray{Float64,1}
    du14::CuArray{Float64,1}
    du15::CuArray{Float64,1}
    du16::CuArray{Float64,1}
    du17::CuArray{Float64,1}
    du0::CuArray{Float64,1}
    du::CuArray{Float64,1}
end



# N = Nx*Ny
cu_zeros = CuArray(zeros(N))
iGm = intermediates_GPU_mutable(Nx,Ny,N,CuArray(zeros(N)),CuArray(zeros(N)),CuArray(zeros(N)),CuArray(zeros(N)),CuArray(zeros(N)),CuArray(zeros(N)),CuArray(zeros(N)),CuArray(zeros(N)),CuArray(zeros(N)),CuArray(zeros(N)),CuArray(zeros(N)),CuArray(zeros(N)),CuArray(zeros(N)),CuArray(zeros(N)),CuArray(zeros(N)),CuArray(zeros(N)),CuArray(zeros(N)),CuArray(zeros(N)),CuArray(zeros(N)),CuArray(zeros(N)),CuArray(zeros(N)),CuArray(zeros(N)));


# function myMAT_beta_GPU!(du_GPU::AbstractVector, u_GPU::AbstractVector, container, var_test) # , intermediates_GPU_mutable)
function myMAT_beta_GPU!(du_GPU::AbstractVector, u_GPU::AbstractVector, iGm, var_test) # , intermediates_GPU_mutable)
    # @unpack N, y_D2x, y_D2y, y_Dx, y_Dy, y_Hxinv, y_Hyinv, yv2f1, yv2f2, yv2f3, yv2f4, yv2fs, yf2v1, yf2v2, yf2v3, yf2v4, yf2vs, y_Bx, y_By, y_BxSx, y_BySy, y_BxSx_tran, y_BySy_tran, y_Hx, y_Hy = container
    @unpack Nx,Ny,N,hx,hy,alpha1,alpha2,alpha3,alpha4,beta = var

    # N = Nx*Ny
    # cu_zeros = CuArray(zeros(N))
    # iGm = intermediates_GPU_mutable(Nx,Ny,N,cu_zeros,cu_zeros,cu_zeros,cu_zeros,cu_zeros,cu_zeros,cu_zeros,cu_zeros,cu_zeros,cu_zeros,cu_zeros,cu_zeros,cu_zeros,cu_zeros,cu_zeros,cu_zeros,cu_zeros,cu_zeros,cu_zeros,cu_zeros,cu_zeros,cu_zeros);

    TILE_DIM_1 = 4
    TILE_DIM_2 = 16

    griddim_x = (div(Nx,TILE_DIM_1) + 1, div(Ny,TILE_DIM_2) + 1)
    griddim_y = (div(Nx,TILE_DIM_2) + 1, div(Ny,TILE_DIM_1) + 1)

    blockdim_x = (TILE_DIM_1,TILE_DIM_2)
    blockdim_y = (TILE_DIM_2,TILE_DIM_1)

    # @show typeof(u)
    # @show typeof(iGm.du_x)
    # @show blockdim_x
    # @show griddim_x
    # @show size(u)
    # @show size(iGm.du_x)
    @cuda threads=blockdim_x blocks=griddim_x D2x_GPU_shared(u_GPU,iGm.du_x, Nx, Ny, hx, Val(TILE_DIM_1), Val(TILE_DIM_2))
    # @show Array(iGm.du_x)
    # output = Array(iGm.du_x)
    # output_GPU = iGm.du_x
    synchronize()
    @cuda threads=blockdim_y blocks=griddim_y D2y_GPU_shared(u_GPU,iGm.du_y, Nx, Ny, hy, Val(TILE_DIM_2), Val(TILE_DIM_1))
    synchronize()
    iGm.du_ops = iGm.du_x + iGm.du_y
    output2 = Array(du_ops)
    @cuda threads=blockdim_y blocks=griddim_y BySy_GPU_shared(u_GPU,iGm.du1, Nx, Ny, hy, Val(TILE_DIM_2), Val(TILE_DIM_1))
    # @show iGm.du_x

    synchronize()
    iGm.du2 .= CuArray(VOLtoFACE_beta!(Array(iGm.du1),1,Nx,Ny,N,yv2fs))
    @cuda threads=blockdim_y blocks=griddim_y Hyinv_GPU_shared(iGm.du2,iGm.du3,Nx,Ny,hy, Val(TILE_DIM_2), Val(TILE_DIM_1))
    synchronize()
    iGm.du3 = alpha1 * iGm.du3

    iGm.du5 = VOLtoFACE_beta!(Array(iGm.du1),2,Nx,Ny,N,yv2fs)
    @cuda threads=blockdim_y blocks=griddim_y Hyinv_GPU_shared(iGm.du5,iGm.du6,Nx,Ny,hy, Val(TILE_DIM_2), Val(TILE_DIM_1))
    synchronize()
    iGm.du6 = alpha2 * iGm.du6

    iGm.du7 = CuArray(VOLtoFACE_beta!(Array(u_GPU),3,Nx,Ny,N,yv2fs))
    @cuda threads=blockdim_x blocks=griddim_x BxSx_tran_GPU_shared(iGm.du7,iGm.du8,Nx,Ny,hx,Val(TILE_DIM_1), Val(TILE_DIM_2))
    synchronize()
    @cuda threads=blockdim_x blocks=griddim_x Hxinv_GPU_shared(iGm.du8,iGm.du9,Nx,Ny,hx, Val(TILE_DIM_1), Val(TILE_DIM_2))
    synchronize()
    iGm.du9 = beta * iGm.du9

    @cuda threads=blockdim_x blocks=griddim_x Hxinv_GPU_shared(iGm.du7,iGm.du11,Nx,Ny,hx, Val(TILE_DIM_1), Val(TILE_DIM_2))
    synchronize()
    iGm.du11 =alpha3 * iGm.du11

    iGm.du12 = CuArray(VOLtoFACE_beta!(Array(u_GPU),4,Nx,Ny,N,yv2fs))
    synchronize()
    @cuda threads=blockdim_x blocks=griddim_x BxSx_tran_GPU_shared(iGm.du12,iGm.du13,Nx,Ny,hx,Val(TILE_DIM_1), Val(TILE_DIM_2))
    synchronize()
    @cuda threads=blockdim_x blocks=griddim_x Hxinv_GPU_shared(iGm.du13,iGm.du14,Nx,Ny,hx,Val(TILE_DIM_1), Val(TILE_DIM_2))
    synchronize()
    iGm.du14 = beta * iGm.du14
    @cuda threads=blockdim_x blocks=griddim_x Hxinv_GPU_shared(iGm.du12,iGm.du16,Nx,Ny,hx,Val(TILE_DIM_1), Val(TILE_DIM_2))
    synchronize()
    iGm.du16 = alpha4 * iGm.du16
    synchronize()
    iGm.du0 = iGm.du_ops + iGm.du3 + iGm.du6 + iGm.du9 + iGm.du11 + iGm.du14 + iGm.du16
    synchronize()
    # comment: starting this line, iGm.du17 is not returned with correct solution
    # @cuda threads=blockdim_y blocks=griddim_y Hy_GPU_shared(iGm.du0,iGm.du17,Nx,Ny,hx,Val(TILE_DIM_1),Val(TILE_DIM_2))
    # synchronize()
    @cuda threads=blockdim_x blocks=griddim_x Hy_GPU_shared(iGm.du0,iGm.du17,Nx,Ny,hx,Val(TILE_DIM_1),Val(TILE_DIM_2)) # Looks likes something wrong with griddim_y and blockdim_y
    synchronize()
    @cuda threads=blockdim_x blocks=griddim_x Hx_GPU_shared(iGm.du17,iGm.du,Nx,Ny,hx,Val(TILE_DIM_1),Val(TILE_DIM_2))
    synchronize()
    iGm.du = -1.0 * iGm.du
    # return Array(iGm.du_x)
    # @show output
    # output_final = copy(iGm.du);
    # @show output_final[1:10]
    return Array(iGm.du)
    # return output_final
    # return output2
end

Random.seed!(1234)
u = randn(N);
u_GPU = CuArray(u);

du = similar(u);
du_GPU = similar(u_GPU);

# myMAT_beta_GPU!(du_GPU,u_GPU,iGm,var)


function myMAT_beta!(du::AbstractVector, u::AbstractVector,variable,intermediate)
    D2x_beta!(u,variable.Nx,variable.Ny,variable.N,variable.hx,variable.hy,intermediate.du_x)
    D2y_beta!(u,variable.Nx,variable.Ny,variable.N,variable.hx,variable.hy,intermediate.du_y)

    intermediate.du_ops .= intermediate.du_x + intermediate.du_y
    BySy_beta!(u,variable.Nx,variable.Ny,variable.N,variable.hx,variable.hy,intermediate.du1)
    VOLtoFACE_beta!(intermediate.du1,1,variable.Nx,variable.Ny,variable.N,intermediate.Vol2Face)
    Hyinv_beta!(intermediate.Vol2Face[1],variable.Nx,variable.Ny,variable.N,variable.hx,variable.hy,variable.alpha1,intermediate.du3)     #compute action of P1  .= for faster assignment

    VOLtoFACE_beta!(intermediate.du1,2,variable.Nx,variable.Ny,variable.N,intermediate.Vol2Face)
    Hyinv_beta!(intermediate.Vol2Face[2],variable.Nx,variable.Ny,variable.N,variable.hx,variable.hy, variable.alpha2,intermediate.du6)
    
    #compute action of P2
    VOLtoFACE_beta!(u,3,variable.Nx,variable.Ny,variable.N,intermediate.Vol2Face)
    BxSx_tran_beta!(intermediate.Vol2Face[3],variable.Nx,variable.Ny,variable.N,variable.hx,variable.hy,intermediate.du8)
    Hxinv_beta!(intermediate.du8,variable.Nx,variable.Ny,variable.N,variable.hx,variable.hy, variable.beta, intermediate.du9)
    Hxinv_beta!(intermediate.Vol2Face[3],variable.Nx,variable.Ny,variable.N,variable.hx,variable.hy,variable.alpha3,intermediate.du11)   #compute action of P3

    VOLtoFACE_beta!(u,4,variable.Nx,variable.Ny,variable.N,intermediate.Vol2Face)
    BxSx_tran_beta!(intermediate.Vol2Face[4],variable.Nx,variable.Ny,variable.N,variable.hx,variable.hy,intermediate.du13)
    Hxinv_beta!(intermediate.du13,variable.Nx,variable.Ny,variable.N,variable.hx,variable.hy,variable.beta,intermediate.du14)
    Hxinv_beta!(intermediate.Vol2Face[4],variable.Nx,variable.Ny,variable.N,variable.hx,variable.hy,variable.alpha4,intermediate.du16)  #compute action of P4
    intermediate.du0 .= intermediate.du_ops .+ intermediate.du3 .+ intermediate.du6 .+ intermediate.du9 .+ intermediate.du11 .+ intermediate.du14 .+ intermediate.du16 #Collect together
    Hy_beta!(intermediate.du0,variable.Nx,variable.Ny,variable.N,variable.hx,variable.hy,intermediate.du17)
	Hx_beta!(intermediate.du17,variable.Nx,variable.Ny,variable.N,variable.hx,variable.hy, -1.0,intermediate.du)
    return intermediate.du
end


function Generate(variable)
    # @unpack Nx,Ny,N,hx,hy,x,y,alpha1,alpha2,alpha3,alpha4,beta = var
    Nx = variable.Nx
    Ny = variable.Ny
    N = variable.N
    hx = variable.hx
    hy = variable.hy
    x = variable.x
    y = variable.y
    alpha1 = variable.alpha1
    alpha2 = variable.alpha2
    alpha3 = variable.alpha3
    alpha4 = variable.alpha4
    beta = variable.beta

    g1 = -pi .* cos.(pi .* x)
    g2 = pi .* cos.(pi .* x .+ pi)
    g3 = sin.(pi .* y)
    g4 = sin.(pi .+ pi .* y)

    f = spzeros(Nx,Ny)
    exactU = spzeros(Nx,Ny)

    for i = 1:Nx
    	for j = 1:Ny
    		f[j,i] = -pi^2 .* sin.(pi .* x[i] + pi .* y[j]) - pi^2 .* sin.(pi .* x[i] + pi .* y[j])
    		exactU[j,i] = sin.(pi .* x[i] + pi .* y[j])
    	end
    end

    f = f[:]
    exact = exactU[:]

    #Construct vector b
    b0 = FACEtoVOL(g1,1,Nx,Ny)
    b1 = alpha1*Hyinv(b0,Nx,Ny,hy)

    b2 = FACEtoVOL(g2,2,Nx,Ny)
    b3 = alpha2*Hyinv(b2,Nx,Ny,hy)

    b4 = FACEtoVOL(g3,3,Nx,Ny)
    b5 = alpha3*Hxinv(b4,Nx,Ny,hx)
    b6 = BxSx_tran(b4,Nx,Ny,hx)
    b7 = beta*Hxinv(b6,Nx,Ny,hx)

    b8 = FACEtoVOL(g4,4,Nx,Ny)
    b9 = alpha4*Hxinv(b8,Nx,Ny,hx)
    b10 = BxSx_tran(b8,Nx,Ny,hx)
    b11 = beta*Hxinv(b10,Nx,Ny,hx)

    bb = b1  + b3  + b5 + b7 + b9 + b11 + f

    #Modify b for PD system
    b12 = Hx(bb,Nx,Ny,hx)
    b = -Hy(b12,Nx,Ny,hy)
    return b,exact
end

b,exact = Generate(variable)



@with_kw struct var_cgs
    Nx = 1001
    Ny = 1001
    N = Nx*Ny
    u = randn(N)
    du = similar(u)
end


var_cg = var_cgs()

#@unpack u, du, N = var_cg
Random.seed!(1234)
du = zeros(N);
u = zeros(N);



# function conjugate(myMAT_beta!,b,container,var,intermediate,maxIteration)
#     @unpack N, y_D2x, y_D2y, y_Dx, y_Dy, y_Hxinv, y_Hyinv, yv2f1, yv2f2, yv2f3, yv2f4, yv2fs, yf2v1, yf2v2, yf2v3, yf2v4, yf2vs, y_Bx, y_By, y_BxSx, y_BySy, y_BxSx_tran, y_BySy_tran, y_Hx, y_Hy = container
#     @unpack Nx,Ny,N,hx,hy,alpha1,alpha2,alpha3,alpha4,beta = var
#     @unpack du_ops,du1,du2,du3,du4,du5,du6,du7,du8,du9,du10,du11,du12,du13,du14,du15,du16,du17,du0 = intermediate

#     u = zeros(N);
#     tol = 1e-16
#     r = b - myMAT_beta!(du,u,container,var,intermediate,maxIteration)
#     p = r
#     rsold = r'*r

#     counts = 0
#     for i = 1:N
#         Ap = myMAT_beta!(du,p,container,var,intermediate,maxIteration)   # can't simply translate MATLAB code, p = r create a link from p to r, once p modified, r will be modified
#         alpha = rsold / (p'*Ap)
#         u = u + alpha * p
#         #axpy!(alpha,p,u)
#         r = r - alpha * Ap
#         #axpy!(-alpha,Ap,r)
#         rsnew = r'*r
#         if sqrt(rsnew) < tol
#             break
#         end
#         p = r + (rsnew/rsold) * p
#         rsold = rsnew;
#         counts += 1
#         #return rsold;
#     end
#     return u, counts
# end

r = similar(u)
function conjugate_beta(myMAT_beta!,r,b,variable,intermediate,maxIteration)
    # @unpack N, y_D2x, y_D2y, y_Dx, y_Dy, y_Hxinv, y_Hyinv, yv2f1, yv2f2, yv2f3, yv2f4, yv2fs, yf2v1, yf2v2, yf2v3, yf2v4, yf2vs, y_Bx, y_By, y_BxSx, y_BySy, y_BxSx_tran, y_BySy_tran, y_Hx, y_Hy = container
    # @unpack Nx,Ny,N,hx,hy,alpha1,alpha2,alpha3,alpha4,beta = var
    # @unpack du_ops,du1,du2,du3,du4,du5,du6,du7,du8,du9,du10,du11,du12,du13,du14,du15,du16,du17,du0 = intermediate

    N = variable.N
    u = zeros(N);
    du = zeros(N);
    tol = 1e-16

    r .= b .- myMAT_beta!(du,u,variable,intermediate)
    p = copy(r)
    Ap = similar(u)
    rsold = r'*r
    counts = 0
    # maxIteration = 1000
    for i = 1:maxIteration
        Ap .= myMAT_beta!(du,p,variable,intermediate)   # can't simply translate MATLAB code, p = r create a link from p to r, once p modified, r will be modified
        alpha = rsold / (p'*Ap)
        #u = u + alpha * p
        axpy!(alpha,p,u) # BLAS function
        #r = r - alpha * Ap
        axpy!(-alpha,Ap,r)
        rsnew = r'*r
        if sqrt(rsnew) < tol
            break
        end
        #p = r + (rsnew/rsold) * p
        #p .= r .+ (rsnew/rsold) .*p
        p .= (rsnew/rsold) .* p .+ r

        rsold = rsnew;
        counts += 1
        #return rsold;
    end
    return u, counts
end


# function conjugate_beta_GPU(myMAT_beta_GPU!,r,b,container,var,intermediate,maxIteration)
function conjugate_beta_GPU(myMAT_beta_GPU!,b,var,maxIteration)
    @unpack N, y_D2x, y_D2y, y_Dx, y_Dy, y_Hxinv, y_Hyinv, yv2f1, yv2f2, yv2f3, yv2f4, yv2fs, yf2v1, yf2v2, yf2v3, yf2v4, yf2vs, y_Bx, y_By, y_BxSx, y_BySy, y_BxSx_tran, y_BySy_tran, y_Hx, y_Hy = container
    @unpack Nx,Ny,N,hx,hy,alpha1,alpha2,alpha3,alpha4,beta = var
    # @unpack du_ops,du1,du2,du3,du4,du5,du6,du7,du8,du9,du10,du11,du12,du13,du14,du15,du16,du17,du0 = intermediate
    iGm = intermediates_GPU_mutable(Nx,Ny,N,CuArray(zeros(N)),CuArray(zeros(N)),CuArray(zeros(N)),CuArray(zeros(N)),CuArray(zeros(N)),CuArray(zeros(N)),CuArray(zeros(N)),CuArray(zeros(N)),CuArray(zeros(N)),CuArray(zeros(N)),CuArray(zeros(N)),CuArray(zeros(N)),CuArray(zeros(N)),CuArray(zeros(N)),CuArray(zeros(N)),CuArray(zeros(N)),CuArray(zeros(N)),CuArray(zeros(N)),CuArray(zeros(N)),CuArray(zeros(N)),CuArray(zeros(N)),CuArray(zeros(N)));
    # u = zeros(N);
    # du = zeros(N);
    u_GPU = CuArray(zeros(N))
    du_GPU = CuArray(zeros(N))
    tol = 1e-16

    r = similar(u)
    r .= b .- Array(myMAT_beta_GPU!(du_GPU,u_GPU,iGm,var))
    p = copy(r)
    Ap = similar(u)
    rsold = r'*r
    counts = 0
    # maxIteration = 1000
    for i = 1:maxIteration
        Ap = Array(myMAT_beta_GPU!(du_GPU,CuArray(p),iGm,var))   # can't simply translate MATLAB code, p = r create a link from p to r, once p modified, r will be modified
        Ap = Array(Ap)
        alpha = rsold / (p'*Ap)
        #u = u + alpha * p
        # axpy!(alpha,p,Array(u)) # BLAS function
        u_GPU = u_GPU + alpha * CuArray(p)
        #r = r - alpha * Ap
        # axpy!(-alpha,Ap,r)
        r = r - alpha * Ap
        rsnew = r'*r
        if sqrt(rsnew) < tol
            break
        end
        #p = r + (rsnew/rsold) * p
        #p .= r .+ (rsnew/rsold) .*p
        p .= (rsnew/rsold) .* p .+ r

        rsold = rsnew;
        counts += 1
        #return rsold;
    end
    return u_GPU, counts
end

# (uGPU, countsGPU) = conjugate_beta_GPU(myMAT_beta_GPU!,b,var,100)
# conjugate_beta_GPU(myMAT_beta_GPU!,r,b,container,var,intermediate,100)


(u1,counts1) = conjugate_beta(myMAT_beta!,r,b,variable,intermediate,100)
u1 = copy(u1)
(u2,counts2) = conjugate_beta(myMAT_beta!,r,b,container,var,intermediate,2000)
u2 = copy(u2)
(u3,counts3) = conjugate_beta(myMAT_beta!,r,b,container,var,intermediate,3000)
u3 = copy(u3)
(u4,counts4) = conjugate_beta(myMAT_beta!,r,b,container,var,intermediate,1000)
u4 = copy(u4)
(u5,counts5) = conjugate_beta(myMAT_beta!,r,b,container,var,intermediate,1000)
u5 = copy(u5)

counts1
counts2
counts3
counts4
counts5

err_1 = norm(u1 - exact)
err_2 = norm(u2 - exact)
err_3 = norm(u3 - exact)
err_4 = norm(u4 - exact)
err_5 = norm(u5 - exact)



err_norm1 = norm(u1 - exact)
err_norm2 = norm(u2 - exact)

function cg!(du, u, b, myMAT_beta! , tol=1e-16)

    g = similar(u)
    g .= 0
    x = similar(u)
    x .= 0


    myMAT_beta!(g, x, container, var, intermediate)
    g .-= b
    u .=  -g

    gTg = dot(g, g)
    maxiteration = N
    start_time = time()
    for k = 1:maxiteration
        if gTg < tol^2
            end_time = time()
            total_time = end_time - start_time
            return gTg, k, total_time
        end
    gTg = cg_iteration!(x, g, du, u, myMAT_beta!, gTg)
    end

    end_time = time()
    total_time = end_time - start_time
    return u, gTg, maxiteration+1, total_time
end



# function cg_iteration!(u,du, g, w, myMAT_beta!, gTg) version 1
#     myMAT_beta!(du, u, container,var,intermediate)
#     dTw = dot(du, w)
#     alpha = gTg / dTw
#     u .+= alpha .* du
#     g .+= alpha .* w
#     g1Tg1 = dot(g, g)
#     beta = g1Tg1 / gTg
#     u .= .-g .+ beta .* du
#     # u .= u
#     return g1Tg1
# end

function cg_iteration!(x, g, du, u, myMAT_beta!, gTg) #version 2
    #@unpack u, du, N = var_cg
    myMAT_beta!(du, u, container, var, intermediate)
    dTw = dot(u, du)
    alpha = gTg / dTw
    x .+= alpha .* u
    g .+= alpha .* du
    g1Tg1 = dot(g, g)
    beta = g1Tg1 / gTg
    u .= .-g .+ beta .* u
    x .= x
    return g1Tg1
end
