using CUDA
using Printf
using StaticArrays
using GPUifyLoops: @unroll

# include("deriv_ops.jl")

###
# y_in: input GPU array
# y_out: output GPU array
###

# function D2x_GPU(y_in, y_out, Nx, Ny, h, ::Val{TILE_DIM}) where {TILE_DIM}
# 	tidx = (blockIdx().x - 1) * TILE_DIM + threadIdx().x
# 	N = Nx*Ny
# 	# y_out = zeros(N)
# 	if tidx <= Ny
# 		y_out[tidx] = (y_in[tidx] - 2 * y_in[Ny + tidx] + y_in[2*Ny + tidx]) / h^2
# 	end
# 	sync_threads()

# 	if Ny+1 <= tidx <= N-Ny
# 		y_out[tidx] = (y_in[tidx - Ny] - 2 .* y_in[tidx] + y_in[tidx + Ny]) / h^2
# 	end

# 	sync_threads()

# 	if N-Ny+1 <= tidx <= N
# 		y_out[tidx] = (y_in[tidx - 2*Ny] -2 * y_in[tidx - Ny] + y_in[tidx]) / h^2
# 	end
# 	sync_threads()

# 	nothing
# end

function D2x_GPU(y_in::CuArray,y_out::CuArray,Nx,Ny)
	y_out = similar(y_in)
	TILE_DIM_1 = 4
	TILE_DIM_2 = 16
	h = 1/Nx
	griddim = (div(Nx,TILE_DIM_1) + 1, div(Ny,TILE_DIM_2) + 1)
	blockdim = (TILE_DIM_1,TILE_DIM_2)
	@cuda threads=blockdim blocks=griddim D2x_GPU_shared(y_in, y_out, Nx, Ny, h, Val(TILE_DIM_1), Val(TILE_DIM_2))
	synchronize();
	return y_out
end

function D2x_GPU_shared(y_in, y_out, Nx, Ny, h, ::Val{TILE_DIM1}, ::Val{TILE_DIM2}) where {TILE_DIM1, TILE_DIM2}
    tidx = threadIdx().x
    tidy = threadIdx().y

    # for global memory indexing
    i = (blockIdx().x - 1) * TILE_DIM1 + tidx
    j = (blockIdx().y - 1) * TILE_DIM2 + tidy

    global_index = i + (j - 1) * Ny

    HALO_WIDTH = 2 # For second order derivative

    tile = @cuStaticSharedMem(eltype(y_in), (TILE_DIM1, TILE_DIM2 + 2 * HALO_WIDTH))

    # for tile indexing
    k = tidx
    l = tidy

    
	# Writing pencil-shaped shared memory

	# for tile itself
	# if k <= TILE_DIM1 && l <= TILE_DIM2 && global_index <= Nx*Ny
	if k <= TILE_DIM1 && l <= TILE_DIM2 && i <= Ny && j <= Nx
		# @inbounds tile[k,l+HALO_WIDTH] = y_in[global_index]
		 tile[k,l+HALO_WIDTH] = y_in[global_index]
    end
    
    sync_threads()

	# for left halo
	# if k <= TILE_DIM1 && l <= HALO_WIDTH && HALO_WIDTH*Ny+1 <= global_index <= (Nx+HALO_WIDTH)*Ny
	if k <= TILE_DIM1 && l <= HALO_WIDTH && i <= Ny && HALO_WIDTH+1 <= j <= HALO_WIDTH + Nx 
		# @inbounds tile[k,l] = y_in[global_index - HALO_WIDTH*Ny]
		tile[k,l] = y_in[global_index - HALO_WIDTH*Ny]
	end

	sync_threads()


	# for right halo
	# if k <= TILE_DIM1 && l >= TILE_DIM2 - HALO_WIDTH && HALO_WIDTH*Ny+1 <= global_index <= (Nx-HALO_WIDTH)*Ny
	if k <= TILE_DIM1 && TILE_DIM2 - HALO_WIDTH + 1 <= l <= TILE_DIM2 && i <= Ny && j <= Nx - HALO_WIDTH
		# @inbounds tile[k,l+2*HALO_WIDTH] = y_in[global_index + HALO_WIDTH*Ny]
		tile[k,l+2*HALO_WIDTH] = y_in[global_index + HALO_WIDTH*Ny]
	end

    sync_threads()

    # Finite difference operation starts here

	# Left Boundary
	if k <= TILE_DIM1 && l + HALO_WIDTH <= TILE_DIM2 + 2*HALO_WIDTH -2 && i <= Ny && j == 1
		# @inbounds y_out[global_index] = (tile[k,l + HALO_WIDTH] - 2*tile[k,l + HALO_WIDTH+1] + tile[k,l + HALO_WIDTH+2]) / h^2
		y_out[global_index] = (tile[k,l + HALO_WIDTH] - 2*tile[k,l + HALO_WIDTH+1] + tile[k,l + HALO_WIDTH+2]) / h^2
	end

	# Center
	if k <= TILE_DIM1 && l + HALO_WIDTH <= TILE_DIM2 + 2*HALO_WIDTH - 1 && i <= Ny && 2 <= j <= Nx-1
		# @inbounds y_out[global_index] = (tile[k,l + HALO_WIDTH-1] - 2*tile[k, l + HALO_WIDTH] + tile[k,l + HALO_WIDTH + 1]) / h^2
		y_out[global_index] = (tile[k,l + HALO_WIDTH-1] - 2*tile[k, l + HALO_WIDTH] + tile[k,l + HALO_WIDTH + 1]) / h^2
	end

	# Right Boundary
	if k <= TILE_DIM1 && 3 <= l + HALO_WIDTH <= TILE_DIM2 + 2*HALO_WIDTH && i <= Ny && j == Nx
		@inbounds y_out[global_index] = (tile[k,l + HALO_WIDTH-2] - 2*tile[k,l + HALO_WIDTH - 1] + tile[k,l + HALO_WIDTH]) / h^2
		# y_out[global_index] = (tile[k,l + HALO_WIDTH-2] - 2*tile[k,l + HALO_WIDTH - 1] + tile[k,l + HALO_WIDTH]) / h^2
		# y_out[global_index] = (tile[k,l+HALO_WIDTH-2])
		# y_out[global_index] = 0
	end

    sync_threads()
    
    nothing
end

function Dx_GPU(y_in::CuArray,y_out::CuArray,Nx,Ny)
	y_out = similar(y_in)
	TILE_DIM_1 = 4
	TILE_DIM_2 = 16
	h = 1/Nx
	griddim = (div(Nx,TILE_DIM_1) + 1, div(Ny,TILE_DIM_2) + 1)
	blockdim = (TILE_DIM_1,TILE_DIM_2)
	@cuda threads=blockdim blocks=griddim Dx_GPU_shared(y_in, y_out, Nx, Ny, h, Val(TILE_DIM_1), Val(TILE_DIM_2))
	synchronize();
	return y_out
end

function Dx_GPU_shared(y_in, y_out, Nx, Ny, h, ::Val{TILE_DIM1}, ::Val{TILE_DIM2}) where {TILE_DIM1, TILE_DIM2}
	tidx = threadIdx().x
    tidy = threadIdx().y

    # for global memory indexing
    i = (blockIdx().x - 1) * TILE_DIM1 + tidx
    j = (blockIdx().y - 1) * TILE_DIM2 + tidy

    global_index = i + (j - 1) * Ny

    HALO_WIDTH = 2 # For second order derivative

    tile = @cuStaticSharedMem(eltype(y_in), (TILE_DIM1, TILE_DIM2 + 2 * HALO_WIDTH))

    # for tile indexing
    k = tidx
    l = tidy

    
	# Writing pencil-shaped shared memory

	# for tile itself
	# if k <= TILE_DIM1 && l <= TILE_DIM2 && global_index <= Nx*Ny
	if k <= TILE_DIM1 && l <= TILE_DIM2 && i <= Ny && j <= Nx
		# @inbounds tile[k,l+HALO_WIDTH] = y_in[global_index]
		 tile[k,l+HALO_WIDTH] = y_in[global_index]
    end
    
    sync_threads()

	# for left halo
	# if k <= TILE_DIM1 && l <= HALO_WIDTH && HALO_WIDTH*Ny+1 <= global_index <= (Nx+HALO_WIDTH)*Ny
	if k <= TILE_DIM1 && l <= HALO_WIDTH && i <= Ny && HALO_WIDTH+1 <= j <= HALO_WIDTH + Nx 
		# @inbounds tile[k,l] = y_in[global_index - HALO_WIDTH*Ny]
		tile[k,l] = y_in[global_index - HALO_WIDTH*Ny]
	end

	sync_threads()


	# for right halo
	# if k <= TILE_DIM1 && l >= TILE_DIM2 - HALO_WIDTH && HALO_WIDTH*Ny+1 <= global_index <= (Nx-HALO_WIDTH)*Ny
	if k <= TILE_DIM1 && TILE_DIM2 - HALO_WIDTH + 1<= l <= TILE_DIM2 && i <= Ny && j <= Nx - HALO_WIDTH
		# @inbounds tile[k,l+2*HALO_WIDTH] = y_in[global_index + HALO_WIDTH*Ny]
		tile[k,l+2*HALO_WIDTH] = y_in[global_index + HALO_WIDTH*Ny]
	end

    sync_threads()

    # Finite difference operation starts here

	# Left Boundary
	if k <= TILE_DIM1 && l + HALO_WIDTH <= TILE_DIM2 + 2*HALO_WIDTH -2 && i <= Ny && j == 1
		# @inbounds y_out[global_index] = (tile[k,l + HALO_WIDTH] - 2*tile[k,l + HALO_WIDTH+1] + tile[k,l + HALO_WIDTH+2]) / h^2
		y_out[global_index] = (tile[k,l + HALO_WIDTH+1] - tile[k,l + HALO_WIDTH]) / h
	end

	# Center
	if k <= TILE_DIM1 && l + HALO_WIDTH <= TILE_DIM2 + 2*HALO_WIDTH - 1 && i <= Ny && 2 <= j <= Nx-1
		# @inbounds y_out[global_index] = (tile[k,l + HALO_WIDTH-1] - 2*tile[k, l + HALO_WIDTH] + tile[k,l + HALO_WIDTH + 1]) / h^2
		y_out[global_index] = (tile[k,l + HALO_WIDTH + 1] - tile[k,l + HALO_WIDTH - 1]) / (2*h)
	end

	# Right Boundary
	if k <= TILE_DIM1 && 3 <= l + HALO_WIDTH <= TILE_DIM2 + 2*HALO_WIDTH && i <= Ny && j == Nx
		# @inbounds y_out[global_index] = (tile[k,l + HALO_WIDTH-2] - 2*tile[k,l + HALO_WIDTH - 1] + tile[k,l + HALO_WIDTH]) / h^2
		y_out[global_index] = (tile[k,l + HALO_WIDTH ] - tile[k,l + HALO_WIDTH - 1]) / h
	end

    sync_threads()
    
    nothing
end

function Hxinv_GPU_shared(y_in, y_out, Nx, Ny, h, ::Val{TILE_DIM1}, ::Val{TILE_DIM2}) where {TILE_DIM1, TILE_DIM2}
	tidx = threadIdx().x
    tidy = threadIdx().y

    # for global memory indexing
    i = (blockIdx().x - 1) * TILE_DIM1 + tidx
    j = (blockIdx().y - 1) * TILE_DIM2 + tidy

    global_index = i + (j - 1) * Ny

    HALO_WIDTH = 2 # For second order derivative

    tile = @cuStaticSharedMem(eltype(y_in), (TILE_DIM1, TILE_DIM2 + 2 * HALO_WIDTH))

    # for tile indexing
    k = tidx
    l = tidy

    
	# Writing pencil-shaped shared memory

	# for tile itself
	# if k <= TILE_DIM1 && l <= TILE_DIM2 && global_index <= Nx*Ny
	if k <= TILE_DIM1 && l <= TILE_DIM2 && i <= Ny && j <= Nx
		# @inbounds tile[k,l+HALO_WIDTH] = y_in[global_index]
		 tile[k,l+HALO_WIDTH] = y_in[global_index]
    end
    
    sync_threads()

	# for left halo
	# if k <= TILE_DIM1 && l <= HALO_WIDTH && HALO_WIDTH*Ny+1 <= global_index <= (Nx+HALO_WIDTH)*Ny
	if k <= TILE_DIM1 && l <= HALO_WIDTH && i <= Ny && HALO_WIDTH+1 <= j <= HALO_WIDTH + Nx 
		# @inbounds tile[k,l] = y_in[global_index - HALO_WIDTH*Ny]
		tile[k,l] = y_in[global_index - HALO_WIDTH*Ny]
	end

	sync_threads()


	# for right halo
	# if k <= TILE_DIM1 && l >= TILE_DIM2 - HALO_WIDTH && HALO_WIDTH*Ny+1 <= global_index <= (Nx-HALO_WIDTH)*Ny
	if k <= TILE_DIM1 && TILE_DIM2 - HALO_WIDTH + 1<= l <= TILE_DIM2 && i <= Ny && j <= Nx - HALO_WIDTH
		# @inbounds tile[k,l+2*HALO_WIDTH] = y_in[global_index + HALO_WIDTH*Ny]
		tile[k,l+2*HALO_WIDTH] = y_in[global_index + HALO_WIDTH*Ny]
	end

    sync_threads()

    # Finite difference operation starts here

	# Left Boundary
	if k <= TILE_DIM1 && l + HALO_WIDTH <= TILE_DIM2 + 2*HALO_WIDTH -2 && i <= Ny && j == 1
		# @inbounds y_out[global_index] = (tile[k,l + HALO_WIDTH] - 2*tile[k,l + HALO_WIDTH+1] + tile[k,l + HALO_WIDTH+2]) / h^2
		y_out[global_index] = (2*tile[k,l + HALO_WIDTH]) / h
	end

	# Center
	if k <= TILE_DIM1 && l + HALO_WIDTH <= TILE_DIM2 + 2*HALO_WIDTH - 1 && i <= Ny && 2 <= j <= Nx-1
		# @inbounds y_out[global_index] = (tile[k,l + HALO_WIDTH-1] - 2*tile[k, l + HALO_WIDTH] + tile[k,l + HALO_WIDTH + 1]) / h^2
		y_out[global_index] = (tile[k,l + HALO_WIDTH] ) / (h)
	end

	# Right Boundary
	if k <= TILE_DIM1 && 3 <= l + HALO_WIDTH <= TILE_DIM2 + 2*HALO_WIDTH && i <= Ny && j == Nx
		# @inbounds y_out[global_index] = (tile[k,l + HALO_WIDTH-2] - 2*tile[k,l + HALO_WIDTH - 1] + tile[k,l + HALO_WIDTH]) / h^2
		y_out[global_index] = (2*tile[k,l + HALO_WIDTH ]) / h
	end

    sync_threads()
    
    nothing
end

function Hx_GPU_shared(y_in, y_out, Nx, Ny, h, ::Val{TILE_DIM1}, ::Val{TILE_DIM2}) where {TILE_DIM1, TILE_DIM2}
	tidx = threadIdx().x
    tidy = threadIdx().y

    # for global memory indexing
    i = (blockIdx().x - 1) * TILE_DIM1 + tidx
    j = (blockIdx().y - 1) * TILE_DIM2 + tidy

    global_index = i + (j - 1) * Ny

    HALO_WIDTH = 2 # For second order derivative

    tile = @cuStaticSharedMem(eltype(y_in), (TILE_DIM1, TILE_DIM2 + 2 * HALO_WIDTH))

    # for tile indexing
    k = tidx
    l = tidy

    
	# Writing pencil-shaped shared memory

	# for tile itself
	# if k <= TILE_DIM1 && l <= TILE_DIM2 && global_index <= Nx*Ny
	if k <= TILE_DIM1 && l <= TILE_DIM2 && i <= Ny && j <= Nx
		# @inbounds tile[k,l+HALO_WIDTH] = y_in[global_index]
		 tile[k,l+HALO_WIDTH] = y_in[global_index]
    end
    
    sync_threads()

	# for left halo
	# if k <= TILE_DIM1 && l <= HALO_WIDTH && HALO_WIDTH*Ny+1 <= global_index <= (Nx+HALO_WIDTH)*Ny
	if k <= TILE_DIM1 && l <= HALO_WIDTH && i <= Ny && HALO_WIDTH+1 <= j <= HALO_WIDTH + Nx 
		# @inbounds tile[k,l] = y_in[global_index - HALO_WIDTH*Ny]
		tile[k,l] = y_in[global_index - HALO_WIDTH*Ny]
	end

	sync_threads()


	# for right halo
	# if k <= TILE_DIM1 && l >= TILE_DIM2 - HALO_WIDTH && HALO_WIDTH*Ny+1 <= global_index <= (Nx-HALO_WIDTH)*Ny
	if k <= TILE_DIM1 && TILE_DIM2 - HALO_WIDTH + 1 <= l <= TILE_DIM2 && i <= Ny && j <= Nx - HALO_WIDTH
		# @inbounds tile[k,l+2*HALO_WIDTH] = y_in[global_index + HALO_WIDTH*Ny]
		tile[k,l+2*HALO_WIDTH] = y_in[global_index + HALO_WIDTH*Ny]
	end

    sync_threads()

    # Finite difference operation starts here

	# Left Boundary
	if k <= TILE_DIM1 && l + HALO_WIDTH <= TILE_DIM2 + 2*HALO_WIDTH -2 && i <= Ny && j == 1
		# @inbounds y_out[global_index] = (tile[k,l + HALO_WIDTH] - 2*tile[k,l + HALO_WIDTH+1] + tile[k,l + HALO_WIDTH+2]) / h^2
		y_out[global_index] = (h*tile[k,l + HALO_WIDTH]) / 2
	end

	# Center
	if k <= TILE_DIM1 && l + HALO_WIDTH <= TILE_DIM2 + 2*HALO_WIDTH - 1 && i <= Ny && 2 <= j <= Nx-1
		# @inbounds y_out[global_index] = (tile[k,l + HALO_WIDTH-1] - 2*tile[k, l + HALO_WIDTH] + tile[k,l + HALO_WIDTH + 1]) / h^2
		y_out[global_index] = h*(tile[k,l + HALO_WIDTH] )
	end

	# Right Boundary
	if k <= TILE_DIM1 && 3 <= l + HALO_WIDTH <= TILE_DIM2 + 2*HALO_WIDTH && i <= Ny && j == Nx
		# @inbounds y_out[global_index] = (tile[k,l + HALO_WIDTH-2] - 2*tile[k,l + HALO_WIDTH - 1] + tile[k,l + HALO_WIDTH]) / h^2
		y_out[global_index] = (h*tile[k,l + HALO_WIDTH ]) / 2
	end

    sync_threads()
    
    nothing
end

function D2y_GPU(y_in, y_out, Nx, Ny, h, ::Val{TILE_DIM}) where {TILE_DIM}
	tidx = (blockIdx().x - 1) * TILE_DIM + threadIdx().x
	N = Nx*Ny
	if 2 <= tidx <= N-1
		@inbounds y_out[tidx] = (y_in[tidx-1] - 2y_in[tidx] + y_in[tidx + 1]) / h^2
	end


	if 1 <= tidx <= N && mod(tidx,Ny) == 0
		@inbounds y_out[tidx] = (y_in[tidx] - 2y_in[tidx - 1] + y_in[tidx - 2]) / h^2
		@inbounds y_out[tidx-Ny+1] = (y_in[tidx-Ny+1] - 2y_in[tidx - Ny + 2] + y_in[tidx - Ny + 3]) / h^2
	end

	sync_threads()
	nothing
end

function D2y_GPU_shared(y_in, y_out, Nx, Ny, h, ::Val{TILE_DIM1}, ::Val{TILE_DIM2}) where {TILE_DIM1, TILE_DIM2}
    tidx = threadIdx().x
    tidy = threadIdx().y

	i = (blockIdx().x - 1) * TILE_DIM1 + tidx
	j = (blockIdx().y - 1) * TILE_DIM2 + tidy

	global_index = i + (j-1)*Nx

	HALO_WIDTH = 2
	tile = @cuStaticSharedMem(eltype(y_in),(TILE_DIM1+2*HALO_WIDTH,TILE_DIM2))

	k = tidx
	l = tidy

    # Writing pencil-shaped shared memory

    # for tile itself
	# if k <= TILE_DIM1 && l <= TILE_DIM2 && global_index <= Nx*Ny
	if k <= TILE_DIM1 && l <= TILE_DIM2 && i <= Ny && j <= Nx
		# @inbounds tile[k+HALO_WIDTH,l] = y_in[global_index]
		tile[k+HALO_WIDTH,l] = y_in[global_index]
	end

	sync_threads()

	# For upper halo
	# if k <= HALO_WIDTH && l <= TILE_DIM2 && HALO_WIDTH + 1 <= global_index <= Nx*Ny + HALO_WIDTH
	if k <= HALO_WIDTH && l <= TILE_DIM2 && HALO_WIDTH + 1 <= i <= Ny + HALO_WIDTH && j <= Nx
		# @inbounds tile[k,l] = y_in[global_index - HALO_WIDTH]
		tile[k,l] = y_in[global_index - HALO_WIDTH]
	end

	sync_threads()

	# For lower halo
	# if k >= TILE_DIM1 - HALO_WIDTH && l <= TILE_DIM2 && HALO_WIDTH + 1 <= global_index <= Nx*Ny - HALO_WIDTH
	if  TILE_DIM1 - HALO_WIDTH + 1 <= k <= TILE_DIM1 && l <= TILE_DIM2 && i <= Ny - HALO_WIDTH && j <= Nx
		# @inbounds tile[k+2*HALO_WIDTH,l] = y_in[global_index + HALO_WIDTH]
		tile[k+2*HALO_WIDTH,l] = y_in[global_index + HALO_WIDTH]
	end

    sync_threads()
    
    # Finite Difference Operations starts 

    #Upper Boundary
	if k + HALO_WIDTH <= TILE_DIM1 + 2*HALO_WIDTH -2 && l <= TILE_DIM2 && i == 1 && j <= Ny
		# @inbounds y_out[global_index] = (tile[k+HALO_WIDTH,l] - 2*tile[k+HALO_WIDTH+1,l] + tile[k+HALO_WIDTH+2,l]) / h^2
		y_out[global_index] = (tile[k+HALO_WIDTH,l] - 2*tile[k+HALO_WIDTH+1,l] + tile[k+HALO_WIDTH+2,l]) / h^2
	end

	sync_threads()

	#Center
	if k + HALO_WIDTH <= TILE_DIM1 + 2*HALO_WIDTH - 1 && l <= TILE_DIM2 && 2 <= i <= Nx-1 && j <= Ny
		# @inbounds y_out[global_index] = (tile[k+HALO_WIDTH-1,l] - 2*tile[k+HALO_WIDTH,l] + tile[k+HALO_WIDTH+1,l]) / h^2
		y_out[global_index] = (tile[k+HALO_WIDTH-1,l] - 2*tile[k+HALO_WIDTH,l] + tile[k+HALO_WIDTH+1,l]) / h^2
	end

	sync_threads()

	#Lower Boundary
	if 3 <= k + HALO_WIDTH <= TILE_DIM1 + 2*HALO_WIDTH && l <= TILE_DIM2 && i == Nx && j <= Ny
		# @inbounds y_out[global_index] = (tile[k+HALO_WIDTH-2,l] - 2*tile[k+HALO_WIDTH-1,l] + tile[k+HALO_WIDTH,l]) / h^2
		y_out[global_index] = (tile[k+HALO_WIDTH-2,l] - 2*tile[k+HALO_WIDTH-1,l] + tile[k+HALO_WIDTH,l]) / h^2
		# y_out[global_index] = (tile[k+HALO_WIDTH-2,l] - 2*tile[k+HALO_WIDTH-1,l])
		# y_out[global_index] = tile[k+HALO_WIDTH,l]
		# y_out[global_index] = 0
	end
	
	# if i == 1 && j <= Nx
	# 	y_out[global_index] = y_out[global_index+1]
	# end

	# sync_threads()

	# if i == Ny && j <= Nx
	# 	y_out[global_index] = y_out[global_index-1]
	# 	# y_out[global_index] = 0.0
	# end
    
    sync_threads()

    nothing

end

function Dy_GPU_shared(y_in, y_out, Nx, Ny, h, ::Val{TILE_DIM1}, ::Val{TILE_DIM2}) where {TILE_DIM1, TILE_DIM2}
	tidx = threadIdx().x
    tidy = threadIdx().y

	i = (blockIdx().x - 1) * TILE_DIM1 + tidx
	j = (blockIdx().y - 1) * TILE_DIM2 + tidy

	global_index = i + (j-1)*Nx

	HALO_WIDTH = 2
	tile = @cuStaticSharedMem(eltype(y_in),(TILE_DIM1+2*HALO_WIDTH,TILE_DIM2))

	k = tidx
	l = tidy

    # Writing pencil-shaped shared memory

    # for tile itself
	# if k <= TILE_DIM1 && l <= TILE_DIM2 && global_index <= Nx*Ny
	if k <= TILE_DIM1 && l <= TILE_DIM2 && i <= Ny && j <= Nx
		# @inbounds tile[k+HALO_WIDTH,l] = y_in[global_index]
		tile[k+HALO_WIDTH,l] = y_in[global_index]
	end

	sync_threads()

	# For upper halo
	# if k <= HALO_WIDTH && l <= TILE_DIM2 && HALO_WIDTH + 1 <= global_index <= Nx*Ny + HALO_WIDTH
	if k <= HALO_WIDTH && l <= TILE_DIM2 && HALO_WIDTH + 1 <= i <= Ny + HALO_WIDTH && j <= Nx
		# @inbounds tile[k,l] = y_in[global_index - HALO_WIDTH]
		tile[k,l] = y_in[global_index - HALO_WIDTH]
	end

	sync_threads()

	# For lower halo
	# if k >= TILE_DIM1 - HALO_WIDTH && l <= TILE_DIM2 && HALO_WIDTH + 1 <= global_index <= Nx*Ny - HALO_WIDTH
	if  TILE_DIM1 - HALO_WIDTH + 1 <= k <= TILE_DIM1 && l <= TILE_DIM2 && i <= Ny - HALO_WIDTH && j <= Nx
		# @inbounds tile[k+2*HALO_WIDTH,l] = y_in[global_index + HALO_WIDTH]
		tile[k+2*HALO_WIDTH,l] = y_in[global_index + HALO_WIDTH]
	end

    sync_threads()
    
    # Finite Difference Operations starts 

    #Upper Boundary
	if k + HALO_WIDTH <= TILE_DIM1 + 2*HALO_WIDTH - 2 && l <= TILE_DIM2 && i == 1 && j <= Ny
		# @inbounds y_out[global_index] = (tile[k+HALO_WIDTH,l] - 2*tile[k+HALO_WIDTH+1,l] + tile[k+HALO_WIDTH+2,l]) / h^2
		y_out[global_index] = (tile[k+HALO_WIDTH+1,l] - tile[k+HALO_WIDTH,l]) / h
	end

	sync_threads()

	#Center
	if k + HALO_WIDTH <= TILE_DIM1 + 2*HALO_WIDTH && l <= TILE_DIM2 && 2 <= i <= Nx-1 && j <= Ny
		# @inbounds y_out[global_index] = (tile[k+HALO_WIDTH-1,l] - 2*tile[k+HALO_WIDTH,l] + tile[k+HALO_WIDTH+1,l]) / h^2
		y_out[global_index] = (tile[k+HALO_WIDTH+1,l] - tile[k+HALO_WIDTH-1,l]) / (2*h)
	end

	sync_threads()

	#Lower Boundary
	if 3 <= k + HALO_WIDTH <= TILE_DIM1 + 2*HALO_WIDTH && l <= TILE_DIM2 && i == Nx && j <= Ny
		# @inbounds y_out[global_index] = (tile[k+HALO_WIDTH-2,l] - 2*tile[k+HALO_WIDTH-1,l] + tile[k+HALO_WIDTH,l]) / h^2
		y_out[global_index] = (tile[k+HALO_WIDTH,l] - tile[k+HALO_WIDTH-1,l]) / h
		# y_out[global_index] = tile[k+HALO_WIDTH-1,l] #- tile[k+HALO_WIDTH-1,l]
 	end
    
    sync_threads()

    nothing
end

function Hyinv_GPU_shared(y_in, y_out, Nx, Ny, h, ::Val{TILE_DIM1}, ::Val{TILE_DIM2}) where {TILE_DIM1, TILE_DIM2}
	tidx = threadIdx().x
    tidy = threadIdx().y

	i = (blockIdx().x - 1) * TILE_DIM1 + tidx
	j = (blockIdx().y - 1) * TILE_DIM2 + tidy

	global_index = i + (j-1)*Nx

	HALO_WIDTH = 2
	tile = @cuStaticSharedMem(eltype(y_in),(TILE_DIM1+2*HALO_WIDTH,TILE_DIM2))

	k = tidx
	l = tidy

    # Writing pencil-shaped shared memory

    # for tile itself
	# if k <= TILE_DIM1 && l <= TILE_DIM2 && global_index <= Nx*Ny
	if k <= TILE_DIM1 && l <= TILE_DIM2 && i <= Ny && j <= Nx
		# @inbounds tile[k+HALO_WIDTH,l] = y_in[global_index]
		tile[k+HALO_WIDTH,l] = y_in[global_index]
	end

	sync_threads()

	# For upper halo
	# if k <= HALO_WIDTH && l <= TILE_DIM2 && HALO_WIDTH + 1 <= global_index <= Nx*Ny + HALO_WIDTH
	if k <= HALO_WIDTH && l <= TILE_DIM2 && HALO_WIDTH + 1 <= i <= Ny + HALO_WIDTH && j <= Nx
		# @inbounds tile[k,l] = y_in[global_index - HALO_WIDTH]
		tile[k,l] = y_in[global_index - HALO_WIDTH]
	end

	sync_threads()

	# For lower halo
	# if k >= TILE_DIM1 - HALO_WIDTH && l <= TILE_DIM2 && HALO_WIDTH + 1 <= global_index <= Nx*Ny - HALO_WIDTH
	if  TILE_DIM1 - HALO_WIDTH <= k <= TILE_DIM1 && l <= TILE_DIM2 && i <= Ny - HALO_WIDTH && j <= Nx
		# @inbounds tile[k+2*HALO_WIDTH,l] = y_in[global_index + HALO_WIDTH]
		tile[k+2*HALO_WIDTH,l] = y_in[global_index + HALO_WIDTH]
	end

    sync_threads()
    
    # Finite Difference Operations starts 

    #Upper Boundary
	if k + HALO_WIDTH <= TILE_DIM1 + 2*HALO_WIDTH && l <= TILE_DIM2 && i == 1 && j <= Ny
		# @inbounds y_out[global_index] = (tile[k+HALO_WIDTH,l] - 2*tile[k+HALO_WIDTH+1,l] + tile[k+HALO_WIDTH+2,l]) / h^2
		y_out[global_index] = (2*tile[k+HALO_WIDTH,l]) / h
	end

	sync_threads()

	#Center
	if k + HALO_WIDTH <= TILE_DIM1 + 2*HALO_WIDTH && l <= TILE_DIM2 && 2 <= i <= Nx-1 && j <= Ny
		# @inbounds y_out[global_index] = (tile[k+HALO_WIDTH-1,l] - 2*tile[k+HALO_WIDTH,l] + tile[k+HALO_WIDTH+1,l]) / h^2
		y_out[global_index] = (tile[k+HALO_WIDTH,l]) / h
	end

	sync_threads()

	#Lower Boundary
	if 3 <= k + HALO_WIDTH <= TILE_DIM1 + 2*HALO_WIDTH && l <= TILE_DIM2 && i == Nx && j <= Ny
		# @inbounds y_out[global_index] = (tile[k+HALO_WIDTH-2,l] - 2*tile[k+HALO_WIDTH-1,l] + tile[k+HALO_WIDTH,l]) / h^2
		y_out[global_index] = (2*tile[k+HALO_WIDTH,l]) / h
    end
    
    sync_threads()

    nothing
end

function Hy_GPU_shared(y_in, y_out, Nx, Ny, h, ::Val{TILE_DIM1}, ::Val{TILE_DIM2}) where {TILE_DIM1, TILE_DIM2}
	tidx = threadIdx().x
    tidy = threadIdx().y

	i = (blockIdx().x - 1) * TILE_DIM1 + tidx
	j = (blockIdx().y - 1) * TILE_DIM2 + tidy

	global_index = i + (j-1)*Ny

	HALO_WIDTH = 2
	tile = @cuStaticSharedMem(eltype(y_in),(TILE_DIM1+2*HALO_WIDTH,TILE_DIM2))

	k = tidx
	l = tidy

    # Writing pencil-shaped shared memory

    # for tile itself
	# if k <= TILE_DIM1 && l <= TILE_DIM2 && global_index <= Nx*Ny
	if k <= TILE_DIM1 && l <= TILE_DIM2 && i <= Ny && j <= Nx
		# @inbounds tile[k+HALO_WIDTH,l] = y_in[global_index]
		tile[k+HALO_WIDTH,l] = y_in[global_index]
	end

	sync_threads()

	# For upper halo
	# if k <= HALO_WIDTH && l <= TILE_DIM2 && HALO_WIDTH + 1 <= global_index <= Nx*Ny + HALO_WIDTH
	if k <= HALO_WIDTH && l <= TILE_DIM2 && HALO_WIDTH + 1 <= i <= Ny + HALO_WIDTH && j <= Nx
		# @inbounds tile[k,l] = y_in[global_index - HALO_WIDTH]
		tile[k,l] = y_in[global_index - HALO_WIDTH]
	end

	sync_threads()

	# For lower halo
	# if k >= TILE_DIM1 - HALO_WIDTH && l <= TILE_DIM2 && HALO_WIDTH + 1 <= global_index <= Nx*Ny - HALO_WIDTH
	if  TILE_DIM1 - HALO_WIDTH + 1 <= k <= TILE_DIM1 && l <= TILE_DIM2 && i <= Ny - HALO_WIDTH && j <= Nx
		# @inbounds tile[k+2*HALO_WIDTH,l] = y_in[global_index + HALO_WIDTH]
		tile[k+2*HALO_WIDTH,l] = y_in[global_index + HALO_WIDTH]
	end

    sync_threads()
    
    # Finite Difference Operations starts 

    #Upper Boundary
	# if k + HALO_WIDTH <= TILE_DIM1 + 2*HALO_WIDTH -2 && l <= TILE_DIM2 && i == 1 && j <= Ny
	if k + HALO_WIDTH <= TILE_DIM1 + 2*HALO_WIDTH && l <= TILE_DIM2 && i == 1 && j <= Nx
		# @inbounds y_out[global_index] = (tile[k+HALO_WIDTH,l] - 2*tile[k+HALO_WIDTH+1,l] + tile[k+HALO_WIDTH+2,l]) / h^2
		y_out[global_index] = (h*tile[k+HALO_WIDTH,l]) / 2
	end

	# sync_threads()

	#Center
	# if k + HALO_WIDTH <= TILE_DIM1 + 2*HALO_WIDTH - 1 && l <= TILE_DIM2 && 2 <= i <= Nx-1 && j <= Ny
	if k + HALO_WIDTH <= TILE_DIM1 + 2*HALO_WIDTH && l <= TILE_DIM2 && 2 <= i <= Ny-1 && j <= Nx
		# @inbounds y_out[global_index] = (tile[k+HALO_WIDTH-1,l] - 2*tile[k+HALO_WIDTH,l] + tile[k+HALO_WIDTH+1,l]) / h^2
		y_out[global_index] = h * (tile[k+HALO_WIDTH,l]) 
	end

	# sync_threads()

	#Lower Boundary
	if 1 <= k + HALO_WIDTH <= TILE_DIM1 + 2*HALO_WIDTH && l <= TILE_DIM2 && i == Ny && j <= Nx
		# @inbounds y_out[global_index] = (tile[k+HALO_WIDTH-2,l] - 2*tile[k+HALO_WIDTH-1,l] + tile[k+HALO_WIDTH,l]) / h^2
		y_out[global_index] = (h*tile[k+HALO_WIDTH,l]) / 2
    end
    
    sync_threads()

    nothing
end


function Bx_GPU_shared(y_in, y_out, Nx, Ny, h, ::Val{TILE_DIM1}, ::Val{TILE_DIM2}) where {TILE_DIM1, TILE_DIM2}
	tidx = threadIdx().x
    tidy = threadIdx().y

    # for global memory indexing
    i = (blockIdx().x - 1) * TILE_DIM1 + tidx
    j = (blockIdx().y - 1) * TILE_DIM2 + tidy

    global_index = i + (j - 1) * Ny

    HALO_WIDTH = 2 # For second order derivative

    tile = @cuStaticSharedMem(eltype(y_in), (TILE_DIM1, TILE_DIM2 + 2 * HALO_WIDTH))

    # for tile indexing
    k = tidx
    l = tidy

    
	# Writing pencil-shaped shared memory

	# for tile itself
	# if k <= TILE_DIM1 && l <= TILE_DIM2 && global_index <= Nx*Ny
	if k <= TILE_DIM1 && l <= TILE_DIM2 && i <= Ny && j <= Nx
		# @inbounds tile[k,l+HALO_WIDTH] = y_in[global_index]
		 tile[k,l+HALO_WIDTH] = y_in[global_index]
    end
    
    sync_threads()

	# for left halo
	# if k <= TILE_DIM1 && l <= HALO_WIDTH && HALO_WIDTH*Ny+1 <= global_index <= (Nx+HALO_WIDTH)*Ny
	if k <= TILE_DIM1 && l <= HALO_WIDTH && i <= Ny && HALO_WIDTH+1 <= j <= HALO_WIDTH + Nx 
		# @inbounds tile[k,l] = y_in[global_index - HALO_WIDTH*Ny]
		tile[k,l] = y_in[global_index - HALO_WIDTH*Ny]
	end

	sync_threads()


	# for right halo
	# if k <= TILE_DIM1 && l >= TILE_DIM2 - HALO_WIDTH && HALO_WIDTH*Ny+1 <= global_index <= (Nx-HALO_WIDTH)*Ny
	if k <= TILE_DIM1 && TILE_DIM2 - HALO_WIDTH + 1<= l <= TILE_DIM2 && i <= Ny && j <= Nx - HALO_WIDTH
		# @inbounds tile[k,l+2*HALO_WIDTH] = y_in[global_index + HALO_WIDTH*Ny]
		tile[k,l+2*HALO_WIDTH] = y_in[global_index + HALO_WIDTH*Ny]
	end

    sync_threads()

    # Finite difference operation starts here

	# Left Boundary
	if k <= TILE_DIM1 && l + HALO_WIDTH <= TILE_DIM2 + 2*HALO_WIDTH -2 && i <= Ny && j == 1
		# @inbounds y_out[global_index] = (tile[k,l + HALO_WIDTH] - 2*tile[k,l + HALO_WIDTH+1] + tile[k,l + HALO_WIDTH+2]) / h^2
		y_out[global_index] = -1.0
	end

	# Center
	if k <= TILE_DIM1 && l + HALO_WIDTH <= TILE_DIM2 + 2*HALO_WIDTH - 1 && i <= Ny && 2 <= j <= Nx-1
		# @inbounds y_out[global_index] = (tile[k,l + HALO_WIDTH-1] - 2*tile[k, l + HALO_WIDTH] + tile[k,l + HALO_WIDTH + 1]) / h^2
		y_out[global_index] = 0.0
	end

	# Right Boundary
	if k <= TILE_DIM1 && 3 <= l + HALO_WIDTH <= TILE_DIM2 + 2*HALO_WIDTH && i <= Ny && j == Nx
		# @inbounds y_out[global_index] = (tile[k,l + HALO_WIDTH-2] - 2*tile[k,l + HALO_WIDTH - 1] + tile[k,l + HALO_WIDTH]) / h^2
		y_out[global_index] = 1.0
	end

    sync_threads()
    
    nothing
end

function BxSx_GPU_shared(y_in, y_out, Nx, Ny, h, ::Val{TILE_DIM1}, ::Val{TILE_DIM2}) where {TILE_DIM1, TILE_DIM2}
	tidx = threadIdx().x
    tidy = threadIdx().y

    # for global memory indexing
    i = (blockIdx().x - 1) * TILE_DIM1 + tidx
    j = (blockIdx().y - 1) * TILE_DIM2 + tidy

    global_index = i + (j - 1) * Ny

    HALO_WIDTH = 2 # For second order derivative

    tile = @cuStaticSharedMem(eltype(y_in), (TILE_DIM1, TILE_DIM2 + 2 * HALO_WIDTH))

    # for tile indexing
    k = tidx
    l = tidy

    
	# Writing pencil-shaped shared memory

	# for tile itself
	# if k <= TILE_DIM1 && l <= TILE_DIM2 && global_index <= Nx*Ny
	if k <= TILE_DIM1 && l <= TILE_DIM2 && i <= Ny && j <= Nx
		# @inbounds tile[k,l+HALO_WIDTH] = y_in[global_index]
		 tile[k,l+HALO_WIDTH] = y_in[global_index]
    end
    
    sync_threads()

	# for left halo
	# if k <= TILE_DIM1 && l <= HALO_WIDTH && HALO_WIDTH*Ny+1 <= global_index <= (Nx+HALO_WIDTH)*Ny
	if k <= TILE_DIM1 && l <= HALO_WIDTH && i <= Ny && HALO_WIDTH+1 <= j <= HALO_WIDTH + Nx 
		# @inbounds tile[k,l] = y_in[global_index - HALO_WIDTH*Ny]
		tile[k,l] = y_in[global_index - HALO_WIDTH*Ny]
	end

	sync_threads()


	# for right halo
	# if k <= TILE_DIM1 && l >= TILE_DIM2 - HALO_WIDTH && HALO_WIDTH*Ny+1 <= global_index <= (Nx-HALO_WIDTH)*Ny
	if k <= TILE_DIM1 && TILE_DIM2 - HALO_WIDTH + 1 <= l <= TILE_DIM2 && i <= Ny && j <= Nx - HALO_WIDTH
		# @inbounds tile[k,l+2*HALO_WIDTH] = y_in[global_index + HALO_WIDTH*Ny]
		tile[k,l+2*HALO_WIDTH] = y_in[global_index + HALO_WIDTH*Ny]
	end

    sync_threads()

    # Finite difference operation starts here

	# Left Boundary
	if k <= TILE_DIM1 && l + HALO_WIDTH <= TILE_DIM2 + 2*HALO_WIDTH -2 && i <= Ny && j == 1
		# @inbounds y_out[global_index] = (tile[k,l + HALO_WIDTH] - 2*tile[k,l + HALO_WIDTH+1] + tile[k,l + HALO_WIDTH+2]) / h^2
		y_out[global_index] = (1.5 * tile[k,l+HALO_WIDTH] - 2.0 * tile[k,l+HALO_WIDTH+1] + 0.5*tile[k,l+HALO_WIDTH+2]) / h
	end

	# Center
	if k <= TILE_DIM1 && l + HALO_WIDTH <= TILE_DIM2 + 2*HALO_WIDTH - 1 && i <= Ny && 2 <= j <= Nx-1
		# @inbounds y_out[global_index] = (tile[k,l + HALO_WIDTH-1] - 2*tile[k, l + HALO_WIDTH] + tile[k,l + HALO_WIDTH + 1]) / h^2
		y_out[global_index] = 0.0
	end

	# Right Boundary
	if k <= TILE_DIM1 && 3 <= l + HALO_WIDTH <= TILE_DIM2 + 2*HALO_WIDTH && i <= Ny && j == Nx
		# @inbounds y_out[global_index] = (tile[k,l + HALO_WIDTH-2] - 2*tile[k,l + HALO_WIDTH - 1] + tile[k,l + HALO_WIDTH]) / h^2
		y_out[global_index] = (0.5 * tile[k,l+HALO_WIDTH-2] - 2.0 * tile[k,l+HALO_WIDTH-1] + 1.5 * tile[k,l+HALO_WIDTH]) / h
	end

    sync_threads()
    
    nothing
end

function By_GPU_shared(y_in, y_out, Nx, Ny, h, ::Val{TILE_DIM1}, ::Val{TILE_DIM2}) where {TILE_DIM1, TILE_DIM2}
	tidx = threadIdx().x
    tidy = threadIdx().y

	i = (blockIdx().x - 1) * TILE_DIM1 + tidx
	j = (blockIdx().y - 1) * TILE_DIM2 + tidy

	global_index = i + (j-1)*Nx

	HALO_WIDTH = 2
	tile = @cuStaticSharedMem(eltype(y_in),(TILE_DIM1+2*HALO_WIDTH,TILE_DIM2))

	k = tidx
	l = tidy

    # Writing pencil-shaped shared memory

    # for tile itself
	# if k <= TILE_DIM1 && l <= TILE_DIM2 && global_index <= Nx*Ny
	if k <= TILE_DIM1 && l <= TILE_DIM2 && i <= Ny && j <= Nx
		# @inbounds tile[k+HALO_WIDTH,l] = y_in[global_index]
		tile[k+HALO_WIDTH,l] = y_in[global_index]
	end

	sync_threads()

	# For upper halo
	# if k <= HALO_WIDTH && l <= TILE_DIM2 && HALO_WIDTH + 1 <= global_index <= Nx*Ny + HALO_WIDTH
	if k <= HALO_WIDTH && l <= TILE_DIM2 && HALO_WIDTH + 1 <= i <= Ny && j <= Nx
		# @inbounds tile[k,l] = y_in[global_index - HALO_WIDTH]
		tile[k,l] = y_in[global_index - HALO_WIDTH]
	end

	sync_threads()

	# For lower halo
	# if k >= TILE_DIM1 - HALO_WIDTH && l <= TILE_DIM2 && HALO_WIDTH + 1 <= global_index <= Nx*Ny - HALO_WIDTH
	if  TILE_DIM1 - HALO_WIDTH <= k <= TILE_DIM1 && l <= TILE_DIM2 && i <= Ny - HALO_WIDTH && j <= Nx
		# @inbounds tile[k+2*HALO_WIDTH,l] = y_in[global_index + HALO_WIDTH]
		tile[k+2*HALO_WIDTH,l] = y_in[global_index + HALO_WIDTH]
	end

    sync_threads()
    
    # Finite Difference Operations starts 

    #Upper Boundary
	if k + HALO_WIDTH <= TILE_DIM1 + 2*HALO_WIDTH && l <= TILE_DIM2 && i == 1 && j <= Ny
		# @inbounds y_out[global_index] = (tile[k+HALO_WIDTH,l] - 2*tile[k+HALO_WIDTH+1,l] + tile[k+HALO_WIDTH+2,l]) / h^2
		y_out[global_index] = -1.0
	end

	sync_threads()

	#Center
	if k + HALO_WIDTH <= TILE_DIM1 + 2*HALO_WIDTH && l <= TILE_DIM2 && 2 <= i <= Nx-1 && j <= Ny
		# @inbounds y_out[global_index] = (tile[k+HALO_WIDTH-1,l] - 2*tile[k+HALO_WIDTH,l] + tile[k+HALO_WIDTH+1,l]) / h^2
		y_out[global_index] = 0.0
	end

	sync_threads()

	#Lower Boundary
	if 3 <= k + HALO_WIDTH <= TILE_DIM1 + 2*HALO_WIDTH && l <= TILE_DIM2 && i == Nx && j <= Ny
		# @inbounds y_out[global_index] = (tile[k+HALO_WIDTH-2,l] - 2*tile[k+HALO_WIDTH-1,l] + tile[k+HALO_WIDTH,l]) / h^2
		y_out[global_index] = 1.0
    end
    
    sync_threads()

	nothing
end

function BySy_GPU_shared(y_in, y_out, Nx, Ny, h, ::Val{TILE_DIM1}, ::Val{TILE_DIM2}) where {TILE_DIM1, TILE_DIM2}
    tidx = threadIdx().x
    tidy = threadIdx().y

	i = (blockIdx().x - 1) * TILE_DIM1 + tidx
	j = (blockIdx().y - 1) * TILE_DIM2 + tidy

	global_index = i + (j-1)*Nx

	HALO_WIDTH = 2
	tile = @cuStaticSharedMem(eltype(y_in),(TILE_DIM1+2*HALO_WIDTH,TILE_DIM2))

	k = tidx
	l = tidy

    # Writing pencil-shaped shared memory

    # for tile itself
	# if k <= TILE_DIM1 && l <= TILE_DIM2 && global_index <= Nx*Ny
	if k <= TILE_DIM1 && l <= TILE_DIM2 && i <= Ny && j <= Nx
		# @inbounds tile[k+HALO_WIDTH,l] = y_in[global_index]
		tile[k+HALO_WIDTH,l] = y_in[global_index]
	end

	sync_threads()

	# For upper halo
	# if k <= HALO_WIDTH && l <= TILE_DIM2 && HALO_WIDTH + 1 <= global_index <= Nx*Ny + HALO_WIDTH
	if k <= HALO_WIDTH && l <= TILE_DIM2 && HALO_WIDTH + 1 <= i <= Ny + HALO_WIDTH && j <= Nx
		# @inbounds tile[k,l] = y_in[global_index - HALO_WIDTH]
		tile[k,l] = y_in[global_index - HALO_WIDTH]
	end

	sync_threads()

	# For lower halo
	# if k >= TILE_DIM1 - HALO_WIDTH && l <= TILE_DIM2 && HALO_WIDTH + 1 <= global_index <= Nx*Ny - HALO_WIDTH
	if  TILE_DIM1 - HALO_WIDTH <= k <= TILE_DIM1 && l <= TILE_DIM2 && i <= Ny - HALO_WIDTH && j <= Nx
		# @inbounds tile[k+2*HALO_WIDTH,l] = y_in[global_index + HALO_WIDTH]
		tile[k+2*HALO_WIDTH,l] = y_in[global_index + HALO_WIDTH]
	end

    sync_threads()
    
    # Finite Difference Operations starts 

    #Upper Boundary
	if k + HALO_WIDTH <= TILE_DIM1 + 2*HALO_WIDTH && l <= TILE_DIM2 && i == 1 && j <= Ny
		# @inbounds y_out[global_index] = (tile[k+HALO_WIDTH,l] - 2*tile[k+HALO_WIDTH+1,l] + tile[k+HALO_WIDTH+2,l]) / h^2
		y_out[global_index] = (1.5*tile[k+HALO_WIDTH,l] - 2*tile[k+HALO_WIDTH+1,l] + 0.5*tile[k+HALO_WIDTH+2,l]) / h
	end

	sync_threads()

	#Center
	if k + HALO_WIDTH <= TILE_DIM1 + 2*HALO_WIDTH && l <= TILE_DIM2 && 2 <= i <= Nx-1 && j <= Ny
		# @inbounds y_out[global_index] = (tile[k+HALO_WIDTH-1,l] - 2*tile[k+HALO_WIDTH,l] + tile[k+HALO_WIDTH+1,l]) / h^2
		y_out[global_index] = 0.0
	end

	sync_threads()

	#Lower Boundary
	if 3 <= k + HALO_WIDTH <= TILE_DIM1 + 2*HALO_WIDTH && l <= TILE_DIM2 && i == Nx && j <= Ny
		# @inbounds y_out[global_index] = (tile[k+HALO_WIDTH-2,l] - 2*tile[k+HALO_WIDTH-1,l] + tile[k+HALO_WIDTH,l]) / h^2
		y_out[global_index] = (0.5 * tile[k+HALO_WIDTH-2,l] - 2*tile[k+HALO_WIDTH-1,l] + 1.5*tile[k+HALO_WIDTH,l]) / h
    end
    
    sync_threads()

    nothing

end

function BxSx_tran_GPU_shared(y_in, y_out, Nx, Ny, h, ::Val{TILE_DIM1}, ::Val{TILE_DIM2}) where {TILE_DIM1, TILE_DIM2}
	tidx = threadIdx().x
    tidy = threadIdx().y

    # for global memory indexing
    i = (blockIdx().x - 1) * TILE_DIM1 + tidx
    j = (blockIdx().y - 1) * TILE_DIM2 + tidy

    global_index = i + (j - 1) * Ny

    HALO_WIDTH = 2 # For second order derivative

    tile = @cuStaticSharedMem(eltype(y_in), (TILE_DIM1, TILE_DIM2 + 2 * HALO_WIDTH))

    # for tile indexing
    k = tidx
    l = tidy

    
	# Writing pencil-shaped shared memory

	# for tile itself
	# if k <= TILE_DIM1 && l <= TILE_DIM2 && global_index <= Nx*Ny
	if k <= TILE_DIM1 && l <= TILE_DIM2 && i <= Ny && j <= Nx
		# @inbounds tile[k,l+HALO_WIDTH] = y_in[global_index]
		 tile[k,l+HALO_WIDTH] = y_in[global_index]
    end
    
    sync_threads()

	# for left halo
	# if k <= TILE_DIM1 && l <= HALO_WIDTH && HALO_WIDTH*Ny+1 <= global_index <= (Nx+HALO_WIDTH)*Ny
	if k <= TILE_DIM1 && l <= HALO_WIDTH && i <= Ny && HALO_WIDTH+1 <= j <= HALO_WIDTH + Nx 
		# @inbounds tile[k,l] = y_in[global_index - HALO_WIDTH*Ny]
		tile[k,l] = y_in[global_index - HALO_WIDTH*Ny]
	end

	sync_threads()


	# for right halo
	# if k <= TILE_DIM1 && l >= TILE_DIM2 - HALO_WIDTH && HALO_WIDTH*Ny+1 <= global_index <= (Nx-HALO_WIDTH)*Ny
	if k <= TILE_DIM1 && TILE_DIM2 - HALO_WIDTH <= l <= TILE_DIM2 && i <= Ny && j <= Nx - HALO_WIDTH
		# @inbounds tile[k,l+2*HALO_WIDTH] = y_in[global_index + HALO_WIDTH*Ny]
		tile[k,l+2*HALO_WIDTH] = y_in[global_index + HALO_WIDTH*Ny]
	end

    sync_threads()

	# Finite difference operation starts here
	

	# Center
	if k <= TILE_DIM1 && l + HALO_WIDTH <= TILE_DIM2 + 2*HALO_WIDTH - 1 && i <= Ny && 4 <= j <= Nx-3
		# @inbounds y_out[global_index] = (tile[k,l + HALO_WIDTH-1] - 2*tile[k, l + HALO_WIDTH] + tile[k,l + HALO_WIDTH + 1]) / h^2
		y_out[global_index] = 0.0
	end

	sync_threads()

	# Left Boundary
	if k <= TILE_DIM1 && l + HALO_WIDTH <= TILE_DIM2 + 2*HALO_WIDTH -2 && i <= Ny && j == 1
		# @inbounds y_out[global_index] = (tile[k,l + HALO_WIDTH] - 2*tile[k,l + HALO_WIDTH+1] + tile[k,l + HALO_WIDTH+2]) / h^2
		y_out[global_index] = (1.5 * tile[k,l+HALO_WIDTH]) / h
		y_out[global_index + Ny] = (-2.0 * tile[k,l+HALO_WIDTH])/h
		y_out[global_index + 2*Ny] = (0.5 * tile[k,l+HALO_WIDTH])/h
	end
	sync_threads()

	# if k <= TILE_DIM1 && l + HALO_WIDTH <= TILE_DIM2 + 2*HALO_WIDTH -2 && i <= Ny && j == 2
	# 	# @inbounds y_out[global_index] = (tile[k,l + HALO_WIDTH] - 2*tile[k,l + HALO_WIDTH+1] + tile[k,l + HALO_WIDTH+2]) / h^2
	# 	y_out[global_index] = (-2.0 * tile[k,l+HALO_WIDTH]) / h
	# end

	# if k <= TILE_DIM1 && l + HALO_WIDTH <= TILE_DIM2 + 2*HALO_WIDTH -2 && i <= Ny && j == 3
	# 	# @inbounds y_out[global_index] = (tile[k,l + HALO_WIDTH] - 2*tile[k,l + HALO_WIDTH+1] + tile[k,l + HALO_WIDTH+2]) / h^2
	# 	y_out[global_index] = (0.5 * tile[k,l+HALO_WIDTH]) / h
	# end

	# Right Boundary
	if k <= TILE_DIM1 && 3 <= l + HALO_WIDTH <= TILE_DIM2 + 2*HALO_WIDTH && i <= Ny && j == Nx
		# @inbounds y_out[global_index] = (tile[k,l + HALO_WIDTH-2] - 2*tile[k,l + HALO_WIDTH - 1] + tile[k,l + HALO_WIDTH]) / h^2
		y_out[global_index] = (1.5 * tile[k,l+HALO_WIDTH]) / h
		y_out[global_index - Ny] = (-2.0 * tile[k, l+HALO_WIDTH])/h
		y_out[global_index - 2*Ny] = (0.5 * tile[k, l+HALO_WIDTH])/h
	end
	sync_threads()

	# if k <= TILE_DIM1 && 3 <= l + HALO_WIDTH <= TILE_DIM2 + 2*HALO_WIDTH && i <= Ny && j == Nx - 1
	# 	# @inbounds y_out[global_index] = (tile[k,l + HALO_WIDTH-2] - 2*tile[k,l + HALO_WIDTH - 1] + tile[k,l + HALO_WIDTH]) / h^2
	# 	y_out[global_index] = (-2.0 * tile[k,l+HALO_WIDTH]) / h
	# end

	# if k <= TILE_DIM1 && 3 <= l + HALO_WIDTH <= TILE_DIM2 + 2*HALO_WIDTH && i <= Ny && j == Nx - 2
	# 	# @inbounds y_out[global_index] = (tile[k,l + HALO_WIDTH-2] - 2*tile[k,l + HALO_WIDTH - 1] + tile[k,l + HALO_WIDTH]) / h^2
	# 	y_out[global_index] = (0.5 * tile[k,l+HALO_WIDTH]) / h
	# end
    
    nothing
end

function BySy_tran_GPU_shared(y_in, y_out, Nx, Ny, h, ::Val{TILE_DIM1}, ::Val{TILE_DIM2}) where {TILE_DIM1, TILE_DIM2}
    tidx = threadIdx().x
    tidy = threadIdx().y

	i = (blockIdx().x - 1) * TILE_DIM1 + tidx
	j = (blockIdx().y - 1) * TILE_DIM2 + tidy

	global_index = i + (j-1)*Nx

	HALO_WIDTH = 2
	tile = @cuStaticSharedMem(eltype(y_in),(TILE_DIM1+2*HALO_WIDTH,TILE_DIM2))

	k = tidx
	l = tidy

    # Writing pencil-shaped shared memory

    # for tile itself
	# if k <= TILE_DIM1 && l <= TILE_DIM2 && global_index <= Nx*Ny
	if k <= TILE_DIM1 && l <= TILE_DIM2 && i <= Ny && j <= Nx
		# @inbounds tile[k+HALO_WIDTH,l] = y_in[global_index]
		tile[k+HALO_WIDTH,l] = y_in[global_index]
	end

	sync_threads()

	# For upper halo
	# if k <= HALO_WIDTH && l <= TILE_DIM2 && HALO_WIDTH + 1 <= global_index <= Nx*Ny + HALO_WIDTH
	if k <= HALO_WIDTH && l <= TILE_DIM2 && HALO_WIDTH + 1 <= i <= Ny && j <= Nx
		# @inbounds tile[k,l] = y_in[global_index - HALO_WIDTH]
		tile[k,l] = y_in[global_index - HALO_WIDTH]
	end

	sync_threads()

	# For lower halo
	# if k >= TILE_DIM1 - HALO_WIDTH && l <= TILE_DIM2 && HALO_WIDTH + 1 <= global_index <= Nx*Ny - HALO_WIDTH
	if  TILE_DIM1 - HALO_WIDTH <= k <= TILE_DIM1 && l <= TILE_DIM2 && i <= Ny - HALO_WIDTH && j <= Nx
		# @inbounds tile[k+2*HALO_WIDTH,l] = y_in[global_index + HALO_WIDTH]
		tile[k+2*HALO_WIDTH,l] = y_in[global_index + HALO_WIDTH]
	end

    sync_threads()
    
    # Finite Difference Operations starts 

    #Upper Boundary
	if k + HALO_WIDTH <= TILE_DIM1 + 2*HALO_WIDTH && l <= TILE_DIM2 && i == 1 && j <= Ny
		# @inbounds y_out[global_index] = (tile[k+HALO_WIDTH,l] - 2*tile[k+HALO_WIDTH+1,l] + tile[k+HALO_WIDTH+2,l]) / h^2
		y_out[global_index] = (1.5*tile[k+HALO_WIDTH,l]) / h
		y_out[global_index + 1] = (-2.0*tile[k+HALO_WIDTH,l]) / h
		y_out[global_index + 2] = (0.5*tile[k+HALO_WIDTH,l])/h
	end

	sync_threads()

	#Center
	if k + HALO_WIDTH <= TILE_DIM1 + 2*HALO_WIDTH && l <= TILE_DIM2 && 4 <= i <= Nx-3 && j <= Ny
		# @inbounds y_out[global_index] = (tile[k+HALO_WIDTH-1,l] - 2*tile[k+HALO_WIDTH,l] + tile[k+HALO_WIDTH+1,l]) / h^2
		y_out[global_index] = 0.0
	end

	sync_threads()

	#Lower Boundary
	if 3 <= k + HALO_WIDTH <= TILE_DIM1 + 2*HALO_WIDTH && l <= TILE_DIM2 && i == Nx && j <= Ny
		# @inbounds y_out[global_index] = (tile[k+HALO_WIDTH-2,l] - 2*tile[k+HALO_WIDTH-1,l] + tile[k+HALO_WIDTH,l]) / h^2
		y_out[global_index] = (1.5*tile[k+HALO_WIDTH,l]) / h
		y_out[global_index-1] = (-2.0 * tile[k+HALO_WIDTH,l]) / h
		y_out[global_index-2] = (0.5 * tile[k+HALO_WIDTH,l])/h
    end
    
    sync_threads()

    nothing

end

# function FACEtoVOL_GPU_shared(y_out, u_face, face, Nx, Ny, ::Val{TILE_DIM}) where {TILE_DIM} #, ::Val{TILE_DIM1}, ::Val{TILE_DIM2}) where {TILE_DIM1, TILE_DIM2}
# 	# tidx = threadIdx().x
# 	# i = (blockIdx().x - 1)*TILE_DIM + tidx
# 	i = (blockIdx().x - 1) * TILE_DIM + threadIdx().x
# 	N = Nx*Ny

# 	if i <= N
# 		y_out[i] = 0.0
# 	end

# 	sync_threads()

# 	if face == 1 && i <= N && mod(i,Ny) == 1 
# 		y_out[i] = u_face
# 	end

# 	sync_threads()

# 	if face == 2 && i <= N && mod(i,Ny) == 0
# 		y_out[i] = u_face
# 	end

# 	sync_threads()

# 	if face == 3 && i <= Ny
# 		y_out[i] = u_face
# 	end

# 	sync_threads()

# 	if face == 4 && N-Ny+1 <= i <= N
# 		y_out[i] = u_face
# 	end
# 	sync_threads()

# 	nothing
# end

# function VOLtoFACE_GPU_shared(y_in, y_out,  face, Nx, Ny, ::Val{TILE_DIM}) where {TILE_DIM} #, ::Val{TILE_DIM1}, ::Val{TILE_DIM2}) where {TILE_DIM1, TILE_DIM2}
# 	# tidx = threadIdx().x
# 	# i = (blockIdx().x - 1)*TILE_DIM + tidx
# 	i = (blockIdx().x - 1) * TILE_DIM + threadIdx().x
# 	N = Nx*Ny

# 	if i <= N
# 		y_out[i] = 0.0
# 	end

# 	sync_threads()

# 	if face == 1 && i <= N && mod(i,Ny) == 1 
# 		y_out[i] = u_face
# 	end

# 	sync_threads()

# 	if face == 2 && i <= N && mod(i,Ny) == 0
# 		y_out[i] = u_face
# 	end

# 	sync_threads()

# 	if face == 3 && i <= Ny
# 		y_out[i] = u_face
# 	end

# 	sync_threads()

# 	if face == 4 && N-Ny+1 <= i <= N
# 		y_out[i] = u_face
# 	end
# 	sync_threads()

# 	nothing
# end

# tester_function : This gives evaluation on CPU, GPU, GPU_shared 
function tester_function(f,Nx,TILE_DIM_1,TILE_DIM_2,TILE_DIM)
    Ny = Nx
	@show f
	@eval gpu_function = $(Symbol(f,"_GPU"))
	@eval gpu_function_shared = $(Symbol(f,"_GPU_shared"))
	@show gpu_function
    @show gpu_function_shared
    h = 1/Nx
	# TILE_DIM_1 = 16
	# TILE_DIM_2 = 2

	u = randn(Nx*Ny)
	y_in = CuArray(u)
	y_out = similar(y_in)
	y_out2 = similar(y_in)

	griddim = (div(Nx,TILE_DIM_1) + 1, div(Ny,TILE_DIM_2) + 1)
	blockdim = (TILE_DIM_1,TILE_DIM_2)

	# TILE_DIM = 32
	THREAD_NUM = TILE_DIM
    BLOCK_NUM = div(Nx * Ny,TILE_DIM)+1 
    
	y = f(u,Nx,Ny,h)
	@cuda threads=THREAD_NUM blocks=BLOCK_NUM gpu_function(y_in, y_out, Nx, Ny, h, Val(TILE_DIM))
    @cuda threads=blockdim blocks=griddim gpu_function_shared(y_in, y_out2, Nx, Ny, h, Val(TILE_DIM_1), Val(TILE_DIM_2))
	@show y ≈ Array(y_out)
	@show y ≈ Array(y_out2)
	@show y - Array(y_out2)
	
	rep_times = 10

	t_y = time_ns()
	for i in 1:rep_times
		y = f(u,Nx,Ny,h)
	end
	t_y_end = time_ns()
	t1 = t_y_end - t_y

	memsize = length(u) * sizeof(eltype(u))
	@show Float64(t1)
	@printf("CPU Through-put %20.2f\n", 2 * memsize * rep_times / t1)


	println()

	t_y_out = time_ns()
	for i in 1:rep_times
		@cuda threads=THREAD_NUM blocks=BLOCK_NUM gpu_function(y_in, y_out, Nx, Ny, h, Val(TILE_DIM))
		# @cuda threads=THREAD_NUM blocks=BLOCK_NUM D2y_GPU_v2(y_in, y_out, Nx, Ny, h, Val(TILE_DIM))
	end
	synchronize()
	t_y_out_end = time_ns()
	t2 = t_y_out_end - t_y_out
	@show Float64(t2)
	@show Float64(t1)/Float64(t2)
	@printf("GPU Through-put (naive) %20.2f\n", 2 * memsize * rep_times / t2)

	println()

	t_y_out2 = time_ns()
	for i in 1:rep_times
		@cuda threads=blockdim blocks=griddim gpu_function_shared(y_in, y_out2, Nx, Ny, h, Val(TILE_DIM_1), Val(TILE_DIM_2))
	end
	synchronize()
	t_y_out2_end = time_ns()
	t3 = t_y_out2_end - t_y_out2

	@show Float64(t3)
	@show Float64(t1)/Float64(t3)
	@printf("GPU Through-put (shared memory)%20.2f\n", 2 * memsize * rep_times / t3)

end

# tester_function_v2 : This gives evaluation on CPU, GPU_shared with error output
function tester_function_v2(f,Nx,TILE_DIM_1,TILE_DIM_2)
    Ny = Nx
	@show f
	# @eval gpu_function = $(Symbol(f,"_GPU"))
	@eval gpu_function_shared = $(Symbol(f,"_GPU_shared"))
	# @show gpu_function
    @show gpu_function_shared
    h = 1/Nx
	# TILE_DIM_1 = 16
	# TILE_DIM_2 = 2

	u = randn(Nx*Ny)
	y_in = CuArray(u)
	# y_out = similar(y_in)
	y_out2 = similar(y_in)

	griddim = (div(Nx,TILE_DIM_1) + 1, div(Ny,TILE_DIM_2) + 1)
	blockdim = (TILE_DIM_1,TILE_DIM_2)

	# TILE_DIM = 32
	# THREAD_NUM = TILE_DIM
    # BLOCK_NUM = div(Nx * Ny,TILE_DIM)+1 
    
	y = f(u,Nx,Ny,h)
	# @cuda threads=THREAD_NUM blocks=BLOCK_NUM gpu_function(y_in, y_out, Nx, Ny, h, Val(TILE_DIM))
    @cuda threads=blockdim blocks=griddim gpu_function_shared(y_in, y_out2, Nx, Ny, h, Val(TILE_DIM_1), Val(TILE_DIM_2))
	# @show y ≈ Array(y_out)
	@show y ≈ Array(y_out2)

	@show u
	println()
	@show y
	println()
	@show Array(y_out2)
	println()
	@show y - Array(y_out2)

	@show y ≈ Array(y_out2)
	
	rep_times = 10

	t_y = time_ns()
	for i in 1:rep_times
		y = f(u,Nx,Ny,h)
	end
	t_y_end = time_ns()
	t1 = t_y_end - t_y

	memsize = length(u) * sizeof(eltype(u))
	@show Float64(t1)
	@printf("CPU Through-put %20.2f\n", 2 * memsize * rep_times / t1)


	println()

	# t_y_out = time_ns()
	# for i in 1:rep_times
	# 	@cuda threads=THREAD_NUM blocks=BLOCK_NUM gpu_function(y_in, y_out, Nx, Ny, h, Val(TILE_DIM))
	# 	# @cuda threads=THREAD_NUM blocks=BLOCK_NUM D2y_GPU_v2(y_in, y_out, Nx, Ny, h, Val(TILE_DIM))
	# end
	# synchronize()
	# t_y_out_end = time_ns()
	# t2 = t_y_out_end - t_y_out
	# @show Float64(t2)
	# @show Float64(t1)/Float64(t2)
	# @printf("GPU Through-put (naive) %20.2f\n", 2 * memsize * rep_times / t2)

	# println()

	t_y_out2 = time_ns()
	for i in 1:rep_times
		@cuda threads=blockdim blocks=griddim gpu_function_shared(y_in, y_out2, Nx, Ny, h, Val(TILE_DIM_1), Val(TILE_DIM_2))
	end
	synchronize()
	t_y_out2_end = time_ns()
	t3 = t_y_out2_end - t_y_out2

	@show Float64(t3)
	@show Float64(t1)/Float64(t3)
	@printf("GPU Through-put (shared memory)%20.2f\n", 2 * memsize * rep_times / t3)

end

# tester_function_v3 : This gives evaluation on CPU, GPU_shared with error output turned off
function tester_function_v3(f,Nx,TILE_DIM_1,TILE_DIM_2)
    Ny = Nx
	@show f
	# @eval gpu_function = $(Symbol(f,"_GPU"))
	@eval gpu_function_shared = $(Symbol(f,"_GPU_shared"))
	# @show gpu_function
    @show gpu_function_shared
    h = 1/Nx
	# TILE_DIM_1 = 16
	# TILE_DIM_2 = 2

	u = randn(Nx*Ny)
	y_in = CuArray(u)
	# y_out = similar(y_in)
	y_out2 = similar(y_in)

	griddim = (div(Nx,TILE_DIM_1) + 1, div(Ny,TILE_DIM_2) + 1)
	blockdim = (TILE_DIM_1,TILE_DIM_2)

	@show blockdim
	@show griddim

	# TILE_DIM = 32
	# THREAD_NUM = TILE_DIM
    # BLOCK_NUM = div(Nx * Ny,TILE_DIM)+1 
    
	y = f(u,Nx,Ny,h)
	# @cuda threads=THREAD_NUM blocks=BLOCK_NUM gpu_function(y_in, y_out, Nx, Ny, h, Val(TILE_DIM))
    @cuda threads=blockdim blocks=griddim gpu_function_shared(y_in, y_out2, Nx, Ny, h, Val(TILE_DIM_1), Val(TILE_DIM_2))
	# @show y ≈ Array(y_out)
	@show y ≈ Array(y_out2)

	# @show y - Array(y_out2)
	
	rep_times = 10

	t_y = time_ns()
	for i in 1:rep_times
		y = f(u,Nx,Ny,h)
	end
	t_y_end = time_ns()
	t1 = t_y_end - t_y

	memsize = length(u) * sizeof(eltype(u))
	@show Float64(t1)
	@printf("CPU Through-put %20.2f\n", 2 * memsize * rep_times / t1)


	println()

	# t_y_out = time_ns()
	# for i in 1:rep_times
	# 	@cuda threads=THREAD_NUM blocks=BLOCK_NUM gpu_function(y_in, y_out, Nx, Ny, h, Val(TILE_DIM))
	# 	# @cuda threads=THREAD_NUM blocks=BLOCK_NUM D2y_GPU_v2(y_in, y_out, Nx, Ny, h, Val(TILE_DIM))
	# end
	# synchronize()
	# t_y_out_end = time_ns()
	# t2 = t_y_out_end - t_y_out
	# @show Float64(t2)
	# @show Float64(t1)/Float64(t2)
	# @printf("GPU Through-put (naive) %20.2f\n", 2 * memsize * rep_times / t2)

	# println()

	t_y_out2 = time_ns()
	for i in 1:rep_times
		@cuda threads=blockdim blocks=griddim gpu_function_shared(y_in, y_out2, Nx, Ny, h, Val(TILE_DIM_1), Val(TILE_DIM_2))
	end
	synchronize()
	t_y_out2_end = time_ns()
	t3 = t_y_out2_end - t_y_out2

	@show Float64(t3)
	@show Float64(t1)/Float64(t3)
	@printf("GPU Through-put (shared memory)%20.2f\n", 2 * memsize * rep_times / t3)

end


function tester_function_FV(f,u_face,face,Nx,TILE_DIM)
	Ny = Nx
	N = Ny*Nx
	@show f
	# @eval gpu_function = $(Symbol(f,"_GPU"))
	@eval gpu_function_shared = $(Symbol(f,"_GPU_shared"))
	# @show gpu_function
    @show gpu_function_shared
	y = zeros(N)
	y_out = CuArray(y)

	# u_face = 2
	# face = 2

	THREAD_NUM = TILE_DIM
	BLOCK_NUM = (N + TILE_DIM-1, TILE_DIM)

	y = f(u_face, face, Nx, Ny)

	@cuda threads=THREAD_NUM blocks=BLOCK_NUM gpu_function_shared(y_out,u_face,face,Nx,Ny,Val(TILE_DIM))

	@show y ≈ Array(y_out)
	# @show y
	# @show Array(y_out)
	# @show y - Array(y_out)

	rep_times = 10
	memsize = length(y) * sizeof(eltype(y))

	t_y = time_ns()
	for i in 1:rep_times
		y .= f(u_face, face, Nx, Ny)
	end
	t_y_end = time_ns()
	t1 = t_y_end - t_y
	@show Float64(t1)
	@printf("CPU Through-put %20.2f\n", 2 * memsize * rep_times / t1)


	t_y_out = time_ns()
	for i in rep_times
		@cuda threads=THREAD_NUM blocks=BLOCK_NUM gpu_function_shared(y_out,u_face,face,Nx,Ny,Val(TILE_DIM))
	end
	synchronize()
	t_y_out_end = time_ns()
	t2 = t_y_out_end - t_y_out
	@show Float64(t2)
	@printf("GPU Through-put (naive) %20.2f\n", 2 * memsize * rep_times / t2)

end

# tester_function(D2x, 1000, 16, 4, 32)
# tester_function(D2y, 1000, 4, 16, 32)

# tester_function_v3(Dx, 10, 16, 4)

# tester_function_v3(Dx, 10, 16, 4)
# tester_function_v3(Dy, 10, 16, 4)


# tester_function_FV(FACEtoVOL,1,1,10,32)

# Nx = 10
# Ny = Nx
# h = 1/Nx
# TILE_DIM_1 = 4
# TILE_DIM_2 = 16

# u = randn(Nx*Ny);
# y_in = CuArray(u);
# y_out = similar(y_in);
# y_out2 = similar(y_in);

# y = D2y(u,Nx,Ny,h)

# griddim = (div(Nx,TILE_DIM_1) + 1, div(Ny,TILE_DIM_2) + 1)
# blockdim = (TILE_DIM_1,TILE_DIM_2)

# # TILE_DIM = 32
# # THREAD_NUM = 32
# # BLOCK_NUM = div(Nx * Ny,TILE_DIM) + 1

# @cuda threads=blockdim blocks=griddim D2y_GPU_v7(y_in, y_out, Nx, Ny, h, Val(TILE_DIM_1), Val(TILE_DIM_2))
# Array(y_out)

# Array(y_out) ≈ y

# D2y(u,Nx,Ny,h)

