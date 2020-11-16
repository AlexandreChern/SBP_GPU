#functions for calling the matrix vector products
#i.e. function A(y_in,Nx,Ny,h) computes the product A*u and stores it in y

#Right now only for  the second-order accurate SBP operators for constant coefficients from
#Mattsson and Nordstrom, 2004.

#Written for the 2D domain (x,y) \in (a, b) \times (c, d),
#stacking the grid function in the vertical direction,
#so that, e.g., \partial u / \partial x \approx D \kron I and
#\partial u / \partial y \approx I \kron D
#the faces of the 2D domain are label bottom to top, left to right, i.e.
#side 1 is y = c, side 2 is y = d
#side 3 is x = a, side 4 is x = b

#D2x and D2y compute approximations to
#\partial^2 u / \partial x^2 and
#\partial^2 u / \partial y^2, respectively

#FACEtoVOL(face,u_face,Nx,Ny) maps the face value u_face to a full length vector at the
#nodes corresponding to face

#BxSx \approx the traction on faces 3 and 4
#BySy \approx the traction on faces 1 and 2

#BxSx_tran and BySy_tran are the transposes of BxSx and BySy


# variables order: y_in, Nx, Ny, N, hx, hy, y_out(container)

##
# y_in: input Array
# y_out: output Array
##


function D2x_beta!(y_in::Array{Float64,1}, Nx::Int64, Ny::Int64, N::Int64, hx::Float64, hy::Float64, y_out::Array{Float64,1})
	#N = Nx*Ny
	#hx = Float64(1/(Nx-1))
	#hy = Float64(1/(Ny-1))
	#y = similar(y_in)
	#y = similar(y_in)
	@inbounds  for idx = 1:Ny
		y_out[idx] = (y_in[idx] - 2*y_in[Ny + idx] + y_in[2*Ny + idx]) / hx^2
	end

	@inbounds for idx1 = Ny+1:N-Ny
		y_out[idx1] = (y_in[idx1 - Ny] - 2 * y_in[idx1] + y_in[idx1 + Ny]) / hx^2
	end

	@inbounds for idx2 = N-Ny+1:N
		y_out[idx2] = (y_in[idx2 - 2*Ny] -2 * y_in[idx2 - Ny] + y_in[idx2]) / hx^2
	end
	# return y_out
end



function D2y_beta!(y_in::Array{Float64,1}, Nx::Int64, Ny::Int64, N::Int64, hx::Float64, hy::Float64, y_out::Array{Float64,1})
	#N = Nx*Ny
	#hx = Float64(1/(Nx-1))
	#hy = Float64(1/(Ny-1))
	@inbounds for idx = 1:Ny:N-Ny+1
		y_out[idx] = (y_in[idx] - 2 * y_in[idx + 1] + y_in[idx + 2]) / hy^2
	end

	@inbounds for idx1 = Ny:Ny:N
		y_out[idx1] = (y_in[idx1 - 2] - 2 * y_in[idx1 .- 1] + y_in[idx1]) / hy^2
	end

	@inbounds for j = 1:Nx
		@inbounds for idx = 2+(j-1)*Ny:j*Ny-1
			y_out[idx] = (y_in[idx - 1] - 2 * y_in[idx] + y_in[idx + 1]) / hy^2
		end
	end

	# return y_out

end





function Dx_beta!(y_in::Array{Float64,1}, Nx::Int64, Ny::Int64, N::Int64, hx::Float64, hy::Float64, y_out::Array{Float64,1})

	@inbounds for idx = 1:Ny
		y_out[idx] = (y_in[idx + Ny] - y_in[idx]) / hx
	end

	@inbounds for idx1 = Ny+1:N-Ny
		y_out[idx1] = (y_in[idx1 + Ny]-y_in[idx1 - Ny]) / (2*hx)
	end

	@inbounds for idx2 = N-Ny+1:N
		y_out[idx2] = (y_in[idx2]-y_in[idx2 .- Ny]) ./ hx
	end
	# return y_out
end




function Dy_beta!(y_in::Array{Float64,1}, Nx::Int64, Ny::Int64, N::Int64, hx::Float64, hy::Float64, y_out::Array{Float64,1})

	@inbounds for idx = 1:Ny:N-Ny+1
		y_out[idx] = (y_in[idx + 1] - y_in[idx]) / h
	end

	@inbounds for idx = Ny:Ny:N
		y_out[idx] = (y_in[idx] - y_in[idx - 1]) /h
	end

	@inbounds for j = 1:Nx
		@inbounds for idx = 2+(j-1)*Ny:j*Ny-1
		 	y_out[idx] = (y_in[idx + 1] - y_in[idx - 1]) / (2*h)
		end
	end
	# return y_out
end



function Hxinv_beta!(y_in::Array{Float64,1}, Nx::Int64, Ny::Int64, N::Int64, hx::Float64, hy::Float64, coef::Float64, y_out::Array{Float64,1})
	#N = Nx*Ny
	#y = similar(y_in)
	@inbounds for idx = 1:Ny
		y_out[idx] = coef * (2*y_in[idx]) * (1/hx)
	end

	@inbounds for idx1 = Ny+1:N-Ny
		y_out[idx1] = coef * (1*y_in[idx1]) * (1/hx)
	end

	@inbounds for idx2 = N-Ny+1:N
		y_out[idx2] = coef * (2*y_in[idx2]) * (1/hx)
	end

	# return y_out
end



function Hyinv_beta!(y_in::Array{Float64,1}, Nx::Int64, Ny::Int64, N::Int64, hx::Float64, hy::Float64, coef::Float64, y_out::Array{Float64,1})
	#N = Nx*Ny
	#y = similar(y_in)
	#hx = Float64(1/(Nx-1))
	#hy = Float64(1/(Ny-1))
	@inbounds for idx = 1:Ny:N-Ny+1
		y_out[idx] = coef * (2*y_in[idx]) * (1/hy)
	end

	@inbounds for idx1 = Ny:Ny:N
		y_out[idx1] = coef * (2*y_in[idx1]) * (1/hy)
	end

	@inbounds for i = 1:Nx
		@inbounds for idx2 = 2+(i-1)*Ny:i*Ny-1
			y_out[idx2] = coef * (y_in[idx2]) * (1/hy)
		end
	end
	# return y_out
end



function Hx_beta!(y_in::Array{Float64,1}, Nx::Int64, Ny::Int64, N::Int64, hx::Float64, hy::Float64,coef::Float64 ,y_out::Array{Float64,1})
	@inbounds for idx = 1:Ny
		y_out[idx] = coef * hx*y_in[idx]/2
	end

	@inbounds for idx1 = Ny+1:N-Ny
		y_out[idx1] = coef* hx*y_in[idx1]
	end

	@inbounds for idx2 = N-Ny+1:N
		y_out[idx2] = coef * hx*y_in[idx2]/2
	end
	# return y_out
end



function Hy_beta!(y_in::Array{Float64,1}, Nx::Int64, Ny::Int64, N::Int64, hx::Float64, hy::Float64, y_out::Array{Float64,1})
	#N = Nx*Ny
	#y = similar(y_in)

	@inbounds for idx = 1:Ny:N-Ny+1
		y_out[idx] = hy*y_in[idx]/2
	end

	@inbounds for idx1 = Ny:Ny:N
		y_out[idx1] = hy*y_in[idx1]/2
	end

	@inbounds for i = 1:Nx
		@inbounds for idx2 = 2 + (i-1)*Ny:i*Ny-1
			y_out[idx2] = hy*y_in[idx2]
		end
	end
	# return y_out
end



function FACEtoVOL_beta!(y_in_face::Array{Float64,1},  face::Int64, Nx::Int64, Ny::Int64, N::Int64, y_outs::Array{Array{Float64,1},1})
	if face == 1
		idx = 1:Ny:N-Ny+1
	elseif face==2
		idx = Ny:Ny:N
	elseif face==3
		idx = 1:Ny
	elseif face == 4
		idx = N-Ny+1:N
	else
	end
	y_out = y_outs[face]
	y_out[idx] = y_in_face
	# return y_out
end



function VOLtoFACE_beta!(y_in::Array{Float64,1}, face::Int64,Nx::Int64, Ny::Int64, N::Int64, y_outs::Array{Array{Float64,1},1}) ## Has some issue
	if face == 1
			idx = 1:Ny:N-Ny+1
	elseif face == 2
			idx = Ny:Ny:N
	elseif face == 3
			idx = 1:Ny
	elseif face == 4
			idx = N-Ny+1:N
	else
	end
    y_out = y_outs[face]
	y_out[idx] = y_in[idx]

	# return y_out
end


function Bx_beta!(Nx::Int64,Ny::Int64, N::Int64, y_out::Array{Float64,1})

	@inbounds for idx=1:Ny
		y_out[idx] = -1
	end

	@inbounds for idx = N-Ny+1:N
		y_out[idx] = 1
	end
	# return y_out
end


function By_beta!(Nx::Int64,Ny::Int64,N::Int64,y_By::Array{Float64,1})
	@inbounds for idx = 1:Ny:N-Ny+1
		y_By[idx] = -1
	end

	@inbounds for idx1 = Ny:Ny:N
		y_By[idx1] = 1
	end
	# return y_By
end


function BxSx_beta!(y_in::Array{Float64,1},Nx::Int64,Ny::Int64,N::Int64,hx::Float64,hy::Float64,y_out::Array{Float64,1})
	@inbounds for idx = 1:Ny
		y_out[idx] = (1/hx) * (1.5 * y_in[idx] - 2 * y_in[idx + Ny] + 0.5 * y_in[idx + 2*Ny])
		y_out[N-Ny + idx] = (1/hx) * (0.5 * y_in[N-3*Ny + idx] - 2 * y_in[N-2*Ny + idx] + 1.5 * y_in[N-Ny + idx])
	end
	# return y_out
end


function BySy_beta!(y_in::Array{Float64,1}, Nx::Int64, Ny::Int64, N::Int64, hx::Float64, hy::Float64,y_out::Array{Float64,1})
	@inbounds for idx = 1:Ny:N-Ny+1
		y_out[idx] = (1/hy) * (1.5 * y_in[idx] - 2 * y_in[idx .+ 1] + 0.5 * y_in[idx .+ 2])
	end

	@inbounds for idx = Ny:Ny:N
		y_out[idx] = (1/hy) * (0.5 * y_in[idx - 2] - 2 * y_in[idx - 1] + 1.5 * y_in[idx])
	end

	# return y_out
end



function BxSx_tran_beta!(y_in::Array{Float64,1},Nx::Int64,Ny::Int64,N::Int64,hx::Float64,hy::Float64,y_out::Array{Float64,1}) # be careful with += expression
	#hx = Float64(1/(Nx-1))
   	#hy = Float64(1/(Ny-1))
	@inbounds for idx1 = 1:Ny
		y_out[idx1] = (1.5 * y_in[idx1]) * (1/hx)

		# for idx = Ny+1:2*Ny
		y_out[idx1+Ny] = (-2 * y_in[idx1]) * (1/hx)
		# end


		# for idx  = 2*Ny+1:3*Ny
		y_out[idx1+2Ny] = (0.5 * y_in[idx1]) * (1/hx)
		# end
	end

	@inbounds for idxN = N-Ny+1:N
		y_out[idxN] = (1.5 * y_in[idxN]) * (1/hx)

		# for idx = N-2*Ny+1:N-Ny
		y_out[idxN-Ny] = (-2 * y_in[idxN]) * (1/hx)
		# end

		# for idx = N-3*Ny+1:N-2*Ny
		y_out[idxN-2Ny] = (0.5 * y_in[idxN]) * (1/hx)
		# end
	end
	# return y_out
end




function BySy_tran_beta!(y_in::Array{Float64,1}, Nx::Int64, Ny::Int64, N::Int64, hx::Float64, hy::Float64, y_out::Array{Float64,1}) # Be Careful about double foor loops
	@inbounds for idx1 = 1:Ny:N-Ny+1
		y_out[idx1] = (1.5 * y_in[idx1]) * (1/hy)

		# for idx = Ny+1:2*Ny
		y_out[idx1+1] = (-2 * y_in[idx1]) * (1/hy)
		# end


		# for idx  = 2*Ny+1:3*Ny
		y_out[idx1+2] = (0.5 * y_in[idx1]) * (1/hy)
		# end
	end

	@inbounds for idxN = Ny:Ny:N
		y_out[idxN] = (1.5 * y_in[idxN]) * (1/hy)

		# for idx = N-2*Ny+1:N-Ny
		y_out[idxN-1] = (-2 * y_in[idxN]) * (1/hy)
		# end

		# for idx = N-3*Ny+1:N-2*Ny
		y_out[idxN-2] = (0.5 * y_in[idxN]) * (1/hy)
		# end
	end

	# return y_out
end




# Previous function from deriv_ops.jl

# function FACEtoVOL(y_in_face, face, Nx, Ny)
# 	N = Nx*Ny
# 	y = zeros(N)

# 	if face == 1
# 		idx = 1:Ny:N-Ny+1
# 	elseif face == 2
# 		idx = Ny:Ny:N
# 	elseif face == 3
# 		idx = 1:Ny
# 	elseif face == 4
# 		idx = N-Ny+1:N
# 	else
# 	end

# 	y[idx] = u_face

# 	# return y

# end

# function VOLtoFACE(y_in, face, Nx, Ny)
# 	N = Nx*Ny
#         y = zeros(N)

#         if face == 1
#                 idx = 1:Ny:N-Ny+1
#         elseif face == 2
#                 idx = Ny:Ny:N
#         elseif face == 3
#                 idx = 1:Ny
#         elseif face == 4
#                 idx = N-Ny+1:N
#         else
#         end

# 	y[idx] = y_in[idx]
#         # return y
# end

# function Hxinv(y_in, Nx, Ny, h)
# 	N = Nx*Ny
# 	y = zeros(N)

# 	idx = 1:Ny
# 	y[idx] = (2*y_in[idx]) .* (1/h)

# 	idx = Ny+1:N-Ny
# 	y[idx] = (1*y_in[idx]) .* (1/h)

# 	idx = N-Ny+1:N
# 	y[idx] = (2*y_in[idx]) .* (1/h)

# 	# return y
# end

# function Hyinv(y_in, Nx, Ny, h)
# 	N = Nx*Ny
# 	y = zeros(N)

# 	idx = 1:Ny:N-Ny+1
# 	y[idx] = (2*y_in[idx]) .* (1/h)

# 	idx = Ny:Ny:N
# 	y[idx] = (2*y_in[idx]) .* (1/h)

# 	for i = 1:Nx
# 		idx = 2+(i-1).*Ny:i*Ny-1
# 		y[idx] = y_in[idx] .* (1/h)
# 	end

# 	# return y

# end

# function Hx(y_in, Nx, Ny, h)
# 	N = Nx*Ny
#         y = zeros(N)

#         idx = 1:Ny
# 	y[idx] = h .* (1/2)*y_in[idx]

#         idx = Ny+1:N-Ny
#         y[idx] = h .* 1*y_in[idx]

#         idx = N-Ny+1:N
# 	y[idx] = h .* (1/2)*y_in[idx]

#         # return y


# end

# function Hy(y_in, Nx, Ny, h)
# 	N = Nx*Ny
#         y = zeros(N)

#         idx = 1:Ny:N-Ny+1
# 	y[idx] = h .* (1/2)*y_in[idx]

#         idx = Ny:Ny:N
# 	y[idx] = h .* (1/2)*y_in[idx]

#         for i = 1:Nx
#                 idx = 2+(i-1).*Ny:i*Ny-1
#                 y[idx] = h .* y_in[idx]
#         end

#         # return y

# end

# function BxSx_tran(y_in, Nx, Ny, h)
# 	N = Nx*Ny
# 	y = zeros(N)

# 	idx1 = 1:Ny
# 	y[idx1] += (1.5 .* y_in[idx1]) .* (1/h)
# 	idx = Ny+1:2*Ny
# 	y[idx] += (-2 .* y_in[idx1]) .* (1/h)
# 	idx  = 2*Ny+1:3*Ny
# 	y[idx] += (0.5 .* y_in[idx1]) .* (1/h)

# 	idxN = N-Ny+1:N
# 	y[idxN] += (1.5 .* y_in[idxN]) .* (1/h)
# 	idx = N-2*Ny+1:N-Ny
# 	y[idx] += (-2 .* y_in[idxN]) .* (1/h)
# 	idx = N-3*Ny+1:N-2*Ny
# 	y[idx] += (0.5 .* y_in[idxN]) .* (1/h)

# 	# return y
# end
