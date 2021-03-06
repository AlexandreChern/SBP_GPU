#functions for calling the matrix vector products
#i.e. function A(y_in,Nx,Ny,h) computes the product A*u and stores it in y

#Right now only_out[for  the second-order accurate SBP operators for constant coefficients from
#Mattsson and Nordstrom, 2004.

#Written for the 2D domain (x,y) \in (a, b) \times (c, d),
#stacking the grid function in the vertical direction, 
#so that, e.g., \partial u / \partial x \approx D \kron I and 
#\partial u / \partial y_out[\approx I \kron D
#the faces of the 2D domain are label bottom to top, left to right, i.e.
#side 1 is y_out[= c, side 2 is y_out[= d
#side 3 is x = a, side 4 is x = b

#D2x and D2y_out[compute approximations to 
#\partial^2 u / \partial x^2 and 
#\partial^2 u / \partial y^2, respectively

#FACEtoVOL(face,y_in_face,Nx,Ny) maps the face value y_in_face to a full length vector at the 
#nodes corresponding to face

#BxSx \approx the traction on faces 3 and 4
#BySy_out[\approx the traction on faces 1 and 2

#BxSx_tran and BySy_tran are the transposes of BxSx and BySy





function D2x_p4(y_in, Nx, Ny, h)
	N = Nx*Ny
	y_out = zeros(N)
	idx = 1:Ny
	y_out[idx] = (y_in[idx] - 2 .* y_in[Ny.+ idx] + y_in[2*Ny .+ idx]) ./ h^2

	idx1 = Ny+1:N-Ny
	y_out[idx1] = (y_in[idx1 .- Ny] - 2 .* y_in[idx1] + y_in[idx1 .+ Ny]) ./ h^2

	idx2 = N-Ny+1:N
	y_out[idx2] = (y_in[idx2 .- 2*Ny] -2 .* y_in[idx2 .- Ny] + y_in[idx2]) ./ h^2

	return y_out
end



function D2y_p4(y_in, Nx, Ny, h)
	N = Nx*Ny
	y_out = zeros(N)
	idx = 1:Ny:N-Ny+1
	y_out[idx] = (y_in[idx] - 2 .* y_in[idx .+ 1] + y_in[idx .+ 2]) ./ h^2

	idx1 = Ny:Ny:N
	y_out[idx1] = (y_in[idx1 .- 2] - 2 .* y_in[idx1 .- 1] + y_in[idx1]) ./ h^2

	for j = 1:Nx
		idx = 2+(j-1)*Ny:j*Ny-1
		y_out[idx] = (y_in[idx .- 1] - 2 .* y_in[idx] + y_in[idx .+ 1]) ./ h^2
	end

	return y_out

end

function Dx_p4(y_in, Nx, Ny, h)
	d  = [1/12 -2/3 0 2/3 -1/12]
    bd = [-24/17  59/34  -4/17  -3/34  0     0;
           -1/2    0      1/2    0     0     0;
            4/43 -59/86   0     59/86 -4/43  0;
            3/98   0    -59/98   0    32/49 -4/49]
	N = Nx*Ny
	y_out = zeros(N)

	for row_num = 1:size(bd)[1]
		idx = (row_num - 1)*Ny+1:(row_num)*Ny
		y_out[idx] = (bd[row_num,1] * y_in[idx .+ (1 - row_num)*Ny] + bd[row_num,2] * y_in[idx .+ Ny*(2-row_num)] + bd[row_num,3] * y_in[idx .+ Ny*(3-row_num)] + bd[row_num,4] * y_in[idx .+ Ny*(4-row_num)] + bd[row_num,5] * y_in[idx .+ Ny*(5-row_num)] + bd[row_num,6] * y_in[idx .+ Ny*(6-row_num)]) ./ h
	end

	idx1 = (size(bd)[1])*Ny+1:N-(size(bd)[1])*Ny
	y_out[idx1] = (d[1]*y_in[idx1 .- 2*Ny] + d[2]*y_in[idx1 .- Ny] + d[3]*y_in[idx1] + d[4]*y_in[idx1 .+ Ny] + d[5] * y_in[idx1 .+ 2*Ny]) ./ (h)

	for row_num = 1:size(bd)[1]
		idx2 = N-(row_num)*Ny+1:N .- (row_num - 1)*Ny
		y_out[idx2] = -(bd[row_num,1] * y_in[idx2 .+ (row_num - 1)*Ny] + bd[row_num,2] * y_in[idx2 .- Ny*(2-row_num)] + bd[row_num,3] * y_in[idx2 .- Ny*(3-row_num)] + bd[row_num,4]*y_in[idx2 .- Ny*(4-row_num)] + bd[row_num,5] * y_in[idx2 .- Ny*(5-row_num)] + bd[row_num,6] * y_in[idx2 .- Ny*(6-row_num)]) ./ h
	end

	return y_out
end

function Dy_p4(y_in, Nx, Ny, h)
	d  = [1/12 -2/3 0 2/3 -1/12]
    bd = [-24/17  59/34  -4/17  -3/34  0     0;
           -1/2    0      1/2    0     0     0;
            4/43 -59/86   0     59/86 -4/43  0;
            3/98   0    -59/98   0    32/49 -4/49]
	N = Nx*Ny
	y_out = zeros(N)

	idx = 1:Ny:N-Ny+1
	y_out[idx] = (y_in[idx .+ 1] - y_in[idx]) ./ h

	idx = Ny:Ny:N
	y_out[idx] = (y_in[idx] - y_in[idx .- 1]) ./h

	for j = 1:Nx
		idx = 2+(j-1)*Ny:j*Ny-1
		y_out[idx] = (y_in[idx .+ 1] - y_in[idx .- 1]) ./ (2*h)
	end

	return y_out
end

function Hxinv_p4(y_in, Nx, Ny, h)
	N = Nx*Ny
	y_out = zeros(N)

	idx = 1:Ny
	y_out[idx] = (2*y_in[idx]) .* (1/h)

	idx = Ny+1:N-Ny
	y_out[idx] = (1*y_in[idx]) .* (1/h)

	idx = N-Ny+1:N
	y_out[idx] = (2*y_in[idx]) .* (1/h)

	return y_out
end

function Hyinv_p4(y_in, Nx, Ny, h)
	N = Nx*Ny
	y_out = zeros(N)

	idx = 1:Ny:N-Ny+1
	y_out[idx] = (2*y_in[idx]) .* (1/h)

	idx = Ny:Ny:N
	y_out[idx] = (2*y_in[idx]) .* (1/h)

	for i = 1:Nx
		idx = 2+(i-1).*Ny:i*Ny-1
		y_out[idx] = y_in[idx] .* (1/h)
	end

	return y_out
	
end

function Hx_p4(y_in, Nx, Ny, h)
	N = Nx*Ny
    y_out = zeros(N)

    idx = 1:Ny
	y_out[idx] = h .* (1/2)*y_in[idx]

    idx = Ny+1:N-Ny
    y_out[idx] = h .* 1*y_in[idx]

    idx = N-Ny+1:N
	y_out[idx] = h .* (1/2)*y_in[idx]

    return y_out

        
end

function Hy_p4(y_in, Nx, Ny, h)
	N = Nx*Ny
    y_out = zeros(N)

    idx = 1:Ny:N-Ny+1
	y_out[idx] = h .* (1/2)*y_in[idx]

    idx = Ny:Ny:N
	y_out[idx] = h .* (1/2)*y_in[idx]

    for i = 1:Nx
        idx = 2+(i-1).*Ny:i*Ny-1
        y_out[idx] = h .* y_in[idx]
    end

    return y_out
        
end

function FACEtoVOL(y_in_face, face, Nx, Ny)
	N = Nx*Ny
	y_out = zeros(N)

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

	y_out[idx] = y_in_face

	return y_out
	
end

function VOLtoFACE(y_in, face, Nx, Ny)
	N = Nx*Ny
    y_out = zeros(N)

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

	y_out[idx] = y_in[idx]
    return y
end

function Bx(Nx,Ny)
	N = Nx*Ny
	y_out = zeros(N)

	idx = 1:Ny
	y_out[idx] = -1 .* ones(Ny)

	idx = N-Ny+1:N
	y_out[idx] = 1 .* ones(Ny)
	return y_out
end

function By(Nx,Ny)
	N = Nx*Ny
	y_out = zeros(N)

	idx = 1:Ny:N-Ny+1
	y_out[idx] = -1 .* ones(Ny)

	idx = Ny:Ny:N
	y_out[idx] = 1 .* ones(Ny)
	return y_out
end

function BxSx_p4(y_in, Nx, Ny, h)
	N = Nx*Ny
	y_out = zeros(N)

	idx = 1:Ny
	y_out[idx] = (1/h) .* (1.5 .* y_in[idx] - 2 .* y_in[idx .+ Ny] + 0.5 .* y_in[idx .+ 2*Ny])
	y_out[N-Ny.+ idx] = (1/h) .* (0.5 .* y_in[N-3*Ny .+ idx] - 2 .* y_in[N-2*Ny .+ idx] + 1.5 .* y_in[N-Ny .+ idx])

	return y_out
	
end

function BySy_p4(y_in, Nx, Ny, h)
	N = Nx*Ny
	y_out = zeros(N)

	idx = 1:Ny:N-Ny+1
	y_out[idx] = (1/h) .* (1.5 .* y_in[idx] - 2 .* y_in[idx .+ 1] + 0.5 .* y_in[idx .+ 2])

	idx = Ny:Ny:N
	y_out[idx] = (1/h) .* (0.5 .* y_in[idx .- 2] - 2 .* y_in[idx .- 1] + 1.5 .* y_in[idx])
	
	return y_out
end

function BxSx_tran_p4(y_in, Nx, Ny, h)
	N = Nx*Ny
	y_out = zeros(N)

	idx1 = 1:Ny
	y_out[idx1] += (1.5 .* y_in[idx1]) .* (1/h)
	idx = Ny+1:2*Ny
	y_out[idx] += (-2 .* y_in[idx1]) .* (1/h)
	idx  = 2*Ny+1:3*Ny
	y_out[idx] += (0.5 .* y_in[idx1]) .* (1/h)

	idxN = N-Ny+1:N
	y_out[idxN] += (1.5 .* y_in[idxN]) .* (1/h)
	idx = N-2*Ny+1:N-Ny
	y_out[idx] += (-2 .* y_in[idxN]) .* (1/h)
	idx = N-3*Ny+1:N-2*Ny
	y_out[idx] += (0.5 .* y_in[idxN]) .* (1/h)
	
	return y_out
end


function BySy_tran_p4(y_in, Nx, Ny, h)
	N = Nx*Ny
	y_out = zeros(N)

	idx1 = 1:Ny:N-Ny+1
	y_out[idx1] += (1.5 .* y_in[idx1]) .* (1/h)
	idx = 2:Ny:N-Ny+2
	y_out[idx] += (-2 .* y_in[idx1]) .* (1/h)
	idx = 3:Ny:N-Ny+3
	y_out[idx] += (0.5 .* y_in[idx1]) .* (1/h)

	idxN = Ny:Ny:N
	y_out[idxN] += (1.5 .* y_in[idxN]) .* (1/h)
	idx = Ny-1:Ny:N-1
	y_out[idx] += (-2 .* y_in[idxN]) .* (1/h)
	idx = Ny-2:Ny:N-2
	y_out[idx] += (0.5 .* y_in[idxN]) .* (1/h)

	return y_out
end




