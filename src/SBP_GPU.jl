module SBP_GPU

# Write your package code here.
function test_function()
    println("Hello Julia's World!")
end


include("deriv_ops.jl")
include("deriv_ops_beta.jl")
include("deriv_ops_GPU.jl")

# export D2x, D2x_GPU_shared, tester_function_v3
export Bx, BxSx, BxSx_tran, By, BySy, BySy_tran, D2x, D2y, Dx, Dy, Hx, Hxinv, Hy, Hyinv, FACEtoVOL, VOLtoFACE
export Bx_beta, BxSx_beta, BxSx_tran_beta, By_beta, BySy_beta, BySy_tran_beta, D2x_beta, D2y_beta, Dx_beta, Dy_beta, Hx_beta, Hxinv_beta, Hy_beta, Hyinv_beta, FACEtoVOL_beta, VOLtoFACE_beta
export Bx_GPU_shared, BxSx_GPU_shared, By_GPU_shared, BySy_GPU_shared, BySy_tran_GPU_shared, D2x_GPU_shared, D2y_GPU_shared, Dx_GPU_shared, Dy_GPU_shared, Hx_GPU_shared, Hxinv_GPU_shared, Hy_GPU_shared

end