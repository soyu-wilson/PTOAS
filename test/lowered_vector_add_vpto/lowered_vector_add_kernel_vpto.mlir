module attributes {pto.backend = "vpto", pto.target_arch = "a5"} {
  module attributes {pto.backend = "vpto", pto.kernel_kind = #pto.kernel_kind<vector>, pto.target_arch = "a5"} {
    func.func @vector_add_kernel_outlined_vf_0(%arg0: !pto.ptr<f32, ub>, %arg1: !pto.ptr<f32, ub>, %arg2: !pto.ptr<f32, ub>) attributes {no_inline} {
      %c64_i16 = arith.constant 64 : i16
      %c256_i16 = arith.constant 256 : i16
      %c0_i16 = arith.constant 0 : i16
      %c0 = arith.constant 0 : index
      pto.vecscope {
        %0 = pto.pset_b32 "PAT_ALL" : !pto.mask<b32>
        scf.for %arg3 = %c0_i16 to %c256_i16 step %c64_i16  : i16 {
          %1 = arith.index_cast %arg3 : i16 to index
          %2 = pto.addptr %arg0, %1 : <f32, ub> -> <f32, ub>
          %3 = pto.addptr %arg1, %1 : <f32, ub> -> <f32, ub>
          %4 = pto.addptr %arg2, %1 : <f32, ub> -> <f32, ub>
          %result = pto.vlds %2[%c0] : !pto.ptr<f32, ub> -> !pto.vreg<64xf32>
          %result_0 = pto.vlds %3[%c0] : !pto.ptr<f32, ub> -> !pto.vreg<64xf32>
          %5 = pto.vadd %result, %result_0, %0 : !pto.vreg<64xf32>, !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg<64xf32>
          pto.vsts %5, %4[%c0], %0 : !pto.vreg<64xf32>, !pto.ptr<f32, ub>, !pto.mask<b32>
        }
      }
      return
    }
    func.func @vector_add_kernel(%arg0: !pto.ptr<ui8, gm>, %arg1: !pto.ptr<ui8, gm>, %arg2: !pto.ptr<f32, gm>, %arg3: !pto.ptr<f32, gm>, %arg4: !pto.ptr<f32, gm>, %arg5: i32, %arg6: i32, %arg7: i32) attributes {pto.kernel} {
      %false = arith.constant false
      %c0_i64 = arith.constant 0 : i64
      %c1024_i64 = arith.constant 1024 : i64
      %c1_i64 = arith.constant 1 : i64
      %c-1152921504606846977_i64 = arith.constant -1152921504606846977 : i64
      %c281474976710656_i64 = arith.constant 281474976710656 : i64
      %c1152921504606846976_i64 = arith.constant 1152921504606846976 : i64
      %0 = pto.get_ctrl : i64
      %1 = arith.andi %0, %c-1152921504606846977_i64 : i64
      pto.set_ctrl %1 : i64
      %2 = pto.get_ctrl : i64
      %3 = arith.ori %2, %c281474976710656_i64 : i64
      pto.set_ctrl %3 : i64
      %4 = pto.castptr %c0_i64 : i64 -> !pto.ptr<f32, ub>
      pto.copy_gm_to_ubuf %arg2, %4, %c0_i64, %c1_i64, %c1024_i64, %c0_i64, %c0_i64, %false, %c0_i64, %c0_i64, %c0_i64 : !pto.ptr<f32, gm>, !pto.ptr<f32, ub>, i64, i64, i64, i64, i64, i1, i64, i64, i64
      %5 = pto.castptr %c1024_i64 : i64 -> !pto.ptr<f32, ub>
      pto.copy_gm_to_ubuf %arg3, %5, %c0_i64, %c1_i64, %c1024_i64, %c0_i64, %c0_i64, %false, %c0_i64, %c0_i64, %c0_i64 : !pto.ptr<f32, gm>, !pto.ptr<f32, ub>, i64, i64, i64, i64, i64, i1, i64, i64, i64
      pto.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
      pto.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
      call @vector_add_kernel_outlined_vf_0(%4, %5, %4) : (!pto.ptr<f32, ub>, !pto.ptr<f32, ub>, !pto.ptr<f32, ub>) -> ()
      pto.set_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
      pto.wait_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
      pto.copy_ubuf_to_gm %4, %arg4, %c0_i64, %c1_i64, %c1024_i64, %c0_i64, %c0_i64, %c0_i64 : !pto.ptr<f32, ub>, !pto.ptr<f32, gm>, i64, i64, i64, i64, i64, i64
      %6 = pto.get_ctrl : i64
      %7 = arith.ori %6, %c1152921504606846976_i64 : i64
      pto.set_ctrl %7 : i64
      pto.barrier <PIPE_ALL>
      return
    }
  }
}
