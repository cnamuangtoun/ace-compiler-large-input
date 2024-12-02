//-*-c++-*-
//=============================================================================
//
// Copyright (c) XXXX-XXXX
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//=============================================================================

#ifndef NN_VECTOR_TENSOR2VECTOR_HANDLER_H
#define NN_VECTOR_TENSOR2VECTOR_HANDLER_H
#include <iostream>
#include <sstream>
#include "air/base/transform_util.h"
#include "nn/core/null_handler.h"
#include "nn/vector/tensor2vector_ctx.h"
#include "nn/vector/tensor2vector_util.h"
#include "nn/vector/vector_opcode.h"
#include "nn/vector/vector_utils.h"

namespace nn {
namespace vector {

class TENSOR2VECTOR_HANDLER : public nn::core::NULL_HANDLER {
public:
  TENSOR2VECTOR_HANDLER() {}

  template <typename RETV, typename VISITOR>
  RETV Handle_add(VISITOR* visitor, air::base::NODE_PTR node) {
    TENSOR2VECTOR_CTX& ctx = visitor->Context();
    TIMING_UTIL        timing(ctx, node->Spos(), "Tensor::add", false);
    TENSOR2VECTOR_UTIL vgen(ctx);
    NODE_PTR           new_ld0 = visitor->template Visit<RETV>(node->Child(0));
    NODE_PTR           new_ld1 = visitor->template Visit<RETV>(node->Child(1));
    NODE_PTR           new_add = vgen.New_add(new_ld0, new_ld1, node->Spos());
    return new_add;
  }

  template <typename RETV, typename VISITOR>
  RETV Handle_mul(VISITOR* visitor, air::base::NODE_PTR node) {
    TENSOR2VECTOR_CTX& ctx = visitor->Context();
    TIMING_UTIL        timing(ctx, node->Spos(), "Tensor::mul", false);
    TENSOR2VECTOR_UTIL vgen(ctx);
    NODE_PTR           new_ld0 = visitor->template Visit<RETV>(node->Child(0));
    NODE_PTR           new_ld1 = visitor->template Visit<RETV>(node->Child(1));
    NODE_PTR           new_mul = vgen.New_mul(new_ld0, new_ld1, node->Spos());
    return new_mul;
  }

  template <typename RETV, typename VISITOR>
  RETV Handle_relu(VISITOR* visitor, air::base::NODE_PTR node) {
    TENSOR2VECTOR_CTX& ctx = visitor->Context();
    CONTAINER*         cntr = ctx.Container();
    SPOS               spos     = node->Spos();
    TENSOR2VECTOR_UTIL vgen(ctx);


    NODE_PTR new_ld = visitor->template Visit<RETV>(node->Child(0));

    std::cout << "start relu handle \n"; 

    // // Get original conv2d input shape. Assuming NCHW&padding now.
    new_ld->Print_tree(std::cout);
    NODE_PTR orig_input = new_ld->Child(0);
    std::vector<NODE_PTR> input_chunks;

    // Retrieve the "is_chunked" attribute from orig_input
    const int* num_channels;
    const int* chunks_per_channel;
    const int* is_chunked = orig_input->Rtype()->Attr<int>("is_chunked");
    if (is_chunked && *is_chunked) {
      // Retrieve "chunks_per_channel" from orig_input
      chunks_per_channel = orig_input->Rtype()->Attr<int>("chunks_per_channel");
      ctx.Trace(TF_LOWER, "Processing chunked input with ", *chunks_per_channel, " chunks per channel.\n");
      num_channels = orig_input->Rtype()->Attr<int>("num_channels");
      
      input_chunks = CollectChunks<RETV>(new_ld, visitor);

      ctx.Trace(TF_LOWER, "Collected ", input_chunks.size(), " input chunks.\n");
      std::cout << "input is chunked \n";
    } else {
      ctx.Trace(TF_LOWER, "Input is not chunked.\n");
      std::cout << "???? \n";
    }

    std::cout << "relu input size: " << input_chunks.size() << "\n";

    std::vector<NODE_PTR> outputs;
    for (auto input : input_chunks) {
      NODE_PTR new_ld = visitor->template Visit<RETV>(input);
      NODE_PTR op     = visitor->Context().Container()->Clone_node(node);
      op->Set_child(0, new_ld->Id());
      op->Rtype()->Set_attr<int>("is_chunked", orig_input->Rtype()->Attr<int>("is_chunked"), 1);
      op->Rtype()->Set_attr<int>("chunks_per_channel", orig_input->Rtype()->Attr<int>("chunks_per_channel"), 1);
      op->Rtype()->Set_attr<int>("num_channels", orig_input->Rtype()->Attr<int>("num_channels"), 1);
      outputs.push_back(op);
    }

    return vgen.New_concat_output_node(cntr, outputs, spos, 0);
  }

  template <typename RETV, typename VISITOR>
  RETV Handle_flatten(VISITOR* visitor, air::base::NODE_PTR node) {
    TENSOR2VECTOR_CTX& ctx = visitor->Context();
    TIMING_UTIL        timing(ctx, node->Spos(), "Tensor::flatten", false);
    CONTAINER*         cntr = ctx.Container();
    TENSOR2VECTOR_UTIL vgen(ctx);
    GLOB_SCOPE*        gscope     = cntr->Glob_scope();
    SPOS               spos       = node->Spos();
    NODE_PTR           orig_input = node->Child(0);
    NODE_PTR           new_op0    = visitor->template Visit<RETV>(orig_input);
    // TODO: handle flatten axis. Now flatten to [1,x].
    // Only set load. Other op rtype is conistent.
    std::vector<int64_t> shape = orig_input->Rtype()->Cast_to_arr()->Shape();
    if ((orig_input->Opcode() ==
         air::base::OPCODE(air::core::CORE, air::core::OPCODE::LD)) &&
        (shape.size() > 1)) {
      int64_t size = 1;
      for (auto s : shape) size *= s;
      ctx.Trace(TF_LOWER, "WARN: flatten new_input ", shape.size(),
                " is not 1D! \n");
      ctx.Trace_cmd(TF_LOWER, Trace_node, new_op0);
      std::vector<int64_t> shape1d{size};
      NODE_PTR new_op0_reshape = vgen.New_reshape(new_op0, shape1d, spos);
      return new_op0_reshape;
    }
    return new_op0;
  }

  template <typename RETV, typename VISITOR>
  RETV Handle_reshape(VISITOR* visitor, air::base::NODE_PTR node) {
    // TODO: use handle_flatten now, will improve later!
    return Handle_flatten<RETV, VISITOR>(visitor, node);
  }

  template <typename RETV, typename VISITOR>
  RETV Handle_conv(VISITOR* visitor, air::base::NODE_PTR node) {
    TENSOR2VECTOR_CTX& ctx = visitor->Context();
    if (ctx.Improve_ss_insert()) {
      ctx.Incr_num_op_ca_t2vsh();
    }
    
    if (ctx.Toeplitz()) {
      return Handle_conv_toeplitz<RETV>(visitor, node);
    }
    
    TIMING_UTIL        timing(ctx, node->Spos(), "Tensor::conv", false);
    CONTAINER*         cntr = ctx.Container();
    TENSOR2VECTOR_UTIL vgen(ctx);
    GLOB_SCOPE*        gscope   = cntr->Glob_scope();
    SPOS               spos     = node->Spos();
    CONST_TYPE_PTR     s32_type = gscope->Prim_type(PRIMITIVE_TYPE::INT_S32);
    NODE_PTR orig_input = node->Child(0);
    int64_t  batch = 0, channel_in = 0, input_height = 0, input_width = 0;
    Get_array_nchw(orig_input->Rtype(), batch, channel_in, input_height,
                   input_width);

    int64_t output_height = input_height;
    int64_t output_width  = input_width;

    ctx.Trace(TF_LOWER, "conv orig_input shape: [", batch, ", ", channel_in,
              ", ", input_height, ", ", input_width, "]\n");
    AIR_ASSERT_MSG(batch == 1, "Conv only supports batch=1");

    NODE_PTR weight_node = node->Child(1);
    int64_t  channel_out = 0, channel_in_kernel = 0, kernel_height = 0,
            kernel_width = 0;
    Get_array_nchw(weight_node->Rtype(), channel_out, channel_in_kernel,
                   kernel_height, kernel_width);
    AIR_ASSERT_MSG(channel_in == channel_in_kernel,
                   "channel_in == channel_in_kernel");
    ctx.Trace(TF_LOWER, "conv kernel shape: [", channel_out, ", ", channel_in,
              ", ", kernel_height, ", ", kernel_width, "]\n");

    NODE_PTR new_input   = visitor->template Visit<RETV>(node->Child(0));
    NODE_PTR new_input1d = new_input;
    AIR_ASSERT_MSG(new_input->Rtype()->Is_array(),
                   "conv new_input is not an array type");
    if (new_input->Rtype()->Cast_to_arr()->Shape().size() > 1) {
      ctx.Trace(TF_LOWER, "conv new_input is not 1D! Reshaping to 1D.\n");
      ctx.Trace_cmd(TF_LOWER, Trace_node, new_input);
      //  Insert reshape input to 1D
      std::vector<int64_t> input1d_shape(
          1, channel_in * input_height * input_height);
      new_input1d = vgen.New_reshape(new_input, input1d_shape, spos);
    }

    // transpose_im2col weight
    ctx.Trace_cmd(TF_LOWER, Trace_float_array, weight_node->Const(),
                  "conv_weight");
    const float* cptr = weight_node->Const()->Array_ptr<float>();
    FPVEC        weight(
        cptr, cptr + channel_out * channel_in * kernel_height * kernel_width);

    int stride = 1;
    // The following code just solves the second conv in LeNet.
    // Get_num_op_ca_t2vsh() == 3 means conv(1)-avgpool(2)-conv(3)
    // It's duplication length (ceil(16/6)*6*32*32) exceeds 32768
    // due to "no gap handling" in avgpool.
    // Need a whole analysis to handle this case.
    // So Never modify the weight_pad code currently.
    if (ctx.Improve_ss_insert() && (ctx.Get_num_op_ca_t2vsh() == 3)) {
      stride = 2;
      AIR_ASSERT_MSG(channel_in == 6, "for lenet second conv");
      FPVEC weight_pad(
          channel_out * (channel_in + 2) * kernel_height * kernel_width, 0);
      for (int i = 0; i < channel_out; i++)
        for (int j = 0; j < channel_in; j++)
          for (int k = 0; k < kernel_height * kernel_width; k++)
            weight_pad[i * (channel_in + 2) * kernel_height * kernel_width +
                       j * kernel_height * kernel_width + k] =
                weight[i * channel_in * kernel_height * kernel_width +
                       j * kernel_height * kernel_width + k];
      channel_in += 2;
      weight = std::move(weight_pad);
    }
    int64_t kernel_size = kernel_height * kernel_width;
    // Handle case where channel_out%channel_in != 0
    // weight cannot mul all input channel. So expand input and weight.
    if ((channel_out >= channel_in) && ctx.Conv_fast() &&
        channel_out % channel_in != 0) {
      ctx.Trace(TF_LOWER, "channel_out%channel_in != 0 -> padding channel_in=",
                channel_in, "\n");
      int channel_in_new = channel_in;
      while (channel_out % channel_in_new != 0) channel_in_new++;
      ctx.Trace(TF_LOWER, "padding channel_in_new=", channel_in_new, "\n");
      FPVEC weight_pad(channel_out * channel_in_new * kernel_size, 0);
      for (int i = 0; i < channel_out; i++)
        for (int j = 0; j < channel_in; j++)
          for (int k = 0; k < kernel_size; k++)
            weight_pad[i * channel_in_new * kernel_size + j * kernel_size + k] =
                weight[i * channel_in * kernel_size + j * kernel_size + k];
      channel_in = channel_in_new;
      weight     = std::move(weight_pad);
      ctx.Trace(TF_LOWER, "padding weight.size()=", weight.size(), "\n");
    }

    FPMAT conv1_im2col_kernel(
        channel_in * kernel_height * kernel_width,
        FPVEC(channel_out * input_height * input_width, 0.0));
    std::vector<int> ra(kernel_height * kernel_width, 0);
    Get_im2col_kernel(weight, channel_in, input_height, input_width,
                      channel_out, kernel_height, kernel_width, 1, stride, ra,
                      conv1_im2col_kernel);

    std::vector<int> real_strides = Get_attr_int(node, "strides");
    AIR_ASSERT_MSG(real_strides.size() == 2, "conv stride size only support 2");
    AIR_ASSERT_MSG(real_strides[0] == real_strides[1],
                   "the value of conv stride should be equal currently");
    std::vector<int> real_pads = Get_attr_int(node, "pads");
    AIR_ASSERT_MSG(real_pads.size() == 4, "conv padding size only support 4");
    ctx.Trace(TF_LOWER, "conv stride is ", real_strides[0], "\n");
    ctx.Trace(TF_LOWER, "conv padding is ", real_pads[0], "\n");
    if ((real_strides[0] > 1) && (real_pads[0] != 0)) {
      Masking_padding_stride_data_in_mat(
          channel_in * kernel_height * kernel_width, input_height, input_width,
          channel_out, real_pads[0], real_strides[0], conv1_im2col_kernel);
    } else if (real_pads[0] == 0) {
      Masking_no_padding_stride_data_in_mat(
          channel_in * kernel_height * kernel_width, input_height, input_width,
          kernel_height, kernel_width, channel_out, real_pads[0],
          real_strides[0], conv1_im2col_kernel);
    }

    if ((channel_out >= channel_in) && ctx.Conv_fast()) {
      for (int i = 1; i < channel_in; i++) {
        for (int j = 0; j < kernel_size; j++) {
          rotate(conv1_im2col_kernel[i * kernel_size + j].begin(),
                 conv1_im2col_kernel[i * kernel_size + j].begin() +
                     conv1_im2col_kernel[i * kernel_size + j].size() -
                     i * input_height * input_width,
                 conv1_im2col_kernel[i * kernel_size + j].end());
        }
      }
    }

    FPVEC weight_im2col_vec;
    for (int i = 0; i < channel_in * kernel_height * kernel_width; i++)
      for (int j = 0; j < channel_out * input_height * input_width; j++)
        weight_im2col_vec.push_back(conv1_im2col_kernel[i][j]);

    // New weight_im2col_const
    int64_t weight_im2col_size = channel_in * kernel_height * kernel_width *
                                 channel_out * input_height * input_width;
    std::vector<int64_t> weight_im2col_shape{
        channel_in * kernel_height * kernel_width,
        channel_out * input_height * input_width};
    std::string weight_im2col_str =
        New_array_name("weight_im2col_float", weight_im2col_shape);
    CONSTANT_PTR weight_im2col_const = New_array_const(
        gscope, weight_im2col_str.c_str(), weight_im2col_size,
        weight_node->Rtype()->Cast_to_arr()->Elem_type(), weight_im2col_shape,
        (void*)weight_im2col_vec.data(), spos);
    NODE_PTR new_weight = cntr->New_ldc(weight_im2col_const, spos);

    // Expand bias const: TODO: add has broadcast, to sihe?
    NODE_PTR     bias_node = node->Child(2);
    const float* bias_ptr  = bias_node->Const()->Array_ptr<float>();
    FPVEC        bias_expand(channel_out * output_height * output_width);
    for (int i = 0; i < channel_out; i++) {
      for (int j = 0; j < output_height * output_width; j++) {
        bias_expand[i * output_height * output_width + j] = bias_ptr[i];
      }
    }

    if ((real_strides[0] > 1) && (real_pads[0] != 0)) {
      Masking_padding_stride_data_in_vec(output_height, output_width,
                                         channel_out, real_pads[0],
                                         real_strides[0], bias_expand);
    } else if (real_pads[0] == 0) {
      Masking_no_padding_stride_data_in_vec(
          output_height, output_width, channel_out, real_pads[0],
          real_strides[0], kernel_height, kernel_width, bias_expand);
    }

    int64_t bias_expand_size = channel_out * output_height * output_width;
    std::vector<int64_t> bias_expand_shape{bias_expand_size};
    CONSTANT_PTR         bias_expand_const =
        New_array_const(gscope, "bias_expand", bias_expand_size,
                        bias_node->Rtype()->Cast_to_arr()->Elem_type(),
                        bias_expand_shape, (void*)bias_expand.data(), spos);
    NODE_PTR new_bias = cntr->New_ldc(bias_expand_const, spos);

    NODE_PTR new_node;
    if ((channel_out >= channel_in) && ctx.Conv_fast())
      new_node = vgen.New_conv_metakernel_fast(
          new_input1d, new_weight, new_bias, ra, channel_in, channel_out,
          output_height, output_width, kernel_height * kernel_width, spos);
    else
      new_node = vgen.New_conv_metakernel(
          new_input1d, new_weight, new_bias, ra, channel_in, channel_out,
          output_height, output_width, kernel_height * kernel_width, stride,
          spos);
    return new_node;
  }

  template <typename RETV, typename VISITOR>
  RETV Handle_gemm(VISITOR* visitor, air::base::NODE_PTR node) {
    TENSOR2VECTOR_CTX& ctx = visitor->Context();
    TIMING_UTIL        timing(ctx, node->Spos(), "Tensor::gemm", false);
    CONTAINER*         cntr = ctx.Container();
    TENSOR2VECTOR_UTIL vgen(ctx);
    // TODO: pad for shape consistence.
    NODE_PTR     new_ld0   = visitor->template Visit<RETV>(node->Child(0));
    SPOS         spos      = node->Spos();
    CONSTANT_PTR op1_const = node->Child(1)->Const();
    AIR_ASSERT_MSG(node->Child(1)->Rtype()->Is_array(),
                   "operand1 is not an array type");
    ARRAY_TYPE_PTR op1_ty_arr = node->Child(1)->Rtype()->Cast_to_arr();

    std::vector<int> op1_shape;
    for (DIM_ITER dim_iter = op1_ty_arr->Begin_dim();
         dim_iter != op1_ty_arr->End_dim(); ++dim_iter) {
      op1_shape.push_back((*dim_iter)->Ub_val());
    }
    AIR_ASSERT_MSG(op1_shape.size() == 2, "operand1_const dim %d != 2",
                   op1_shape.size());

    int          height = op1_shape[0];
    int          width  = op1_shape[1];
    const float* cptr   = op1_const->Array_ptr<float>();

    // Read the op1_const
    ctx.Trace_cmd(TF_LOWER, Trace_float_array, op1_const, "gemm weight");
    FPMAT op1_mat(height);
    for (int i = 0; i < height; i++) {
      op1_mat[i].resize(width);
      for (int j = 0; j < width; j++) {
        op1_mat[i][j] = cptr[i * width + j];
      }
    }

    // Assuming GEMM:: transB = 1
    // transpose_diag(op1_const)
    FPVEC diag_vec;
    int   padw;
    for (padw = width; padw <= height * width; padw++) {
      if (padw % height == 0) break;
    }

    int64_t old_height = height;
    if (ctx.Gemm_fast()) {
      AIR_ASSERT_MSG(((height & (height - 1)) == 0) &&
                         (((height / width) & (height / width - 1)) == 0),
                     "TODO: gemm_fast only supports height=%d=2^n now. "
                     "Example: 8x4, 2048x1024",
                     height);
      if (old_height > width) {
        padw   = height;
        height = width;
      } else {
        padw = width;
      }
    }

    int h1 = 0, h2 = 0;
    Get_block_size(height, h1, h2);
    ctx.Trace(TF_LOWER, "Handle_gemm Get_block_size: height=", height,
              ", h1=", h1, ", h2=", h2, "\n");

    if (ctx.Gemm_fast()) {
      // weight tiling
      for (int i = 0; i < height; i++) {
        FPVEC diag;
        if (old_height > width) {
          FPMAT slice_weight0(op1_mat.begin(), op1_mat.begin() + height);
          diag = Transpose_diagonal(slice_weight0, i, height);
          rotate(diag.begin(), diag.begin() + diag.size() - (i / h1) * h1,
                 diag.end());
          for (size_t slice = 1; slice < old_height / height; slice++) {
            FPMAT slice_weight(op1_mat.begin() + height * slice,
                               op1_mat.begin() + height * (slice + 1));
            FPVEC diag2 = Transpose_diagonal(slice_weight, i, height);
            rotate(diag2.begin(), diag2.begin() + diag2.size() - (i / h1) * h1,
                   diag2.end());
            diag = diag + diag2;
          }
        } else {
          diag = Transpose_diagonal(op1_mat, i, padw);
          rotate(diag.begin(), diag.begin() + diag.size() - (i / h1) * h1,
                 diag.end());
        }
        diag_vec = diag_vec + diag;
      }
    } else {
      for (int i = 0; i < height; i++) {
        FPVEC diag = Transpose_diagonal(op1_mat, i, padw);
        diag_vec   = diag_vec + diag;
      }
    }
    AIR_ASSERT_MSG(diag_vec.size() == padw * height,
                   "operand1_diag size != height*padw  %d != %d*%d",
                   diag_vec.size(), height, padw);

    // new 2d diag_array const: height*padw
    GLOB_SCOPE*          gscope = cntr->Glob_scope();
    std::vector<int64_t> weight_diag_shape{height, padw};
    CONSTANT_PTR         diag_const = New_array_const(
        gscope, "weight_diag", height * padw, op1_ty_arr->Elem_type(),
        weight_diag_shape, (void*)diag_vec.data(), spos);
    // new_ld1 is LD operand_diag
    NODE_PTR new_weight = cntr->New_ldc(diag_const, spos);
    NODE_PTR new_bias   = visitor->template Visit<RETV>(node->Child(2));

    NODE_PTR new_node;
    if (ctx.Gemm_fast()) {
      new_node = vgen.New_gemm_metakernel_fast(new_ld0, new_weight, new_bias,
                                               spos, old_height > width);
    } else
      new_node = vgen.New_gemm_metakernel(new_ld0, new_weight, new_bias, spos);
    return new_node;
  }

  template <typename RETV, typename VISITOR>
  RETV Handle_average_pool(VISITOR* visitor, air::base::NODE_PTR node) {
    TENSOR2VECTOR_CTX& ctx = visitor->Context();
    if (ctx.Improve_ss_insert()) {
      ctx.Incr_num_op_ca_t2vsh();
    }
    TIMING_UTIL        timing(ctx, node->Spos(), "Tensor::avg_pool", false);
    CONTAINER*         cntr = ctx.Container();
    TENSOR2VECTOR_UTIL vgen(ctx);
    // TODO: pad for shape consistence.
    AIR_ASSERT_MSG(node->Num_child() == 1,
                   "average pool operator only support 1 child");
    NODE_PTR input = visitor->template Visit<RETV>(node->Child(0));
    SPOS     spos  = node->Spos();

    AIR_ASSERT_MSG(input->Rtype()->Is_array(), "operand is not an array type");
    ARRAY_TYPE_PTR       op_ty_arr = input->Rtype()->Cast_to_arr();
    std::vector<int64_t> op_shape  = op_ty_arr->Shape();
    AIR_ASSERT_MSG(op_shape.size() == 4, "input shape should be 4");
    // std::cout << ">> input shape size: " << op_shape.size() << std::endl;
    // for (auto dim : op_shape) {
    //   std::cout << dim << std::endl;
    // }
    std::vector<int> kernel_shape = Get_attr_int(node, "kernel_shape");
    std::vector<int> strides      = Get_attr_int(node, "strides");

    int64_t stride = strides[0];
    int64_t ks     = kernel_shape[0];
    int64_t c_in   = op_shape[1];
    int64_t h      = op_shape[2];
    int64_t w      = op_shape[3];

    AIR_ASSERT_MSG(stride == 2 || stride == 4,
                   "average pool only support stride and ks equals 2 or 4!");

    GLOB_SCOPE* gscope = cntr->Glob_scope();
    FUNC_SCOPE* fscope = cntr->Parent_func_scope();

    std::string add_row_str =
        std::string("tmp_row_add") + std::to_string(ctx.Get_num_vloop());
    ADDR_DATUM_PTR add_row_var =
        fscope->New_var(op_ty_arr, add_row_str.c_str(), spos);

    CONST_TYPE_PTR s32_type = gscope->Prim_type(PRIMITIVE_TYPE::INT_S32);

    NODE_PTR extra_rool_node = input;
    if (ctx.Improve_ss_insert() && (ctx.Get_num_op_ca_t2vsh() == 4)) {
      // TODO: hard code here!
      stride = 4;
      ks     = 4;
      std::vector<int> extra_roll_num{6 * 32 + 6};
      extra_rool_node =
          vgen.New_roll(input, cntr->New_intconst(s32_type, 6 * 32 + 6, spos),
                        extra_roll_num, spos);
    }

    // calculate sum of element in kernel shape
    std::vector<int> roll_num1{(int)ks / 2 * (int)w};
    NODE_PTR         tmp_roll_node1 = vgen.New_roll(
        extra_rool_node, cntr->New_intconst(s32_type, ks / 2 * w, spos),
        roll_num1, spos);

    NODE_PTR add_row_node = vgen.New_add(extra_rool_node, tmp_roll_node1, spos);
    STMT_PTR add_row_stmt = cntr->New_st(add_row_node, add_row_var, spos);

    ctx.Prepend(add_row_stmt);
    NODE_PTR ld_row_add = cntr->New_ld(add_row_var, spos);

    std::vector<int> roll_num2{(int)ks / 2};
    NODE_PTR         tmp_roll_node2 =
        vgen.New_roll(ld_row_add, cntr->New_intconst(s32_type, ks / 2, spos),
                      roll_num2, spos);
    NODE_PTR add_col_node = vgen.New_add(ld_row_add, tmp_roll_node2, spos);

    // masking [1,0,1,0..., 0,0,0,0..., 1,0,1,0..., 0,0,0,0...]
    FPVEC avg_value_mask = Get_avg_value_mask(c_in, h, w, ks);

    std::vector<int64_t> avg_mask_shape{c_in * h * w};
    std::string          mask_name =
        std::string("avg_value_mask") + std::to_string(ctx.Get_num_vloop());
    CONSTANT_PTR mask_const =
        New_array_const(gscope, mask_name, c_in * h * w, op_ty_arr->Elem_type(),
                        avg_mask_shape, (void*)avg_value_mask.data(), spos);
    // new_ldc is LD avg value mask
    NODE_PTR avg_value_mask_node = cntr->New_ldc(mask_const, spos);

    ctx.Trace_cmd(TF_LOWER, Trace_float_array, avg_value_mask_node->Const(),
                  "avg_pool_mask");

    // intermediate average pool result which not considerring stride
    NODE_PTR avgpool_inter_node =
        vgen.New_mul(add_col_node, avg_value_mask_node, spos);

    return avgpool_inter_node;
  }

  template <typename RETV, typename VISITOR>
  RETV Handle_max_pool(VISITOR* visitor, air::base::NODE_PTR node) {
    return Handle_average_pool<RETV, VISITOR>(visitor, node);
  }

  template <typename RETV, typename VISITOR>
  RETV Handle_global_average_pool(VISITOR* visitor, air::base::NODE_PTR node) {
    TENSOR2VECTOR_CTX& ctx = visitor->Context();
    TIMING_UTIL timing(ctx, node->Spos(), "Tensor::global_avg_pool", false);
    CONTAINER*  cntr = ctx.Container();
    TENSOR2VECTOR_UTIL vgen(ctx);

    AIR_ASSERT_MSG(node->Num_child() == 1,
                   "global average pool operator only support 1 child");
    NODE_PTR input_node = visitor->template Visit<RETV>(node->Child(0));
    SPOS     spos       = node->Spos();

    AIR_ASSERT_MSG(input_node->Rtype()->Is_array(),
                   "operand is not an array type");
    ARRAY_TYPE_PTR       op_ty_arr = input_node->Rtype()->Cast_to_arr();
    std::vector<int64_t> op_shape  = op_ty_arr->Shape();
    AIR_ASSERT_MSG(op_shape.size() == 4, "input shape should be 4");

    int64_t c_in = op_shape[1];
    int64_t h    = op_shape[2];
    int64_t w    = op_shape[3];

    GLOB_SCOPE* gscope = cntr->Glob_scope();
    FUNC_SCOPE* fscope = cntr->Parent_func_scope();

    CONST_TYPE_PTR s32_type = gscope->Prim_type(PRIMITIVE_TYPE::INT_S32);

    // 1. use rotate and add to sum element in channel
    STMT_PTR  sum_loop_stmt = vgen.New_loop("sum_index", 0, log2(h * w), spos);
    STMT_LIST sum_body_sl =
        STMT_LIST::Enclosing_list(sum_loop_stmt->Node()->Child(3)->End_stmt());

    // input_roll = ROLL(input, pow(2, i)); pow(2,i) == (1<< i)
    NODE_PTR shl_node1 = cntr->New_bin_arith(
        air::base::OPCODE(air::core::CORE, air::core::OPCODE::SHL),
        cntr->New_intconst(s32_type, 1, spos),
        cntr->New_ld(sum_loop_stmt->Node()->Iv(), spos), spos);

    std::vector<int> input_roll_num;
    for (int i = 0; i < log2(h * w); i++) {
      input_roll_num.push_back(pow(2, i));
    }
    NODE_PTR tmp_node = input_node;
    NODE_PTR input_roll_node =
        vgen.New_roll(tmp_node, shl_node1, input_roll_num, spos);
    tmp_node = vgen.New_add(tmp_node, input_roll_node, spos);

    STMT_PTR sum_store;
    if (input_node->Opcode() ==
        air::base::OPCODE(air::core::CORE, air::core::OPCODE::LDP)) {
      sum_store = cntr->New_stp(tmp_node, input_node->Preg(), spos);
    } else {
      sum_store = cntr->New_st(tmp_node, input_node->Addr_datum(), spos);
    }
    sum_body_sl.Append(sum_store);
    ctx.Prepend(sum_loop_stmt);

    // 2. calculate global average value with gap
    // 2.1 prepare the mask which will be used to calculate global avg value
    FPVEC clear_zero_mask = Get_global_avg_value_mask(c_in, h, w);

    std::vector<int64_t> cz_mask_shape{c_in * h * w};
    CONSTANT_PTR         cz_mask_const = New_array_const(
        gscope, "clear_zero_mask", c_in * h * w, op_ty_arr->Elem_type(),
        cz_mask_shape, (void*)clear_zero_mask.data(), spos);
    NODE_PTR cz_mask_node = cntr->New_ldc(cz_mask_const, spos);

    // 2.2 caculate global average value with gap
    std::string result_str =
        std::string("gap_result") + std::to_string(ctx.Get_num_vloop());
    ADDR_DATUM_PTR gap_result_var =
        fscope->New_var(op_ty_arr, result_str.c_str(), spos);

    NODE_PTR ld_sum_node;
    if (input_node->Opcode() ==
        air::base::OPCODE(air::core::CORE, air::core::OPCODE::LDP)) {
      ld_sum_node = cntr->New_ldp(input_node->Preg(), spos);
    } else {
      ld_sum_node = cntr->New_ld(input_node->Addr_datum(), spos);
    }
    NODE_PTR cz_result_node = vgen.New_mul(ld_sum_node, cz_mask_node, spos);
    STMT_PTR cz_result_stmt =
        cntr->New_st(cz_result_node, gap_result_var, spos);
    ctx.Prepend(cz_result_stmt);

    // 3. use rotate and add to combine and compact
    vgen.Gen_combine_cross_channel(gap_result_var, c_in, h, w, 1, 1, spos);

    // 4. clear other channel invalid data for later use
    // TODO: tag valid data to input, then make subsequent op to do clear data.
    // This can reduce level use.
    vgen.Gen_clear_data_stmt(gap_result_var, h * w, op_ty_arr->Elem_type(),
                             spos);

    NODE_PTR load_result_node = cntr->New_ld(gap_result_var, spos);
    return load_result_node;
  }

  template <typename RETV, typename VISITOR>
  RETV Handle_strided_slice(VISITOR* visitor, air::base::NODE_PTR node) {
    TENSOR2VECTOR_CTX& ctx  = visitor->Context();
    CONTAINER*         cntr = ctx.Container();
    TENSOR2VECTOR_UTIL vgen(ctx);
    AIR_ASSERT_MSG(node->Num_child() == 4,
                   "strided_slice operator only support 4 child");
    NODE_PTR input = visitor->template Visit<RETV>(node->Child(0));
    SPOS     spos  = node->Spos();

    AIR_ASSERT_MSG(input->Rtype()->Is_array(), "operand is not an array type");
    ARRAY_TYPE_PTR       op_ty_arr = input->Rtype()->Cast_to_arr();
    std::vector<int64_t> op_shape  = op_ty_arr->Shape();
    AIR_ASSERT_MSG(op_shape.size() == 4, "input shape should be 4");
    ctx.Trace(TF_LOWER, ">> input shape size: ", op_shape.size(), "\n");

    CONSTANT_PTR start_indice_const = node->Child(1)->Const();
    int64_t      start_row = 0, start_column = 0;
    Get_const_array_value(start_indice_const, start_row, start_column);
    AIR_ASSERT_MSG(start_row == start_column, "only support same start indice");

    CONSTANT_PTR slice_size_const = node->Child(2)->Const();
    int64_t      ss_height = 0, ss_width = 0;
    Get_const_array_value(slice_size_const, ss_height, ss_width);

    CONSTANT_PTR stride_size_const = node->Child(3)->Const();
    int64_t      stride_row = 0, stride_col = 0;
    Get_const_array_value(stride_size_const, stride_row, stride_col);
    AIR_ASSERT_MSG(stride_row == stride_col, "only support same stride size");

    std::vector<int> channel = Get_attr_int(node, "channel");
    AIR_ASSERT_MSG(channel.size() == 1,
                   "strided slice only contains 1 channel attribute");

    // TODO: kernel shape, height, width should also get from attributes
    // or child node?
    int padsize      = start_row;
    int kernal_shape = stride_row;
    if (start_row != 0) {
      kernal_shape = 2 * padsize + 1;
    }
    int64_t actual_height = op_shape[2];
    int64_t actual_width  = op_shape[3];

    GLOB_SCOPE* gscope = cntr->Glob_scope();
    FUNC_SCOPE* fscope = cntr->Parent_func_scope();

    std::string orig_result_str =
        std::string("orig_result") + std::to_string(ctx.Get_num_vloop());
    ADDR_DATUM_PTR orig_result_var =
        fscope->New_var(op_ty_arr, orig_result_str.c_str(), spos);
    STMT_PTR st_stmt = cntr->New_st(input, orig_result_var, spos);
    ctx.Prepend(st_stmt);

    NODE_PTR ld_result = cntr->New_ld(orig_result_var, spos);

    NODE_PTR extraced_node = vgen.New_extract_valid_data(
        ld_result, channel[0], padsize, actual_height, actual_width, ss_height,
        ss_width, kernal_shape, stride_row, (start_row == 0), spos);

    return extraced_node;
  }

  // template <typename RETV, typename VISITOR>
  // RETV Handle_conv_toeplitz(VISITOR* visitor, air::base::NODE_PTR node) {
  //   TENSOR2VECTOR_CTX& ctx = visitor->Context();
  //   TIMING_UTIL timing(ctx, node->Spos(), "Tensor::conv_toeplitz", false);
  //   CONTAINER* cntr = ctx.Container();
  //   TENSOR2VECTOR_UTIL vgen(ctx);
  //   GLOB_SCOPE* gscope = cntr->Glob_scope();
  //   SPOS spos = node->Spos();
  //   CONST_TYPE_PTR s32_type = gscope->Prim_type(PRIMITIVE_TYPE::INT_S32);

  //   // Get original conv2d input shape (Assuming NCHW format)
  //   NODE_PTR orig_input = node->Child(0);
  //   int64_t batch = 0, channel_in = 0, input_height = 0, input_width = 0;
  //   Get_array_nchw(orig_input->Rtype(), batch, channel_in, input_height, input_width);

  //   // Set output dimensions to match input dimensions
  //   int64_t output_height = input_height;
  //   int64_t output_width = input_width;

  //   ctx.Trace(TF_LOWER, "conv_toeplitz input shape: [", batch, ", ", channel_in, 
  //             ", ", input_height, ", ", input_width, "]\n");
  //   AIR_ASSERT_MSG(batch == 1, "Conv only supports batch=1");

  //    // Get kernel (weight) node and its shape
  //   NODE_PTR weight_node = node->Child(1);
  //   int64_t channel_out = 0, channel_in_kernel = 0, kernel_height = 0, kernel_width = 0;
  //   Get_array_nchw(weight_node->Rtype(), channel_out, channel_in_kernel, kernel_height, kernel_width);
  //   AIR_ASSERT_MSG(channel_in == channel_in_kernel, "channel_in == channel_in_kernel");
  //   ctx.Trace(TF_LOWER, "conv_toeplitz kernel shape: [", channel_out, ", ", channel_in, ", ", kernel_height, ", ", kernel_width, "]\n");

  //   // Get padding (assuming symmetric padding)
  //   std::vector<int> pads = Get_attr_int(node, "pads");
  //   AIR_ASSERT_MSG(pads.size() == 4, "conv padding size only support 4");
  //   int pad_top = pads[0];
  //   int pad_left = pads[1];

  //   // Visit input node
  //   NODE_PTR new_input   = visitor->template Visit<RETV>(node->Child(0));
  //   NODE_PTR new_input1d = new_input;
  //   AIR_ASSERT_MSG(new_input->Rtype()->Is_array(),
  //                  "conv new_input is not an array type");
  //   if (new_input->Rtype()->Cast_to_arr()->Shape().size() > 1) {
  //     ctx.Trace(TF_LOWER, "conv new_input is not 1D! Reshaping to 1D.\n");
  //     ctx.Trace_cmd(TF_LOWER, Trace_node, new_input);
  //     //  Insert reshape input to 1D
  //     std::vector<int64_t> input1d_shape(
  //         1, channel_in * input_height * input_height);
  //     new_input1d = vgen.New_reshape(new_input, input1d_shape, spos);
  //   }

  //   // Get weight data
  //   ctx.Trace_cmd(TF_LOWER, Trace_float_array, weight_node->Const(),
  //                 "conv_weight");
  //   const float* cptr = weight_node->Const()->Array_ptr<float>();
  //   FPVEC        weight(
  //       cptr, cptr + channel_out * channel_in * kernel_height * kernel_width);

  //   int64_t weight_size = channel_out * channel_in * kernel_height * kernel_width;

  //   // Compute Toeplitz matrix dimensions
  //   int64_t h_o = output_height;
  //   int64_t w_o = output_width;
  //   int64_t howoco = h_o * w_o * channel_out;
  //   int64_t hiwici = input_height * input_width * channel_in;

  //   // Initialize Toeplitz matrix data in FPMAT format
  //   FPMAT T_data(howoco, FPVEC(hiwici, 0.0));

  //   // Construct the Toeplitz matrix
  //   for (int64_t o = 0; o < channel_out; ++o) {
  //     for (int64_t i = 0; i < h_o; ++i) {
  //       for (int64_t j = 0; j < w_o; ++j) {
  //         int64_t row_index = o * h_o * w_o + i * w_o + j;
  //         for (int64_t c = 0; c < channel_in; ++c) {
  //           int64_t kernel_index_base = (o * channel_in + c) * kernel_height * kernel_width;
  //           for (int64_t m = 0; m < kernel_height; ++m) {
  //             for (int64_t n = 0; n < kernel_width; ++n) {
  //               int64_t input_i = i + m - pad_top;
  //               int64_t input_j = j + n - pad_left;
  //               if (0 <= input_i && input_i < input_height && 0 <= input_j && input_j < input_width) {
  //                 int64_t col_index = c * input_height * input_width + input_i * input_width + input_j;
  //                 int64_t kernel_index = kernel_index_base + m * kernel_width + n;
  //                 T_data[row_index][col_index] = weight[kernel_index];
  //               }
  //             }
  //           }
  //         }
  //       }
  //     }
  //   }

  //   int64_t width = howoco;
  //   int64_t height = hiwici;

  //   FPVEC diag_vec;
  //   std::vector<int> ra;
  //   int   padw;
  //   for (padw = width; padw <= height * width; padw++) {
  //     if (padw % height == 0) break;
  //   }

  //   for (int i = 0; i < padw; i++) {
  //     FPVEC diag = Transpose_diagonal(T_data, i, padw);
      
  //     // Check if diag is non-zero
  //     bool is_non_zero = false;
  //     for (const auto& val : diag) {
  //         if (val != 0.0f) {
  //             is_non_zero = true;
  //             break;
  //         }
  //     }
  //     if (is_non_zero) {
  //       diag_vec = diag_vec + diag;
  //       // Store the roll amount
  //       ra.push_back(i);
  //     }
  //   }

  //   int n = ra.size(); // Total number of rotations (number of diagonals)
  //   int n2 = n / channel_in;
  //   int n1 = n / n2;

  //   ctx.Trace(TF_LOWER, "n1: ", n1, " n2: ", n2, "\n");

  //   for (int i = 0; i < n; i++) {
  //     int64_t roll_multiplier = ra[i] / n1;
  //     int64_t roll_amount = roll_multiplier * n1;

  //     // Iterator for current start of rotation
  //     auto cur_index = diag_vec.begin() + i * padw;
      
  //     // Rotate
  //     rotate(cur_index, cur_index + roll_amount, cur_index + padw);
  //   }

  //   ctx.Trace(TF_LOWER, "ra.size()", ra.size(), "\n");
  //   ctx.Trace(TF_LOWER, "padw: ", padw, "\n");

  //   int new_height = ra.size();

  //   // Create a constant for the Toeplitz matrix
  //   std::vector<int64_t> weight_diag_shape{new_height, padw};
  //   std::string T_const_name = New_array_name("Toeplitz_matrix", weight_diag_shape);
  //   CONSTANT_PTR T_const = New_array_const(gscope, T_const_name.c_str(), new_height * padw,
  //                                         weight_node->Rtype()->Cast_to_arr()->Elem_type(),
  //                                         weight_diag_shape, (void*)diag_vec.data(), spos);
  //   NODE_PTR T_node = cntr->New_ldc(T_const, spos);

  //   // Visit bias node
  //   NODE_PTR bias_node = node->Child(2);
  //   NODE_PTR new_bias   = visitor->template Visit<RETV>(bias_node);

  //   NODE_PTR new_node = vgen.New_gemm_metakernel_toeplitz(new_input1d, T_node, new_bias, output_height, output_width, n1, n2, ra, spos);

  //   return new_node;
  // }

  template <typename RETV, typename VISITOR>
  RETV Handle_conv_toeplitz(VISITOR* visitor, air::base::NODE_PTR node) {
    TENSOR2VECTOR_CTX& ctx = visitor->Context();
    if (ctx.Improve_ss_insert()) {
      ctx.Incr_num_op_ca_t2vsh();
    }
    TIMING_UTIL        timing(ctx, node->Spos(), "Tensor::conv", false);
    CONTAINER*         cntr = ctx.Container();
    TENSOR2VECTOR_UTIL vgen(ctx);
    GLOB_SCOPE*        gscope   = cntr->Glob_scope();
    FUNC_SCOPE*        fscope   = cntr->Parent_func_scope();
    SPOS               spos     = node->Spos();
    CONST_TYPE_PTR     s32_type = gscope->Prim_type(PRIMITIVE_TYPE::INT_S32);

    std::cout << "start conv toeplitz handle \n"; 
    node->Print_tree(std::cout);

    // Get original conv2d input shape. Assuming NCHW&padding now.
    NODE_PTR orig_input = node->Child(0)->Child(0);
    std::vector<NODE_PTR> input_chunks;

    // Retrieve the "is_chunked" attribute from orig_input
    const int* num_channels;
    const int* chunks_per_channel;
    const int* is_chunked = orig_input->Rtype()->Attr<int>("is_chunked");
    if (is_chunked && *is_chunked) {
      // Retrieve "chunks_per_channel" from orig_input
      chunks_per_channel = orig_input->Rtype()->Attr<int>("chunks_per_channel");
      ctx.Trace(TF_LOWER, "Processing chunked input with ", *chunks_per_channel, " chunks per channel.\n");
      num_channels = orig_input->Rtype()->Attr<int>("num_channels");
      
      input_chunks = CollectChunks<RETV>(node->Child(0), visitor);

      ctx.Trace(TF_LOWER, "Collected ", input_chunks.size(), " input chunks.\n");
    } else {
      ctx.Trace(TF_LOWER, "Input is not chunked.\n");
    }

    int num_input_blocks = static_cast<int>(input_chunks.size());

    orig_input = input_chunks[0];

    int64_t  batch = 0, channel_in = 0, input_height = 0, input_width = 0;
    Get_array_nchw(orig_input->Rtype(), batch, channel_in, input_height,
                   input_width);

    channel_in *= *num_channels;
    input_height *= *chunks_per_channel;

    int64_t output_height = input_height;
    int64_t output_width  = input_width;

    ctx.Trace(TF_LOWER, "conv orig_input shape: [", batch, ", ", channel_in,
              ", ", input_height, ", ", input_width, "]\n");
    AIR_ASSERT_MSG(batch == 1, "Conv only supports batch=1");

    NODE_PTR weight_node = node->Child(1);
    int64_t  channel_out = 0, channel_in_kernel = 0, kernel_height = 0,
            kernel_width = 0;
    Get_array_nchw(weight_node->Rtype(), channel_out, channel_in_kernel,
                   kernel_height, kernel_width);
    AIR_ASSERT_MSG(channel_in == channel_in_kernel,
                   "channel_in == channel_in_kernel");
    ctx.Trace(TF_LOWER, "conv kernel shape: [", channel_out, ", ", channel_in,
              ", ", kernel_height, ", ", kernel_width, "]\n");

    std::vector<int> pads = Get_attr_int(node, "pads");
    AIR_ASSERT_MSG(pads.size() == 4, "conv padding size only support 4");
    int padding = pads[0];

    std::vector<NODE_PTR> new_input1d_vec;

    for (size_t i = 0; i < input_chunks.size(); ++i) {
      NODE_PTR current_input = input_chunks[i];

      // Get input shape
      int64_t batch = 0, channel_in = 0, input_height = 0, input_width = 0;
      Get_array_nchw(current_input->Rtype(), batch, channel_in, input_height, input_width);

      ctx.Trace(TF_LOWER, "Processing chunk ", i, " with shape: [", batch, ", ", channel_in, ", ", input_height, ", ", input_width, "]\n");

      AIR_ASSERT_MSG(batch == 1, "Conv only supports batch=1");

      // Reshape input to 1D if necessary
      NODE_PTR new_input1d = current_input;
      if (current_input->Rtype()->Cast_to_arr()->Shape().size() > 1) {
        ctx.Trace(TF_LOWER, "Chunk ", i, " is not 1D! Reshaping to 1D.\n");
        std::vector<int64_t> input1d_shape(1, channel_in * input_height * input_width);
        new_input1d_vec.push_back(vgen.New_reshape(current_input, input1d_shape, spos));
      }
    }

    // transpose_im2col weight
    ctx.Trace_cmd(TF_LOWER, Trace_float_array, weight_node->Const(),
                  "conv_weight");
    const float* cptr = weight_node->Const()->Array_ptr<float>();
    FPVEC        weight(
        cptr, cptr + channel_out * channel_in * kernel_height * kernel_width);

    int stride = 1;
    int64_t kernel_size = kernel_height * kernel_width;

    // std::vector<int> ra(kernel_height * kernel_width, 0);
    std::vector<std::vector<FPVEC>> T_blocks;
    std::vector<std::pair<int, int>> output_block_indices;
    std::vector<std::pair<int, int>> input_block_indices;

    Construct_toeplitz_matrix_blocks(channel_in, input_height, input_width, weight, 
                              kernel_height, kernel_width, padding, channel_out, 
                              num_input_blocks,  T_blocks, output_block_indices, 
                              input_block_indices);

    std::vector<std::vector<NODE_PTR>> T_block_nodes(T_blocks.size(), 
    std::vector<NODE_PTR>(num_input_blocks));

    std::vector<std::vector<std::vector<int>>> T_block_ra(T_blocks.size(), 
    std::vector<std::vector<int>>(num_input_blocks));
    
    for (size_t i = 0; i < T_blocks.size(); ++i) {
      for (size_t j = 0; j < T_blocks[i].size(); ++j) {
        const auto& block = T_blocks[i][j];
        int block_rows = output_block_indices[i].second - output_block_indices[i].first;
        int block_cols = input_block_indices[j].second - input_block_indices[j].first;

        // Convert T_block_flat to 2D matrix
        FPMAT T_block_matrix(block_rows, FPVEC(block_cols));
        for (int r = 0; r < block_rows; ++r) {
          for (int c = 0; c < block_cols; ++c) {
            T_block_matrix[r][c] = block[r * block_cols + c];
          }
        }

        FPVEC diag_vec;
        std::vector<int> ra;
        int new_height;

        for (int i = 0; i < block_cols; i++) {
          FPVEC diag = Transpose_diagonal(T_block_matrix, i, block_cols);
          // Check if diag is non-zero
          bool is_non_zero = false;
          for (const auto& val : diag) {
              if (val != 0.0f) {
                  is_non_zero = true;
                  break;
              }
          }
          if (is_non_zero) {
            diag_vec = diag_vec + diag;
            // Store the roll amount
            ra.push_back(i);
          }
        }

        T_block_ra[i][j] = ra;

        new_height = ra.size();

        ctx.Trace(TF_LOWER, "ra.size()", ra.size(), "\n");
        ctx.Trace(TF_LOWER, "padw: ", block_cols, "\n");

        // Create constant node
        std::vector<int64_t> block_shape = { new_height, block_cols };
        std::string block_name = "T_block_diag_" + std::to_string(i) + "_" + std::to_string(j);
        CONSTANT_PTR block_const = New_array_const(
          gscope, block_name.c_str(), new_height * block_cols,
          weight_node->Rtype()->Cast_to_arr()->Elem_type(),
          block_shape, (void*)diag_vec.data(), spos);
        NODE_PTR block_node = cntr->New_ldc(block_const, spos);

        ctx.Trace_cmd(TF_LOWER, Trace_float_array, block_node->Const(),
                  "T_block_diag_" + std::to_string(i) + "_" + std::to_string(j));

        T_block_nodes[i][j] = block_node;
      }
    }

    NODE_PTR     bias_node = node->Child(2);
    const float* bias_ptr  = bias_node->Const()->Array_ptr<float>();
    FPVEC        bias_expand(channel_out * output_height * output_width);
    int slot_capacity = 512;
    size_t num_output_chunks = channel_out * *chunks_per_channel;

    // Initialize vector to hold bias chunks
    std::vector<FPVEC> bias_chunks(num_output_chunks);

    // Precompute the number of elements in each chunk
    std::vector<size_t> elements_in_chunk(*chunks_per_channel);
    for (size_t i = 0; i < *chunks_per_channel; ++i) {
      size_t h_start = i * (slot_capacity / input_width);
      size_t rows_in_chunk = (h_start + slot_capacity / input_width > input_height)
                                ? input_height - h_start
                                : slot_capacity / input_width;
      elements_in_chunk[i] = rows_in_chunk * input_width;
    }

    // Expand and chunk the bias
    size_t chunk_idx = 0;
    for (int k = 0; k < channel_out; ++k) { // For each output channel
      for (size_t i = 0; i < *chunks_per_channel; ++i) { // For each chunk in height
        size_t elem_count = elements_in_chunk[i];

        // Create bias chunk filled with bias_ptr[k]
        FPVEC bias_chunk(elem_count, bias_ptr[k]);

        bias_chunks[chunk_idx++] = bias_chunk;
      }
    }

    // Create constant nodes for bias chunks
    std::vector<NODE_PTR> bias_chunk_nodes(num_output_chunks);
    chunk_idx = 0;
    for (int k = 0; k < channel_out; ++k) { // For each output channel
      for (size_t i = 0; i < *chunks_per_channel; ++i) { // For each chunk in height
        size_t elem_count = elements_in_chunk[i];

        // Get the bias chunk
        FPVEC& bias_chunk = bias_chunks[chunk_idx];

        // Create constant node for bias chunk
        std::vector<int64_t> bias_chunk_shape{elem_count};
        std::string bias_chunk_name = "bias_chunk_" + std::to_string(chunk_idx);
        CONSTANT_PTR bias_chunk_const = New_array_const(
            gscope, bias_chunk_name.c_str(), elem_count,
            bias_node->Rtype()->Cast_to_arr()->Elem_type(),
            bias_chunk_shape, (void*)bias_chunk.data(), spos);
        NODE_PTR bias_chunk_node = cntr->New_ldc(bias_chunk_const, spos);

        ctx.Trace_cmd(TF_LOWER, Trace_float_array, bias_chunk_node->Const(), bias_chunk_name);

        // Store the bias chunk node
        bias_chunk_nodes[chunk_idx] = bias_chunk_node;

        chunk_idx++;
      }
    }

    int64_t bias_expand_size = channel_out * output_height * output_width;
    std::vector<int64_t> bias_expand_shape{bias_expand_size};
    CONSTANT_PTR         bias_expand_const =
        New_array_const(gscope, "bias_expand", bias_expand_size,
                        bias_node->Rtype()->Cast_to_arr()->Elem_type(),
                        bias_expand_shape, (void*)bias_expand.data(), spos);
    NODE_PTR new_bias = cntr->New_ldc(bias_expand_const, spos);

    ctx.Trace_cmd(TF_LOWER, Trace_float_array, new_bias->Const(), "bias");
    
    std::vector<NODE_PTR> output_nodes(T_blocks.size());
    for (size_t i = 0; i < T_blocks.size(); ++i) {
      TYPE_PTR res_type =
          New_array_type(gscope, "type_output_block_n", ctx.Get_num_vloop(),
                        s32_type, input_chunks[0]->Rtype()->Cast_to_arr()->Shape(), spos);

      ADDR_DATUM_PTR Y_block =
          vgen.Gen_store_zero_to_var_stmt("output_block_n", res_type, spos);
      for (size_t j = 0; j < input_chunks.size(); ++j) {
        NODE_PTR T_block_node = T_block_nodes[i][j];
        std::vector<int>& ra = T_block_ra[i][j];
        NODE_PTR X_block = new_input1d_vec[j];

        std::vector<int64_t>actual_shape = input_chunks[j]->Rtype()->Cast_to_arr()->Shape();

        NODE_PTR block_output = vgen.New_gemm_metakernel_toeplitz(X_block, 
          T_block_node, actual_shape[1], actual_shape[3], actual_shape[2], 
          ra, spos);
        NODE_PTR vadd_node = vgen.New_add(cntr->New_ld(Y_block, spos), 
          block_output, spos);
        STMT_PTR vadd_store = cntr->New_st(vadd_node, Y_block, spos);
        ctx.Prepend(vadd_store);
      }
      
      NODE_PTR bias_chunk_node = bias_chunk_nodes[i];
      NODE_PTR Y_block_with_bias = vgen.New_add(cntr->New_ld(Y_block, spos), bias_chunk_node, spos);

      // Set chunking attributes
      Y_block_with_bias->Rtype()->Set_attr<int>("is_chunked", orig_input->Rtype()->Attr<int>("is_chunked"), 1);
      Y_block_with_bias->Rtype()->Set_attr<int>("chunks_per_channel", orig_input->Rtype()->Attr<int>("chunks_per_channel"), 1);
      Y_block_with_bias->Rtype()->Set_attr<int>("num_channels", orig_input->Rtype()->Attr<int>("num_channels"), 1);

      output_nodes[i] = Y_block_with_bias;
    }
    // Return the output nodes
    return vgen.New_concat_output_node(cntr, output_nodes, spos, 0);
  }

  template <typename RETV, typename VISITOR>
  std::vector<NODE_PTR> CollectChunks(NODE_PTR root, VISITOR* visitor) {
    std::vector<NODE_PTR> chunks;

    // Recursive helper function
    std::function<void(NODE_PTR)> traverse = [&](NODE_PTR current) {

      if (current->Opcode() != air::base::OPCODE(nn::core::NN, nn::core::OPCODE::CONCAT)) {
        chunks.push_back(visitor->template Visit<RETV>(current));
        return;
      }
      if (current->Opcode() == air::base::OPCODE(nn::core::NN, nn::core::OPCODE::CONCAT)) {
        // Traverse left and right children
        traverse(current->Child(0));
        traverse(current->Child(1));
      }
    };

    traverse(root);
    return chunks;
  }
};

}  // namespace vector
}  // namespace nn

#endif  // NN_VECTOR_HANDLER_H
