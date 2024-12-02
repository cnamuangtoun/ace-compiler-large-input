//-*-c++-*-
//=============================================================================
//
// Copyright (c) XXXX-XXXX
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//=============================================================================

#include "nn/onnx2air/air_type.h"
#include "nn/onnx2air/air_gen.h"
#include "nn/onnx2air/air_utils.h"
#include "onnx.pb.h"

namespace nn {
namespace onnx2air {

TYPE_PTR
AIRTYGEN::Convert_builtin_type(int32_t data_type) {
  std::unordered_map<int, PRIMITIVE_TYPE> onx_ty_map = {
      {onnx::TensorProto_DataType_FLOAT,  PRIMITIVE_TYPE::FLOAT_32},
      {onnx::TensorProto_DataType_DOUBLE, PRIMITIVE_TYPE::FLOAT_64},
      {onnx::TensorProto_DataType_BOOL,   PRIMITIVE_TYPE::BOOL    },
      {onnx::TensorProto_DataType_INT8,   PRIMITIVE_TYPE::INT_S8  },
      {onnx::TensorProto_DataType_UINT8,  PRIMITIVE_TYPE::INT_U8  },
      {onnx::TensorProto_DataType_INT16,  PRIMITIVE_TYPE::INT_S16 },
      {onnx::TensorProto_DataType_UINT16, PRIMITIVE_TYPE::INT_U16 },
      {onnx::TensorProto_DataType_INT32,  PRIMITIVE_TYPE::INT_S32 },
      {onnx::TensorProto_DataType_UINT32, PRIMITIVE_TYPE::INT_U32 },
      {onnx::TensorProto_DataType_INT64,  PRIMITIVE_TYPE::INT_S64 },
      {onnx::TensorProto_DataType_UINT64, PRIMITIVE_TYPE::INT_U64 }
  };
  if (onx_ty_map.count(data_type))
    return Get_airgen()->Get_glob()->Prim_type(onx_ty_map[data_type]);
  AIR_ASSERT_MSG(false, ("Unsupported builtin type:\n"));
  return Null_ptr;
}

TYPE_PTR
AIRTYGEN::Convert_io_tensor_type(onnx::ValueInfoProto& vi) {
  onnx::TypeProto        tp            = vi.type();
  onnx::TypeProto_Tensor tpt           = tp.tensor_type();
  int32_t                datatype      = tpt.elem_type();
  TYPE_PTR               base_type_idx = Convert_builtin_type(datatype);
  onnx::TensorShapeProto tsp           = tpt.shape();
  std::vector<int>       data_dim;
  for (onnx::TensorShapeProto_Dimension d : tsp.dim()) {
    int dim_size = 1;
    if (isalpha(d.dim_param()[0])) {
      if (d.dim_value()) {
        dim_size = d.dim_value();
      }
    } else {
      dim_size = d.dim_value();
    }
    data_dim.push_back(dim_size);
  }
  if (tsp.dim().size() == 0) data_dim.push_back(1);
  TYPE_PTR tensor_type = Create_tensor_type(base_type_idx, data_dim, Get_airgen()->Get_glob());

  int is_chunked;
  if (ShouldChunkTensor(tensor_type)) {
    is_chunked = 1;
    tensor_type->Set_attr<int>("is_chunked", &is_chunked, 1);
    
    // Calculate chunking information
    int num_channels = GetNumChannels(tensor_type);
    std::cout << "num_channels: " << num_channels << "\n";
    tensor_type->Set_attr<int>("num_channels", &num_channels, 1);
    int chunks_per_channel = CalculateChunksPerChannel(tensor_type);
    tensor_type->Set_attr<int>("chunks_per_channel", &chunks_per_channel, 1);
  } else {
    is_chunked = 0;
    tensor_type->Set_attr<int>("is_chunked", &is_chunked, 1);
  }

  return tensor_type;
}

bool AIRTYGEN::ShouldChunkTensor(TYPE_PTR tensor_type) {
  // Calculate the total number of elements in the tensor
  int total_elements = 1;
  for (int dim_size : tensor_type->Cast_to_arr()->Shape()) {
    total_elements *= dim_size;
  }

  int slot_capacity = 512;

  // Determine if chunking is necessary
  return total_elements > slot_capacity;
}

int AIRTYGEN::GetNumChannels(TYPE_PTR tensor_type) {
  // Assuming NCHW format, channel dimension is at index 1
  const std::vector<long int> shape = tensor_type->Cast_to_arr()->Shape();
  if (shape.size() > 1) {
    return shape[1];
  } else {
    // Handle tensors without a channel dimension
    return 1;
  }
}

int AIRTYGEN::CalculateChunksPerChannel(TYPE_PTR tensor_type) {
  int H = tensor_type->Cast_to_arr()->Shape()[2]; // Height
  int W = tensor_type->Cast_to_arr()->Shape()[3]; // Width
  int slot_capacity = 512;
  int rows_per_chunk = slot_capacity / W;

  int num_channels = GetNumChannels(tensor_type);
  int num_chunks_per_channel = (H + rows_per_chunk - 1) / rows_per_chunk;

  return num_chunks_per_channel;
}

TYPE_PTR
AIRTYGEN::Convert_tensor_type(const onnx::TensorProto& tensor) {
  int32_t          datatype      = tensor.data_type();
  TYPE_PTR         base_type_ptr = Convert_builtin_type(datatype);
  std::vector<int> data_dim;
  for (int dim : tensor.dims()) data_dim.push_back(dim);
  if (tensor.dims().size() == 0) data_dim.push_back(1);
  return Create_tensor_type(base_type_ptr, data_dim, Get_airgen()->Get_glob());
}
}  // namespace onnx2air
}  // namespace nn
