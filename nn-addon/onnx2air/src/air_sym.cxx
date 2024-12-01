//-*-c++-*-
//=============================================================================
//
// Copyright (c) XXXX-XXXX
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//=============================================================================

#include "nn/onnx2air/air_sym.h"

#include <stdio.h>

#include "nn/onnx2air/air_const.h"
#include "nn/onnx2air/air_gen.h"
#include "nn/onnx2air/air_utils.h"
#include "nn/onnx2air/air_type.h"
#include "onnx.pb.h"

namespace nn {
namespace onnx2air {

FUNC_SCOPE* AIRSYMGEN::Convert_func_sym(onnx::GraphProto& onnx_graph) {
  std::vector<NAME_TYPE_PAIR> param_node;
  for (auto vi : onnx_graph.input()) {
    param_node.push_back({vi.name(), _airgen->Tg().Convert_io_tensor_type(vi)});
  }

  std::vector<NAME_TYPE_PAIR> ret_node;
  for (auto vi : onnx_graph.output()) {
    ret_node.push_back({vi.name(), _airgen->Tg().Convert_io_tensor_type(vi)});
  }

  FUNC_SCOPE* func_scope = Create_function(FUNC_NAME, param_node, ret_node);
  if (func_scope == nullptr) return nullptr;
  func_scope->Owning_func()->Entry_point()->Set_program_entry();
  Get_airgen()->Set_func_scope(func_scope);
  for (auto tensor : onnx_graph.initializer())
    Get_airgen()->Sg().Convert_init_sym(tensor);

  return func_scope;
}

FUNC_SCOPE* AIRSYMGEN::Create_function(const char*                  func_name,
                                       std::vector<NAME_TYPE_PAIR>& param_node,
                                       std::vector<NAME_TYPE_PAIR>& ret_node) {
  GLOB_SCOPE* glob     = _airgen->Get_glob();
  STR_PTR     name_str = glob->New_str(func_name);
  BLOCK_PTR   scope    = glob->Comp_env();
  SPOS        spos     = glob->Unknown_simple_spos();

  SIGNATURE_TYPE_PTR sig = glob->New_sig_type();
  for (auto arg : param_node) {
    TYPE_PTR type = arg._ty_ptr;
    const int* is_chunked = type->Attr<int>("is_chunked");
    if (*is_chunked) {
      // Retrieve chunking information
      const int* num_channels = type->Attr<int>("num_channels");
      const int* chunks_per_channel = type->Attr<int>("chunks_per_channel");
      TYPE_PTR base_type = type->Cast_to_arr()->Elem_type();
      std::vector<int64_t> data_dim = type->Cast_to_arr()->Shape();
      int slot_capacity = 512;
      int rows_per_chunk = slot_capacity / data_dim[3];

      // Create parameters for each chunk
      for (int c = 0; c < *num_channels; ++c) {
        int rows_consumed = 0;
        for (int i = 0; i < *chunks_per_channel; ++i) {
          rows_consumed += rows_per_chunk;
          std::vector<int> data_dim_new(data_dim.size());
          std::copy(data_dim.begin(), data_dim.end(), data_dim_new.begin());
          data_dim_new[1] = 1;
          if (rows_consumed <= data_dim[2]) {
            data_dim_new[2] = rows_per_chunk;
          } else {
            data_dim_new[2] = data_dim[2] - (rows_consumed - rows_per_chunk);
          }
          TYPE_PTR new_type = Create_tensor_type(base_type, data_dim_new, Get_airgen()->Get_glob());
          new_type->Set_attr<int>("is_chunked", is_chunked, 1);
          new_type->Set_attr<int>("num_channels", num_channels, 1);
          new_type->Set_attr<int>("chunks_per_channel", chunks_per_channel, 1);
          std::string chunk_name = arg._name + "_channel_" + std::to_string(c) + "_chunk_" + std::to_string(i);
          STR_PTR param_str = glob->New_str(chunk_name.c_str());
          glob->New_param(param_str, new_type, sig, spos);
        }
      }
    } else {
      STR_PTR param_str = glob->New_str(arg._name.c_str());
      glob->New_param(param_str, type, sig, spos);
    }
  }
  sig->Set_complete();

  FUNC_PTR func = glob->New_func(name_str, spos);
  func->Set_parent(glob->Comp_env_id());
  ENTRY_PTR func_ent = glob->New_global_entry_point(sig, func, name_str, spos);

  FUNC_SCOPE* func_scope = &glob->New_func_scope(func);
  STMT_PTR    entry_stmt = func_scope->Container().New_func_entry(spos);
  int         formal_idx = 0;
  for (auto arg : param_node) {
    TYPE_PTR type = arg._ty_ptr;
    const int* is_chunked = type->Attr<int>("is_chunked");
    if (*is_chunked) {
      // Retrieve chunking information
      const int* num_channels = type->Attr<int>("num_channels");
      const int* chunks_per_channel = type->Attr<int>("chunks_per_channel");

      std::vector<ADDR_DATUM_PTR> chunked_formals;

      // Create parameters for each chunk
      for (int c = 0; c < *num_channels; ++c) {
        for (int i = 0; i < *chunks_per_channel; ++i) {
          ADDR_DATUM_PTR formal = func_scope->Formal(formal_idx);
          std::string chunk_name = arg._name + "_channel_" + std::to_string(c) + "_chunk_" + std::to_string(i);
          Put_result(chunk_name, NAME_MAP::New_sym(formal));
          _input_sts.push_back(formal);
          ++formal_idx;

          // Store formal parameter for the original tensor name
          chunked_formals.push_back(formal);
        }
      }

      // Store the list of chunked formals under the original name
      _chunked_parameters[arg._name] = chunked_formals;
    } else {
      ADDR_DATUM_PTR formal = func_scope->Formal(formal_idx);
      Put_result(arg._name, NAME_MAP::New_sym(formal));
      _input_sts.push_back(formal);
      ++formal_idx;
    }
  }

  // for (auto ret : ret_node) {
  //   // Generates the output symbol in the onnx graph.
  //   TYPE_PTR type = ret._ty_ptr;
  //   const int* is_chunked = type->Attr<int>("is_chunked");
  //   if (*is_chunked) {
  //     // Retrieve chunking information
  //     const int* num_channels = type->Attr<int>("num_channels");
  //     const int* chunks_per_channel = type->Attr<int>("chunks_per_channel");
  //     TYPE_PTR base_type = type->Cast_to_arr()->Elem_type();
  //     std::vector<int64_t> data_dim = type->Cast_to_arr()->Shape();
  //     int slot_capacity = 512;
  //     int rows_per_chunk = slot_capacity / data_dim[3];
      
  //     std::vector<ADDR_DATUM_PTR> chunked_rets;
  //     // Create parameters for each chunk
  //     for (int c = 0; c < *num_channels; ++c) {
  //       int rows_consumed = 0;
  //       for (int i = 0; i < *chunks_per_channel; ++i) {
  //         rows_consumed += rows_per_chunk;
  //         std::vector<int> data_dim_new(data_dim.size());
  //         std::copy(data_dim.begin(), data_dim.end(), data_dim_new.begin());
  //         data_dim_new[1] = 1;
  //         if (rows_consumed <= data_dim[2]) {
  //           data_dim_new[2] = rows_per_chunk;
  //         } else {
  //           data_dim_new[2] = data_dim[2] - (rows_consumed - rows_per_chunk);
  //         }
  //         TYPE_PTR new_type = Create_tensor_type(base_type, data_dim_new, Get_airgen()->Get_glob());
  //         new_type->Set_attr<int>("is_chunked", is_chunked, 1);
  //         new_type->Set_attr<int>("num_channels", num_channels, 1);
  //         new_type->Set_attr<int>("chunks_per_channel", chunks_per_channel, 1);
  //         std::string chunk_name = ret._name + "_channel_" + std::to_string(c) + "_chunk_" + std::to_string(i);
  //         std::cout << "Generate chunk name: " <<  chunk_name << "\n";
  //         ADDR_DATUM_PTR st_ptr = Generate_sym(chunk_name, new_type, func_scope);
  //         chunked_rets.push_back(st_ptr);
  //       }
  //     }
  //     ADDR_DATUM_PTR st_ptr = Generate_sym(ret._name, ret._ty_ptr, func_scope);
  //     _output_sts.push_back(st_ptr);
  //     _chunked_parameters[ret._name] = chunked_rets;
  //   } else {
  //     ADDR_DATUM_PTR st_ptr = Generate_sym(ret._name, ret._ty_ptr, func_scope);
  //     _output_sts.push_back(st_ptr);
  //   }
  // }

  for (auto ret : ret_node) {
    // Generates the output symbol in the onnx graph.
    ADDR_DATUM_PTR st_ptr = Generate_sym(ret._name, ret._ty_ptr, func_scope);
    _output_sts.push_back(st_ptr);
  }

  // TODO. At present only one output variable is supported.
  AIR_ASSERT_MSG(
      ret_node.size() <= 1,
      ("This functionality only supports single output variable.\n"));
  // If there is no output variable, the compiler will not generate
  // any return statement at the end of the function.
  if (ret_node.size() > 0) glob->New_ret_param(ret_node[0]._ty_ptr, sig);

  if (ret_node.size() == 0) {
    CMPLR_USR_MSG(U_CODE::No_Output_Var, func_name);
    return nullptr;
  }

  return func_scope;
}

int AIRSYMGEN::GetRowsPerChunk(TYPE_PTR type) {
  int W = type->Cast_to_arr()->Shape()[3];
  int slot_capacity = 512;
  return slot_capacity / W;
}

CONSTANT_PTR AIRSYMGEN::Get_cst(const std::string name) {
  std::unordered_map<std::string, CONSTANT_PTR>::iterator it;
  it = _cst_map.find(name);
  if (it != _cst_map.end()) return it->second;
  return Null_ptr;
}

void AIRSYMGEN::Put_cst(const std::string name, CONSTANT_PTR cst) {
  _cst_map[name] = cst;
}

CONSTANT_PTR AIRSYMGEN::Get_operator_cst(const std::string name) {
  std::unordered_map<std::string, CONSTANT_PTR>::iterator it;
  it = _onx_const_map.find(name);
  if (it != _cst_map.end()) return it->second;
  return Null_ptr;
}

void AIRSYMGEN::Put_operator_cst(const std::string name, CONSTANT_PTR cst) {
  _onx_const_map[name] = cst;
}

PREG_PTR AIRSYMGEN::Get_preg(const std::string name) {
  std::unordered_map<std::string, PREG_PTR>::iterator it;
  it = _preg_map.find(name);
  if (it != _preg_map.end()) return it->second;
  return Null_ptr;
}

void AIRSYMGEN::Put_preg(const std::string name, PREG_PTR preg) {
  _preg_map[name] = preg;
}

NAME_MAP AIRSYMGEN::Get_result(const std::string name) {
  CONSTANT_PTR cst = Get_operator_cst(name);
  if (cst != air::base::Null_ptr) return NAME_MAP::New_cst(cst);

  ADDR_DATUM_PTR sym = Get_st(name);
  if (sym != air::base::Null_ptr) return NAME_MAP::New_sym(sym);

  cst = Get_cst(name);
  if (cst != air::base::Null_ptr) return NAME_MAP::New_cst(cst);

  PREG_PTR preg = Get_preg(name);
  if (preg != air::base::Null_ptr) return NAME_MAP::New_preg(preg);

  // Check if the name corresponds to chunked parameter
  auto it1 = _chunked_parameters.find(name);
  if (it1 != _chunked_parameters.end()) {
    return NAME_MAP::New_sym_list(it1->second);
  }

 // Check if the name corresponds to chunked preg
  auto it2 = _chunked_pregs.find(name);
  if (it2 != _chunked_pregs.end()) {
    return NAME_MAP::New_preg_list(it2->second);
  }

  // Name not found
  return NAME_MAP::New_none();
}

void AIRSYMGEN::Put_result(const std::string name, const NAME_MAP& res) {
  if (res.Is_sym())
    Put_st(name, res.Sym());
  else if (res.Is_preg())
    Put_preg(name, res.Preg());
  else if (res.Is_cst())
    Put_cst(name, res.Cst());

  AIR_ASSERT_MSG(true, "Result is empty.");
}

ADDR_DATUM_PTR
AIRSYMGEN::Get_st(const std::string name) {
  std::unordered_map<std::string, ADDR_DATUM_PTR>::iterator it;
  it = _st_map.find(name);
  if (it != _st_map.end()) return it->second;
  return Null_ptr;
}

void AIRSYMGEN::Put_st(const std::string name, ADDR_DATUM_PTR st_ptr) {
  _st_map[name] = st_ptr;
}

CONSTANT_PTR
AIRSYMGEN::Convert_init_sym(const onnx::TensorProto& tensor) {
  if (onnx::TensorProto_DataLocation() !=
      onnx::TensorProto_DataLocation_DEFAULT)
    AIR_ASSERT_MSG(false, "unhandled: non-default data location in tensor");
  if (tensor.has_segment())
    AIR_ASSERT_MSG(false, "unhandled: segmented data in tensor");
  TYPE_PTR ty_ptr = Get_airgen()->Tg().Convert_tensor_type(tensor);

  AIRCONSTGEN  const_gen(_airgen);
  CONSTANT_PTR cst = const_gen.Convert_const(tensor, ty_ptr);
  Put_result(tensor.name(), NAME_MAP::New_cst(cst));
  return cst;
}
ADDR_DATUM_PTR
AIRSYMGEN::Generate_sym(std::string sym_name, TYPE_PTR ty_ptr,
                        FUNC_SCOPE* func_scope) {
  NAME_MAP st_ptr = Get_result(sym_name);
  AIR_ASSERT_MSG(st_ptr.Is_none(), "Expect emtpy result.");

  STR_PTR        sym_ptr = Get_airgen()->Get_glob()->New_str(sym_name.c_str());
  SPOS           spos    = Get_airgen()->Get_glob()->Unknown_simple_spos();
  ADDR_DATUM_PTR var     = func_scope->New_var(ty_ptr, sym_ptr, spos);
  Put_result(sym_name, NAME_MAP::New_sym(var));
  return var;
}

NAME_MAP
AIRSYMGEN::Get_sym_or_preg(std::string sym_name, TYPE_PTR ty_ptr,
                           FUNC_SCOPE* func_scope) {
  NAME_MAP res = Get_result(sym_name);
  if (res.Is_preg() || res.Is_sym()) return res;

  AIR_ASSERT_MSG(res.Is_none(), "Expect emtpy result.");

  // Except input/output variables, the compiler generates the pregs
  // for the output of onnx ops.
  PREG_PTR preg = func_scope->New_preg(ty_ptr);
  Put_result(sym_name, NAME_MAP::New_preg(preg));
  return NAME_MAP::New_preg(preg);
}

NAME_MAP
AIRSYMGEN::Get_tensor_sym_or_preg(std::string sym_name, TYPE_PTR base_ty,
                                  std::vector<int>& dim,
                                  FUNC_SCOPE*       func_scope) {
  TYPE_PTR tensor_ty_ptr =
      Create_tensor_type(base_ty, dim, Get_airgen()->Get_glob());
  return Get_sym_or_preg(sym_name, tensor_ty_ptr, func_scope);
}

}  // namespace onnx2air
}  // namespace nn
