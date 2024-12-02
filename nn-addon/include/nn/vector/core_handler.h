//-*-c++-*-
//=============================================================================
//
// Copyright (c) XXXX-XXXX
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//=============================================================================

#ifndef NN_VECTOR_CORE_HANDLER_H
#define NN_VECTOR_CORE_HANDLER_H

#include "air/base/container.h"
#include "air/base/st.h"
#include "air/core/default_handler.h"
#include "nn/vector/tensor2vector_ctx.h"
#include "air/core/opcode.h"
#include <functional>

namespace nn {
namespace vector {

using namespace air::base;

//! @brief Core handler for Vector Lowering
class CORE_HANDLER : public air::core::DEFAULT_HANDLER {
public:
  template <typename RETV, typename VISITOR>
  RETV Handle_ld(VISITOR* visitor, air::base::NODE_PTR node) {
    TENSOR2VECTOR_CTX& ctx  = visitor->Context();
    CONTAINER*     cntr = visitor->Context().Container();
    ADDR_DATUM_PTR data =
        cntr->Parent_func_scope()->Addr_datum(node->Addr_datum_id());

    SYM_LIST_MAP t2v_sym_list_map   = ctx.Get_t2v_sym_list_map();
    SYM_LIST_MAP::iterator iter = t2v_sym_list_map.find(data->Id().Value());
    std::cout << "Handle_ld - Original Sym ID: " << data->Id().Value() << "\n";
    NODE_PTR new_load;

    if (iter != t2v_sym_list_map.end()) {
      std::vector<NODE_PTR> load_list;
      for (const auto& cur_iter : iter->second) {
        std::cout << "can find \n";
        ADDR_DATUM_PTR used_sym =
          cntr->Parent_func_scope()->Addr_datum(ADDR_DATUM_ID(cur_iter));
        NODE_PTR new_load = cntr->New_ld(used_sym, node->Spos());
        const int* is_chunked = new_load->Rtype()->Attr<int>("is_chunked");
        if (is_chunked && *is_chunked) {
          std::cout << "ldp chunked \n";
        } else {
          std::cout << "ldp not chunked \n";
        }
        load_list.push_back(new_load);
      }
      ConcatInputsToTree(cntr, load_list, node->Spos(), 0, new_load);
    } else {
      std::cout << "cannot find \n";
      new_load = cntr->New_ld(data, node->Spos());
    }
    
    new_load->Print_tree(std::cout);
    return RETV(new_load);
  }

  template <typename RETV, typename VISITOR>
  RETV Handle_retv(VISITOR* visitor, air::base::NODE_PTR node) {
    TENSOR2VECTOR_CTX& ctx  = visitor->Context();
    CONTAINER*         cntr = visitor->Context().Container();
    NODE_PTR child = Node(visitor->template Visit<RETV>(node->Child(0)));
    std::vector<NODE_PTR> inputs;

    std::cout << "handle retv: \n";

    const int* is_chunked = child->Rtype()->Attr<int>("is_chunked");
    if (is_chunked && *is_chunked) {
      std::cout << "concat node is chunked \n";
    } else {
      std::cout << "concat node is not chunked \n";
    }

    if (child->Opcode() == air::base::OPCODE(nn::core::NN, nn::core::OPCODE::CONCAT)) {
      inputs = CollectChunks<RETV>(child, visitor);
      std::cout << "chunk input size: " << inputs.size() << "\n";
    } else {
      STMT_PTR new_store = cntr->New_retv(child, node->Spos());
      return RETV(new_store->Node());
    }

    std::cout << "input size st: " << inputs.size() << "\n";
    STMT_PTR new_store;
    int leaf_index = 0;
    for (auto leaf_node : inputs) {
      const int* is_chunked = leaf_node->Rtype()->Attr<int>("is_chunked");
      if (is_chunked && *is_chunked) {
        std::cout << "chunked \n";
      } else {
        std::cout << "not chunked \n";
      }
      STMT_PTR new_store = cntr->New_retv(leaf_node, node->Spos());
      ctx.Prepend(new_store);
      new_store->Node()->Print_tree(std::cout);
    }

    return RETV();
  }

  template <typename RETV, typename VISITOR>
  RETV Handle_st(VISITOR* visitor, air::base::NODE_PTR node) {
    TENSOR2VECTOR_CTX& ctx  = visitor->Context();
    CONTAINER*         cntr = visitor->Context().Container();
    NODE_PTR child = Node(visitor->template Visit<RETV>(node->Child(0)));
    std::vector<NODE_PTR> inputs;

    std::cout << "handle st: \n";
    ADDR_DATUM_PTR data =
        cntr->Parent_func_scope()->Addr_datum(node->Addr_datum_id());
    data->Set_type(child->Rtype());

    const int* is_chunked = child->Rtype()->Attr<int>("is_chunked");
    if (is_chunked && *is_chunked) {
      std::cout << "concat node is chunked \n";
    } else {
      std::cout << "concat node is not chunked \n";
    }

    if (child->Opcode() == air::base::OPCODE(nn::core::NN, nn::core::OPCODE::CONCAT)) {
      inputs = CollectChunks<RETV>(child, visitor);
      std::cout << "chunk input size: " << inputs.size() << "\n";
    } else {
      STMT_PTR new_store = cntr->New_st(child, data, node->Spos());
      return RETV(new_store->Node());
    }

    std::cout << "input size st: " << inputs.size() << "\n";
    STMT_PTR new_store;
    std::vector<uint32_t> sym_ids;
    int leaf_index = 0;
    for (auto leaf_node : inputs) {
      const int* is_chunked = leaf_node->Rtype()->Attr<int>("is_chunked");
      if (is_chunked && *is_chunked) {
        std::cout << "chunked \n";
      } else {
        std::cout << "not chunked \n";
      }
      const int* chunks_per_channel = leaf_node->Rtype()->Attr<int>("chunks_per_channel");
      std::cout << "st name: " << data->Name()->Char_str() << "\n";
      const std::string chunk_name = 
        std::string(data->Name()->Char_str()) + 
        "_channel_" + 
        std::to_string(leaf_index / *chunks_per_channel) + 
        "_chunk_" + 
        std::to_string(leaf_index % *chunks_per_channel);
      ADDR_DATUM_PTR var  = cntr->Parent_func_scope()->New_var(leaf_node->Rtype(), chunk_name.c_str(), node->Spos());
      leaf_index++;
      STMT_PTR new_store = cntr->New_st(leaf_node, var, node->Spos());
      sym_ids.push_back(var->Id().Value());
      ctx.Prepend(new_store);
      new_store->Node()->Print_tree(std::cout);
    }

    ctx.Insert_t2v_sym_list_map({data->Id().Value(), sym_ids});

    return RETV();
  }

  template <typename RETV, typename VISITOR>
  RETV Handle_stp(VISITOR* visitor, air::base::NODE_PTR node) {
    TENSOR2VECTOR_CTX& ctx  = visitor->Context();
    CONTAINER*         cntr = visitor->Context().Container();
    NODE_PTR child = Node(visitor->template Visit<RETV>(node->Child(0)));
    PREG_PTR preg  = cntr->Parent_func_scope()->Preg(node->Preg_id());
    std::vector<NODE_PTR> inputs;

    std::cout << "Handle_stp - Original Preg ID: " << preg->Id().Value() << "\n";

    std::cout << "handle stp: \n";
    child->Print_tree(std::cout);

    const int* is_chunked = child->Rtype()->Attr<int>("is_chunked");
    if (is_chunked && *is_chunked) {
      std::cout << "concat node is chunked \n";
    } else {
      std::cout << "concat node is not chunked \n";
    }

    if (child->Opcode() == air::base::OPCODE(nn::core::NN, nn::core::OPCODE::CONCAT)) {
      inputs = CollectChunks<RETV>(child, visitor);
      std::cout << "chunk input size: " << inputs.size() << "\n";
    } else {
      STMT_PTR new_store = cntr->New_stp(child, preg, node->Spos());
      return RETV(new_store->Node());
    }
    
    std::cout << "input size stp: " << inputs.size() << "\n";
    STMT_PTR new_store;
    std::vector<uint32_t> preg_ids;
    for (auto leaf_node : inputs) {
      const int* is_chunked = leaf_node->Rtype()->Attr<int>("is_chunked");
      if (is_chunked && *is_chunked) {
        std::cout << "chunked \n";
      } else {
        std::cout << "not chunked \n";
      }
      PREG_PTR new_type_preg =
        cntr->Parent_func_scope()->New_preg(leaf_node->Rtype());
      new_store = cntr->New_stp(leaf_node, new_type_preg, node->Spos());
      preg_ids.push_back(new_type_preg->Id().Value());
      ctx.Prepend(new_store);
      new_store->Node()->Print_tree(std::cout);
    }

    ctx.Insert_t2v_preg_list_map({preg->Id().Value(), preg_ids});

    return RETV();
  }

  template <typename RETV, typename VISITOR>
  RETV Handle_ldp(VISITOR* visitor, air::base::NODE_PTR node) {
    TENSOR2VECTOR_CTX& ctx  = visitor->Context();
    CONTAINER*         cntr = visitor->Context().Container();
    PREG_PTR orig_preg      = cntr->Parent_func_scope()->Preg(node->Preg_id());

    PREG_LIST_MAP t2v_preg_list_map   = ctx.Get_t2v_preg_list_map();
    PREG_LIST_MAP::iterator iter = t2v_preg_list_map.find(orig_preg->Id().Value());
    std::cout << "Handle_ldp - Original Preg ID: " << orig_preg->Id().Value() << "\n";
    NODE_PTR new_load;

    if (iter != t2v_preg_list_map.end()) {
      std::vector<NODE_PTR> load_list;
      for (const auto& cur_iter : iter->second) {
        std::cout << "can find \n";
        PREG_PTR used_preg =
          cntr->Parent_func_scope()->Preg(PREG_ID(cur_iter));
        NODE_PTR new_load = cntr->New_ldp(used_preg, node->Spos());
        const int* is_chunked = new_load->Rtype()->Attr<int>("is_chunked");
        if (is_chunked && *is_chunked) {
          std::cout << "ldp chunked \n";
        } else {
          std::cout << "ldp not chunked \n";
        }
        load_list.push_back(new_load);
      }
      ConcatInputsToTree(cntr, load_list, node->Spos(), 0, new_load);
    } else {
      std::cout << "cannot find \n";
      new_load = cntr->New_ldp(orig_preg, node->Spos());
    }


    new_load->Print_tree(std::cout);
    return RETV(new_load);
  }

private:
  template <typename RETV>
  NODE_PTR Node(RETV retv) {
    AIR_ASSERT(retv.Num_node() == 1);
    return retv.Node();
  }

  NODE_PTR Node(NODE_PTR node) { return node; }

  template <typename RETV, typename VISITOR>
  std::vector<NODE_PTR> CollectChunks(NODE_PTR root, VISITOR* visitor) {
    std::vector<NODE_PTR> chunks;

    // Recursive helper function
    std::function<void(NODE_PTR)> traverse = [&](NODE_PTR current) {
        if (current->Opcode() != air::base::OPCODE(nn::core::NN, nn::core::OPCODE::CONCAT)) {
            NODE_PTR visited_node = current;
            chunks.push_back(current);
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

  void ConcatInputsToTree(CONTAINER* cntr, 
                        std::vector<NODE_PTR>& input, 
                        const SPOS& spos, int ignore_n, 
                        NODE_PTR& concatenated_tree) {

    size_t num_inputs = input.size();
    if (num_inputs < 2) {
      throw std::invalid_argument("Insufficient input nodes for concatenation");
    } else if (num_inputs == 2) {
      concatenated_tree = cntr->New_bin_arith(
          air::base::OPCODE(nn::core::NN, nn::core::OPCODE::CONCAT), input[0], input[1], spos);
      return;
    }

    // Helper function to recursively build the tree of concatenated nodes
    std::function<NODE_PTR(size_t, size_t)> build_concat_tree = [&](size_t start, size_t end) -> NODE_PTR {
      if (start == end) {
        return input[start];
      }

      size_t mid = start + (end - start) / 2;
      NODE_PTR left = build_concat_tree(start, mid);
      NODE_PTR right = build_concat_tree(mid + 1, end);

      // Create a concatenation node for the left and right subtrees
      return cntr->New_bin_arith(
          air::base::OPCODE(nn::core::NN, nn::core::OPCODE::CONCAT), left, right, spos);
    };

    // Build the tree for the inputs except the last n
    concatenated_tree = build_concat_tree(0, num_inputs - ignore_n - 1);
  }
};

}  // namespace vector
}  // namespace nn

#endif  // NN_VECTOR_CORE_HANDLER_H
