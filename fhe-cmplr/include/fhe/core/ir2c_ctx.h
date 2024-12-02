//-*-c++-*-
//=============================================================================
//
// Copyright (c) XXXX-XXXX
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//=============================================================================

#ifndef FHE_CORE_IR2C_CTX_H
#define FHE_CORE_IR2C_CTX_H

#include "air/core/ir2c_ctx.h"
#include "fhe/core/lower_ctx.h"
#include "fhe/poly/poly2c_config.h"

namespace fhe {

namespace core {

/**
 * @brief Context for IR to C in fhe-cmplr
 *
 */
class IR2C_CTX : public air::core::IR2C_CTX {
public:
  /**
   * @brief Construct a new ir2c ctx object
   *
   * @param os Output stream
   */
  IR2C_CTX(std::ostream& os, const LOWER_CTX& lower_ctx,
           const fhe::poly::POLY2C_CONFIG& cfg)
      : air::core::IR2C_CTX(os), _lower_ctx(lower_ctx), _config(cfg) {}

  bool Is_poly_type(air::base::TYPE_ID type) {
    return _lower_ctx.Is_poly_type(type);
  }
  bool Is_plain_type(air::base::TYPE_ID type) {
    return _lower_ctx.Is_plain_type(type);
  }
  bool Is_cipher_type(air::base::TYPE_ID type) {
    return _lower_ctx.Is_cipher_type(type);
  }
  bool Is_cipher3_type(air::base::TYPE_ID type) {
    return _lower_ctx.Is_cipher3_type(type);
  }

  const core::LOWER_CTX& Lower_ctx() { return _lower_ctx; }

  void        Add_output_name(const char* name) { _output_names.push_back(name); }
  std::vector<std::string> Output_names() const { return _output_names; }

  DECLARE_POLY2C_CONFIG_ACCESS_API(_config)

private:
  // lower context
  const core::LOWER_CTX&          _lower_ctx;
  std::vector<std::string>        _output_names;
  const fhe::poly::POLY2C_CONFIG& _config;

};  // IR2C_CTX

}  // namespace core

}  // namespace fhe

#endif  // FHE_CORE_IR2C_CTX_H
