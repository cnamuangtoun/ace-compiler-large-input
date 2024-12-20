//-*-c++-*-
//=============================================================================
//
// Copyright (c) XXXX-XXXX
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//=============================================================================

#ifndef NAME_MAP_H
#define NAME_MAP_H

#include "nn/onnx2air/onnx2air_decl.h"
#include "onnx.pb.h"

namespace nn {
namespace onnx2air {

enum RK { R_NONE, R_SYM, R_PREG, R_CST, R_SYM_LIST, R_PREG_LIST };

class NAME_MAP {
private:
  NAME_MAP(void);                        // REQUIRED UNDEFINED UNWANTED methods
  NAME_MAP& operator=(const NAME_MAP&);  // REQUIRED UNDEFINED UNWANTED methods
  ADDR_DATUM_PTR _sym;
  PREG_PTR       _preg;
  CONSTANT_PTR   _cst;
  std::vector<ADDR_DATUM_PTR> _sym_list;
  std::vector<PREG_PTR> _preg_list;
  uint16_t       _kind : 3;

  NAME_MAP(RK kind) { _kind = kind; }
  NAME_MAP(ADDR_DATUM_PTR sym) {
    _kind = R_SYM;
    _sym  = sym;
  }
  NAME_MAP(PREG_PTR preg) {
    _kind = R_PREG;
    _preg = preg;
  }
  NAME_MAP(CONSTANT_PTR cst) {
    _kind = R_CST;
    _cst  = cst;
  }
  NAME_MAP(const std::vector<ADDR_DATUM_PTR>& sym_list) {
    _kind = R_SYM_LIST;
    _sym_list = sym_list;
  }
  NAME_MAP(const std::vector<PREG_PTR>& preg_list) {
    _kind = R_PREG_LIST;
    _preg_list = preg_list;
  }

public:
  static NAME_MAP New_none() { return NAME_MAP(R_NONE); }
  static NAME_MAP New_sym(ADDR_DATUM_PTR sym) { return NAME_MAP(sym); }
  static NAME_MAP New_preg(PREG_PTR preg) { return NAME_MAP(preg); }
  static NAME_MAP New_cst(CONSTANT_PTR cst) { return NAME_MAP(cst); }
  static NAME_MAP New_sym_list(const std::vector<ADDR_DATUM_PTR>& sym_list) { return NAME_MAP(sym_list); }
  static NAME_MAP New_preg_list(const std::vector<PREG_PTR>& preg_list) { return NAME_MAP(preg_list); }

  RK             Kind() const { return (RK)_kind; }
  ADDR_DATUM_PTR Sym() const {
    AIR_ASSERT_MSG(_kind == R_SYM, ("not sym"));
    return _sym;
  }
  PREG_PTR Preg() const {
    AIR_ASSERT_MSG(_kind == R_PREG, ("not preg"));
    return _preg;
  }
  CONSTANT_PTR Cst() const {
    AIR_ASSERT_MSG(_kind == R_CST, ("not constant"));
    return _cst;
  }
  const std::vector<ADDR_DATUM_PTR>& Sym_list() const {
    AIR_ASSERT_MSG(_kind == R_SYM_LIST, ("not symbol list"));
    return _sym_list;
  }
  const std::vector<PREG_PTR>& Preg_list() const {
    AIR_ASSERT_MSG(_kind == R_PREG_LIST, ("not preg list"));
    return _preg_list;
  }

  bool Is_none() const { return _kind == R_NONE; }
  bool Is_sym() const { return _kind == R_SYM; }
  bool Is_preg() const { return _kind == R_PREG; }
  bool Is_cst() const { return _kind == R_CST; }
  bool Is_sym_list() const { return _kind == R_SYM_LIST; }
  bool Is_preg_list() const { return _kind == R_PREG_LIST; }
  NAME_MAP(const NAME_MAP& inp) {
    this->_kind = inp.Kind();
    switch (inp.Kind()) {
      case R_NONE:
        break;
      case R_SYM:
        this->_sym = inp.Sym();
        break;
      case R_PREG:
        this->_preg = inp.Preg();
        break;
      case R_CST:
        this->_cst = inp.Cst();
        break;
      case R_SYM_LIST:
        this->_sym_list = inp.Sym_list();
        break;
      case R_PREG_LIST:
        this->_preg_list = inp.Preg_list();
        break;
    }
  }
};

}  // namespace onnx2air
}  // namespace nn

#endif /* NAME_MAP_H */
