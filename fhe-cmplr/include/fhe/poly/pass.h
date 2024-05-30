//-*-c++-*-
//=============================================================================
//
// Copyright (c) XXXX-XXXX
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//=============================================================================

#ifndef FHE_POLY_PASS_H
#define FHE_POLY_PASS_H

#include "air/driver/pass.h"
#include "config.h"

namespace fhe {

namespace driver {
// forward declaration for FHE_COMPILER
class FHE_COMPILER;
}  // namespace driver

namespace poly {

class POLY_PASS : public air::driver::PASS<POLY_CONFIG> {
public:
  POLY_PASS();

  R_CODE Init(driver::FHE_COMPILER* driver);

  R_CODE Pre_run();

  R_CODE Run();

  void Post_run();

  void Fini();

  const char* Name() const { return "POLY"; }

private:
  driver::FHE_COMPILER* _driver;
};  // POLY_PASS

}  // namespace poly

}  // namespace fhe

#endif  // FHE_POLY_PASS_H
