//-*-c-*-
//=============================================================================
//
// Copyright (c) XXXX-XXXX
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//=============================================================================

// This file should be auto generated by onnx2c.py,
// it's used as driver for testing ONNX.

#include <math.h>

#include "common/rtlib.h"

double Expected_data[] = {-0.15436399, -0.13319215, -0.110327594, -0.1612797, -0.14571834, -0.16106397, -0.19733712, -0.12234347, -0.17777261, -0.14520718, -0.077035144, -0.17484064, -0.09567858, -0.12718305, -0.17221114, -0.14658874, -0.13679962, -0.19026464, -0.1524792, -0.1447199, -0.1614807, -0.12190082, -0.101006545, -0.14811803, -0.14102896};
int Expected_len = 25;

/**
* @brief generate input data for testing ONNX
*
*
* @param n
* @param c
* @param h
* @param w
* @param data, data pointer
* @return TENSOR input data
*/
TENSOR *Generate_input_data(size_t n, size_t c, size_t h, size_t w, double *data) {
    return Alloc_tensor(n, c, h, w, data);
}

/**
 * @brief validate output vector with expect vector
 *
 *
 * @param result double *
 * @param expect double *
 * @param len int
 * @return return true if value match
 */
bool Validate_output_data(double *result, double *expect, int len) {
  double error = 1e-2;
  for(int i = 0; i < len; i++) {
    if (fabs(result[i] - expect[i]) > error) {
      printf("index: %d, value: %f != %f\n", i, result[i], expect[i]);
      return false;
    }
  }
  return true;
}

int main(int argc, char* argv[]) {
  Prepare_context();

  double input1[]={0.043394215404987335, -0.07600603252649307, 0.0019050687551498413, 0.023958586156368256, 0.05224526673555374, -0.028081431984901428, -0.07631033658981323, -0.020811118185520172, 0.08970362693071365, 0.0710049495100975, -0.032893672585487366, 0.05039609223604202, 0.07348894327878952, 0.03136231750249863, -0.04622630029916763, -0.03717760741710663, 0.049590133130550385, 0.01088794320821762, -0.07103017717599869, 0.03597470372915268, 0.06565859168767929, -0.06523159146308899, -0.07067469507455826, -0.006697818636894226, -0.07269861549139023, -0.004511617124080658, 0.05935443192720413, -0.07280044257640839, 0.07323040813207626, -0.062452901154756546, -0.043437134474515915, 0.09852307289838791, -0.00980927050113678, 0.03554714471101761, 0.02939034253358841, -0.09672939777374268, -0.08927831798791885, 0.0902530774474144, -0.09666434675455093, 0.02791690081357956, 0.02378436177968979, 0.06685518473386765, 0.07093756645917892, -0.06632350385189056, -0.07704348862171173, -0.05161556228995323, 0.08880496770143509, -0.06435519456863403, 0.008372046053409576, -0.0024577006697654724, 0.09891655296087265, -0.0035670846700668335, -0.05870095640420914, -0.0622553713619709, 0.08176072686910629, -0.013611532747745514, -0.013970211148262024, -0.09113384783267975, 0.0708378329873085, -0.035615965723991394, -0.0718405619263649, 0.0218452587723732, -0.015116319060325623, -0.017574355006217957, -0.04090462997555733, 0.08681564778089523, -0.05436859279870987, 0.010283686220645905, 0.07772687822580338, -0.08594685792922974, 0.06731108576059341, -0.001478336751461029, -0.08333279937505722, -0.09305226057767868, -0.048956241458654404};
  TENSOR *input_data1 = Generate_input_data(1, 3, 5, 5, input1);
  printf("input");
  Print_tensor(stdout, input_data1);
  Prepare_input(input_data1, "input");
  Free_tensor(input_data1);

  Run_main_graph();

  double  *result = Handle_output("output");

  Finalize_context();

  bool    res    = Validate_output_data(result, Expected_data, Expected_len);
  free(result);
  if (res) {
    printf("SUCCESS!\n");
  } else {
    printf("FAILED!\n");
    return 1;
  }

  return 0;
}

#include "eg_rtseal_conv2d_keep_shape.inc"
