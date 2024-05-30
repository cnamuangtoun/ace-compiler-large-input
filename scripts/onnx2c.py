#!/usr/bin/env python3

import os
import argparse
import torch
import onnx
import onnxruntime

from enum import Enum, unique

# input onnx file(required), image input data(optional, input shape can get from onnx, content use random data)
# output file name/path,
# run onnx model generate expected output
# generate main c program
# invoke interface from FHE RTLib(or main_graph.c generated by fhe compiler)
# validate output fhe data with expected output(C implement)

SHAPE_DIMENSION = 4


@unique
class InputType(Enum):
    NEG_ONE = 0
    ONE = 1
    INCREMENT = 2  # from 0~n
    RANDOM = 3


comment_info = '''//-*-c-*-
//=============================================================================
//
// Copyright (c) Ant Group Co., Ltd
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//=============================================================================

// This file should be auto generated by onnx2c.py,
// it's used as driver for testing ONNX.
'''

include_header_stmt = '''
#include <math.h>

#include "common/rtlib.h"
'''

gid_stmt = '''
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
'''

vod_stmt = r'''
/**
 * @brief validate output vector with expect vector
 *
 *
 * @param result double *
 * @param expect double *
 * @param len int
 * @return return true if value match
 */
bool Validate_output_data_absolute_error(double *result, double *expect, int len) {
  bool print_all = false;
  const char* print_all_str = getenv("PRINT_ALL");
  if(print_all_str != NULL) {
    printf("Value of PRINT_ALL: %s\n", print_all_str);
    print_all = true;
  }

  const char* absolute_error_str = getenv("ABS_ERROR");
  double absolute_error = 0.0001;
  if(absolute_error_str != NULL) {
    printf("Value of ABS_ERROR: %s\n", absolute_error_str);
    absolute_error = atof(absolute_error_str);
  }
  printf("expect absolute error less than: %f\n", absolute_error);
  int count = 0;
  for(int i = 0; i < len; i++) {
    double result_absolute_error = fabs(result[i] - expect[i]);
    if(print_all) {
      printf("index: %d, result: %f, expect: %f, result absolute error=%f, ", i, result[i], expect[i], result_absolute_error);
      if(result_absolute_error > absolute_error) {
        count++;
        printf("%d failed\n", count);
      } else {
        printf("ok\n");
      }
    } else {
      if(result_absolute_error > absolute_error) {
        printf("index: %d, value: %f != %f, result absolute error=%f\n", i, result[i], expect[i], result_absolute_error);
        return false;
      }
    }
  }
  if(print_all && (count != 0)) {
    return false;
  }
  return true;
}

bool Validate_output_data_relative_error(double *result, double *expect, int len) {
  bool print_all = false;
  const char* print_all_str = getenv("PRINT_ALL");
  if(print_all_str != NULL) {
    printf("Value of PRINT_ALL: %s\n", print_all_str);
    print_all = true;
  }

  const char* relative_error_str = getenv("REL_ERROR");
  double relative_error = 0.001;
  if(relative_error_str != NULL) {
    printf("Value of REL_ERROR: %s\n", relative_error_str);
    relative_error = atof(relative_error_str);
  }
  printf("expect relative error less than: %f\n", relative_error);
  int count = 0;
  for(int i = 0; i < len; i++) {
    double result_relative_error = fabs(result[i] - expect[i])/expect[i];
    if(print_all) {
      printf("index: %d, result: %f, expect: %f, result relative error=%f, ", i, result[i], expect[i], result_relative_error);
      if(result_relative_error > relative_error) {
        count++;
        printf("%d failed\n", count);
      } else {
        printf("ok\n");
      }
    } else {
      if (result_relative_error > relative_error) {
        printf("index: %d, value: %f != %f, result relative error: %f\n", i, result[i], expect[i], result_relative_error);
        return false;
      }
    }
  }
  if(print_all && (count != 0)) {
    return false;
  }
  return true;
}
'''


def get_parser():
    parser = argparse.ArgumentParser(description='generate c program')
    parser.add_argument('--model-path', '-mp', type=str, dest='model_path', required=True, help='Path of onnx file')
    parser.add_argument('--input-all-negone', '-iano', dest='input_all_neg_one', action='store_true', default=False,
                        help='input is all -1, by default is random input value')
    parser.add_argument('--input-all-one', '-iao', dest='input_all_one', action='store_true', required=False,
                        help='input is all 1, by default is random input value')
    parser.add_argument('--input-increment', '-ii', dest='input_increment', action='store_true', required=False,
                        help='input is from 0~n, stride is 1, by default is random input value')
    parser.add_argument('--input-path', '-ip', type=str, dest='input_path', required=False,
                        help='Path of input image, not implemented yet')
    parser.add_argument('--output-path', '-op', type=str, dest='output_path', required=False,
                        help='Path of output file, default is main.c in current path')
    return parser


def write_main_c_program(global_var: str, main_func: str, output_file_path: None):
    assert (len(global_var) != 0)
    assert (len(main_func) != 0)

    if output_file_path is None:
        output_file_path = "main.c"

    with open(output_file_path, "w") as output_file:
        print(comment_info)
        output_file.write(comment_info)
        print(include_header_stmt)
        output_file.write(include_header_stmt)

        print(global_var)
        output_file.write(global_var)

        print(gid_stmt)
        output_file.write(gid_stmt)
        print(vod_stmt)
        output_file.write(vod_stmt)

        print(main_func)
        output_file.write(main_func)


class InputTensor:
    def __init__(self, name: str, n: int, c: int, h: int, w: int, data: list):
        self.name = name
        self.n = n
        self.c = c
        self.h = h
        self.w = w
        self.data = data

    def __len__(self):
        assert (self.n * self.c * self.h * self.w == len(self.data))
        return len(self.data)


def get_input_data(shape: list, input_type: InputType = InputType.RANDOM):
    if input_type == InputType.NEG_ONE:
        input_data = torch.negative(torch.ones(shape))
    elif input_type == InputType.ONE:
        input_data = torch.ones(shape)
    elif input_type == InputType.INCREMENT:
        up_val = 1.0
        for val in shape:
            up_val *= val
        input_data = torch.arange(0.0, up_val).reshape(shape)
    else:
        input_data = torch.distributions.uniform.Uniform(-0.1, 0.1).sample(shape)
    return input_data


def format_input_data_and_get_expected_data(onnx_file_path: str, input_type: InputType = InputType.RANDOM,
                                            input_file_path: str = None):
    # Load onnx file
    onnx_model = onnx.load(onnx_file_path)
    onnx.checker.check_model(onnx_model)

    ort_session = onnxruntime.InferenceSession(onnx_file_path)

    shape_info = [1, 1, 1, 1]
    inputs = []
    input_tensors = []
    if input_file_path is None:
        assert (len(ort_session.get_inputs()) > 0)
        for i in range(len(ort_session.get_inputs())):
            input_shape = ort_session.get_inputs()[i].shape  # list
            input_dim = len(input_shape)
            assert (input_dim <= SHAPE_DIMENSION)
            for j in range(input_dim):
                shape_info[SHAPE_DIMENSION - input_dim + j] = input_shape[j]
            input_data = get_input_data(input_shape, input_type)
            input_name = ort_session.get_inputs()[i].name
            # TODO: here input_data.flatten is a tensor, not a list
            input_tensor = InputTensor(input_name, shape_info[0], shape_info[1], shape_info[2], shape_info[3],
                                       input_data.flatten())
            inputs.append(input_data)
            input_tensors.append(input_tensor)
        assert (len(input_tensors) > 0)
    else:
        # TODO: construct read from file statements
        print("read from input file has not supported yet")

    def to_numpy(tensor):
        return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

    # compute ONNX Runtime output prediction
    ort_inputs = {}
    for i in range(len(ort_session.get_inputs())):
        # print(ort_session.get_inputs()[i].name)
        ort_inputs[ort_session.get_inputs()[i].name] = to_numpy(inputs[i])
    ort_outs = ort_session.run(None, ort_inputs)

    return input_tensors, ort_outs[0].flatten()


def generate_global_var(expected_output: list):
    # global variable
    # expected data init in C
    expected_data_stmt = "\ndouble Expected_data[] = {"
    for i in range(len(expected_output)):
        expected_data_stmt += str(expected_output[i])
        if i != len(expected_output) - 1:
            expected_data_stmt += ", "
    expected_data_stmt += "};\n"
    expected_data_len_stmt = "int Expected_len = %s;\n" % str(len(expected_output))
    return expected_data_stmt + expected_data_len_stmt


def generate_main_func(input_tensor: list):
    input_var_template = "input%d"
    input_data_var_template = "input_data%d"
    input_init_template = "  double %s[]={"
    generate_input_template = "  TENSOR *%s = Generate_input_data(%s, %s, %s, %s, %s);\n"
    printf_template = '  printf("%s");\n'
    print_tensor_template = '  Print_tensor(stdout, %s);\n'
    free_tensor_template = '  Free_tensor(%s);\n'

    prepare_context = "  Prepare_context();\n\n"
    prepare_input_template = '  Prepare_input(%s, "%s");\n'
    return_stmt = "  return 0;"

    main_body = r'''  Run_main_graph();

  double  *result = Handle_output("output");

  Finalize_context();

  bool    res_relative    = Validate_output_data_relative_error(result, Expected_data, Expected_len);
  bool    res_absolute    = Validate_output_data_absolute_error(result, Expected_data, Expected_len);
  free(result);
  if (res_relative || res_absolute) {
    printf("SUCCESS!\n");
  } else {
    printf("FAILED!\n");
  }
'''

    main_method_template = "\nint main(int argc, char* argv[]) {\n%s\n%s\n%s\n}"

    begin_main_body = prepare_context
    end_main_body = return_stmt
    i = 0
    for it_item in input_tensor:
        i += 1
        input_var = input_var_template % i
        input_data_var = input_data_var_template % i
        input_init = input_init_template % input_var
        j = 0
        for ele in it_item.data:
            j += 1
            input_init += str(ele.item())
            if j != len(it_item):
                input_init += ", "
        input_init += "};\n"
        generate_input_stmt = generate_input_template % (input_data_var, it_item.n, it_item.c, it_item.h, it_item.w, input_var)
        print_stmt = printf_template % it_item.name
        print_tensor_stmt = print_tensor_template % input_data_var
        prepare_input_stmt = prepare_input_template % (input_data_var, it_item.name)
        free_tensor_stmt = free_tensor_template % input_data_var

        input_stmts = input_init + generate_input_stmt + print_stmt + print_tensor_stmt + prepare_input_stmt + free_tensor_stmt
        begin_main_body += input_stmts
    main_method = main_method_template % (begin_main_body, main_body, end_main_body)
    return main_method


def main():
    parser = get_parser()

    args = parser.parse_args()
    path = os.path.dirname(os.path.realpath(args.model_path))
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
    if args.input_path is None:
        print("input file path is not provided(not implement yet), will use random input content")
    if args.output_path is None:
        print("output file path is not provided, will output main.c to current path")

    input_type = InputType.RANDOM
    if args.input_all_neg_one:
        input_type = InputType.NEG_ONE
    elif args.input_all_one:
        input_type = InputType.ONE
    elif args.input_increment:
        input_type = InputType.INCREMENT

    formatted_input, expected_output = format_input_data_and_get_expected_data(args.model_path, input_type,
                                                                               args.input_path)
    global_var = generate_global_var(expected_output)
    main_func = generate_main_func(formatted_input)
    write_main_c_program(global_var, main_func, args.output_path)


if __name__ == '__main__':
    main()

