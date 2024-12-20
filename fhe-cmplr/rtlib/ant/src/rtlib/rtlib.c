//-*-c-*-
//=============================================================================
//
// Copyright (c) XXXX-XXXX
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//=============================================================================

#include "ckks/cipher_eval.h"
#include "common/io_api.h"
#include "common/rt_api.h"
#include "rtlib/context.h"
#include "util/ciphertext.h"
#include "util/ckks_decryptor.h"
#include "util/ckks_encoder.h"
#include "util/ckks_encryptor.h"
#include "util/plaintext.h"
#include "util/polynomial.h"

VALUE_LIST* Pre_encode_scheme(TENSOR* image, DATA_SCHEME* scheme) {
  // TODO: post decode from DATA_SCHEME
  size_t      len = TENSOR_SIZE(image);
  VALUE_LIST* res = Alloc_value_list(DCMPLX_TYPE, len);
  size_t      idx = 0;
  FOR_ALL_TENSOR_ELEM(image, n, c, h, w) {
    DCMPLX_VALUE_AT(res, idx) = TENSOR_ELEM(image, n, c, h, w);
    idx++;
  }  IS_TRUE(idx == len, "invalid length of vector");
  return res;
}

double* Post_decode_scheme(VALUE_LIST* vec, DATA_SCHEME* scheme) {
  // TODO: post decode from DATA_SCHEME
  IS_TRUE(LIST_TYPE(vec) == DCMPLX_TYPE, "invalid type");
  double* data = (double*)malloc(LIST_LEN(vec) * sizeof(double));
  FOR_ALL_ELEM(vec, idx) { data[idx] = creal(Get_dcmplx_value_at(vec, idx)); }
  return data;
}

void Prepare_input(TENSOR* input, const char* name) {
  DATA_SCHEME* scheme    = Get_encode_scheme(0);
  VALUE_LIST*  input_vec = Pre_encode_scheme(input, scheme);

  // encode & encrypt
  PLAINTEXT* plain = Alloc_plaintext();
  ENCODE(plain, (CKKS_ENCODER*)Context->_encoder, input_vec);
  CIPHER ciph = Alloc_ciphertext();
  Encrypt_msg(ciph, (CKKS_ENCRYPTOR*)Context->_encryptor, plain);

  Io_set_input(name, 0, ciph);
  Free_value_list(input_vec);
  Free_plaintext(plain);
}

void Prepare_input_large(TENSOR* input, const char* name) {
  size_t C = TENSOR_C(input); // Number of channels
  size_t H = TENSOR_H(input); // Height of the image
  size_t W = TENSOR_W(input); // Width of the image

  size_t slot_capacity = 512;

  printf("C: %d, H: %d, W: %d\n", C, H, W);

  for (size_t c = 0; c < C; ++c) { // Iterate over each channel
    size_t chunk_idx = 0;
    for (size_t h = 0; h < H; h += slot_capacity / W) { // Divide the height
      size_t rows_to_process = (h + slot_capacity / W > H)
                                    ? H - h
                                    : slot_capacity / W;

      size_t chunk_len = rows_to_process * W; // Number of elements in this chunk
      printf("chunk_len: %d\n", chunk_len);

      VALUE_LIST* chunk_vec = Alloc_value_list(DCMPLX_TYPE, chunk_len);

      size_t idx = 0;
      for (size_t rh = 0; rh < rows_to_process; ++rh) {
        for (size_t w = 0; w < W; ++w) {
          DCMPLX_VALUE_AT(chunk_vec, idx++) = TENSOR_ELEM(input, 0, c, h + rh, w);
        }
      }

      char chunk_name[64];
      sprintf(chunk_name, "%s_channel_%zu_chunk_%zu", name, c, chunk_idx);
      printf("input chunk name: %s\n", chunk_name);

      PLAINTEXT* plain = Alloc_plaintext();
      ENCODE(plain, (CKKS_ENCODER*)Context->_encoder, chunk_vec);

      CIPHER ciph = Alloc_ciphertext();
      Encrypt_msg(ciph, (CKKS_ENCRYPTOR*)Context->_encryptor, plain);
      Io_set_input(chunk_name, 0, ciph);
      chunk_idx++;

      Free_value_list(chunk_vec);
      Free_plaintext(plain);
    }
  }
}

double* Handle_output(const char* name) {
  // decrypt & decode
  printf("Get output nam: %s\n", name);
  CIPHER ciph = (CIPHER)Io_get_output(name, 0);
  IS_TRUE(ciph != NULL, "not find data");
  PLAINTEXT* plain = Alloc_plaintext();
  Decrypt(plain, (CKKS_DECRYPTOR*)Context->_decryptor, ciph, NULL);
  VALUE_LIST* res = Alloc_value_list(DCMPLX_TYPE, plain->_slots);
  Decode(res, (CKKS_ENCODER*)Context->_encoder, plain);

  DATA_SCHEME* scheme = Get_decode_scheme(0);
  double*      ret    = Post_decode_scheme(res, scheme);

  Free_ciphertext(ciph);
  Free_plaintext(plain);
  Free_value_list(res);
  return ret;
}

CIPHERTEXT Get_input_data(const char* name, size_t idx) {
  CIPHERTEXT* data = (CIPHERTEXT*)Io_get_input(name, idx);
  IS_TRUE(data != NULL, "not find data");
  CIPHERTEXT ret = *data;
  free(data);  // only free data itself, won't free poly
  return ret;
}

void Set_output_data(const char* name, size_t idx, CIPHER data) {
  printf("Set output nam: %s\n", name);
  CIPHER output = Alloc_ciphertext();
  Copy_ciph(output, data);
  Free_ciph_poly(data, 1);
  Io_set_output(name, idx, output);
}
