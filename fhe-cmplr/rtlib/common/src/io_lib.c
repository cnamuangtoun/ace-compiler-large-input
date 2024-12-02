//-*-c-*-
//=============================================================================
//
// Copyright (c) XXXX-XXXX
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//=============================================================================

#include <stdlib.h>
#include <string.h>

#include "common/error.h"
#include "common/io_api.h"
#include "common/rt_api.h"

typedef struct {
  const char* _name;
  int         _count;
  void*       _ct[];
} IO_DATA;

typedef struct MetadataEntry {
    char* name;                  
    char* data;                  
    struct MetadataEntry* next;  
} MetadataEntry;

static IO_DATA** Input_data;
static IO_DATA** Output_data;
#pragma omp threadprivate(Input_data)
#pragma omp threadprivate(Output_data)

static MetadataEntry* metadata_head = NULL;

void Io_set_metadata(const char* name, const char* data) {
    // Create a new metadata entry
    MetadataEntry* entry = (MetadataEntry*)malloc(sizeof(MetadataEntry));
    if (!entry) {
        fprintf(stderr, "Memory allocation failed in Io_set_metadata\n");
        exit(EXIT_FAILURE);
    }
    entry->name = strdup(name);
    entry->data = strdup(data);
    entry->next = metadata_head;
    metadata_head = entry;
}

char* Io_get_metadata(const char* name) {
  MetadataEntry* current = metadata_head;
  MetadataEntry* previous = NULL;

  while (current != NULL) {
    if (strcmp(current->name, name) == 0) {
    // Found the entry
    char* data = current->data;

    // Remove the entry from the list
    if (previous == NULL) {
      // Entry is at the head
      metadata_head = current->next;
    } else {
      previous->next = current->next;
    }

    // Free the entry structure but not the data
    free(current->name);
    free(current);

    return data; // Caller is responsible for freeing 'data'
  }
    previous = current;
    current = current->next;
  }
  return NULL; // Not found
}

static void Io_set_data(IO_DATA** data, const char* name, size_t idx,
                        void* ct) {
  printf("set data: %s \n", name);
  while (*data != NULL) {
    if (strcmp((*data)->_name, name) == 0) {
      FMT_ASSERT(idx < (*data)->_count, "index out of bounds\n");
      (*data)->_ct[idx] = ct;
      return;
    }
    ++data;
  }
  FMT_ASSERT(false, "fail to find %s.\n", name);
}

static void* Io_get_data(IO_DATA** data, const char* name, size_t idx) {
  printf("get data: %s \n", name);
  while (*data != NULL) {
    if (strcmp((*data)->_name, name) == 0) {
      FMT_ASSERT(idx < (*data)->_count, "index out of bounds\n");
      return (*data)->_ct[idx];
    }
    ++data;
  }
  FMT_ASSERT(false, "fail to find %s.\n", name);
  return NULL;
}

static void* Io_create_data(const char* name, size_t count) {
  size_t   sz   = sizeof(IO_DATA) + count * sizeof(void*);
  IO_DATA* data = (IO_DATA*)malloc(sz);
  data->_name   = name;
  data->_count  = count;
  memset(data->_ct, 0, count * sizeof(void*));
  return data;
}

void Io_init() {
  IS_TRUE(Input_data == NULL && Output_data == NULL, "invalid data pointer");
  int isize = Get_input_count();
  IS_TRUE(isize > 0, "invalid input count");
  Input_data = (IO_DATA**)malloc((isize + 1) * sizeof(IO_DATA*));
  for (int i = 0; i < isize; ++i) {
    DATA_SCHEME* scheme = Get_encode_scheme(i);
    Input_data[i]       = Io_create_data(scheme->_name, scheme->_count);
  }
  Input_data[isize] = NULL;

  int osize = Get_output_count();
  IS_TRUE(osize > 0, "invalid output count");
  Output_data = (IO_DATA**)malloc((osize + 1) * sizeof(IO_DATA*));
  for (int i = 0; i < osize; ++i) {
    DATA_SCHEME* scheme = Get_decode_scheme(i);
    Output_data[i]      = Io_create_data(scheme->_name, scheme->_count);
  }
  Output_data[osize] = NULL;
}

void Io_fini() {
  int isize = Get_input_count();
  for (int i = 0; i < isize; ++i) {
    free(Input_data[i]);
  }
  free(Input_data);
  Input_data = NULL;

  int osize = Get_output_count();
  for (int i = 0; i < osize; ++i) {
    free(Output_data[i]);
  }
  free(Output_data);
  Output_data = NULL;
}

void Io_set_input(const char* name, size_t idx, void* ct) {
  if (Input_data == NULL) {
    Io_init();
  }
  Io_set_data(Input_data, name, idx, ct);
}

void* Io_get_input(const char* name, size_t idx) {
  return Io_get_data(Input_data, name, idx);
}

void Io_set_output(const char* name, size_t idx, void* ct) {
  if (Output_data == NULL) {
    Io_init();
  }
  Io_set_data(Output_data, name, idx, ct);
}

void* Io_get_output(const char* name, size_t idx) {
  return Io_get_data(Output_data, name, idx);
}
