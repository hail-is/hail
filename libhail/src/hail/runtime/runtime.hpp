#ifndef HAIL_RUNTIME_RUNTIME_HPP_INCLUDED
#define HAIL_RUNTIME_RUNTIME_HPP_INCLUDED 1

extern "C" {

char *hl_runtime_region_allocate(char *region, size_t align, size_t size);
void hl_runtime_print_float64(double d);
void hl_runtime_print_bool(bool d);


}

#endif
