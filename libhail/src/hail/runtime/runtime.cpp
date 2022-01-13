#include "hail/allocators.hpp"
#include "hail/runtime/runtime.hpp"

using namespace hail;

char *
hl_runtime_region_allocate(char *region, size_t align, size_t size) {
  RawArenaAllocator &raw_arena = *reinterpret_cast<RawArenaAllocator *>(region);
  return (char *)raw_arena.allocate(align, size);
}
