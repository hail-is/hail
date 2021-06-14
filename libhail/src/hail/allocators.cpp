#include "hail/allocators.hpp"

namespace hail {

RawArenaAllocator::RawArenaAllocator(HeapAllocator &heap)
  : heap(heap),
    free_start(nullptr),
    block_end(nullptr) {}

void
RawArenaAllocator::free() {
  for (void *chunk : chunks)
    heap.free(chunk);
  chunks.clear();

  for (void *block : blocks)
    heap.free(block);
  blocks.clear();

  free_start = nullptr;
  block_end = nullptr;
}

void *
RawArenaAllocator::allocate_in_new_block(size_t align, size_t size) {
  assert(is_power_of_two(align) && align <= 8);
  assert(size <= block_size);

  char *new_block = (char *)heap.malloc(block_size);
  blocks.push_back(new_block);
  free_start = new_block + size;
  block_end = new_block + block_size;

  return new_block;
}

void *
RawArenaAllocator::allocate_as_chunk(size_t align, size_t size) {
  assert(is_power_of_two(align) && align <= 8);

  void *p = heap.malloc(size);
  assert(is_aligned(p, align));

  chunks.push_back(p);
  return p;
}

void *
RawArenaAllocator::allocate_slow(size_t align, size_t size) {
  if (size > block_size / 4)
    return allocate_as_chunk(align, size);

  return allocate_in_new_block(align, size);
}

void
ArenaAllocator::free() {
  for (auto &d : destructors)
    d.invoke();
  destructors.clear();

  raw_arena.free();
}

}
