#include <hail/allocators.hpp>

namespace hail {

void
ArenaAllocator::free() {
  for (auto &d : destructors)
    d.invoke();
  destructors.clear();

  for (void *chunk : chunks)
    heap.free(chunk);
  chunks.clear();

  for (void *block : blocks)
    heap.free(block);
  blocks.clear();
}

void *
ArenaAllocator::allocate_in_new_block(size_t align, size_t size) {
  assert(is_power_of_two(align) && align <= 8);
  assert(size <= block_size);

  char *new_block = static_cast<char *>(heap.malloc(block_size));
  assert(is_aligned(new_block, align));

  blocks.push_back(new_block);
  free_start = new_block + size;
  block_end = new_block + block_size;

  return new_block;
}

void *
ArenaAllocator::allocate_as_chunk(size_t align, size_t size) {
  assert(is_power_of_two(align) && align <= 8);

  void *p = heap.malloc(size);
  assert(is_aligned(p, align));

  chunks.push_back(p);
  return p;
}

}
