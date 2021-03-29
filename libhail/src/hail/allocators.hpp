#ifndef HAIL_ALLOCATORS_HPP
#define HAIL_ALLOCATORS_HPP 1

#include <cstdlib>
#include <cassert>
#include <cstdint>

#include <vector>

namespace hail {

constexpr inline bool
is_power_of_two(size_t n) {
  return n != 0 && (n & (n - 1)) == 0;
}

inline bool
is_aligned(void *p, size_t align) {
  assert(is_power_of_two(align));
  return (reinterpret_cast<uintptr_t>(p) & (align - 1)) == 0;
}

inline char *
make_aligned(void *p, size_t align) {
  assert(is_power_of_two(align));
  return reinterpret_cast<char *>((reinterpret_cast<uintptr_t>(p) + (align - 1)) & ~(align - 1));
}

inline size_t
make_aligned(size_t i, size_t align) {
  assert(is_power_of_two(align));
  return (i + (align - 1)) & ~(align - 1);
}

/* A standard malloc/free heap allocator. */
class HeapAllocator {
  size_t n_unfreed;

public:
  HeapAllocator()
    : n_unfreed(0) {}
  ~HeapAllocator() {
    assert(n_unfreed == 0);
  }

  void *malloc(size_t size) {
    ++n_unfreed;
    return ::malloc(size);
  }

  void free(void *p) {
    --n_unfreed;
    ::free(p);
  }
};

class RawArenaAllocator {
  static const size_t block_size = 64 * 1024;

  HeapAllocator &heap;
  std::vector<void *> blocks;
  std::vector<void *> chunks;

  char *free_start;
  char *block_end;

public:
  RawArenaAllocator(HeapAllocator &heap);
  ~RawArenaAllocator() { free(); }

  void *allocate_in_new_block(size_t align, size_t size);

  void *allocate_as_chunk(size_t align, size_t size);

  void *allocate_slow(size_t align, size_t size);

  void *allocate(size_t align, size_t size) {
    char *p = make_aligned(free_start, align);
    if (size <= block_end - p) {
      free_start = p + size;
      return p;
    }

    return allocate_slow(align, size);
  }

  void free();
};


/* Arena allocator with high-level `make<T>(...)` object
   constructor */
class ArenaAllocator {
  class DestructorClosure {
    void *p;
    void (*destructor)(void *);
  public:
    DestructorClosure(void *p, void (*destructor)(void *))
      : p(p),
	destructor(destructor) {}

    void invoke() {
      destructor(p);
    }
  };

  std::vector<DestructorClosure> destructors;

public:
  // FIXME public?  pass in?
  RawArenaAllocator raw_arena;

  ArenaAllocator(HeapAllocator &heap)
    : raw_arena(heap) {}
  ~ArenaAllocator() { free(); }

  void *allocate(size_t align, size_t size) {
    return raw_arena.allocate(align, size);
  }

  template<typename T, typename... Args> T *
  make(Args &&... args) {
    void *p = raw_arena.allocate(alignof(T), sizeof(T));
    new (p) T(std::forward<Args>(args)...);
    destructors.emplace_back(p,
			     [](void *p) {
			       static_cast<T *>(p)->~T();
			     });
    return static_cast<T *>(p);
  }

  void free();
};

}

#endif
