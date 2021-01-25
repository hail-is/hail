#ifndef HAIL_ALLOCATORS_HPP
#define HAIL_ALLOCATORS_HPP 1

#include <cstdlib>
#include <cassert>
#include <cstdint>

#include <vector>

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

template<typename A> class ArenaAllocator;

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

/* Arena allocator with high-level `make<T>(...)` object
   constructor */
template<class A>
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

  static const size_t block_size = 8 * 1024;

  A &base;
  std::vector<DestructorClosure> destructors;
  std::vector<void *> blocks;
  std::vector<void *> chunks;

  char *free_start;
  char *block_end;

public:
  ArenaAllocator(A &base)
    : base(base),
      free_start(nullptr),
      block_end(nullptr) {}
   ~ArenaAllocator() { free(); }

  void *allocate_in_new_block(size_t align, size_t size);

  void *allocate_as_chunk(size_t align, size_t size);

  void *allocate(size_t align, size_t size) {
    if (size > block_size / 4)
      return allocate_as_chunk(align, size);

    char *p = make_aligned(free_start, align);
    if (size <= block_end - p) {
      free_start = p + size;
      return p;
    }

    return allocate_in_new_block(align, size);
  }

  void free();

  template<typename T, typename... Args> T *
  make(Args &&... args) {
    void *p = allocate(alignof(T), sizeof(T));
    new (p) T(std::forward<Args>(args)...);
    destructors.emplace_back(p,
			     [](void *p) {
			       static_cast<T *>(p)->~T();
			     });
    return static_cast<T *>(p);
  }
};

template<class A> void
ArenaAllocator<A>::free() {
  for (auto &d : destructors)
    d.invoke();
  destructors.clear();

  for (void *chunk : chunks)
    base.free(chunk);
  chunks.clear();

  for (void *block : blocks)
    base.free(block);
  blocks.clear();
}

template<class A> void *
ArenaAllocator<A>::allocate_in_new_block(size_t align, size_t size) {
  assert(is_power_of_two(align) && align <= 8);
  assert(size <= block_size);

  char *new_block = static_cast<char *>(base.malloc(block_size));
  assert(is_aligned(new_block, align));

  blocks.push_back(new_block);
  free_start = new_block + size;
  block_end = new_block + block_size;

  return new_block;
}

template<class A> void *
ArenaAllocator<A>::allocate_as_chunk(size_t align, size_t size) {
  assert(is_power_of_two(align) && align <= 8);

  void *p = base.malloc(size);
  assert(is_aligned(p, align));

  chunks.push_back(p);
  return p;
}

#endif
