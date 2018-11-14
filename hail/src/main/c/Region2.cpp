#include "hail/Region2.h"
#include "hail/Upcalls.h"
#include <memory>
#include <vector>
#include <utility>

namespace hail {

RegionPool::Region::Region(RegionPool * pool) :
pool_(pool),
block_offset_(0),
current_block_(pool_->get_block()),
used_blocks_(std::vector<std::shared_ptr<RegionPool::Block>>()),
parents_(std::vector<std::shared_ptr<Region2>>()) { }

char * Region2::allocate_new_block() {
  used_blocks_.push_back(std::move(current_block_));
  current_block_ = pool_->get_block();
  block_offset_ = 0;
  return current_block_->buf_;
}

char * Region2::allocate_big_chunk(ssize_t n) {
  auto chunk = pool_->get_sized_block(n);
  used_blocks_.push_back(chunk);
  return chunk->buf_;
}

void Region2::clear() {
  block_offset_ = 0;
  used_blocks_.clear();
  parents_.clear();
}

std::shared_ptr<Region2> Region2::get_region() {
  return pool_->get_region();
}

void Region2::add_reference_to(std::shared_ptr<Region2> region) {
  parents_.push_back(region);
}

void RegionPool::RegionDeleter::operator()(Region2* p) const {
  p->clear();
  pool_->free_regions_.push_back(std::move(p));
}

void RegionPool::BlockDeleter::operator()(Block* p) const {
  pool_->free_blocks_.push_back(std::move(p));
}

void RegionPool::SizedBlockDeleter::operator()(Block* p) const {
  auto b = pool_->free_sized_blocks_.cbegin();
  auto end = pool_->free_sized_blocks_.cend();
  while ((b != end) && ((*b)->size_ <= p->size_)) {
    ++b;
  }
  pool_->free_sized_blocks_.insert(b, std::move(p));
}

RegionPool::RegionPool() :
free_regions_(std::vector<Region2 *>()),
free_blocks_(std::vector<Block *>()),
free_sized_blocks_(std::vector<Block *>()),
del_(RegionDeleter(this)),
block_del_(BlockDeleter(this)),
sized_block_del_(SizedBlockDeleter(this)) { }

RegionPool::~RegionPool() {
  for (Region2 * region : free_regions_) {
    delete region;
  }
  for (Block * block : free_blocks_) {
    delete block;
  }
  for (Block * block : free_sized_blocks_) {
    delete block;
  }
}

std::shared_ptr<RegionPool::Block> RegionPool::new_block() {
  return std::shared_ptr<RegionPool::Block>(new Block(block_size), block_del_);
}

std::shared_ptr<RegionPool::Block> RegionPool::new_sized_block(ssize_t size) {
  return std::shared_ptr<RegionPool::Block>(new Block(size), sized_block_del_);
}

std::shared_ptr<Region2> RegionPool::new_region() {
  return std::shared_ptr<Region2>(new Region2(this), del_);
}

std::shared_ptr<RegionPool::Block> RegionPool::get_block() {
  if (free_blocks_.empty()) {
    return new_block();
  }
  RegionPool::Block * block = std::move(free_blocks_.back());
  free_blocks_.pop_back();
  return std::shared_ptr<Block>(block, block_del_);
}

std::shared_ptr<RegionPool::Block> RegionPool::get_sized_block(ssize_t size) {
  auto b = free_sized_blocks_.cbegin();
  auto end = free_sized_blocks_.cend();
  while ((b != end) && ((*b)->size_ < size)) {
    ++b;
  }
  if (b == end) {
    return new_sized_block(size);
  }
  RegionPool::Block * block = std::move(*b);
  free_sized_blocks_.erase(b);
  return std::shared_ptr<Block>(block, sized_block_del_);
}

std::shared_ptr<Region2> RegionPool::get_region() {
  if (free_regions_.empty()) {
    return new_region();
  }
  Region2 * region = std::move(free_regions_.back());
  free_regions_.pop_back();
  return std::shared_ptr<Region2>(region, del_);
}

}