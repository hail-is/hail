#include <algorithm>
#include <array>
#include <random>
#include <vector>
#include <iostream>

template<int log2_buffer_size>
class ApproximateQuantiles {
  static constexpr int buffer_size = 1 << log2_buffer_size;
  std::random_device rd{};
  std::mt19937 gen{rd()};
  std::uniform_int_distribution<int> coin_dist{0, 1};
  int coin() { return coin_dist(gen); }
  std::vector<int> ends{0};
  std::vector<std::array<int, buffer_size>> buffers{1};

  void check_compact(int current) {
    if (ends[current] >= buffer_size) {
      compact(current);
      check_compact(current + 1);
    }
  }

  void sort_buffer(int index) {
    std::sort(std::begin(buffers[index]),
              std::begin(buffers[index]) + ends[index]);
  }

  void compact(int current) {
    size_t next = current + 1;
    if (next == buffers.size()) {
      buffers.emplace_back();
      ends.emplace_back();
    }
    auto next_end = ends[next];
    sort_buffer(current);
    for (int i = coin();
         i < ends[current];
         i += 2) {
      buffers[next][next_end + i/2] = buffers[current][i];
    }
    ends[next] = next_end + ends[current] / 2;
    ends[current] = 0;
  }
public:
  ApproximateQuantiles() {}
  void accept(int x) {
    buffers[0][ends[0]] = x;
    ++ends[0];
    check_compact(0);
  }
  void finalize() {
    for (size_t i = 0; i < buffers.size() - 1; ++i) {
      compact(i);
    }
    sort_buffer(buffers.size() - 1);
  }
  void write() {
    for (size_t i = 0; i < buffers.size() - 1; ++i) {
      auto &buffer = buffers[i];
      std::cout << "buffer " << i << " = [";
      for (auto x : buffer) {
        std::cout << x << " ";
      }
      std::cout << "] "  << ends[i] << std::endl;
    }
  }
  int rank(int element) {
    auto summary = buffers.back();
    auto lower_bound = std::lower_bound(begin(summary),
                                        begin(summary) + ends.back(),
                                        element);
    int buffer_weight = (1 << (buffers.size() - 1));
    return (lower_bound - std::begin(summary)) * buffer_weight;
  }
};
