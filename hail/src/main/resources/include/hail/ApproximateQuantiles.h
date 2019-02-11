#include <algorithm>
#include <array>
#include <random>
#include <vector>
#include <iostream>

template<int log2_buffer_size>
class ApproximateQuantiles {
  static constexpr int buffer_size = 1 << log2_buffer_size;
  // std::random_device rd;
  std::mt19937 gen{};
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
    std::cout << "buffer = [";
    for (auto x : buffers[index]) {
      std::cout << x << " ";
    }
    std::cout << "]" << std::endl;
    std::sort(std::begin(buffers[index]),
              std::begin(buffers[index]) + ends[index]);
    std::cout << "sorted buffer = [";
    for (auto x : buffers[index]) {
      std::cout << x << " ";
    }
    std::cout << "]" << std::endl;
  }

  void compact(int current) {
    int next = current + 1;
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
    ends[current] = 0;
    ends[next] = next_end + ends[current] / 2;
  }
public:
  ApproximateQuantiles() {}
  void accept(int x) {
    buffers[0][ends[0]] = x;
    ++ends[0];
    check_compact(0);
  }
  void finalize() {
    for (int i = 0; i < buffers.size() - 1; ++i) {
      compact(i);
    }
    sort_buffer(buffers.size());
  }
  void write() {
    for (auto &buffer : buffers) {
      std::cout << "buffer = [";
      for (auto x : buffer) {
        std::cout << x << " ";
      }
      std::cout << "]" << std::endl;
    }
  }
  int rank(int element) {
    auto summary = buffers.back();
    auto lower_bound = std::lower_bound(begin(summary),
                                       begin(summary) + ends.back(),
                                       element);
    return lower_bound - std::begin(summary);
  }
};
