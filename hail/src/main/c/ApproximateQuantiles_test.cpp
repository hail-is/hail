#include "catch.hpp"
#include "hail/ApproximateQuantiles.h"
#include <vector>
#include <ctime>
#include <numeric>
#include <iterator>
#include <iostream>
#include <random>

TEST_CASE("quantiles of small vector are correct") {
  ApproximateQuantiles<4> aq{};
  std::vector<int> values{0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20};

  SECTION("no segfaults") {
    for (auto x : values) {
      aq.accept(x);
    }
    aq.finalize();
    aq.write();
    for (auto x : values) {
      std::cout << "the rank of " << x << " is " << aq.rank(x) << "\n";
    }
  }
}

