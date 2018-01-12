package is.hail.utils

object BinarySearch {
  // return smallest elem such that key <= elem
  def binarySearch(length: Int,
    // key.compare(elem)
    compare: (Int) => Int): Int = {
    assert(length > 0)

    var low = 0
    var high = length - 1
    while (low < high) {
      val mid = (low + high) / 2
      assert(mid >= low && mid < high)

      // key <= elem
      if (compare(mid) <= 0) {
        high = mid
      } else {
        low = mid + 1
      }
    }
    assert(low == high)
    assert(low >= 0 && low < length)

    // key <= low
    assert(compare(low) <= 0 || low == length - 1)
    // low == 0 || (low - 1) > key
    assert(low == 0
      || compare(low - 1) > 0)

    low
  }
}