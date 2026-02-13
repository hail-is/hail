package is.hail.utils

class UnionFind(initialCapacity: Int = 32) {
  private var a: Array[Int] = new Array[Int](initialCapacity)
  private var rank: Array[Int] = new Array[Int](initialCapacity)
  private var count: Int = 0

  def size: Int = count

  private def ensure(i: Int): Unit = {
    if (i >= a.length) {
      var newLength = a.length << 1
      while (i >= newLength)
        newLength = newLength << 1
      val a2 = new Array[Int](newLength)
      Array.copy(a, 0, a2, 0, a.length)
      a = a2
      val rank2 = new Array[Int](newLength)
      Array.copy(rank, 0, rank2, 0, rank.length)
      rank = rank2
    }
  }

  def makeSet(i: Int): Unit = {
    ensure(i)
    a(i) = i
    count += 1
  }

  def find(x: Int): Int = {
    require(x < a.length)
    var representative = x
    while (representative != a(representative))
      representative = a(representative)
    var current = x
    while (representative != current) {
      val temp = a(current)
      a(current) = representative
      current = temp
    }
    current
  }

  def union(x: Int, y: Int): Unit = {
    val xroot = find(x)
    val yroot = find(y)

    if (xroot != yroot) {
      count -= 1
      if (rank(xroot) < rank(yroot)) {
        a(xroot) = yroot
      } else if (rank(xroot) > rank(yroot)) {
        a(yroot) = xroot
      } else {
        a(xroot) = yroot
        rank(yroot) += 1
      }
    }
  }

  def sameSet(x: Int, y: Int): Boolean = {
    require(x < a.length && y < a.length)
    find(x) == find(y)
  }
}
