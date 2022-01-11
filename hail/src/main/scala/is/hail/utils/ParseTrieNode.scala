package is.hail.utils

import scala.collection.mutable.ArrayBuffer

/**
  * A character trie used for parsing membership in a set literal from a character sequence.
  *
  * The children of a node are represented in an unordered array; linear search is used to
  * traverse the tree with the assumption that the average number of children per node is
  * small (small enough that a O(n log n) binary search or O(1) hash lookup would be more
  * expensive)
  */
class ParseTrieNode(val prev: Char, var hasEnd: Boolean, var children: ArrayBuffer[ParseTrieNode]) {
  def search(next: Char): ParseTrieNode = {
    var i = 0
    while (i < children.size) {
      val child = children(i)
      if (child.prev == next)
        return child
      i += 1
    }
    null
  }

}

object ParseTrieNode {
  def generate(data: Array[String]): ParseTrieNode = {
    val root = new ParseTrieNode(null.asInstanceOf[Char], false, new ArrayBuffer[ParseTrieNode])

    def insert(s: String): Unit = {
      var idx = 0
      var node = root
      while (idx < s.length) {
        val next = s(idx)
        val buff = node.children
        var continue = true
        var i = 0
        while (continue) {
          if (i >= buff.size) {
            node = new ParseTrieNode(next, false, new ArrayBuffer[ParseTrieNode])
            buff += node
            continue = false
          } else {
            val child = buff(i)
            if (child.prev == next) {
              node = child
              continue = false
            }
          }
          i += 1
        }

        idx += 1
      }

      node.hasEnd = true
    }

    data.foreach(insert)

    root
  }
}