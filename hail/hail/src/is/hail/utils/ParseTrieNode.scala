package is.hail.utils

import scala.collection.mutable.ArrayBuffer

/** A character trie used for parsing membership in a set literal from a character sequence.
  *
  * The children of a node are represented in an unordered array; linear search is used to traverse
  * the tree with the assumption that the average number of children per node is small (small enough
  * that a O(n log n) binary search or O(1) hash lookup would be more expensive)
  */

class ParseTrieNode(
  val value: String,
  val children: Array[ParseTrieNode],
  val nextChar: Array[Char],
) {

  def search(next: Char): ParseTrieNode = {
    var i = 0
    while (i < children.length) {
      if (nextChar(i) == next)
        return children(i)
      i += 1
    }
    null
  }
}

object ParseTrieNode {

  def generate(data: Array[String]): ParseTrieNode = {
    class ParseTrieNodeBuilder(
      var value: String,
      var children: ArrayBuffer[ParseTrieNodeBuilder],
      var nextChar: ArrayBuffer[Char],
    ) {

      def result(): ParseTrieNode =
        new ParseTrieNode(value, children.toArray.map(_.result()), nextChar.toArray)
    }

    val root =
      new ParseTrieNodeBuilder(null, new ArrayBuffer[ParseTrieNodeBuilder], new ArrayBuffer[Char])

    def insert(s: String): Unit = {
      var idx = 0
      var node = root
      while (idx < s.length) {
        val next = s(idx)
        val buff = node.children
        val charBuff = node.nextChar
        var continue = true
        var i = 0
        while (continue) {
          if (i >= charBuff.size) {
            node.nextChar += next
            node = new ParseTrieNodeBuilder(
              null,
              new ArrayBuffer[ParseTrieNodeBuilder],
              new ArrayBuffer[Char],
            )
            buff += node
            continue = false
          } else {
            val child = buff(i)
            if (charBuff(i) == next) {
              node = child
              continue = false
            }
          }
          i += 1
        }

        idx += 1
      }

      node.value = s
    }

    data.foreach(insert)

    root.result()
  }
}
