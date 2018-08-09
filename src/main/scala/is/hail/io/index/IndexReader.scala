package is.hail.io.index

import java.util
import java.util.Map.Entry

import is.hail.annotations._
import is.hail.expr.Parser
import is.hail.io._
import is.hail.utils._
import org.apache.hadoop.conf.Configuration
import org.apache.hadoop.fs.FSDataInputStream
import org.apache.spark.sql.Row
import org.json4s.Formats
import org.json4s.jackson.JsonMethods

object IndexReader {
  def readMetadata(hConf: Configuration, metadataFile: String): IndexMetadata = {
    val jv = hConf.readFile(metadataFile) { in => JsonMethods.parse(in) }
    implicit val formats: Formats = defaultJSONFormats
    jv.extract[IndexMetadata]
  }
}

class IndexReader(hConf: Configuration, path: String, cacheCapacity: Int = 8) extends AutoCloseable {
  private val is = hConf.unsafeReader(path + "/index").asInstanceOf[FSDataInputStream]
  private val codecSpec = CodecSpec.default

  private val metadata = IndexReader.readMetadata(hConf, path + "/metadata.json.gz")
  val branchingFactor = metadata.branchingFactor
  val height = metadata.height
  val nKeys = metadata.nKeys
  val attributes = metadata.attributes

  val keyType = Parser.parseType(metadata.keyType)
  val ordering = keyType.ordering
  val leafType = LeafNodeBuilder.typ(keyType)
  val internalType = InternalNodeBuilder.typ(keyType)

  private val leafDecoder = codecSpec.buildDecoder(leafType, leafType)(is)
  private val internalDecoder = codecSpec.buildDecoder(internalType, internalType)(is)

  private val region = new Region()
  private val rv = RegionValue(region)

  var cacheHits = 0L
  var cacheMisses = 0L

  @transient private[this] lazy val cache = new util.LinkedHashMap[Long, IndexNode](cacheCapacity, 0.75f, true) {
    override def removeEldestEntry(eldest: Entry[Long, IndexNode]): Boolean = size() > cacheCapacity
  }

  private def readInternalNode(offset: Long): InternalNode = {
    if (cache.containsKey(offset)) {
      cacheHits += 1
      cache.get(offset).asInstanceOf[InternalNode]
    } else {
      cacheMisses += 1
      is.seek(offset)
      assert(internalDecoder.readByte() == 1)
      rv.setOffset(internalDecoder.readRegionValue(region))
      val node = InternalNode(SafeRow(internalType, rv))
      cache.put(offset, node)
      region.clear()
      node
    }
  }

  private def readLeafNode(offset: Long): LeafNode = {
    if (cache.containsKey(offset)) {
      cacheHits += 1
      cache.get(offset).asInstanceOf[LeafNode]
    } else {
      cacheMisses += 1
      is.seek(offset)
      assert(leafDecoder.readByte() == 0)
      rv.setOffset(leafDecoder.readRegionValue(region))
      val node = LeafNode(SafeRow(leafType, rv))
      cache.put(offset, node)
      region.clear()
      node
    }
  }

  private def binarySearchLeftmost(n: Int, key: Annotation, f: (Int) => Annotation): Int = {
    var left = 0
    var right = n
    while (left < right) {
      val mid = left + (right - left) / 2
      if (ordering.lt(f(mid), key))
        left = mid + 1
      else
        right = mid
    }
    left // if left < n and A[left] == key, leftmost element that equals key. Otherwise, left is the insertion point of key in A.
  }

  private def binarySearchRightmost(n: Int, key: Annotation, f: (Int) => Annotation): Int = {
    var left = 0
    var right = n
    while (left < right) {
      val mid = left + (right - left) / 2
      if (ordering.gt(f(mid), key))
        right = mid
      else
        left = mid + 1
    }
    left // if left > 0 and A[left - 1] == key, rightmost element that equals key. Otherwise, left is the insertion point of key in A.
  }

  private def queryByKeyAllMatches(key: Annotation, level: Int, offset: Long, ab: ArrayBuilder[LeafChild]) {
    if (level == 0) {
      val node = readLeafNode(offset)
      val n = node.children.length
      var idx = binarySearchLeftmost(node.children.length, key, { i => node.children(i).key })
      while (idx < n && ordering.equiv(node.children(idx).key, key)) {
        ab += node.children(idx)
        idx += 1
      }
    } else {
      val node = readInternalNode(offset)
      val n = node.children.length

      var idx = binarySearchLeftmost(node.children.length, key, { i => node.children(i).lastKey })
      while (idx < n && {
        val child = node.children(idx)
        ordering.gteq(key, child.firstKey) && ordering.lteq(key, child.lastKey)
      }) {
        queryByKeyAllMatches(key, level - 1, node.children(idx).childOffset, ab)
        idx += 1
      }
    }
  }

  private def queryByKeyGtEq(key: Annotation, level: Int, offset: Long): Option[LeafChild] = {
    if (level == 0) {
      val node = readLeafNode(offset)
      val n = node.children.length
      val idx = binarySearchRightmost(node.children.length, key, { i => node.children(i).key })
      if (idx > 0 && ordering.equiv(node.children(idx - 1).key, key))
        Some(node.children(idx - 1))
      else if (idx == n)
        None
      else
        Some(node.children(idx))
    } else {
      val node = readInternalNode(offset)
      val n = node.children.length
      val idx = binarySearchRightmost(node.children.length, key, { i => node.children(i).firstKey })
      if (idx > 0 && ordering.lteq(key, node.children(idx - 1).lastKey))
        queryByKeyGtEq(key, level - 1, node.children(idx - 1).childOffset)
      else if (idx == n)
        None
      else
        queryByKeyGtEq(key, level - 1, node.children(idx).childOffset)
    }
  }

  private def queryByKeyGt(key: Annotation, level: Int, offset: Long): Option[LeafChild] = {
    if (level == 0) {
      val node = readLeafNode(offset)
      val n = node.children.length
      val idx = binarySearchRightmost(node.children.length, key, { i => node.children(i).key })
      if (idx == n)
        None
      else
        Some(node.children(idx))
    } else {
      val node = readInternalNode(offset)
      val n = node.children.length
      val idx = binarySearchRightmost(node.children.length, key, { i => node.children(i).firstKey })
      if (idx > 0 && ordering.lt(key, node.children(idx - 1).lastKey))
        queryByKeyGt(key, level - 1, node.children(idx - 1).childOffset)
      else if (idx == n)
        None
      else
        queryByKeyGt(key, level - 1, node.children(idx).childOffset)
    }
  }

  private def queryByKeyLtEq(key: Annotation, level: Int, offset: Long): Option[LeafChild] = {
    if (level == 0) {
      val node = readLeafNode(offset)
      val n = node.children.length
      val idx = binarySearchLeftmost(node.children.length, key, { i => node.children(i).key })
      if (idx < n && ordering.equiv(node.children(idx).key, key))
        Some(node.children(idx))
      else if (idx == 0)
        None
      else
        Some(node.children(idx - 1))
    } else {
      val node = readInternalNode(offset)
      val n = node.children.length
      val idx = binarySearchLeftmost(node.children.length, key, { i => node.children(i).lastKey })
      if (idx < n && ordering.gteq(key, node.children(0).firstKey))
        queryByKeyLtEq(key, level - 1, node.children(idx).childOffset)
      else if (idx == 0)
        None
      else
        queryByKeyLtEq(key, level - 1, node.children(idx - 1).childOffset)
    }
  }

  private def queryByKeyLt(key: Annotation, level: Int, offset: Long): Option[LeafChild] = {
    if (level == 0) {
      val node = readLeafNode(offset)
      val n = node.children.length
      val idx = binarySearchLeftmost(node.children.length, key, { i => node.children(i).key })
      if (idx == 0)
        None
      else
        Some(node.children(idx - 1))
    } else {
      val node = readInternalNode(offset)
      val n = node.children.length
      val idx = binarySearchLeftmost(node.children.length, key, { i => node.children(i).lastKey })
      if (idx < n && ordering.gt(key, node.children(idx).firstKey))
        queryByKeyLt(key, level - 1, node.children(idx).childOffset)
      else if (idx == 0)
        None
      else
        queryByKeyLt(key, level - 1, node.children(idx - 1).childOffset)
    }
  }

  def queryByKeyAllMatches(key: Annotation): Array[LeafChild] = {
    val ab = new ArrayBuilder[LeafChild]()
    if (nKeys != 0)
      queryByKeyAllMatches(key, height - 1, metadata.rootOffset, ab)
    ab.result()
  }

  def queryByKeyAllMatchesOffsets(keys: Array[Annotation]): Array[Long] = {
    val ab = new ArrayBuilder[Long]()
    val lcab = new ArrayBuilder[LeafChild]()
    var lastKey: Annotation = null

    if (nKeys != 0) {
      keys.sortWith({ (a1, a2) => ordering.lt(a1, a2) }).foreach { k =>
        if (k != lastKey) {
          lcab.clear()
          queryByKeyAllMatches(k, height - 1, metadata.rootOffset, lcab)
          var i = 0
          while (i < lcab.size) {
            ab += lcab(i).offset
            i += 1
          }
          lastKey = k
        }
      }
    }
    ab.result()
  }

  def queryByKey(key: Annotation, greater: Boolean = true, closed: Boolean = true): Option[LeafChild] = {
    if (nKeys == 0)
      return None

    if (greater && closed)
      queryByKeyGtEq(key, height - 1, metadata.rootOffset)
    else if (greater && !closed)
      queryByKeyGt(key, height - 1, metadata.rootOffset)
    else if (!greater && closed)
      queryByKeyLtEq(key, height - 1, metadata.rootOffset)
    else
      queryByKeyLt(key, height - 1, metadata.rootOffset)
  }

  private def queryByIndex(idx: Long, level: Int, offset: Long): LeafChild = {
    if (level == 0) {
      val node = readLeafNode(offset)
      val localIdx = idx - node.firstKeyIndex
      node.children(localIdx.toInt)
    } else {
      val node = readInternalNode(offset)
      val blockStartIdx = node.blockFirstKeyIndex
      val nKeysPerBlock = math.pow(branchingFactor, level).toLong
      val localIdx = (idx - blockStartIdx) / nKeysPerBlock
      queryByIndex(idx, level - 1, node.children(localIdx.toInt).childOffset)
    }
  }

  def queryByIndex(idx: Long): LeafChild = {
    require(idx >= 0 && idx < nKeys)
    queryByIndex(idx, height - 1, metadata.rootOffset)
  }

  def close() {
    region.close()
    leafDecoder.close()
    internalDecoder.close()
    log.info(s"Index reader cache hit rate: ${ cacheHits.toDouble / (cacheHits + cacheMisses) }")
  }
}

final case class InternalChild(childOffset: Long, firstKey: Annotation, firstKeyOffset: Long, lastKey: Annotation)

object InternalNode {
  def apply(r: Row): InternalNode = {
    val blockFirstKeyIndex = r.getLong(0)
    val children = r.get(1).asInstanceOf[IndexedSeq[Row]].map(r => InternalChild(r.getLong(0), r.get(1), r.getLong(2), r.get(3)))
    InternalNode(blockFirstKeyIndex, children)
  }
}

final case class InternalNode(blockFirstKeyIndex: Long, children: IndexedSeq[InternalChild]) extends IndexNode

final case class LeafChild(key: Annotation, offset: Long)

object LeafNode {
  def apply(r: Row): LeafNode = {
    val firstKeyIndex = r.getLong(0)
    val keys = r.get(1).asInstanceOf[IndexedSeq[Row]].map(r => LeafChild(r.get(0), r.getLong(1)))
    LeafNode(firstKeyIndex, keys)
  }
}

final case class LeafNode(firstKeyIndex: Long, children: IndexedSeq[LeafChild]) extends IndexNode

sealed trait IndexNode
