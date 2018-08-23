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
  private[io] val metadata = IndexReader.readMetadata(hConf, path + "/metadata.json.gz")
  val branchingFactor = metadata.branchingFactor
  val height = metadata.height
  val nKeys = metadata.nKeys
  val attributes = metadata.attributes
  val indexRelativePath = metadata.indexPath

  val version = SemanticVersion(metadata.fileVersion)
  val keyType = Parser.parseType(metadata.keyType)
  val annotationType = Parser.parseType(metadata.annotationType)
  val leafType = LeafNodeBuilder.typ(keyType, annotationType)
  val internalType = InternalNodeBuilder.typ(keyType, annotationType)
  val ordering = keyType.ordering

  private val is = hConf.unsafeReader(path + "/" + indexRelativePath).asInstanceOf[FSDataInputStream]
  private val codecSpec = CodecSpec.default
  
  private val leafDecoder = codecSpec.buildDecoder(leafType, leafType)(is)
  private val internalDecoder = codecSpec.buildDecoder(internalType, internalType)(is)

  private val region = new Region()
  private val rv = RegionValue(region)

  private var cacheHits = 0L
  private var cacheMisses = 0L

  @transient private[this] lazy val cache = new util.LinkedHashMap[Long, IndexNode](cacheCapacity, 0.75f, true) {
    override def removeEldestEntry(eldest: Entry[Long, IndexNode]): Boolean = size() > cacheCapacity
  }

  private[io] def readInternalNode(offset: Long): InternalNode = {
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

  private[io] def readLeafNode(offset: Long): LeafNode = {
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

  private def findStartIndex(key: Annotation, level: Int, offset: Long): Long = {
    if (level == 0) {
      val node = readLeafNode(offset)
      val idx = binarySearchLeftmost(node.children.length, key, { i => node.children(i).key })
      if (idx == 0 && !ordering.equiv(key, node.children(0).key))
        -1
      else
        node.firstIndex + idx - 1
    } else {
      val node = readInternalNode(offset)
      val children = node.children
      val n = children.length
      val idx = binarySearchLeftmost(n, key, { i => children(i).lastKey })

      if (idx < n && ordering.gt(key, children(idx).firstKey))
        findStartIndex(key, level - 1, children(idx).indexFileOffset)
      else if (idx == 0) // insertion point is first bin but first child key is greater than key
        -1
      else
        findStartIndex(key, level - 1, children(idx - 1).indexFileOffset)
    }
  }

  // Returns an iterator starting from the largest leaf child less than key
  // If no leaf child is larger than key, the first value returned by the iterator is null
  private def iterateFrom(key: Annotation): Iterator[LeafChild] = new Iterator[LeafChild] {
    var idx = findStartIndex(key, height - 1, metadata.rootOffset)
    assert(idx >= -1 && idx < nKeys)

    def next(): LeafChild = {
      val lc = if (idx == -1) null else queryByIndex(idx)
      idx += 1
      lc
    }

    def hasNext: Boolean = idx < nKeys
  }

  private def queryByKeyLt(key: Annotation): Option[LeafChild] = {
    val it = iterateFrom(key)
    assert(it.hasNext)
    Option(it.next())
  }

  private def queryByKeyGt(key: Annotation): Option[LeafChild] = {
    val it = iterateFrom(key)
    assert(it.hasNext)
    it.next() // need to skip first value
    var current = if (it.hasNext) it.next() else null

    while (current != null && ordering.lteq(current.key, key)) {
      current = if (it.hasNext) it.next() else null
    }
    Option(current)
  }

  private def queryByKeyLtEq(key: Annotation): Option[LeafChild] = {
    val it = iterateFrom(key)

    val current = if (it.hasNext) it.next() else null
    val next = if (it.hasNext) it.next() else null

    if (next != null && ordering.equiv(next.key, key))
      Some(next)
    else
      Option(current)
  }

  private def queryByKeyGtEq(key: Annotation): Option[LeafChild] = {
    val it = iterateFrom(key)

    var current = if (it.hasNext) it.next() else null
    var next = if (it.hasNext) it.next() else null

    while (next != null && ordering.lteq(next.key, key)) {
      current = next
      next = if (it.hasNext) it.next() else null
    }

    if (current != null && ordering.equiv(current.key, key))
      Some(current)
    else
      Option(next)
  }

  def queryByKeyAllMatches(key: Annotation): Array[LeafChild] = {
    val ab = new ArrayBuilder[LeafChild]()
    val it = iterateFrom(key)
    assert(it.hasNext)
    it.next() // need to skip first value
    var current = if (it.hasNext) it.next() else null

    while (current != null && ordering.equiv(current.key, key)) {
      ab += current
      current = if (it.hasNext) it.next() else null
    }

    ab.result()
  }

  def queryByKey(key: Annotation, greater: Boolean = true, closed: Boolean = true): Option[LeafChild] = {
    if (greater && closed)
      queryByKeyGtEq(key)
    else if (greater && !closed)
      queryByKeyGt(key)
    else if (!greater && closed)
      queryByKeyLtEq(key)
    else
      queryByKeyLt(key)
  }

  private def queryByIndex(index: Long, level: Int, offset: Long): LeafChild = {
    if (level == 0) {
      val node = readLeafNode(offset)
      val localIdx = index - node.firstIndex
      node.children(localIdx.toInt)
    } else {
      val node = readInternalNode(offset)
      val firstIndex = node.firstIndex
      val nKeysPerChild = math.pow(branchingFactor, level).toLong
      val localIndex = (index - firstIndex) / nKeysPerChild
      queryByIndex(index, level - 1, node.children(localIndex.toInt).indexFileOffset)
    }
  }

  def queryByIndex(index: Long): LeafChild = {
    require(index >= 0 && index < nKeys)
    queryByIndex(index, height - 1, metadata.rootOffset)
  }

  def iterator: Iterator[LeafChild] = new Iterator[LeafChild] {
    var pos = 0L

    def next(): LeafChild = {
      val lc = queryByIndex(pos)
      pos += 1
      lc
    }

    def hasNext: Boolean = nKeys > 0 && pos < nKeys
  }

  def close() {
    region.close()
    leafDecoder.close()
    internalDecoder.close()
    log.info(s"Index reader cache queries: ${ cacheHits + cacheMisses }")
    log.info(s"Index reader cache hit rate: ${ cacheHits.toDouble / (cacheHits + cacheMisses) }")
  }
}

final case class InternalChild(
  indexFileOffset: Long,
  firstKey: Annotation,
  firstRecordOffset: Long,
  firstAnnotation: Annotation,
  lastKey: Annotation)

object InternalNode {
  def apply(r: Row): InternalNode = {
    val firstKeyIndex = r.getLong(0)
    val children = r.get(1).asInstanceOf[IndexedSeq[Row]].map(r => InternalChild(r.getLong(0), r.get(1), r.getLong(2), r.get(3), r.get(4)))
    InternalNode(firstKeyIndex, children)
  }
}

final case class InternalNode(
  firstIndex: Long,
  children: IndexedSeq[InternalChild]) extends IndexNode

final case class LeafChild(
  key: Annotation,
  recordOffset: Long,
  annotation: Annotation)

object LeafNode {
  def apply(r: Row): LeafNode = {
    val firstKeyIndex = r.getLong(0)
    val keys = r.get(1).asInstanceOf[IndexedSeq[Row]].map(r => LeafChild(r.get(0), r.getLong(1), r.get(2)))
    LeafNode(firstKeyIndex, keys)
  }
}

final case class LeafNode(
  firstIndex: Long,
  children: IndexedSeq[LeafChild]) extends IndexNode

sealed trait IndexNode
