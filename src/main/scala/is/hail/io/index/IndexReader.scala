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

  // Returns smallest i, 0 <= i < n, for which p(i) holds, or returns n if p(i) is false for all i.
  // Assumes all i for which p(i) is true are greater than all i for which p(i) is false.
  // Returned value j has the property that p(i) is false for all i in [0, j),
  // and p(i) is true for all i in [j, n).
  private def findPartitionPoint(n: Int, p: (Int) => Boolean): Int = {
    var left = 0
    var right = n
    while (left < right) {
      val mid = left + (right - left) / 2
      if (p(mid))
        right = mid
      else
        left = mid + 1
    }
    left
  }

  // Returns smallest i, 0 <= i < n, for which f(i) >= key, or returns n if f(i) < key for all i
  private[io] def binarySearchLowerBound(n: Int, key: Annotation, f: (Int) => Annotation): Int =
    findPartitionPoint(n, i => ordering.gteq(f(i), key))

  // Returns smallest i, 0 <= i < n, for which f(i) > key, or returns n if f(i) > key for all i
  private[io] def binarySearchUpperBound(n: Int, key: Annotation, f: (Int) => Annotation): Int =
    findPartitionPoint(n, i => ordering.gt(f(i), key))

  private def lowerBound(key: Annotation, level: Int, offset: Long): Long = {
    if (level == 0) {
      val node = readLeafNode(offset)
      val idx = binarySearchLowerBound(node.children.length, key, { i => node.children(i).key })
      node.firstIndex + idx
    } else {
      val node = readInternalNode(offset)
      val children = node.children
      val n = children.length
      if (n == 0)
        return node.firstIndex
      val idx = binarySearchLowerBound(n, key, { i => children(i).lastKey })
      lowerBound(key, level - 1, children(math.min(idx, n - 1)).indexFileOffset)
    }
  }

  private[io] def lowerBound(key: Annotation): Long =
    lowerBound(key, height - 1, metadata.rootOffset)

  private def upperBound(key: Annotation, level: Int, offset: Long): Long = {
    if (level == 0) {
      val node = readLeafNode(offset)
      val idx = binarySearchUpperBound(node.children.length, key, { i => node.children(i).key })
      node.firstIndex + idx
    } else {
      val node = readInternalNode(offset)
      val children = node.children
      val n = children.length
      if (n == 0)
        return node.firstIndex
      val idx = binarySearchUpperBound(n, key, { i => children(i).firstKey })
      if (idx > 0 && ordering.lteq(key, children(idx - 1).lastKey)) // upper bound returns one past the child index where the key may reside
        upperBound(key, level - 1, children(idx - 1).indexFileOffset)
      else
        upperBound(key, level - 1, children(math.min(idx, n - 1)).indexFileOffset)
    }
  }

  private[io] def upperBound(key: Annotation): Long =
    upperBound(key, height - 1, metadata.rootOffset)

  def queryByKey(key: Annotation): Array[LeafChild] = {
    val ab = new ArrayBuilder[LeafChild]()
    keyIterator(key).foreach(ab += _)
    ab.result()
  }

  def keyIterator(key: Annotation): Iterator[LeafChild] = new Iterator[LeafChild] {
    var pos = lowerBound(key)
    var current: LeafChild = _

    def next(): LeafChild = current

    def hasNext(): Boolean = {
      pos < nKeys && {
        current = queryByIndex(pos)
        pos += 1
        ordering.equiv(current.key, key)
      }
    }
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

  def queryByInterval(interval: Interval): Iterator[LeafChild] =
    queryByInterval(interval.start, interval.end, interval.includesStart, interval.includesEnd)

  def queryByInterval(start: Annotation, end: Annotation, includesStart: Boolean, includesEnd: Boolean): Iterator[LeafChild] = {
    require(Interval.isValid(ordering, start, end, includesStart, includesEnd))
    val startIdx = if (includesStart) lowerBound(start) else upperBound(start)
    val endIdx = if (includesEnd) upperBound(end) else lowerBound(end)
    iterator(startIdx, endIdx)
  }

  def iterator: Iterator[LeafChild] = iterator(0, nKeys)

  def iterator(start: Long, end: Long) = new Iterator[LeafChild] {
    assert(start >= 0 && end <= nKeys && start <= end)
    var pos = start

    def next(): LeafChild = {
      val lc = queryByIndex(pos)
      pos += 1
      lc
    }

    def hasNext: Boolean = pos < end
  }

  def iterateFrom(key: Annotation): Annotation =
    iterator(lowerBound(key), nKeys)

  def iterateUntil(key: Annotation): Annotation =
    iterator(0, lowerBound(key))

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
