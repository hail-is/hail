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


  private def queryByKey(key: Annotation, level: Int, offset: Long, ordering: ExtendedOrdering): Option[LeafChild] = {
    def searchLeafNode(node: LeafNode): Option[LeafChild] = {
      var left = 0
      var right = node.children.length - 1
      while (left <= right) {
        val mid = left + (right - left) / 2
        val midKey = node.children(mid).key

        if (ordering.equiv(key, midKey))
          return Some(node.children(mid))
        else if (ordering.lt(key, midKey))
          right = mid - 1
        else
          left = mid + 1
      }
      None
    }

    def searchInternalNode(node: InternalNode): InternalChild = {
      var left = 0
      var right = node.children.length - 1
      while (left <= right) {
        val mid = left + (right - left) / 2
        val midKey = node.children(mid).firstKey

        if (ordering.equiv(key, midKey))
          return node.children(mid)
        else if (ordering.lt(key, midKey))
          right = mid - 1
        else
          left = mid + 1
      }

      node.children(left - 1)
    }

    if (level == 0)
      searchLeafNode(readLeafNode(offset))
    else
      queryByKey(key, level - 1, searchInternalNode(readInternalNode(offset)).childOffset, ordering)
  }

  def queryByKey(key: Annotation): Option[LeafChild] = {
    queryByKey(key, height - 1, metadata.rootOffset, keyType.ordering)
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
    info(s"Index reader cache hit rate: ${ cacheHits.toDouble / (cacheHits + cacheMisses) }")
  }
}

final case class InternalChild(childOffset: Long, firstKey: Annotation, firstKeyOffset: Long)

object InternalNode {
  def apply(r: Row): InternalNode = {
    val blockFirstKeyIndex = r.getLong(0)
    val children = r.get(1).asInstanceOf[IndexedSeq[Row]].map(r => InternalChild(r.getLong(0), r.get(1), r.getLong(2)))
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
