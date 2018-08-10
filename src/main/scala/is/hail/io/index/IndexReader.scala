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
  private val metadata = IndexReader.readMetadata(hConf, path + "/metadata.json.gz")
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

  private val is = hConf.unsafeReader(path + "/" + indexRelativePath).asInstanceOf[FSDataInputStream]
  private val codecSpec = CodecSpec.default

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

  def close() {
    region.close()
    leafDecoder.close()
    internalDecoder.close()
    log.info(s"Index reader cache queries: ${ cacheHits + cacheMisses }")
    log.info(s"Index reader cache hit rate: ${ cacheHits.toDouble / (cacheHits + cacheMisses) }")
  }
}

final case class InternalChild(indexFileOffset: Long, firstKey: Annotation, firstRecordOffset: Long, firstAnnotation: Annotation)

object InternalNode {
  def apply(r: Row): InternalNode = {
    val firstKeyIndex = r.getLong(0)
    val children = r.get(1).asInstanceOf[IndexedSeq[Row]].map(r => InternalChild(r.getLong(0), r.get(1), r.getLong(2), r.get(3)))
    InternalNode(firstKeyIndex, children)
  }
}

final case class InternalNode(firstIndex: Long, children: IndexedSeq[InternalChild]) extends IndexNode

final case class LeafChild(key: Annotation, recordOffset: Long, annotation: Annotation)

object LeafNode {
  def apply(r: Row): LeafNode = {
    val firstKeyIndex = r.getLong(0)
    val keys = r.get(1).asInstanceOf[IndexedSeq[Row]].map(r => LeafChild(r.get(0), r.getLong(1), r.get(2)))
    LeafNode(firstKeyIndex, keys)
  }
}

final case class LeafNode(firstIndex: Long, children: IndexedSeq[LeafChild]) extends IndexNode

sealed trait IndexNode
