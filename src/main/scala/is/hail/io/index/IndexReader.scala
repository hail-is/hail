package is.hail.io.index

import java.util
import java.util.Map.Entry

import is.hail.annotations.{Annotation, RegionValue, SafeRow}
import is.hail.expr.Parser
import is.hail.io._
import is.hail.rvd.RVDContext
import is.hail.utils._
import org.apache.hadoop.conf.Configuration
import org.apache.hadoop.fs.FSDataInputStream
import org.apache.spark.sql.Row
import org.json4s.Formats
import org.json4s.jackson.JsonMethods

class IndexReader(conf: Configuration, path: String, cacheCapacity: Int = 256) {
  private val is = conf.unsafeReader(path + "/index").asInstanceOf[FSDataInputStream]
  private val codecSpec = CodecSpec.default

  private val metadata = readMetadata(path + "/metadata.json.gz")
  val branchingFactor = metadata.branchingFactor
  val nKeys = metadata.nKeys

  val keyType = Parser.parseType(metadata.keyType)
  val leafType = LeafNodeBuilder.typ(keyType)
  val internalType = InternalNodeBuilder.typ(keyType)

  private val leafDecoder = codecSpec.buildDecoder(leafType, leafType)(is)
  private val internalDecoder = codecSpec.buildDecoder(internalType, internalType)(is)

  private val ctx = RVDContext.default
  private val region = ctx.region
  private val rv = RegionValue(region)

  @transient private[this] lazy val leafCache = new util.LinkedHashMap[Long, LeafNode](cacheCapacity, 0.75f, true) {
    override def removeEldestEntry(eldest: Entry[Long, LeafNode]): Boolean = size() > cacheCapacity
  }

  def readMetadata(metadataFile: String): IndexMetadata = {
    val jv = conf.readFile(metadataFile) { in => JsonMethods.parse(in) }
    implicit val formats: Formats = defaultJSONFormats
    jv.extract[IndexMetadata]
  }

  def readInternalNode(offset: Long): InternalNode = {
    is.seek(offset)
    assert(internalDecoder.readByte() == 1)
    rv.setOffset(internalDecoder.readRegionValue(region))
    InternalNode(SafeRow(internalType, rv))
  }

  def readLeafNode(offset: Long): LeafNode = {
    is.seek(offset)
    assert(leafDecoder.readByte() == 0)
    rv.setOffset(leafDecoder.readRegionValue(region))
    LeafNode(SafeRow(leafType, rv))
  }

  private def queryByIndex(idx: Long, depth: Int, offset: Long): LeafChild = {
    if (depth == 0) {
      val node = readLeafNode(offset)
      val localIdx = idx - node.firstKeyIndex

      if (!leafCache.containsKey(node.firstKeyIndex))
        leafCache.put(node.firstKeyIndex, node)

      node.keys(localIdx.toInt)
    } else {
      val node = readInternalNode(offset)
      val blockStartIdx = node.blockFirstKeyIndex
      val nKeysPerBlock = math.pow(branchingFactor, depth).toLong
      val localIdx = (idx - blockStartIdx) / nKeysPerBlock
      queryByIndex(idx, depth - 1, node.children(localIdx.toInt).childOffset)
    }
  }

  def queryByIndex(idx: Long): LeafChild = {
    require(idx >= 0 && idx < nKeys)

    val hash = idx / branchingFactor * branchingFactor

    if (leafCache.containsKey(hash)) {
      val node = leafCache.get(hash)
      node.keys((idx - node.firstKeyIndex).toInt)
    } else {
      val maxDepth = math.max(IndexUtils.calcDepth(nKeys, branchingFactor) - 1, 1)
      queryByIndex(idx, maxDepth, metadata.rootOffset)
    }
  }

  def close() {
    leafDecoder.close()
    internalDecoder.close()
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

final case class InternalNode(blockFirstKeyIndex: Long, children: IndexedSeq[InternalChild])

final case class LeafChild(key: Annotation, offset: Long) { }

object LeafNode {
  def apply(r: Row): LeafNode = {
    val firstKeyIndex = r.getLong(0)
    val keys = r.get(1).asInstanceOf[IndexedSeq[Row]].map(r => LeafChild(r.get(0), r.getLong(1)))
    LeafNode(firstKeyIndex, keys)
  }
}

final case class LeafNode(firstKeyIndex: Long, keys: IndexedSeq[LeafChild]) { }
