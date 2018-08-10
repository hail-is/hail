package is.hail.io.index

import is.hail.annotations.{Annotation, Region, RegionValueBuilder}
import is.hail.expr.types._
import is.hail.io.CodecSpec
import is.hail.utils._
import is.hail.utils.richUtils.ByteTrackingOutputStream
import org.apache.hadoop.conf.Configuration
import org.json4s.Formats
import org.json4s.jackson.Serialization

case class IndexMetadata(
  fileVersion: Int,
  branchingFactor: Int,
  height: Int,
  keyType: String,
  annotationType: String,
  nKeys: Long,
  indexPath: String,
  rootOffset: Long,
  attributes: Map[String, Any]
)

case class IndexNodeInfo(
  indexFileOffset: Long,
  firstIndex: Long,
  firstKey: Annotation,
  firstRecordOffset: Long,
  firstAnnotation: Annotation
)

object IndexWriter {
  val version: SemanticVersion = SemanticVersion(1, 0, 0)
}

class IndexWriter(
  hConf: Configuration,
  path: String,
  keyType: Type,
  annotationType: TStruct,
  branchingFactor: Int = 1024,
  attributes: Map[String, Any] = Map.empty[String, Any]) extends AutoCloseable {
  require(branchingFactor > 1)

  private var elementIdx = 0L
  private var rootOffset = 0L

  private val region = new Region()
  private val rvb = new RegionValueBuilder(region)

  private val leafNodeBuilder = new LeafNodeBuilder(keyType, annotationType, 0L)
  private val internalNodeBuilders = new ArrayBuilder[InternalNodeBuilder]()

  private val trackedOS = new ByteTrackingOutputStream(hConf.unsafeWriter(path + "/index"))
  private val codecSpec = CodecSpec.default
  private val leafEncoder = codecSpec.buildEncoder(leafNodeBuilder.typ)(trackedOS)
  private val internalEncoder = codecSpec.buildEncoder(InternalNodeBuilder.typ(keyType, annotationType))(trackedOS)

  private def calcDepth: Int =
    math.max(1, (math.log10(elementIdx) / math.log10(branchingFactor)).ceil.toInt) // max necessary for array of length 1 becomes depth = 0

  // last internal node builder is the root (not written): internal node length - root + leaf node == internalNodes.length
  private def height: Int = internalNodeBuilders.length

  private def writeInternalNode(node: InternalNodeBuilder): IndexNodeInfo = {
    val indexFileOffset = trackedOS.bytesWritten

    assert(node.size > 0)
    val child = node.getChild(0)
    val firstIndex = node.firstIdx
    val firstKey = child.firstKey
    val firstRecordOffset = child.firstRecordOffset
    val firstAnnotation = child.firstAnnotation

    internalEncoder.writeByte(1)

    val regionOffset = node.write(rvb)
    internalEncoder.writeRegionValue(region, regionOffset)
    internalEncoder.flush()

    region.clear()
    node.clear(elementIdx)

    IndexNodeInfo(indexFileOffset, firstIndex, firstKey, firstRecordOffset, firstAnnotation)
  }

  private def writeLeafNode(): IndexNodeInfo = {
    val indexFileOffset = trackedOS.bytesWritten

    assert(leafNodeBuilder.size > 0)
    val child = leafNodeBuilder.getChild(0)
    val firstIndex = leafNodeBuilder.firstIdx
    val firstKey = child.key
    val firstRecordOffset = child.recordOffset
    val firstAnnotation = child.annotation

    leafEncoder.writeByte(0)

    val regionOffset = leafNodeBuilder.write(rvb)
    leafEncoder.writeRegionValue(region, regionOffset)
    leafEncoder.flush()

    region.clear()
    leafNodeBuilder.clear(elementIdx)

    IndexNodeInfo(indexFileOffset, firstIndex, firstKey, firstRecordOffset, firstAnnotation)
  }

  private def write(flush: Boolean = false) {
    val nInternalNodes = math.max(calcDepth - 1, 1)

    def writeInternalNodes(level: Int) {
      if (level < nInternalNodes) {
        val node = internalNodeBuilders(level)

        if (node.size == branchingFactor || (flush && node.size > 0)) {
          val info = writeInternalNode(node)
          if (level + 1 == internalNodeBuilders.size)
            internalNodeBuilders += new InternalNodeBuilder(keyType, annotationType, info.firstIndex)
          internalNodeBuilders(level + 1) += info
          writeInternalNodes(level + 1)
        } else if (flush)
          writeInternalNodes(level + 1)
      }
    }

    if (leafNodeBuilder.size == branchingFactor || (flush && leafNodeBuilder.size > 0)) {
      val info = writeLeafNode()
      if (internalNodeBuilders.isEmpty)
        internalNodeBuilders += new InternalNodeBuilder(keyType, annotationType, info.firstIndex)
      internalNodeBuilders(0) += info
      writeInternalNodes(0)
    } else if (flush)
      writeInternalNodes(0)
  }

  private def writeMetadata() = {
    hConf.writeTextFile(path + "/metadata.json.gz") { out =>
      val metadata = IndexMetadata(IndexWriter.version.rep, branchingFactor, height, keyType._toPretty, annotationType._toPretty, elementIdx, "index", rootOffset, attributes)
      implicit val formats: Formats = defaultJSONFormats
      Serialization.write(metadata, out)
    }
  }

  def +=(x: Annotation, offset: Long, annotation: Annotation) {
    leafNodeBuilder += (x, offset, annotation)
    elementIdx += 1
    if (leafNodeBuilder.size == branchingFactor) {
      write()
    }
  }

  def close(): Unit = {
    write(flush = true)
    val rootNodeBuilder = internalNodeBuilders.last
    assert(rootNodeBuilder.size == 1)
    rootOffset = rootNodeBuilder.indexFileOffsets(0)
    trackedOS.close()
    region.close()

    writeMetadata()
  }
}
