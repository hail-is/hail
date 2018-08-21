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
  annotationType: Type,
  branchingFactor: Int = 1024,
  attributes: Map[String, Any] = Map.empty[String, Any]) extends AutoCloseable {
  require(branchingFactor > 1)

  private var elementIdx = 0L
  private val region = new Region()
  private val rvb = new RegionValueBuilder(region)

  private val leafNodeBuilder = new LeafNodeBuilder(keyType, annotationType, 0L)
  private val internalNodeBuilders = new ArrayBuilder[InternalNodeBuilder]()
  internalNodeBuilders += new InternalNodeBuilder(keyType, annotationType, 0L)

  private val trackedOS = new ByteTrackingOutputStream(hConf.unsafeWriter(path + "/index"))
  private val codecSpec = CodecSpec.default
  private val leafEncoder = codecSpec.buildEncoder(leafNodeBuilder.typ)(trackedOS)
  private val internalEncoder = codecSpec.buildEncoder(InternalNodeBuilder.typ(keyType, annotationType))(trackedOS)

  private def height: Int = internalNodeBuilders.length + 1 // have one leaf node layer

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

  private def write(flush: Boolean = false): Long = {
    var offsetLastBlockWritten = 0L

    if (leafNodeBuilder.size == branchingFactor || (flush && leafNodeBuilder.size > 0)) {
      val info = writeLeafNode()
      internalNodeBuilders(0) += info
      offsetLastBlockWritten = info.indexFileOffset
    }

    var level = 0
    while (level < internalNodeBuilders.size && (internalNodeBuilders(level).size == branchingFactor || flush)) {
      val node = internalNodeBuilders(level)
      if (node.size > 0 || (flush && internalNodeBuilders.size == 1)) { // last case is for empty index
        val info = writeInternalNode(node)
        offsetLastBlockWritten = info.indexFileOffset
        if (level + 1 < internalNodeBuilders.size)
          internalNodeBuilders(level + 1) += info
        else if (!flush) {
          internalNodeBuilders += new InternalNodeBuilder(keyType, annotationType, info.firstIndex)
          internalNodeBuilders(level + 1) += info
        }
      }
      level += 1
    }

    offsetLastBlockWritten
  }

  private def writeMetadata(rootOffset: Long) = {
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
    val rootOffset = write(flush = true)
    trackedOS.close()
    region.close()
    writeMetadata(rootOffset)
  }
}
