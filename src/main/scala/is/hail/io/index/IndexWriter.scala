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
  nKeys: Long,
  indexPath: String,
  rootOffset: Long,
  attributes: Map[String, Any]
)

case class IndexNodeInfo(
  fileOffset: Long,
  firstIndex: Long,
  firstKey: Annotation,
  firstKeyOffset: Long,
  lastKey: Annotation
)

object IndexWriter {
  def apply(
    hConf: Configuration,
    path: String,
    keyType: Type,
    branchingFactor: Int = 1024,
    attributes: Map[String, Any] = Map.empty[String, Any]) = new IndexWriter(new SerializableHadoopConfiguration(hConf), path, keyType, branchingFactor, attributes)
}

class IndexWriter(
  hConf: SerializableHadoopConfiguration,
  path: String,
  keyType: Type,
  branchingFactor: Int = 1024,
  attributes: Map[String, Any] = Map.empty[String, Any]) extends AutoCloseable {

  private val indexFile = path + "/index"
  private val metadataFile = path + "/metadata.json.gz"

  private var elementIdx = 0L
  private var rootOffset = 0L

  private val rvb = new RegionValueBuilder()
  private val region = new Region()
  rvb.set(region)

  private val leafNode = new LeafNodeBuilder(keyType)
  private val internalNodes = new ArrayBuilder[InternalNodeBuilder]()

  private val trackedOS = new ByteTrackingOutputStream(hConf.value.unsafeWriter(indexFile))
  private val codecSpec = CodecSpec.default
  private val leafEncoder = codecSpec.buildEncoder(leafNode.typ)(trackedOS)
  private val internalEncoder = codecSpec.buildEncoder(InternalNodeBuilder.typ(keyType))(trackedOS)

  private def calcDepth: Int =
    math.max(1, (math.log10(elementIdx) / math.log10(branchingFactor)).ceil.toInt) // max necessary for array of length 1 becomes depth = 0

  private def height: Int = math.max(calcDepth - 1, 1) + 1 // ensure always at least one internal level and one leaf level

  private def writeInternalNode(node: InternalNodeBuilder): IndexNodeInfo = {
    val fileOffset = trackedOS.bytesWritten
    rootOffset = fileOffset

    assert(node.size > 0)

    val firstChild = node.getChild(0)
    val firstKey = firstChild.firstKey
    val firstKeyOffset = firstChild.firstKeyOffset

    val lastChild = node.getChild(node.size - 1)
    val lastKey = lastChild.lastKey

    val firstIndex = node.firstIndex

    internalEncoder.writeByte(1)

    val regionOffset = node.write(rvb)
    internalEncoder.writeRegionValue(region, regionOffset)
    internalEncoder.flush()

    region.clear()
    node.clear()

    IndexNodeInfo(fileOffset, firstIndex, firstKey, firstKeyOffset, lastKey)
  }

  private def writeLeafNode(idx: Long): IndexNodeInfo = {
    val fileOffset = trackedOS.bytesWritten

    assert(leafNode.size > 0)
    val firstChild = leafNode.getChild(0)
    val firstKey = firstChild.key
    val firstOffset = firstChild.offset

    val lastChild = leafNode.getChild(leafNode.size - 1)
    val lastKey = lastChild.key

    leafEncoder.writeByte(0)

    val regionOffset = leafNode.write(rvb, idx)
    leafEncoder.writeRegionValue(region, regionOffset)
    leafEncoder.flush()

    region.clear()
    leafNode.clear()

    IndexNodeInfo(fileOffset, idx, firstKey, firstOffset, lastKey)
  }

  private def write() {
    def updateInternalNodes(level: Int, info: IndexNodeInfo) {
      if (level == internalNodes.size)
        internalNodes += new InternalNodeBuilder(keyType)

      val node = internalNodes(level)

      if (node.size == 0) {
        node.setFirstIndex(info.firstIndex)
      }

      node += info

      if (node.size == branchingFactor) {
        val newNodeInfo = writeInternalNode(node)
        updateInternalNodes(level + 1, newNodeInfo)
      }
    }

    val idx = elementIdx - leafNode.size
    val newNodeInfo = writeLeafNode(idx)
    updateInternalNodes(0, newNodeInfo)
  }

  private def flush() {
    val nInternalLevels = math.max(calcDepth - 1, 1) // ensure always one internal node

    def flushInternalNodes(level: Int, info: IndexNodeInfo) {
      if (level != nInternalLevels) {
        val node = internalNodes(level)

        if (node.size == 0)
          node.setFirstIndex(info.firstIndex)

        node += info

        assert(node.size > 0 && node.size <= branchingFactor)

        val newNodeInfo = writeInternalNode(node)

        flushInternalNodes(level + 1, newNodeInfo)
      }
    }

    if (leafNode.size > 0)
      write()

    val firstUnwrittenNodeIdx = internalNodes.result().indexWhere(_.size > 0)
    if (firstUnwrittenNodeIdx != -1 && firstUnwrittenNodeIdx < nInternalLevels) {
      val node = internalNodes(firstUnwrittenNodeIdx)
      val newNodeInfo = writeInternalNode(node)
      flushInternalNodes(firstUnwrittenNodeIdx + 1, newNodeInfo)
    }
  }

  private def writeMetadata() = {
    hConf.value.writeTextFile(metadataFile) { out =>
      val metadata = IndexMetadata(1, branchingFactor, height, keyType._toPretty, elementIdx, indexFile, rootOffset, attributes)
      implicit val formats: Formats = defaultJSONFormats
      Serialization.write(metadata, out)
    }
  }

  def +=(x: Any, offset: Long) {
    leafNode += (x, offset)
    elementIdx += 1
    if (leafNode.size == branchingFactor) {
      write()
    }
  }

  def close(): Unit = {
    flush()
    trackedOS.close()
    region.close()

    writeMetadata()
  }
}
