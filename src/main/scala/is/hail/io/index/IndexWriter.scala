package is.hail.io.index

import is.hail.expr.types._
import is.hail.io.CodecSpec
import is.hail.rvd.RVDContext
import is.hail.utils._
import is.hail.utils.richUtils.ByteTrackingOutputStream
import org.apache.hadoop.conf.Configuration
import org.json4s.Formats
import org.json4s.jackson.Serialization

case class IndexMetadata(
  fileVersion: Int,
  branchingFactor: Int,
  keyType: String,
  nKeys: Long,
  indexPath: String,
  rootOffset: Long
)

class IndexWriter(
  conf: Configuration,
  path: String,
  keyType: Type,
  branchingFactor: Int = 1024) {

  private var elementIdx = 0L
  private var rootOffset = 0L

  private val ctx = RVDContext.default
  private val rvb = ctx.rvb
  private val region = ctx.freshRegion
  rvb.set(region)

  private val leafNode = new LeafNodeBuilder(keyType)
  private val internalNodes = new ArrayBuilder[InternalNodeBuilder]()

  private val trackedOS = new ByteTrackingOutputStream(conf.unsafeWriter(path + "/index"))
  private val codecSpec = CodecSpec.default
  private val leafEncoder = codecSpec.buildEncoder(leafNode.typ)(trackedOS)
  private val internalEncoder = codecSpec.buildEncoder(InternalNodeBuilder.typ(keyType))(trackedOS)

  private def writeInternalNode(node: InternalNodeBuilder): (Long, Long, Any, Long) = {
    val fileOffset = trackedOS.bytesWritten
    rootOffset = fileOffset

    val firstKey = node.firstKey
    val firstKeyOffset = node.firstOffset
    val firstIndex = node.firstIndex

    internalEncoder.writeByte(1)

    val regionOffset = node.write(rvb)
    internalEncoder.writeRegionValue(region, regionOffset)
    internalEncoder.flush()

    region.clear()
    node.clear()

    (fileOffset, firstIndex, firstKey, firstKeyOffset)
  }

  private def writeLeafNode(idx: Long): (Long, Any, Long) = {
    val fileOffset = trackedOS.bytesWritten
    val firstKey = leafNode.firstKey
    val firstOffset = leafNode.firstOffset

    leafEncoder.writeByte(0)

    val regionOffset = leafNode.write(rvb, idx)
    leafEncoder.writeRegionValue(region, regionOffset)
    leafEncoder.flush()

    region.clear()
    leafNode.clear()

    (fileOffset, firstKey, firstOffset)
  }

  private def write() {
    def updateInternalNodes(level: Int, fileOffset: Long, firstIndex: Long, firstKey: Any, firstOffset: Long) {
      if (level == internalNodes.size)
        internalNodes += new InternalNodeBuilder(keyType)

      val node = internalNodes(level)

      if (node.size == 0) {
        node.setFirstIndex(firstIndex)
      }

      node += (fileOffset, firstKey, firstOffset)

      if (node.size == branchingFactor) {
        val (fileOffset, firstIndex, firstKey, firstKeyOffset) = writeInternalNode(node)
        updateInternalNodes(level + 1, fileOffset, firstIndex, firstKey, firstKeyOffset)
      }
    }

    val idx = elementIdx - leafNode.size
    val (leafFileOffset, leafFirstKey, leafFirstOffset) = writeLeafNode(idx)
    updateInternalNodes(0, leafFileOffset, idx, leafFirstKey, leafFirstOffset)
  }

  private def flush() {
    val nInternalLevels = math.max(IndexUtils.calcDepth(elementIdx, branchingFactor) - 1, 1) // ensure always one internal node

    def flushInternalNodes(level: Int, fileOffset: Long, firstIndex: Long, firstKey: Any, firstOffset: Long) {
      if (level != nInternalLevels) {
        val node = internalNodes(level)

        if (node.size == 0)
          node.setFirstIndex(firstIndex)

        node += (fileOffset, firstKey, firstOffset)

        assert(node.size > 0 && node.size <= branchingFactor)

        val (nextFileOffset, nextFirstIndex, nextFirstKey, nextFirstKeyOffset) = writeInternalNode(node)

        flushInternalNodes(level + 1, nextFileOffset, nextFirstIndex, nextFirstKey, nextFirstKeyOffset)
      }
    }

    if (leafNode.size > 0) {
      write()
    }
    assert(internalNodes.size > 0)

    var level = 0
    while (level < nInternalLevels) {
      val node = internalNodes(level)
      if (node.size > 0) {
        val (fileOffset, firstIndex, firstKey, firstKeyOffset) = writeInternalNode(node)
        flushInternalNodes(level + 1, fileOffset, firstIndex, firstKey, firstKeyOffset)
      }
      level += 1
    }
  }

  private def writeMetadata() = {
    conf.writeTextFile(path + "/metadata.json.gz") { out =>
      val metadata = IndexMetadata(1, branchingFactor, keyType._toPretty, elementIdx, path, rootOffset)
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
    ctx.close()

    writeMetadata()
  }
}
