package is.hail.io.index

import java.io.OutputStream

import is.hail.annotations.{Annotation, Region, RegionValueBuilder}
import is.hail.expr.types._
import is.hail.expr.types.virtual.Type
import is.hail.io.{CodecSpec, Encoder}
import is.hail.utils._
import is.hail.utils.richUtils.ByteTrackingOutputStream
import is.hail.io.fs.FS
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

  def builder(
    keyType: Type,
    annotationType: Type,
    branchingFactor: Int = 4096,
    attributes: Map[String, Any] = Map.empty[String, Any]
  ): (FS, String) => IndexWriter = {
    val codecSpec = CodecSpec.default
    val makeLeafEncoder = codecSpec.buildEncoder(LeafNodeBuilder.typ(keyType, annotationType).physicalType)
    val makeInternalEncoder = codecSpec.buildEncoder(InternalNodeBuilder.typ(keyType, annotationType).physicalType);
    { (fs, path) =>
      new IndexWriter(
        fs,
        path,
        keyType,
        annotationType,
        makeLeafEncoder,
        makeInternalEncoder,
        branchingFactor,
        attributes)
    }
  }
}


class IndexWriter(
  fs: FS,
  path: String,
  keyType: Type,
  annotationType: Type,
  makeLeafEncoder: (OutputStream) => Encoder,
  makeInternalEncoder: (OutputStream) => Encoder,
  branchingFactor: Int = 4096,
  attributes: Map[String, Any] = Map.empty[String, Any]) extends AutoCloseable {
  require(branchingFactor > 1)

  private var elementIdx = 0L
  private val region = new Region()
  private val rvb = new RegionValueBuilder(region)

  private val leafNodeBuilder = new LeafNodeBuilder(keyType, annotationType, 0L)
  private val internalNodeBuilders = new ArrayBuilder[InternalNodeBuilder]()
  internalNodeBuilders += new InternalNodeBuilder(keyType, annotationType)

  private val trackedOS = new ByteTrackingOutputStream(fs.unsafeWriter(path + "/index"))

  private val leafEncoder = makeLeafEncoder(trackedOS)
  private val internalEncoder = makeInternalEncoder(trackedOS)

  private def height: Int = internalNodeBuilders.length + 1 // have one leaf node layer

  private def writeInternalNode(node: InternalNodeBuilder, level: Int, isRoot: Boolean = false): Long = {
    val indexFileOffset = trackedOS.bytesWritten

    val info = if (node.size > 0) {
      val firstChild = node.getChild(0)
      val firstIndex = firstChild.firstIndex
      val firstKey = firstChild.firstKey
      val firstRecordOffset = firstChild.firstRecordOffset
      val firstAnnotation = firstChild.firstAnnotation
      IndexNodeInfo(indexFileOffset, firstIndex, firstKey, firstRecordOffset, firstAnnotation)
    } else {
      assert(isRoot && level == 0)
      null
    }

    internalEncoder.writeByte(1)

    val regionOffset = node.write(rvb)
    internalEncoder.writeRegionValue(region, regionOffset)
    internalEncoder.flush()

    region.clear()
    node.clear()

    if (!isRoot) {
      if (level + 1 == internalNodeBuilders.length)
        internalNodeBuilders += new InternalNodeBuilder(keyType, annotationType)
      val parent = internalNodeBuilders(level + 1)
      if (parent.size == branchingFactor)
        writeInternalNode(parent, level + 1)
      parent += info
    }

    indexFileOffset
  }

  private def writeLeafNode(): Long = {
    val indexFileOffset = trackedOS.bytesWritten

    assert(leafNodeBuilder.size > 0)

    val firstIndex = leafNodeBuilder.firstIdx
    val firstChild = leafNodeBuilder.getChild(0)
    val firstKey = firstChild.key
    val firstRecordOffset = firstChild.recordOffset
    val firstAnnotation = firstChild.annotation

    leafEncoder.writeByte(0)

    val regionOffset = leafNodeBuilder.write(rvb)
    leafEncoder.writeRegionValue(region, regionOffset)
    leafEncoder.flush()

    region.clear()
    leafNodeBuilder.clear(elementIdx)

    val parent = internalNodeBuilders(0)
    if (parent.size == branchingFactor)
      writeInternalNode(parent, 0)
    parent += IndexNodeInfo(indexFileOffset, firstIndex, firstKey, firstRecordOffset, firstAnnotation)

    indexFileOffset
  }

  private def flush(): Long = {
    var offsetLastBlockWritten = 0L

    if (leafNodeBuilder.size > 0)
      writeLeafNode()

    var level = 0
    while (level < internalNodeBuilders.size) {
      val node = internalNodeBuilders(level)
      val isRoot = level == internalNodeBuilders.size - 1
      if (node.size > 0 || isRoot) {
        offsetLastBlockWritten = writeInternalNode(node, level, isRoot)
      }
      level += 1
    }

    offsetLastBlockWritten
  }

  private def writeMetadata(rootOffset: Long) = {
    fs.writeTextFile(path + "/metadata.json.gz") { out =>
      val metadata = IndexMetadata(
        IndexWriter.version.rep,
        branchingFactor,
        height,
        keyType.parsableString(),
        annotationType.parsableString(),
        elementIdx,
        "index",
        rootOffset,
        attributes)
      implicit val formats: Formats = defaultJSONFormats
      Serialization.write(metadata, out)
    }
  }

  def +=(x: Annotation, offset: Long, annotation: Annotation) {
    if (leafNodeBuilder.size == branchingFactor)
      writeLeafNode()
    leafNodeBuilder += (x, offset, annotation)
    elementIdx += 1
  }

  def close(): Unit = {
    val rootOffset = flush()
    trackedOS.close()
    region.close()
    writeMetadata(rootOffset)
  }
}
