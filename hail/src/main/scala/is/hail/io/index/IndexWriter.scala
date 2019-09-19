package is.hail.io.index

import java.io.OutputStream

import is.hail.annotations.{Annotation, Region, RegionValueBuilder}
import is.hail.expr.types.physical.PType
import is.hail.expr.types.virtual.Type
import is.hail.io.fs.FS
import is.hail.io._
import is.hail.rvd.AbstractRVDSpec
import is.hail.utils._
import is.hail.utils.richUtils.ByteTrackingOutputStream
import org.json4s.Formats
import org.json4s.jackson.Serialization

trait AbstractIndexMetadata {
  def fileVersion: Int

  def branchingFactor: Int

  def height: Int

  def keyType: Type

  def annotationType: Type

  def nKeys: Long

  def indexPath: String

  def rootOffset: Long

  def attributes: Map[String, Any]

  def leafSpec: CodecSpec2

  def intSpec: CodecSpec2
}

case class IndexMetadataUntypedJSON(
  fileVersion: Int,
  branchingFactor: Int,
  height: Int,
  nKeys: Long,
  indexPath: String,
  rootOffset: Long,
  attributes: Map[String, Any]
) {
  def toMetadata(keyType: Type, annotationType: Type): IndexMetadata = IndexMetadata(
    fileVersion, branchingFactor,
    height, keyType, annotationType,
    nKeys, indexPath, rootOffset, attributes)
}

case class IndexMetadata(
  fileVersion: Int,
  branchingFactor: Int,
  height: Int,
  keyType: Type,
  annotationType: Type,
  nKeys: Long,
  indexPath: String,
  rootOffset: Long,
  attributes: Map[String, Any]
) extends AbstractIndexMetadata {
  val baseSpec = LEB128BufferSpec(
      BlockingBufferSpec(32 * 1024,
        LZ4BlockBufferSpec(32 * 1024,
          new StreamBlockBufferSpec)))

  def leafSpec: CodecSpec2 = PackCodecSpec2(LeafNodeBuilder.legacyTyp(keyType.physicalType, annotationType.physicalType), baseSpec)

  def intSpec: CodecSpec2 = PackCodecSpec2(InternalNodeBuilder.legacyTyp(keyType.physicalType, annotationType.physicalType), baseSpec)
}

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
    keyType: PType,
    annotationType: PType,
    branchingFactor: Int = 4096,
    attributes: Map[String, Any] = Map.empty[String, Any]
  ): (FS, String) => IndexWriter = {
    val leafPType = LeafNodeBuilder.typ(keyType, annotationType)
    val makeLeafEnc = CodecSpec.default.makeCodecSpec2(leafPType).buildEncoder(leafPType)

    val intPType = InternalNodeBuilder.typ(keyType, annotationType)
    val makeIntEnc = CodecSpec.default.makeCodecSpec2(intPType).buildEncoder(intPType)


    { (fs: FS, path: String) =>
      new IndexWriter(
        fs,
        path,
        keyType,
        annotationType,
        makeLeafEnc,
        makeIntEnc,
        leafPType,
        intPType,
        branchingFactor,
        attributes)
    }
  }
}


class IndexWriter(
  fs: FS,
  path: String,
  keyType: PType,
  annotationType: PType,
  makeLeafEncoder: (OutputStream) => Encoder,
  makeInternalEncoder: (OutputStream) => Encoder,
  leafPType: PType,
  intPType: PType,
  branchingFactor: Int = 4096,
  attributes: Map[String, Any] = Map.empty[String, Any]) extends AutoCloseable {
  require(branchingFactor > 1)

  private var elementIdx = 0L
  private val region = Region()
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
        keyType.virtualType,
        annotationType.virtualType,
        elementIdx,
        "index",
        rootOffset,
        attributes)
      implicit val formats: Formats = AbstractRVDSpec.formats
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
