package is.hail.io.index

import java.io.InputStream
import java.util
import java.util.Map.Entry

import is.hail.annotations._
import is.hail.expr.types.virtual.Type
import is.hail.expr.ir.IRParser
import is.hail.expr.types.physical.{PStruct, PType}
import is.hail.io._
import is.hail.io.bgen.BgenSettings
import is.hail.utils._
import is.hail.io.fs.FS
import is.hail.rvd.AbstractRVDSpec
import org.apache.hadoop.fs.FSDataInputStream
import org.apache.spark.sql.Row
import org.json4s.Formats
import org.json4s.jackson.JsonMethods

object IndexReaderBuilder {
  def apply(fs: FS, path: String): (FS, String, Int) => IndexReader = {
    val (keyType, annotationType) = IndexReader.readTypes(fs, path)
    IndexReaderBuilder(keyType, annotationType)
  }

  def apply(settings: BgenSettings): (FS, String, Int) => IndexReader =
    IndexReaderBuilder(settings.requestedType.keyType, settings.indexAnnotationType)

  def apply(keyType: Type, annotationType: Type): (FS, String, Int) => IndexReader = {
    val leafPType = LeafNodeBuilder.typ(keyType.physicalType, annotationType.physicalType)
    val intPType = InternalNodeBuilder.typ(keyType.physicalType, annotationType.physicalType)
    val leafCodec = legacyCodecSpec(leafPType)
    val intCodec = legacyCodecSpec(intPType)

    val (leafRetPType, leafDec) = leafCodec.buildDecoder(leafPType.virtualType)
    assert(leafRetPType == leafPType)

    val (intRetPType, intDec) = intCodec.buildDecoder(intPType.virtualType)
    assert(intRetPType == intPType)

    withDecoders(leafDec, intDec, keyType, annotationType, leafPType, intPType)
  }

  def withDecoders(
    leafDec: (InputStream) => Decoder, intDec: (InputStream) => Decoder,
    keyType: Type, annotationType: Type,
    leafPType: PStruct, intPType: PStruct
  ): (FS, String, Int) => IndexReader = {
    (fs, path, cacheCapacity) => new IndexReader(fs, path, cacheCapacity, leafDec,
      intDec, keyType, annotationType, leafPType, intPType)
  }

  def legacyCodecSpec(pt: PType): CodecSpec2 = PackCodecSpec2(pt, LEB128BufferSpec(
    BlockingBufferSpec(32 * 1024,
      LZ4BlockBufferSpec(32 * 1024,
        new StreamBlockBufferSpec))))

}

object IndexReader {
  def readMetadata(fs: FS, path: String): IndexMetadata = {
    val jv = fs.readFile(path + "/metadata.json.gz") { in => JsonMethods.parse(in) }
    implicit val formats: Formats = AbstractRVDSpec.formats
    jv.extract[IndexMetadata]
  }

  def readTypes(fs: FS, path: String): (Type, Type) = {
    val metadata = IndexReader.readMetadata(fs, path)
    (metadata.keyType, metadata.annotationType)
  }

  def apply(fs: FS, path: String, cacheCapacity: Int = 8): IndexReader = {
    val builder = IndexReaderBuilder(fs, path)
    builder(fs, path, cacheCapacity)
  }
}


class IndexReader(fs: FS,
  path: String,
  cacheCapacity: Int = 8,
  leafDecoderBuilder: (InputStream) => Decoder,
  internalDecoderBuilder: (InputStream) => Decoder,
  val keyType: Type,
  val annotationType: Type,
  val leafPType: PStruct,
  val internalPType: PStruct
) extends AutoCloseable {
  private[io] val metadata = IndexReader.readMetadata(fs, path)
  val branchingFactor = metadata.branchingFactor
  val height = metadata.height
  val nKeys = metadata.nKeys
  val attributes = metadata.attributes
  val indexRelativePath = metadata.indexPath
  val ordering = keyType.ordering

  private val is = fs.unsafeReader(path + "/" + indexRelativePath).asInstanceOf[FSDataInputStream]
  private val leafDecoder = leafDecoderBuilder(is)
  private val internalDecoder = internalDecoderBuilder(is)

  private val region = Region()
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
      val node = InternalNode(SafeRow(internalPType, rv))
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
      val node = LeafNode(SafeRow(leafPType, rv))
      cache.put(offset, node)
      region.clear()
      node
    }
  }

  private def lowerBound(key: Annotation, level: Int, offset: Long): Long = {
    if (level == 0) {
      val node = readLeafNode(offset)
      val idx = node.children.lowerBound(key, ordering.lt, _.key)
      node.firstIndex + idx
    } else {
      val node = readInternalNode(offset)
      val children = node.children
      val idx = children.lowerBound(key, ordering.lt, _.firstKey)
      lowerBound(key, level - 1, children(idx - 1).indexFileOffset)
    }
  }

  private[io] def lowerBound(key: Annotation): Long = {
    if (nKeys == 0 || ordering.lteq(key, readInternalNode(metadata.rootOffset).children.head.firstKey))
      0
    else
      lowerBound(key, height - 1, metadata.rootOffset)
  }

  private def upperBound(key: Annotation, level: Int, offset: Long): Long = {
    if (level == 0) {
      val node = readLeafNode(offset)
      val idx = node.children.upperBound(key, ordering.lt, _.key)
      node.firstIndex + idx
    } else {
      val node = readInternalNode(offset)
      val children = node.children
      val n = children.length
      val idx = children.upperBound(key, ordering.lt, _.firstKey)
      upperBound(key, level - 1, children(idx - 1).indexFileOffset)
    }
  }

  private[io] def upperBound(key: Annotation): Long = {
    if (nKeys == 0 || ordering.lt(key, readInternalNode(metadata.rootOffset).children.head.firstKey))
      0
    else
      upperBound(key, height - 1, metadata.rootOffset)
  }

  private def getLeafNode(index: Long, level: Int, offset: Long): LeafNode = {
    if (level == 0) {
      readLeafNode(offset)
    } else {
      val node = readInternalNode(offset)
      val firstIndex = node.firstIndex
      val nKeysPerChild = math.pow(branchingFactor, level).toLong
      val localIndex = (index - firstIndex) / nKeysPerChild
      getLeafNode(index, level - 1, node.children(localIndex.toInt).indexFileOffset)
    }
  }

  private def getLeafNode(index: Long): LeafNode =
    getLeafNode(index, height - 1, metadata.rootOffset)

  def queryByKey(key: Annotation): Array[LeafChild] = {
    val ab = new ArrayBuilder[LeafChild]()
    keyIterator(key).foreach(ab += _)
    ab.result()
  }

  def keyIterator(key: Annotation): Iterator[LeafChild] =
    iterateFrom(key).takeWhile(lc => ordering.equiv(lc.key, key))

  def queryByIndex(index: Long): LeafChild = {
    require(index >= 0 && index < nKeys)
    val node = getLeafNode(index)
    val localIdx = index - node.firstIndex
    node.children(localIdx.toInt)
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
    var localPos = 0
    var leafNode: LeafNode = _

    def next(): LeafChild = {
      assert(hasNext)

      if (leafNode == null || localPos >= leafNode.children.length) {
        leafNode = getLeafNode(pos)
        assert(leafNode.firstIndex <= pos && pos < leafNode.firstIndex + branchingFactor)
        localPos = (pos - leafNode.firstIndex).toInt
      }

      val child = leafNode.children(localPos)
      pos += 1
      localPos += 1
      child
    }

    def hasNext: Boolean = pos < end

    def seek(key: Annotation) {
      val newPos = lowerBound(key)
      assert(newPos >= pos)
      localPos += (newPos - pos).toInt
      pos = newPos
    }
  }

  def iterateFrom(key: Annotation): Iterator[LeafChild] =
    iterator(lowerBound(key), nKeys)

  def iterateUntil(key: Annotation): Iterator[LeafChild] =
    iterator(0, lowerBound(key))

  def close() {
    leafDecoder.close()
    internalDecoder.close()
    log.info(s"Index reader cache queries: ${ cacheHits + cacheMisses }")
    log.info(s"Index reader cache hit rate: ${ cacheHits.toDouble / (cacheHits + cacheMisses) }")
  }
}

final case class InternalChild(
  indexFileOffset: Long,
  firstIndex: Long,
  firstKey: Annotation,
  firstRecordOffset: Long,
  firstAnnotation: Annotation)

object InternalNode {
  def apply(r: Row): InternalNode = {
    val children = r.get(0).asInstanceOf[IndexedSeq[Row]].map(r => InternalChild(r.getLong(0), r.getLong(1), r.get(2), r.getLong(3), r.get(4)))
    InternalNode(children)
  }
}

final case class InternalNode(children: IndexedSeq[InternalChild]) extends IndexNode {
  def firstIndex: Long = {
    assert(children.nonEmpty)
    children.head.firstIndex
  }
}

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
