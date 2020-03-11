package is.hail.io.bgen

import is.hail.annotations._
import is.hail.asm4s.{AsmFunction4, AsmFunction5}
import is.hail.backend.BroadcastValue
import is.hail.expr.ir.PruneDeadFields
import is.hail.expr.types._
import is.hail.expr.types.encoded.{EArray, EBaseStruct, EBinaryOptional, EBinaryRequired, EField, EInt32Optional, EInt32Required, EInt64Required}
import is.hail.expr.types.physical.{PArray, PCall, PCanonicalLocus, PFloat64Required, PInt32, PInt64, PLocus, PString, PStruct}
import is.hail.expr.types.virtual.{Field, TArray, TInt64, TLocus, TString, TStruct, Type}
import is.hail.io.{AbstractTypedCodecSpec, BlockingBufferSpec, HadoopFSDataBinaryReader, LEB128BufferSpec, LZ4HCBlockBufferSpec, StreamBlockBufferSpec, TypedCodecSpec}
import is.hail.io.index.{IndexReader, IndexReaderBuilder, LeafChild}
import is.hail.rvd._
import is.hail.sparkextras._
import is.hail.utils.FastIndexedSeq
import is.hail.variant.ReferenceGenome
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.Row
import org.apache.spark.{OneToOneDependency, Partition, SparkContext, TaskContext}

import scala.language.reflectiveCalls

object BgenSettings {
  def indexKeyType(rg: Option[ReferenceGenome]): TStruct = TStruct(
    "locus" -> rg.map(TLocus(_)).getOrElse(TLocus.representation),
    "alleles" -> TArray(TString))
  val indexAnnotationType: Type = TStruct.empty

  def indexCodecSpecs(rg: Option[ReferenceGenome]): (AbstractTypedCodecSpec, AbstractTypedCodecSpec) = {
    val bufferSpec = LEB128BufferSpec(
      BlockingBufferSpec(32 * 1024,
        LZ4HCBlockBufferSpec(32 * 1024,
          new StreamBlockBufferSpec)))

    val keyVType = indexKeyType(rg)
    val keyEType = EBaseStruct(FastIndexedSeq(
      EField("locus", EBaseStruct(FastIndexedSeq(
        EField("contig", EBinaryRequired, 0),
        EField("position", EInt32Required, 1)
      )), 0),
      EField("alleles", EArray(EBinaryOptional, required = false), 1)),
      required = false
    )

    val annotationVType = TStruct.empty
    val annotationEType = EBaseStruct(FastIndexedSeq(), required = true)

    val leafEType = EBaseStruct(FastIndexedSeq(
      EField("first_idx", EInt64Required, 0),
      EField("keys", EArray(EBaseStruct(FastIndexedSeq(
        EField("key", keyEType, 0),
        EField("offset", EInt64Required, 1),
        EField("annotation", annotationEType, 2)
      ), required = true), required = true), 1)
    ))
    val leafVType = TStruct(FastIndexedSeq(
      Field("first_idx", TInt64, 0),
      Field("keys", TArray(TStruct(FastIndexedSeq(
        Field("key", keyVType, 0),
        Field("offset", TInt64, 1),
        Field("annotation", annotationVType, 2)
      ))), 1)))

    val internalNodeEType = EBaseStruct(FastIndexedSeq(
      EField("children", EArray(EBaseStruct(FastIndexedSeq(
        EField("index_file_offset", EInt64Required, 0),
        EField("first_idx", EInt64Required, 1),
        EField("first_key", keyEType, 2),
        EField("first_record_offset", EInt64Required, 3),
        EField("first_annotation", annotationEType, 4)
      ), required = true), required = true), 0)
    ))

    val internalNodeVType = TStruct(FastIndexedSeq(
      Field("children", TArray(TStruct(FastIndexedSeq(
        Field("index_file_offset", TInt64, 0),
        Field("first_idx", TInt64, 1),
        Field("first_key", keyVType, 2),
        Field("first_record_offset", TInt64, 3),
        Field("first_annotation", annotationVType, 4)
      ))), 0)
    ))

    (TypedCodecSpec(leafEType, leafVType, bufferSpec), (TypedCodecSpec(internalNodeEType, internalNodeVType, bufferSpec)))
  }
}

case class BgenSettings(
  nSamples: Int,
  requestedType: TableType,
  rgBc: Option[BroadcastValue[ReferenceGenome]],
  indexAnnotationType: Type
) {
  require(PruneDeadFields.isSupertype(requestedType, MatrixBGENReader.fullMatrixType(rg).canonicalTableType))

  val entryType: Option[TStruct] = requestedType.rowType
    .fieldOption(MatrixType.entriesIdentifier)
    .map(f => f.typ.asInstanceOf[TArray].elementType.asInstanceOf[TStruct])

  val rowPType: PStruct = PStruct(
    Array(
      "locus" -> PCanonicalLocus.schemaFromRG(rg),
      "alleles" -> PArray(PString()),
      "rsid" -> PString(),
      "varid" -> PString(),
      "offset" -> PInt64(),
      "file_idx" -> PInt32(),
      MatrixType.entriesIdentifier -> PArray(PStruct(
        Array(
          "GT" -> PCall(),
          "GP" -> PArray(PFloat64Required, required = true),
          "dosage" -> PFloat64Required
        ).filter { case (name, _) => entryType.exists(t => t.hasField(name))
        }: _*
      )))
      .filter { case (name, _) => requestedType.rowType.hasField(name) }: _*)

  assert(rowPType.virtualType == requestedType.rowType, s"${ rowPType.virtualType.parsableString() } vs ${ requestedType.rowType.parsableString() }")

  def hasField(name: String): Boolean = requestedType.rowType.hasField(name)

  def hasEntryField(name: String): Boolean = entryType.exists(t => t.hasField(name))

  def rg: Option[ReferenceGenome] = rgBc.map(_.value)
}

object BgenRDD {
  def apply(
    sc: SparkContext,
    partitions: Array[Partition],
    settings: BgenSettings,
    keys: RDD[Row]
  ): ContextRDD[RegionValue] = {
    ContextRDD(new BgenRDD(sc, partitions, settings, keys))
  }

  private[bgen] def decompress(
    input: Array[Byte],
    uncompressedSize: Int
  ): Array[Byte] = is.hail.utils.decompress(input, uncompressedSize)
}

private class BgenRDD(
  sc: SparkContext,
  parts: Array[Partition],
  settings: BgenSettings,
  keys: RDD[Row]
) extends RDD[RVDContext => Iterator[RegionValue]](sc, if (keys == null) Nil else Seq(new OneToOneDependency(keys))) {
  private[this] val f = CompileDecoder(settings)
  private[this] val indexBuilder = {
    val (leafCodec, internalNodeCodec) = BgenSettings.indexCodecSpecs(settings.rg)
    val (leafPType: PStruct, leafDec) = leafCodec.buildDecoder(leafCodec.encodedVirtualType)
    val (intPType: PStruct, intDec) = internalNodeCodec.buildDecoder(internalNodeCodec.encodedVirtualType)
    IndexReaderBuilder.withDecoders(leafDec, intDec, BgenSettings.indexKeyType(settings.rg), BgenSettings.indexAnnotationType, leafPType, intPType)
  }

  protected def getPartitions: Array[Partition] = parts

  def compute(split: Partition, context: TaskContext): Iterator[RVDContext => Iterator[RegionValue]] =
    Iterator.single { (ctx: RVDContext) =>
      split match {
        case p: IndexBgenPartition =>
          assert(keys == null)
          new IndexBgenRecordIterator(ctx, p, settings, f(p.partitionIndex, ctx.partitionRegion)).flatten
        case p: LoadBgenPartition =>
          val index: IndexReader = indexBuilder(p.bcFS.value, p.indexPath, 8)
          context.addTaskCompletionListener { (context: TaskContext) =>
            index.close()
          }
          if (keys == null)
            new BgenRecordIteratorWithoutFilter(ctx, p, settings, f(p.partitionIndex, ctx.partitionRegion), index).flatten
          else {
            val keyIterator = keys.iterator(p.filterPartition, context)
            new BgenRecordIteratorWithFilter(ctx, p, settings, f(p.partitionIndex, ctx.partitionRegion), index, keyIterator).flatten
          }
      }
    }
}

private class IndexBgenRecordIterator(
  ctx: RVDContext,
  p: IndexBgenPartition,
  settings: BgenSettings,
  f: AsmFunction4[Region, BgenPartition, HadoopFSDataBinaryReader, BgenSettings, Long]
) extends Iterator[Option[RegionValue]] {
  private[this] val bfis = p.makeInputStream
  bfis.seek(p.startByteOffset)

  private[this] val rv = RegionValue(ctx.region)

  def next(): Option[RegionValue] = {
    val maybeOffset = f(ctx.region, p, bfis, settings)
    if (maybeOffset == -1) {
      None
    } else {
      rv.setOffset(maybeOffset)
      Some(rv)
    }
  }

  def hasNext: Boolean =
    bfis.getPosition < p.endByteOffset
}

private class BgenRecordIteratorWithoutFilter(
  ctx: RVDContext,
  p: LoadBgenPartition,
  settings: BgenSettings,
  f: AsmFunction4[Region, BgenPartition, HadoopFSDataBinaryReader, BgenSettings, Long],
  index: IndexReader
) extends Iterator[Option[RegionValue]] {
  private[this] val bfis = p.makeInputStream
  private[this] val it = index.iterator(p.startIndex, p.endIndex)
  private[this] val rv = RegionValue(ctx.region)

  def next(): Option[RegionValue] = {
    val recordOffset = it.next().recordOffset
    if (recordOffset != bfis.getPosition)
      bfis.seek(recordOffset)

    val maybeOffset = f(ctx.region, p, bfis, settings)
    if (maybeOffset == -1) {
      None
    } else {
      rv.setOffset(maybeOffset)
      Some(rv)
    }
  }

  def hasNext: Boolean =
    it.hasNext
}

private class BgenRecordIteratorWithFilter(
  ctx: RVDContext,
  p: LoadBgenPartition,
  settings: BgenSettings,
  f: AsmFunction4[Region, BgenPartition, HadoopFSDataBinaryReader, BgenSettings, Long],
  index: IndexReader,
  keys: Iterator[Annotation]
) extends Iterator[Option[RegionValue]] {
  private[this] val bfis = p.makeInputStream
  private[this] val rv = RegionValue(ctx.region)
  private[this] val it = index.iterator(p.startIndex, p.endIndex)
  private[this] var isEnd = false
  private[this] var current: LeafChild = _
  private[this] var key: Annotation = _
  private[this] val ordering = index.keyType.ordering

  def next(): Option[RegionValue] = {
    val recordOffset = current.recordOffset
    if (recordOffset != bfis.getPosition)
      bfis.seek(recordOffset)

    val maybeOffset = f(ctx.region, p, bfis, settings)
    val result = if (maybeOffset == -1) {
      None
    } else {
      rv.setOffset(maybeOffset)
      Some(rv)
    }
    current = null
    result
  }

  def hasNext: Boolean = {
    if (isEnd)
      return false

    if ((current == null && !it.hasNext) || (key == null && !keys.hasNext)) {
      isEnd = true
      return false
    }

    if (key == null)
      key = keys.next()

    if (current == null)
      current = it.next()

    while (current != null && key != null && !ordering.equiv(current.key, key)) {
      if (ordering.lt(key, current.key))
        key = if (keys.hasNext) keys.next() else null
      else {
        it.seek(key)
        current = if (it.hasNext) it.next() else null
      }
    }

    current != null && key != null
  }
}
