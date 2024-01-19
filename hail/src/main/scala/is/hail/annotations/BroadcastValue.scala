package is.hail.annotations

import is.hail.asm4s.HailClassLoader
import is.hail.backend.{BroadcastValue, ExecuteContext}
import is.hail.expr.ir.EncodedLiteral
import is.hail.io.{BufferSpec, Decoder, TypedCodecSpec}
import is.hail.types.physical.{PArray, PStruct, PType}
import is.hail.types.virtual.{TBaseStruct, TStruct}
import is.hail.utils.{formatSpace, log, ArrayOfByteArrayOutputStream}
import is.hail.utils.prettyPrint.ArrayOfByteArrayInputStream

import java.io.InputStream

import org.apache.spark.sql.Row

case class SerializableRegionValue(
  encodedValue: Array[Array[Byte]],
  t: PType,
  makeDecoder: (InputStream, HailClassLoader) => Decoder,
) {
  def readRegionValue(r: Region, theHailClassLoader: HailClassLoader): Long = {
    val dec = makeDecoder(new ArrayOfByteArrayInputStream(encodedValue), theHailClassLoader)
    val offset = dec.readRegionValue(r)
    dec.close()
    offset
  }
}

object BroadcastRow {
  def empty(ctx: ExecuteContext): BroadcastRow = apply(ctx, Row(), TStruct.empty)

  def apply(ctx: ExecuteContext, value: Row, t: TBaseStruct): BroadcastRow = {
    val pType = PType.literalPType(t, value).asInstanceOf[PStruct]
    val offset = pType.unstagedStoreJavaObject(ctx.stateManager, value, ctx.r)
    BroadcastRow(ctx, RegionValue(ctx.r, offset), pType)
  }
}

trait BroadcastRegionValue {
  def ctx: ExecuteContext

  def value: RegionValue

  val t: PType

  lazy val encoding = TypedCodecSpec(t, BufferSpec.wireSpec)

  lazy val (decodedPType, makeDec) = {
    val (pt, md) = encoding.buildDecoder(ctx, t.virtualType)
    assert(pt.virtualType == t.virtualType)
    (pt, md)
  }

  def encodeToByteArrays(theHailClassLoader: HailClassLoader): Array[Array[Byte]] = {
    val makeEnc = encoding.buildEncoder(ctx, t)

    val baos = new ArrayOfByteArrayOutputStream()

    val enc = makeEnc(baos, theHailClassLoader)
    enc.writeRegionValue(value.offset)
    enc.flush()
    enc.close()

    baos.toByteArrays()
  }

  @volatile private[this] var broadcasted: BroadcastValue[SerializableRegionValue] = null

  def broadcast(theHailClassLoader: HailClassLoader): BroadcastValue[SerializableRegionValue] = {
    if (broadcasted == null) {
      this.synchronized {
        if (broadcasted == null) {
          val arrays = encodeToByteArrays(theHailClassLoader)
          val totalSize = arrays.map(_.length).sum
          log.info(
            s"BroadcastRegionValue.broadcast: broadcasting ${arrays.length} byte arrays of total size $totalSize (${formatSpace(totalSize)}"
          )
          val srv = SerializableRegionValue(arrays, decodedPType, makeDec)
          broadcasted = ctx.backend.broadcast(srv)
        }
      }
    }
    broadcasted
  }

  def javaValue: Any

  def safeJavaValue: Any

  override def equals(obj: Any): Boolean = obj match {
    case b: BroadcastRegionValue =>
      t == b.t && (ctx eq b.ctx) && t.unsafeOrdering(ctx.stateManager).compare(value, b.value) == 0
    case _ => false
  }

  override def hashCode(): Int = javaValue.hashCode()
}

case class BroadcastRow(ctx: ExecuteContext, value: RegionValue, t: PStruct)
    extends BroadcastRegionValue {

  def javaValue: UnsafeRow = UnsafeRow.readBaseStruct(t, value.region, value.offset)

  def safeJavaValue: Row = SafeRow.read(t, value).asInstanceOf[Row]

  def cast(newT: PStruct): BroadcastRow = {
    assert(t.virtualType == newT.virtualType)
    if (t == newT)
      return this

    BroadcastRow(
      ctx,
      RegionValue(
        value.region,
        newT.copyFromAddress(ctx.stateManager, value.region, t, value.offset, deepCopy = false),
      ),
      newT,
    )
  }

  def toEncodedLiteral(theHailClassLoader: HailClassLoader): EncodedLiteral =
    EncodedLiteral(encoding, encodeToByteArrays(theHailClassLoader))
}

case class BroadcastIndexedSeq(
  ctx: ExecuteContext,
  value: RegionValue,
  t: PArray,
) extends BroadcastRegionValue {

  def safeJavaValue: IndexedSeq[Row] = SafeRow.read(t, value).asInstanceOf[IndexedSeq[Row]]

  def javaValue: UnsafeIndexedSeq = new UnsafeIndexedSeq(t, value.region, value.offset)

  def cast(newT: PArray): BroadcastIndexedSeq = {
    assert(t.virtualType == newT.virtualType)
    if (t == newT)
      return this

    BroadcastIndexedSeq(
      ctx,
      RegionValue(
        value.region,
        newT.copyFromAddress(ctx.stateManager, value.region, t, value.offset, deepCopy = false),
      ),
      newT,
    )
  }
}
