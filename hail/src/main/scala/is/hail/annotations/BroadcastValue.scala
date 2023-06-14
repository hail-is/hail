package is.hail.annotations

import cats.Functor
import cats.mtl.Ask
import cats.syntax.all.toFunctorOps
import is.hail.Thunk
import is.hail.asm4s.HailClassLoader
import is.hail.backend.{BroadcastValue, ExecuteContext, HailStateManager}
import is.hail.expr.ir.EncodedLiteral
import is.hail.io.{BufferSpec, Decoder, TypedCodecSpec}
import is.hail.types.physical.{PArray, PStruct, PType}
import is.hail.types.virtual.{TBaseStruct, TStruct}
import is.hail.utils.prettyPrint.ArrayOfByteArrayInputStream
import is.hail.utils.{ArrayOfByteArrayOutputStream, forceThunk, formatSpace, log, using}
import org.apache.spark.sql.Row

import java.io.InputStream
import scala.language.higherKinds

case class SerializableRegionValue(
  encodedValue: Array[Array[Byte]], t: PType,
  makeDecoder: (InputStream, HailClassLoader) => Decoder
) {
  def readRegionValue(r: Region, theHailClassLoader: HailClassLoader): Long = {
    val dec = makeDecoder(new ArrayOfByteArrayInputStream(encodedValue), theHailClassLoader)
    val offset = dec.readRegionValue(r)
    dec.close()
    offset
  }
}

object BroadcastRow {
  def empty[M[_]](implicit M: Ask[M, ExecuteContext]): M[BroadcastRow] =
    apply(Row(), TStruct.empty)

  def apply[M[_]](value: Row, t: TBaseStruct)(implicit M: Ask[M, ExecuteContext]): M[BroadcastRow] =
    M.reader { ctx =>
      val pType = PType.literalPType(t, value).asInstanceOf[PStruct]
      val offset = pType.unstagedStoreJavaObject(ctx.stateManager, value, ctx.r)
      BroadcastRow(ctx.stateManager, RegionValue(ctx.r, offset), pType)
    }
}

trait BroadcastRegionValue {

  val stateManager: HailStateManager
  def value: RegionValue

  val t: PType

  lazy val encoding = TypedCodecSpec(t, BufferSpec.wireSpec)

  def encodeToByteArrays[M[_]](implicit M: Ask[M, ExecuteContext]): M[Array[Array[Byte]]] =
    M.reader { ctx =>
      val makeEnc = encoding.buildEncoder(ctx, t)
      val baos = new ArrayOfByteArrayOutputStream()
      using(makeEnc(baos, ctx.theHailClassLoader)) { enc =>
        enc.writeRegionValue(value.offset)
        enc.flush()
      }
      baos.toByteArrays()
    }

  private type HasExecuteContext[F[_]] = Ask[F, ExecuteContext]
  private[this] val broadcasted: Thunk[HasExecuteContext, BroadcastValue[SerializableRegionValue]] =
    new Thunk[HasExecuteContext, BroadcastValue[SerializableRegionValue]] {
      override protected def run[M[_]](implicit M: HasExecuteContext[M]): M[BroadcastValue[SerializableRegionValue]] =
        M.applicative.map2(encodeToByteArrays, M.ask) { (arrays, ctx) =>
          val totalSize = arrays.map(_.length).sum
          log.info(s"BroadcastRegionValue.broadcast: broadcasting ${arrays.length} byte arrays of total size $totalSize (${formatSpace(totalSize)}")
          val (decodedPType, makeDec) = encoding.buildDecoder(ctx, t.virtualType)
          assert(decodedPType.virtualType == t.virtualType)
          val srv = SerializableRegionValue(arrays, decodedPType, makeDec)
          ctx.backend.broadcast(srv)
        }
    }

  def broadcast[M[_]](implicit M: HasExecuteContext[M]): M[BroadcastValue[SerializableRegionValue]] =
    forceThunk(broadcasted)(M.applicative, M)

  def javaValue: Any

  def safeJavaValue: Any

  override def equals(obj: Any): Boolean = obj match {
    case b: BroadcastRegionValue =>
      t == b.t && (stateManager eq b.stateManager) && t.unsafeOrdering(stateManager).compare(value, b.value) == 0
    case _ =>
      false
  }

  override def hashCode(): Int = javaValue.hashCode()
}

case class BroadcastRow(stateManager: HailStateManager, value: RegionValue, t: PStruct)
  extends BroadcastRegionValue {
  def javaValue: UnsafeRow = UnsafeRow.readBaseStruct(t, value.region, value.offset)

  def safeJavaValue: Row = SafeRow.read(t, value).asInstanceOf[Row]

  def cast(newT: PStruct): BroadcastRow = {
    assert(t.virtualType == newT.virtualType)
    if (t == newT)
      return this

    BroadcastRow(
      stateManager,
      RegionValue(value.region, newT.copyFromAddress(stateManager, value.region, t, value.offset, deepCopy = false)),
      newT
    )
  }

  def toEncodedLiteral[M[_]: Functor](implicit M: Ask[M, ExecuteContext]): M[EncodedLiteral] =
    for {bytes <- encodeToByteArrays} yield EncodedLiteral(encoding, bytes)
}

case class BroadcastIndexedSeq(
                                stateManager: HailStateManager,
                                value: RegionValue,
                                t: PArray
) extends BroadcastRegionValue {

  def safeJavaValue: IndexedSeq[Row] = SafeRow.read(t, value).asInstanceOf[IndexedSeq[Row]]

  def javaValue: UnsafeIndexedSeq = new UnsafeIndexedSeq(t, value.region, value.offset)

  def cast(newT: PArray): BroadcastIndexedSeq = {
    assert(t.virtualType == newT.virtualType)
    if (t == newT)
      return this

    BroadcastIndexedSeq(stateManager,
      RegionValue(value.region, newT.copyFromAddress(stateManager, value.region, t, value.offset, deepCopy = false)),
      newT
    )
  }
}
