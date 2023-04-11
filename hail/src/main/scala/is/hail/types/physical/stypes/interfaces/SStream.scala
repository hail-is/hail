package is.hail.types.physical.stypes.interfaces

import is.hail.annotations.Region
import is.hail.asm4s._
import is.hail.expr.ir.{EmitCode, EmitCodeBuilder, EmitMethodBuilder, IEmitCode}
import is.hail.expr.ir.streams.StreamProducer
import is.hail.types.physical.PType
import is.hail.types.physical.stypes._
import is.hail.types.virtual.{TStream, Type}
import is.hail.types.{RIterable, TypeWithRequiredness}
import is.hail.utils.FastIndexedSeq

trait MissingnessAsMethod {
  def isMissing: Boolean
}
trait NoBoxLongIterator {
  def init(partitionRegion: Region, elementRegion: Region)

  // after next() has been called, if eos is true, stream has ended
  // (and value returned by next() is garbage)
  def eos: Boolean

  def next(): Long  // 0L represents missing value

  def close(): Unit
}

object SStream {
  def apply(elementEmitType: EmitType): SimpleSStream = SimpleSStream(elementEmitType)
}

final case class SimpleSStream(elementEmitType: EmitType) extends SStream {
  override def settableTupleTypes(): IndexedSeq[TypeInfo[_]] = throw new NotImplementedError()
  override def fromSettables(settables: IndexedSeq[Settable[_]]): SSettable = throw new NotImplementedError()

  override def fromValues(values: IndexedSeq[Value[_]]): SValue = throw new NotImplementedError()
}

final case class SStreamIteratorLong(elementRequired: Boolean, elementPType: PType, requiresMemoryManagement: Boolean) extends SStream {
  override def settableTupleTypes(): IndexedSeq[TypeInfo[_]] = IndexedSeq(classInfo[NoBoxLongIterator])
  override def fromSettables(settables: IndexedSeq[Settable[_]]): SSettable = new SStreamConcreteSettable(this, coerce[NoBoxLongIterator](settables(0)))

  override def fromValues(values: IndexedSeq[Value[_]]): SValue = new SStreamConcrete(this, coerce[NoBoxLongIterator](values(0)))

  override val elementEmitType: EmitType = EmitType(elementPType.sType, elementRequired)
}

sealed trait SStream extends SType {

  def elementEmitType: EmitType

  def elementType: SType = elementEmitType.st

  override def _coerceOrCopy(cb: EmitCodeBuilder, region: Value[Region], value: SValue, deepCopy: Boolean): SValue = {
    if (deepCopy) throw new NotImplementedError()

    assert(value.st == this)
    value
  }

  override def storageType(): PType = throw new NotImplementedError()

  override def copiedType: SType = throw new NotImplementedError()

  override def containsPointers: Boolean = throw new NotImplementedError()

  override def virtualType: Type = TStream(elementType.virtualType)

  override def castRename(t: Type): SType = throw new UnsupportedOperationException("rename on stream")

  override def _typeWithRequiredness: TypeWithRequiredness = RIterable(elementEmitType.typeWithRequiredness.r)
}

object SStreamValue {
  def apply(producer: StreamProducer): SStreamValue = SStreamControlFlow(SStream(producer.element.emitType), producer)
}


trait SStreamValue extends SUnrealizableValue {
  def st: SStream

  def getProducer(mb: EmitMethodBuilder[_]): StreamProducer

  def defineUnusedLabels(mb: EmitMethodBuilder[_]): Unit
}

class SStreamConcrete(val st: SStreamIteratorLong, val it: Value[NoBoxLongIterator]) extends SStreamValue {

  lazy val valueTuple: IndexedSeq[Value[_]] = FastIndexedSeq(it)

  override def getProducer(mb: EmitMethodBuilder[_]): StreamProducer = {

    new StreamProducer {
      override def method: EmitMethodBuilder[_] = mb
      val elRegion = mb.genFieldThisRef[Region]("stream_iter_elregion")
      val next = mb.genFieldThisRef[Long]("stream_iter_next")
      override val length: Option[EmitCodeBuilder => Code[Int]] = None

      override def initialize(cb: EmitCodeBuilder, outerRegion: Value[Region]): Unit = {
        cb += it.invoke[Region, Region, Unit]("init", outerRegion, elementRegion)
      }

      override val elementRegion: Settable[Region] = elRegion
      override val requiresMemoryManagementPerElement: Boolean = st.requiresMemoryManagement
      override val LproduceElement: CodeLabel = mb.defineAndImplementLabel { cb =>
        cb.assign(next, it.invoke[Long]("next"))
        cb.ifx(it.invoke[Boolean]("eos"), cb.goto(LendOfStream))
        cb.goto(LproduceElementDone)
      }
      override val element: EmitCode = {

        EmitCode.fromI(mb) { cb =>
          IEmitCode(cb, if (st.elementRequired) const(false) else (next cne 0L), st.elementPType.loadCheapSCode(cb, next))
        }
      }

      override def close(cb: EmitCodeBuilder): Unit = it.invoke[Unit]("close")
    }
  }

  override def defineUnusedLabels(mb: EmitMethodBuilder[_]): Unit = () // nothing to do
}

class SStreamConcreteSettable(st: SStreamIteratorLong, val itSettable: Settable[NoBoxLongIterator])
  extends SStreamConcrete(st, itSettable) with SSettable {
  override def store(cb: EmitCodeBuilder, v: SValue): Unit = {
    assert(v.st == st)
    cb.assign(itSettable, v.asInstanceOf[SStreamConcrete].it)
  }

  override def settableTuple(): IndexedSeq[Settable[_]] = FastIndexedSeq(itSettable)
}

case class SStreamControlFlow(st: SimpleSStream, producer: StreamProducer) extends SStreamValue {
  override def getProducer(mb: EmitMethodBuilder[_]): StreamProducer = {
    if (mb != producer.method)
      throw new RuntimeException("stream used in method different from where it was generated -- " +
        s"generated in ${ producer.method.mb.methodName }, used in ${ mb.mb.methodName }")
    producer
  }

  def valueTuple: IndexedSeq[Value[_]] = throw new NotImplementedError()

  override def defineUnusedLabels(mb: EmitMethodBuilder[_]): Unit = {
    (producer.LendOfStream.isImplemented, producer.LproduceElementDone.isImplemented) match {
      case (true, true) =>
      case (false, false) =>

        EmitCodeBuilder.scopedVoid(mb) { cb =>
          cb.define(producer.LendOfStream)
          cb.define(producer.LproduceElementDone)
          cb._fatal("unreachable")
        }

      case (eos, ped) => throw new RuntimeException(s"unrealizable value unused asymmetrically: eos=$eos, ped=$ped")
    }
    producer.element.pv match {
      case ss: SStreamValue => ss.defineUnusedLabels(mb)
      case _ =>
    }
  }

}
