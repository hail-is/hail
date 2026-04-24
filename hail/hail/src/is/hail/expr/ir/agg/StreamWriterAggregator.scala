package is.hail.expr.ir.agg

import is.hail.annotations.Region
import is.hail.asm4s._
import is.hail.asm4s.implicits.{valueToRichCodeOutputBuffer, valueToRichCodeRegion}
import is.hail.backend.ExecuteContext
import is.hail.collection.compat.immutable.ArraySeq
import is.hail.expr.ir._
import is.hail.io.{BufferSpec, InputBuffer, OutputBuffer, TypedCodecSpec}
import is.hail.types.physical._
import is.hail.types.physical.stypes.EmitType
import is.hail.types.physical.stypes.concrete.{SJavaString, SJavaStringValue}
import is.hail.types.virtual._
import is.hail.utils.fatal

class StreamWriterState(override val kb: EmitClassBuilder[_]) extends AggregatorState {
  val outb: Settable[OutputBuffer] = kb.genFieldThisRef[OutputBuffer]()
  val path: Settable[String] = kb.genFieldThisRef[String]()

  override def storageType = PCanonicalStringRequired

  override def createState(cb: EmitCodeBuilder): Unit = {}

  override def newState(cb: EmitCodeBuilder, off: Value[Long]): Unit = {}

  override def load(
    cb: EmitCodeBuilder,
    regionLoader: (EmitCodeBuilder, Value[Region]) => Unit,
    src: Value[Long],
  ): Unit = fatal("makes no sense to load a writer's state")

  override def store(
    cb: EmitCodeBuilder,
    regionStorer: (EmitCodeBuilder, Value[Region]) => Unit,
    dest: Value[Long],
  ): Unit = {}

  override def copyFrom(cb: EmitCodeBuilder, src: Value[Long]): Unit = ???

  override def serialize(codec: BufferSpec): (EmitCodeBuilder, Value[OutputBuffer]) => Unit = ???

  override def deserialize(codec: BufferSpec): (EmitCodeBuilder, Value[InputBuffer]) => Unit = ???
}

class StreamWriterAggregator(spec: TypedCodecSpec) extends StagedAggregator {
  type State = StreamWriterState

  val initOpTypes: IndexedSeq[Type] = ArraySeq(TString)
  val seqOpTypes: IndexedSeq[Type] = ArraySeq(spec.encodedVirtualType)
  val resultEmitType = EmitType(SJavaString, true)

  override protected def _initOp(cb: EmitCodeBuilder, state: State, init: Array[EmitCode]): Unit = {
    val Array(pathEC) = init
    val path = pathEC.toI(cb).getOrFatal(cb, "path cannot be missing").asString.loadString(cb)
    val os = cb.emb.createUnbuffered(path)
    cb.assign(state.path, path)
    cb.assign(state.outb, spec.buildCodeOutputBuffer(os))
  }

  override protected def _seqOp(cb: EmitCodeBuilder, state: State, seq: Array[EmitCode]): Unit = {
    val Array(rowEC) = seq
    val row = rowEC.toI(cb).getOrFatal(cb, "row cannot be missing")
    val encoder = spec.encodedType.buildEncoder(row.st, cb.emb.ecb)
    cb += state.outb.writeByte(1.asInstanceOf[Byte])
    encoder.apply(cb, row, state.outb)
  }

  override protected def _result(cb: EmitCodeBuilder, state: State, region: Value[Region])
    : IEmitCode = {
    cb += state.outb.writeByte(0.asInstanceOf[Byte])
    cb += state.outb.flush()
    cb += state.outb.close()
    IEmitCode.present(cb, new SJavaStringValue(state.path))
  }

  override protected def _combOp(
    ctx: ExecuteContext,
    cb: EmitCodeBuilder,
    region: Value[Region],
    state: State,
    other: State,
  ): Unit = fatal("makes no sense to call a combop on the writer")
}
