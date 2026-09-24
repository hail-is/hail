package is.hail.expr.ir.agg

import is.hail.annotations.Region
import is.hail.asm4s._
import is.hail.asm4s.implicits.valueToRichCodeOutputBuffer
import is.hail.backend.ExecuteContext
import is.hail.expr.ir._
import is.hail.io.{BufferSpec, InputBuffer, OutputBuffer, TypedCodecSpec}
import is.hail.io.index.{IndexType, StagedIndexWriter}
import is.hail.io.index.IndexWriter.DEFAULT_BRANCHING_FACTOR
import is.hail.types._
import is.hail.types.physical._
import is.hail.types.physical.stypes.{EmitType, SValue}
import is.hail.types.physical.stypes.concrete.{SJavaString, SJavaStringValue, SStackStruct}
import is.hail.types.physical.stypes.interfaces.primitive
import is.hail.types.virtual._
import is.hail.utils.fatal

import scala.collection.immutable.ArraySeq

class StreamWriterState(override val kb: EmitClassBuilder[_], indexType: Option[IndexType])
    extends AggregatorState {
  val buffers: Settable[Array[OutputBuffer]] = kb.genFieldThisRef[Array[OutputBuffer]]()
  val part: Settable[String] = kb.genFieldThisRef[String]()

  val indexWriter = indexType.map { typ =>
    val branchingFactor =
      Option(kb.ctx.getFlag("index_branching_factor")).map(_.toInt).getOrElse(
        DEFAULT_BRANCHING_FACTOR
      )
    new StagedIndexWriter(branchingFactor, typ, kb)
  }

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

  override def copyFrom(cb: EmitCodeBuilder, src: Value[Long]): Unit =
    fatal("writer cannot be copied from address")

  override def serialize(codec: BufferSpec): (EmitCodeBuilder, Value[OutputBuffer]) => Unit =
    fatal("writer cannot be serialized")

  override def deserialize(codec: BufferSpec): (EmitCodeBuilder, Value[InputBuffer]) => Unit =
    fatal("writer cannot be deserialized")

  private[agg] def addToIndex(cb: EmitCodeBuilder, codeRow: SValue): Unit =
    indexWriter.foreach { iw =>
      val row = codeRow.asBaseStruct
      val keyType = tcoerce[PStruct](indexType.get.key)
      val annotationType = tcoerce[PStruct](indexType.get.annotation)
      val rowKey = row.subset(keyType.fieldNames: _*)
      val args = (1 to annotationType.size).map(i =>
        EmitCode.present(cb.emb, primitive(cb.memoize(buffers(i).invoke[Long]("indexOffset"))))
      )
      val annotation = SStackStruct.constructFromArgs(
        cb,
        /*region=*/ null,
        annotationType.virtualType,
        args: _*
      )
      iw.add(
        cb,
        IEmitCode.present(cb, rowKey),
        buffers(0).invoke[Long]("indexOffset"),
        IEmitCode.present(cb, annotation),
      )
    }
}

class StreamWriterAggregator(specs: IndexedSeq[TypedCodecSpec], indexed: Boolean)
    extends StagedAggregator {
  require(!specs.isEmpty)
  type State = StreamWriterState

  /* part file name, index directory root (if defined), output roots */
  val initOpTypes: IndexedSeq[Type] =
    ArraySeq.fill(1 + (if (indexed) 1 else 0) + specs.length)(TString)

  val seqOpTypes: IndexedSeq[Type] = specs.map(_.encodedVirtualType)
  val resultEmitType = EmitType(SJavaString, true)

  override protected def _initOp(cb: EmitCodeBuilder, state: State, init: Array[EmitCode]): Unit = {
    val (partEC, ixrootEC, rootECs) = init match {
      case Array(part, roots @ _*) if !indexed =>
        (part, None, roots)
      case Array(part, ixroot, roots @ _*) =>
        require(indexed)
        (part, Some(ixroot), roots)
    }

    val part = partEC.toI(cb).getOrFatal(cb, "part cannot be missing").asString.loadString(cb)
    cb.assign(state.part, part)
    cb.assign(state.buffers, Code.newArray[OutputBuffer](specs.length))

    state.indexWriter.foreach { iw =>
      val root =
        ixrootEC.get.toI(cb).getOrFatal(cb, "index path cannot be missing").asString.loadString(cb)
      val path = cb.memoize(root.concat(part).concat(".idx"))
      iw.init(cb, path, cb.memoize(cb.emb.getObject[Map[String, Any]](Map.empty)))
    }

    rootECs.zipWithIndex.foreach { case (rootEC, i) =>
      val root = rootEC.toI(cb).getOrFatal(cb, "path cannot be missing").asString.loadString(cb)
      val os = cb.emb.createUnbuffered(root.concat(part))
      cb += state.buffers.update(i, specs(i).buildCodeOutputBuffer(os))
    }
  }

  override protected def _seqOp(cb: EmitCodeBuilder, state: State, seq: Array[EmitCode]): Unit = {
    assert(specs.length == seq.length)
    val pairs = (specs zip seq).map { case (spec, valueEC) =>
      val value = valueEC.toI(cb).getOrFatal(cb, "value cannot be missing")
      spec.encodedType.buildEncoder(value.st, cb.emb.ecb) -> value
    }

    val ob = cb.newLocal[OutputBuffer]("scratchbuf")
    state.addToIndex(cb, pairs.head._2)
    pairs.zipWithIndex.foreach { case ((encoder, value), i) =>
      cb.assign(ob, state.buffers(i))
      cb += ob.writeByte(1.asInstanceOf[Byte])
      encoder.apply(cb, value, ob)
    }
  }

  override protected def _result(cb: EmitCodeBuilder, state: State, region: Value[Region])
    : IEmitCode = {
    val i = cb.newLocal[Int]("i")
    cb.for_(
      cb.assign(i, 0),
      i < state.buffers.length(),
      cb.assign(i, i + 1), {
        val ob = cb.memoize[OutputBuffer](state.buffers(i))
        cb += ob.writeByte(0.asInstanceOf[Byte])
        cb += ob.flush()
        cb += ob.close()
      },
    )
    state.indexWriter.foreach(_.close(cb))
    IEmitCode.present(cb, new SJavaStringValue(state.part))
  }

  override protected def _combOp(
    ctx: ExecuteContext,
    cb: EmitCodeBuilder,
    region: Value[Region],
    state: State,
    other: State,
  ): Unit = fatal("makes no sense to call a combop on the writer")
}
