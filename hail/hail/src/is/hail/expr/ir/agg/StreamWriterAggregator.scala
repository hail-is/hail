package is.hail.expr.ir.agg

import is.hail.annotations.Region
import is.hail.asm4s._
import is.hail.asm4s.implicits.{valueToRichCodeOutputBuffer}
import is.hail.backend.ExecuteContext
import is.hail.collection.compat.immutable.ArraySeq
import is.hail.expr.ir._
import is.hail.io.{BufferSpec, InputBuffer, OutputBuffer, TypedCodecSpec}
import is.hail.io.index.StagedIndexWriter
import is.hail.types.physical._
import is.hail.types.physical.stypes.{EmitType, SValue}
import is.hail.types.physical.stypes.concrete.{SJavaString, SJavaStringValue}
import is.hail.types.virtual._
import is.hail.utils.fatal

class StreamWriterState(override val kb: EmitClassBuilder[_], indexKey: Option[PStruct]) extends AggregatorState {
  val outb: Settable[OutputBuffer] = kb.genFieldThisRef[OutputBuffer]()
  val part: Settable[String] = kb.genFieldThisRef[String]()
  val indexWriter = indexKey.map { key =>
    val branchingFactor = Option(kb.ctx.getFlag("index_branching_factor")).map(_.toInt).getOrElse(4096)
    StagedIndexWriter.withDefaults(key, kb, branchingFactor = branchingFactor)
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

  override def copyFrom(cb: EmitCodeBuilder, src: Value[Long]): Unit = ???

  override def serialize(codec: BufferSpec): (EmitCodeBuilder, Value[OutputBuffer]) => Unit = ???

  override def deserialize(codec: BufferSpec): (EmitCodeBuilder, Value[InputBuffer]) => Unit = ???

  private[agg] def addToIndex(cb: EmitCodeBuilder, codeRow: SValue): Unit = indexWriter.foreach { iw =>
    val row = codeRow.asBaseStruct
    val rowKey = row.subset(indexKey.get.fieldNames: _*)
    iw.add(cb, IEmitCode.present(cb, rowKey), outb.invoke[Long]("indexOffset"),
          IEmitCode.present(cb, PCanonicalStruct().loadCheapSCode(cb, 0L)))
  }
}

class StreamWriterAggregator(spec: TypedCodecSpec, indexed: Boolean) extends StagedAggregator {
  type State = StreamWriterState

  val initOpTypes: IndexedSeq[Type] = ArraySeq(
    TString, // partfile base name
    TString, // path root _with_  'directory' separator
  ) ++ (if (indexed) Some(TString) else None) // if indexed, index root path _with_ 'directory' separator
  val seqOpTypes: IndexedSeq[Type] = ArraySeq(spec.encodedVirtualType)
  val resultEmitType = EmitType(SJavaString, true)

  override protected def _initOp(cb: EmitCodeBuilder, state: State, init: Array[EmitCode]): Unit = {
    val (partEC, rootEC, ixrootEC) = init match {
      case Array(root, part) =>
        require(!indexed)
        (root, part, None)
      case Array(root, part, ixroot) =>
        require(indexed)
        (root, part, Some(ixroot))
    }

    val root = rootEC.toI(cb).getOrFatal(cb, "path cannot be missing").asString.loadString(cb)
    val part = partEC.toI(cb).getOrFatal(cb, "part cannot be missing").asString.loadString(cb)
    val os = cb.emb.createUnbuffered(root.concat(part))

    state.indexWriter.foreach { iw =>
      val root = ixrootEC.get.toI(cb).getOrFatal(cb, "index path cannot be missing").asString.loadString(cb)
      val path = cb.memoize(root.concat(part).concat(".idx"))
      iw.init(cb, path, cb.memoize(cb.emb.getObject[Map[String, Any]](Map.empty)))
    }

    cb.assign(state.part, part)
    cb.assign(state.outb, spec.buildCodeOutputBuffer(os))
  }

  override protected def _seqOp(cb: EmitCodeBuilder, state: State, seq: Array[EmitCode]): Unit = {
    val Array(rowEC) = seq
    val row = rowEC.toI(cb).getOrFatal(cb, "row cannot be missing")
    val encoder = spec.encodedType.buildEncoder(row.st, cb.emb.ecb)

    state.addToIndex(cb, row)
    cb += state.outb.writeByte(1.asInstanceOf[Byte])
    encoder.apply(cb, row, state.outb)
  }


  override protected def _result(cb: EmitCodeBuilder, state: State, region: Value[Region])
    : IEmitCode = {
    cb += state.outb.writeByte(0.asInstanceOf[Byte])
    cb += state.outb.flush()
    cb += state.outb.close()
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
