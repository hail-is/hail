package is.hail.expr.ir.agg

import is.hail.annotations.Region
import is.hail.asm4s._
import is.hail.asm4s.implicits.valueToRichCodeOutputBuffer
import is.hail.backend.ExecuteContext
import is.hail.collection.compat.immutable.ArraySeq
import is.hail.expr.ir._
import is.hail.io.{BufferSpec, InputBuffer, OutputBuffer, TypedCodecSpec}
import is.hail.io.index.StagedIndexWriter
import is.hail.types.physical._
import is.hail.types.physical.stypes.{EmitType, SValue}
import is.hail.types.physical.stypes.concrete.{SJavaString, SJavaStringValue, SStackStruct}
import is.hail.types.physical.stypes.interfaces.primitive
import is.hail.types.virtual._
import is.hail.utils.fatal

class StreamSplitWriterState(override val kb: EmitClassBuilder[_], indexKey: PStruct)
    extends AggregatorState {
  val outbRows: Settable[OutputBuffer] = kb.genFieldThisRef[OutputBuffer]()
  val outbEntries: Settable[OutputBuffer] = kb.genFieldThisRef[OutputBuffer]()
  val part: Settable[String] = kb.genFieldThisRef[String]()
  private val iAnnotationType = PCanonicalStruct(required = true, "entries_offset" -> PInt64())

  val indexWriter = {
    val branchingFactor =
      Option(kb.ctx.getFlag("index_branching_factor")).map(_.toInt).getOrElse(4096)
    StagedIndexWriter.withDefaults(
      indexKey,
      kb,
      annotationType = iAnnotationType,
      branchingFactor = branchingFactor,
    )
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

  private[agg] def addToIndex(cb: EmitCodeBuilder, codeRow: SValue): Unit = {
    val row = codeRow.asBaseStruct
    val rowKey = row.subset(indexKey.fieldNames: _*)
    indexWriter.add(
      cb,
      IEmitCode.present(cb, rowKey),
      outbRows.invoke[Long]("indexOffset"),
      IEmitCode.present(
        cb,
        SStackStruct.constructFromArgs(
          cb,
          /*region=*/ null,
          iAnnotationType.virtualType,
          EmitCode.present(
            cb.emb,
            primitive(cb.memoize(outbEntries.invoke[Long]("indexOffset"))),
          ),
        ),
      ),
    )
  }
}

// FIXME fullRowType not needed, pass rows/entries separately
class StreamSplitWriterAggregator(
  fullRowType: TStruct,
  rowSpec: TypedCodecSpec,
  entrySpec: TypedCodecSpec,
) extends StagedAggregator {
  type State = StreamSplitWriterState

  val initOpTypes: IndexedSeq[Type] = ArraySeq(
    TString, // partfile base name,
    TString, // rows path root _with_  'directory' separator
    TString, // entries path root _with_  'directory' separator
    TString, // index path root _with_ 'directory' separator
  )

  // FIXME separate args for rows/entries.
  val seqOpTypes: IndexedSeq[Type] = ArraySeq(fullRowType)
  val resultEmitType = EmitType(SJavaString, true)

  override protected def _initOp(cb: EmitCodeBuilder, state: State, init: Array[EmitCode]): Unit = {
    val Array(partEC, rowRootEC, entryRootEC, ixrootEC) = init

    val part = partEC.toI(cb).getOrFatal(cb, "part name cannot be missing").asString.loadString(cb)
    val rowRoot =
      rowRootEC.toI(cb).getOrFatal(cb, "rows path cannot be missing").asString.loadString(cb)
    val entryRoot =
      entryRootEC.toI(cb).getOrFatal(cb, "entries path cannot be missing").asString.loadString(cb)
    val rowOs = cb.emb.createUnbuffered(rowRoot.concat(part))
    val entryOs = cb.emb.createUnbuffered(entryRoot.concat(part))

    val indexRoot =
      ixrootEC.toI(cb).getOrFatal(cb, "index path cannot be missing").asString.loadString(cb)
    val indexPath = cb.memoize(indexRoot.concat(part).concat(".idx"))
    state.indexWriter.init(cb, indexPath, cb.memoize(cb.emb.getObject[Map[String, Any]](Map.empty)))

    cb.assign(state.part, part)
    cb.assign(state.outbRows, rowSpec.buildCodeOutputBuffer(rowOs))
    cb.assign(state.outbEntries, entrySpec.buildCodeOutputBuffer(entryOs))
  }

  override protected def _seqOp(cb: EmitCodeBuilder, state: State, seq: Array[EmitCode]): Unit = {
    val Array(fullRowEC) = seq
    val fullRow = fullRowEC.toI(cb).getOrFatal(cb, "row cannot be missing")
    val rowsEncoder = rowSpec.encodedType.buildEncoder(fullRow.st, cb.emb.ecb)
    val entriesEncoder = entrySpec.encodedType.buildEncoder(fullRow.st, cb.emb.ecb)

    state.addToIndex(cb, fullRow)
    cb += state.outbRows.writeByte(1.asInstanceOf[Byte])
    cb += state.outbEntries.writeByte(1.asInstanceOf[Byte])
    rowsEncoder.apply(cb, fullRow, state.outbRows)
    entriesEncoder.apply(cb, fullRow, state.outbEntries)
  }

  override protected def _result(cb: EmitCodeBuilder, state: State, region: Value[Region])
    : IEmitCode = {
    cb += state.outbRows.writeByte(0.asInstanceOf[Byte])
    cb += state.outbRows.flush()
    cb += state.outbRows.close()
    cb += state.outbEntries.writeByte(0.asInstanceOf[Byte])
    cb += state.outbEntries.flush()
    cb += state.outbEntries.close()
    state.indexWriter.close(cb)
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
