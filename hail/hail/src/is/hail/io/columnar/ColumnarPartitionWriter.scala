package is.hail.io.columnar

import is.hail.annotations.Region
import is.hail.asm4s._
import is.hail.backend.ExecuteContext
import is.hail.collection.FastSeq
import is.hail.expr.ir.{EmitCode, EmitCodeBuilder, EmitValue, IEmitCode}
import is.hail.expr.ir.defs.PartitionWriter
import is.hail.expr.ir.orderings.CodeOrdering
import is.hail.expr.ir.streams.StreamProducer
import is.hail.io.fs.FS
import is.hail.types.arrow.ArrowSchemaConversion
import is.hail.types.{BaseTypeWithRequiredness, RIterable, RStruct, TypeWithRequiredness}
import is.hail.types.physical._
import is.hail.types.physical.stypes.{EmitType, SType, SValue}
import is.hail.types.physical.stypes.concrete.SJavaString
import is.hail.types.physical.stypes.interfaces.{
  SBaseStructValue, SIndexableValue, SIntervalValue, SNDArray, SNDArrayValue,
}
import is.hail.types.physical.stypes.primitives.{SBooleanValue, SInt64Value}
import is.hail.types.virtual._

object ColumnarPartitionWriter {

  /** The low bits of a Row ID, which hold the row's position within its row partition. The bits
    * above them, up to the sign bit, hold the row partition index (spec section 4.2).
    */
  val RowIdPositionBits: Int = 40

  val MaxRowsPerPartition: Long = 1L << RowIdPositionBits

  val MaxPartitions: Int = 1 << (63 - RowIdPositionBits)
}

/** Writes a stream of structs to one Parquet data file of the columnar format.
  *
  * The file holds the fields of `schema`, preceded by a Row ID field named `rowIdField` if given.
  * Row IDs are `(partitionIndex << RowIdPositionBits) | position`.
  *
  * The context is `{path, partitionIndex}`, where `path` is relative to `root`. The result is the
  * file's path, its row count, and the first and last values of the `key` fields, which are missing
  * for an empty file. `distinct` is whether no two consecutive rows share a key; it is false when
  * `key` is empty.
  */
case class ColumnarPartitionWriter(
  root: String,
  schema: PStruct,
  key: IndexedSeq[String],
  rowIdField: Option[String],
) extends PartitionWriter {
  import ColumnarPartitionWriter._

  lazy val arrowSchema: String =
    ColumnarSchema.serialize(ColumnarSchema.fileSchema(schema, rowIdField))

  private lazy val keyType: TStruct = schema.virtualType.typeAfterSelectNames(key)

  private lazy val keyPType: PCanonicalStruct =
    PType.canonical(keyType).asInstanceOf[PCanonicalStruct]

  private lazy val resultPType: PCanonicalStruct = PCanonicalStruct(
    true,
    "path" -> PCanonicalString(true),
    "rowCount" -> PInt64(true),
    "distinct" -> PBoolean(true),
    "firstKey" -> keyPType,
    "lastKey" -> keyPType,
  )

  override def ctxType: Type = TStruct("path" -> TString, "partitionIndex" -> TInt32)

  override def returnType: Type = resultPType.virtualType

  override def unionTypeRequiredness(
    r: TypeWithRequiredness,
    ctxType: TypeWithRequiredness,
    streamType: RIterable,
  ): Unit = {
    def optional(t: BaseTypeWithRequiredness): Unit = {
      t.union(false)
      t.children.foreach(optional)
    }
    r.union(ctxType.required)
    r.union(streamType.required)
    val rs = r.asInstanceOf[RStruct]
    optional(rs.field("firstKey"))
    optional(rs.field("lastKey"))
  }

  override def consumeStream(
    ctx: ExecuteContext,
    cb: EmitCodeBuilder,
    stream: StreamProducer,
    context: EmitCode,
    region: Value[Region],
  ): IEmitCode =
    context.toI(cb).map(cb) { ctxValue =>
      val ctx = ctxValue.asBaseStruct
      val relPath = cb.memoize(ctx.loadField(cb, "path").getOrAssert(cb).asString.loadString(cb))
      val partitionIndex =
        cb.memoize(ctx.loadField(cb, "partitionIndex").getOrAssert(cb).asInt.value)
      val sink = cb.memoize(
        cb.emb.getObject(ColumnarFileSpec(arrowSchema))
          .invoke[FS, String, ColumnarFileSink](
            "open",
            cb.emb.getFS,
            const(s"$root/").concat(relPath),
          )
      )

      val count = cb.newLocal[Long]("count", 0L)
      val keys =
        if (key.isEmpty) new NoPartitionKeys(keyPType.sType)
        else new StagedPartitionKeys(cb, region, key, keyPType.sType)

      stream.memoryManagedConsume(region, cb) { cb =>
        val row = stream.element.toI(cb)
          .getOrFatal(cb, "columnar writer: a row is missing")
          .asBaseStruct
        cb.if_(
          count >= MaxRowsPerPartition,
          cb._fatal(s"columnar writer: a row partition has more than 2^$RowIdPositionBits rows"),
        )

        cb += sink.invoke[Unit]("startRecord")
        val offset = rowIdField match {
          case Some(name) =>
            field(cb, sink, name, 0) {
              cb += sink.invoke[Long, Unit](
                "addLong",
                (partitionIndex.toL << RowIdPositionBits) | count,
              )
            }
            1
          case None => 0
        }
        schema.fields.foreach { f =>
          writeField(cb, sink, f.name, f.index + offset, f.typ, row.loadField(cb, f.name), f.name)
        }
        cb += sink.invoke[Unit]("endRecord")

        keys.observe(cb, row, isFirst = count ceq 0L)
        cb.assign(count, count + 1L)
      }

      cb += sink.invoke[Unit]("close")

      val (distinct, firstKey, lastKey) = keys.finish(cb)
      resultPType.constructFromFields(
        cb,
        region,
        FastSeq(
          EmitCode.present(cb.emb, SJavaString.construct(cb, relPath)),
          EmitCode.present(cb.emb, new SInt64Value(count)),
          distinct,
          firstKey,
          lastKey,
        ),
        deepCopy = false,
      )
    }

  /** Emits `body` between the start and end of field `name`, at position `index` in its group. */
  private def field(
    cb: EmitCodeBuilder,
    sink: Value[ColumnarFileSink],
    name: String,
    index: Int,
  )(
    body: => Unit
  ): Unit = {
    cb += sink.invoke[String, Int, Unit]("startField", name, index)
    body
    cb += sink.invoke[String, Int, Unit]("endField", name, index)
  }

  /** Emits `body` between the start and end of a group. */
  private def group(cb: EmitCodeBuilder, sink: Value[ColumnarFileSink])(body: => Unit): Unit = {
    cb += sink.invoke[Unit]("startGroup")
    body
    cb += sink.invoke[Unit]("endGroup")
  }

  private def writeField(
    cb: EmitCodeBuilder,
    sink: Value[ColumnarFileSink],
    name: String,
    index: Int,
    t: PType,
    value: IEmitCode,
    path: String,
  ): Unit =
    value.consume(
      cb,
      if (t.required)
        cb._fatal(s"columnar writer: required field '$path' is missing"),
      v => field(cb, sink, name, index)(writeValue(cb, sink, t, v, path)),
    )

  /** Emits a group holding the repeated field `name`, which has values if `nonEmpty`. `body` emits
    * the values, each a group.
    */
  private def repeated(
    cb: EmitCodeBuilder,
    sink: Value[ColumnarFileSink],
    name: String,
    nonEmpty: Code[Boolean],
  )(
    body: => Unit
  ): Unit =
    group(cb, sink) {
      // A repeated field with no values must not be started at all.
      cb.if_(nonEmpty, field(cb, sink, name, 0)(body))
    }

  /** Emits the standard three-level LIST encoding of a list that has elements if `nonEmpty`. `body`
    * emits the elements, each with [[listElement]] or [[missingListElement]].
    */
  private def list(
    cb: EmitCodeBuilder,
    sink: Value[ColumnarFileSink],
    nonEmpty: Code[Boolean],
  )(
    body: => Unit
  ): Unit =
    repeated(cb, sink, ColumnarSchema.ParquetListName, nonEmpty)(body)

  /** Emits a present list element, whose value `body` emits. */
  private def listElement(cb: EmitCodeBuilder, sink: Value[ColumnarFileSink])(body: => Unit): Unit =
    group(cb, sink)(field(cb, sink, ArrowSchemaConversion.ElementFieldName, 0)(body))

  private def missingListElement(cb: EmitCodeBuilder, sink: Value[ColumnarFileSink]): Unit =
    group(cb, sink)(())

  /** Emits an array or set as a LIST. */
  private def writeList(
    cb: EmitCodeBuilder,
    sink: Value[ColumnarFileSink],
    elementType: PType,
    a: SIndexableValue,
    elementPath: String,
  ): Unit =
    list(cb, sink, a.loadLength > 0) {
      a.forEachDefinedOrMissing(cb)(
        (cb, _) =>
          if (elementType.required)
            cb._fatal(s"columnar writer: required field '$elementPath' is missing")
          else
            missingListElement(cb, sink),
        (cb, _, element) =>
          listElement(cb, sink)(writeValue(cb, sink, elementType, element, elementPath)),
      )
    }

  /** Emits a dict as a MAP, failing on a missing dict key (spec section 5.1). */
  private def writeDict(
    cb: EmitCodeBuilder,
    sink: Value[ColumnarFileSink],
    t: PDict,
    d: SIndexableValue,
    path: String,
  ): Unit = {
    val (keyName, valueName) =
      (ArrowSchemaConversion.KeyFieldName, ArrowSchemaConversion.ValueFieldName)
    repeated(cb, sink, ColumnarSchema.ParquetMapEntriesName, d.loadLength > 0) {
      d.forEachDefined(cb) { (cb, _, entry) =>
        val kv = entry.asBaseStruct
        group(cb, sink) {
          kv.loadField(cb, 0).consume(
            cb,
            cb._fatal(
              s"columnar writer: field '$path' has a missing dict key, which can't be stored. " +
                "Filter the missing dict key out before writing."
            ),
            key =>
              field(cb, sink, keyName, 0) {
                writeValue(cb, sink, t.keyType, key, s"$path.$keyName")
              },
          )
          writeField(
            cb,
            sink,
            valueName,
            1,
            t.valueType,
            kv.loadField(cb, 1),
            s"$path.$valueName",
          )
        }
      }
    }
  }

  private def writeInterval(
    cb: EmitCodeBuilder,
    sink: Value[ColumnarFileSink],
    t: PInterval,
    i: SIntervalValue,
    path: String,
  ): Unit = {
    val IndexedSeq(start, end, includesStart, includesEnd) =
      ArrowSchemaConversion.IntervalFieldNames
    group(cb, sink) {
      writeField(cb, sink, start, 0, t.pointType, i.loadStart(cb), s"$path.$start")
      writeField(cb, sink, end, 1, t.pointType, i.loadEnd(cb), s"$path.$end")
      field(cb, sink, includesStart, 2) {
        cb += sink.invoke[Boolean, Unit]("addBoolean", i.includesStart)
      }
      field(cb, sink, includesEnd, 3) {
        cb += sink.invoke[Boolean, Unit]("addBoolean", i.includesEnd)
      }
    }
  }

  /** Emits an ndarray as Arrow's variable shape tensor: its data row-major, whatever its strides,
    * and its shape (spec section 5.4).
    */
  private def writeNDArray(
    cb: EmitCodeBuilder,
    sink: Value[ColumnarFileSink],
    t: PNDArray,
    nd: SNDArrayValue,
    path: String,
  ): Unit = {
    val shape = nd.shapes
    val IndexedSeq(dataName, shapeName) = ArrowSchemaConversion.TensorFieldNames
    shape.foreach { dim =>
      cb.if_(
        dim > Int.MaxValue.toLong,
        cb._fatal(
          s"columnar writer: an ndarray in field '$path' has a dimension longer than " +
            s"${Int.MaxValue}, which can't be stored"
        ),
      )
    }
    group(cb, sink) {
      field(cb, sink, dataName, 0) {
        list(cb, sink, SNDArray.numElements(shape) > 0L) {
          SNDArray.forEachIndexRowMajor(cb, shape, "columnar_writer_ndarray") { (cb, idx) =>
            listElement(cb, sink) {
              writeValue(cb, sink, t.elementType, nd.loadElement(idx, cb), s"$path[]")
            }
          }
        }
      }
      field(cb, sink, shapeName, 1) {
        list(cb, sink, const(t.nDims > 0)) {
          shape.foreach { dim =>
            listElement(cb, sink)(cb += sink.invoke[Int, Unit]("addInt", dim.toI))
          }
        }
      }
    }
  }

  /** Emits a struct or tuple with fields as a group. A tuple's fields are named by their tuple
    * indices, as in its Arrow type.
    */
  private def writeStruct(
    cb: EmitCodeBuilder,
    sink: Value[ColumnarFileSink],
    t: PBaseStruct,
    s: SBaseStructValue,
    path: String,
  ): Unit =
    group(cb, sink) {
      t.fields.foreach { f =>
        writeField(cb, sink, f.name, f.index, f.typ, s.loadField(cb, f.index), s"$path.${f.name}")
      }
    }

  private def writeValue(
    cb: EmitCodeBuilder,
    sink: Value[ColumnarFileSink],
    t: PType,
    v: SValue,
    path: String,
  ): Unit =
    t match {
      case _: PBoolean => cb += sink.invoke[Boolean, Unit]("addBoolean", v.asBoolean.value)
      case _: PInt32 => cb += sink.invoke[Int, Unit]("addInt", v.asInt.value)
      case _: PInt64 => cb += sink.invoke[Long, Unit]("addLong", v.asLong.value)
      case _: PFloat32 => cb += sink.invoke[Float, Unit]("addFloat", v.asFloat.value)
      case _: PFloat64 => cb += sink.invoke[Double, Unit]("addDouble", v.asDouble.value)
      case _: PString =>
        cb += sink.invoke[Array[Byte], Unit]("addBinary", v.asString.toBytes(cb).loadBytes(cb))
      case _: PBinary =>
        cb += sink.invoke[Array[Byte], Unit]("addBinary", v.asBinary.loadBytes(cb))
      case _: PCall => cb += sink.invoke[Int, Unit]("addInt", v.asCall.canonicalCall(cb))
      case _: PLocus => cb += sink.invoke[Long, Unit]("addLong", v.asLocus.packed(cb))
      case t: PInterval => writeInterval(cb, sink, t, v.asInterval, path)
      case t: PNDArray => writeNDArray(cb, sink, t, v.asNDArray, path)
      case t: PDict => writeDict(cb, sink, t, v.asIndexable, path)
      // An array or set; a dict is matched above.
      case t: PContainer => writeList(cb, sink, t.elementType, v.asIndexable, s"$path[]")
      // Parquet can't store a group with no fields (spec section 5.5).
      case t: PBaseStruct if t.size == 0 => cb += sink.invoke[Boolean, Unit]("addBoolean", true)
      case t: PBaseStruct => writeStruct(cb, sink, t, v.asBaseStruct, path)
    }
}

/** Tracks a partition's first and last keys, and whether two consecutive rows share a key. */
sealed abstract private class PartitionKeys {

  /** Records the key of `row`, the partition's next row. */
  def observe(cb: EmitCodeBuilder, row: SBaseStructValue, isFirst: Code[Boolean]): Unit

  /** `(distinct, firstKey, lastKey)`, after the last row. Call once. */
  def finish(cb: EmitCodeBuilder): (EmitCode, EmitCode, EmitCode)
}

/** For a file with no key: not distinct, and no first or last key. */
final private class NoPartitionKeys(keyST: SType) extends PartitionKeys {
  override def observe(cb: EmitCodeBuilder, row: SBaseStructValue, isFirst: Code[Boolean]): Unit =
    ()

  override def finish(cb: EmitCodeBuilder): (EmitCode, EmitCode, EmitCode) =
    (
      EmitCode.present(cb.emb, new SBooleanValue(false)),
      EmitCode.missing(cb.emb, keyST),
      EmitCode.missing(cb.emb, keyST),
    )
}

/** Copies the first key into `region`. The last key seen is copied into a region of its own, which
  * is cleared for each row, and into `region` at the end.
  */
final private class StagedPartitionKeys(
  cb: EmitCodeBuilder,
  region: Value[Region],
  key: IndexedSeq[String],
  keyST: SType,
) extends PartitionKeys {
  private[this] val distinct = cb.newLocal[Boolean]("distinct", true)
  private[this] val firstKey = cb.emb.newEmitLocal("firstKey", EmitType(keyST, false))
  private[this] val lastKey = cb.emb.newEmitLocal("lastKey", EmitType(keyST, false))

  private[this] val lastKeyRegion = cb.newLocal[Region](
    "lastKeyRegion",
    Region.stagedCreate(Region.SMALL, cb.emb.ecb.pool()),
  )

  cb.assign(firstKey, EmitCode.missing(cb.emb, keyST))
  cb.assign(lastKey, EmitCode.missing(cb.emb, keyST))

  override def observe(cb: EmitCodeBuilder, row: SBaseStructValue, isFirst: Code[Boolean]): Unit = {
    val k = row.subset(key: _*)
    cb.if_(
      isFirst,
      cb.assign(firstKey, EmitCode.present(cb.emb, k.castTo(cb, region, keyST, deepCopy = true))),
      cb.if_(
        cb.emb.ecb.getOrderingFunction(keyST, k.st, CodeOrdering.Equiv())(
          cb,
          lastKey,
          EmitValue.present(k),
        ),
        cb.assign(distinct, false),
      ),
    )
    cb += lastKeyRegion.invoke[Unit]("clear")
    cb.assign(
      lastKey,
      EmitCode.present(cb.emb, k.castTo(cb, lastKeyRegion, keyST, deepCopy = true)),
    )
  }

  override def finish(cb: EmitCodeBuilder): (EmitCode, EmitCode, EmitCode) = {
    val lastKeyCopy = cb.memoize(
      lastKey.toI(cb).map(cb)(k => k.castTo(cb, region, keyST, deepCopy = true)),
      "lastKeyCopy",
    )
    cb += lastKeyRegion.invoke[Unit]("invalidate")
    (EmitCode.present(cb.emb, new SBooleanValue(distinct)), firstKey.load, lastKeyCopy.load)
  }
}
