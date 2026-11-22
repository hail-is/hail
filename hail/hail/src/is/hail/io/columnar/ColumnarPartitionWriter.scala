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
import is.hail.types.{BaseTypeWithRequiredness, RIterable, RStruct, TypeWithRequiredness}
import is.hail.types.arrow.ArrowSchemaConversion
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

  /** The most values a leaf field of an entry file may hold in one row group (spec section 4.6.1).
    */
  val MaxLeafCap: Long = 1L << 30

  /** The most bytes a string or binary leaf field of an entry file may hold in one row group (spec
    * section 4.6.1).
    */
  val MaxByteCap: Long = Int.MaxValue.toLong
}

/** The entries of a matrix table, which a [[ColumnarPartitionWriter]] writes to an entry file
  * beside each row file, in the dense layout with one column block (spec section 4.3).
  *
  * Each row holds its entries in field `field`, an array of `entryType` with one element per
  * column. `leafCap` is the most values a leaf field of an entry file may hold in one row group,
  * and `byteCap` the most bytes a string or binary leaf field may hold in one.
  */
case class ColumnarEntries(
  field: String,
  entryType: PStruct,
  entryValidField: String,
  leafCap: Long,
  byteCap: Long,
) {

  /** The fields of an entry file after the Row ID: the entry validity list, then one list per entry
    * field, whose elements are nullable.
    */
  def fileType: PStruct = PCanonicalStruct(
    true,
    (entryValidField -> PCanonicalArray(PBoolean(true), true)) +:
      entryType.fields.map(f => f.name -> PCanonicalArray(f.typ.setRequired(false), true)): _*
  )
}

/** Writes a stream of structs to one Parquet data file of the columnar format, and, given
  * `entries`, each struct's entries to an entry file beside it. Both files have the same row
  * groups.
  *
  * The file holds the fields of `schema`, preceded by a Row ID field named `rowIdField` if given.
  * Row IDs are `(partitionIndex << RowIdPositionBits) | position`. An entry file requires a Row ID
  * field, which it shares.
  *
  * The context is `{path, partitionIndex}`, where `path` is relative to `root`. With entries, it
  * also has the entry file's `entryPath` and the `columnCount`, the length of every row's entries.
  * The result is the file's path, its row count, and the first and last values of the `key` fields,
  * which are missing for an empty file. `distinct` is whether no two consecutive rows share a key;
  * it is false when `key` is empty. With entries, it also has the `entryPath`.
  */
case class ColumnarPartitionWriter(
  root: String,
  schema: PStruct,
  key: IndexedSeq[String],
  rowIdField: Option[String],
  entries: Option[ColumnarEntries] = None,
) extends PartitionWriter {
  import ColumnarPartitionWriter._

  assert(entries.isEmpty || rowIdField.isDefined, "an entry file requires a Row ID field")

  lazy val arrowSchema: String =
    ColumnarSchema.serialize(ColumnarSchema.fileSchema(schema, rowIdField))

  lazy val entriesArrowSchema: Option[String] =
    entries.map(e => ColumnarSchema.serialize(ColumnarSchema.fileSchema(e.fileType, rowIdField)))

  private lazy val keyType: TStruct = schema.virtualType.typeAfterSelectNames(key)

  private lazy val keyPType: PCanonicalStruct =
    PType.canonical(keyType).asInstanceOf[PCanonicalStruct]

  private lazy val resultPType: PCanonicalStruct = PCanonicalStruct(
    true,
    FastSeq(
      "path" -> PCanonicalString(true),
      "rowCount" -> PInt64(true),
      "distinct" -> PBoolean(true),
      "firstKey" -> keyPType,
      "lastKey" -> keyPType,
    ) ++ entries.map(_ => "entryPath" -> PCanonicalString(true)): _*
  )

  override def ctxType: Type = TStruct(
    FastSeq("path" -> TString, "partitionIndex" -> TInt32) ++
      entries.toSeq.flatMap(_ => FastSeq("entryPath" -> TString, "columnCount" -> TInt32)): _*
  )

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

  private def open(cb: EmitCodeBuilder, arrowSchema: String, relPath: Value[String])
    : Value[ColumnarFileSink] =
    cb.memoize(
      cb.emb.getObject(ColumnarFileSpec(arrowSchema))
        .invoke[FS, String, ColumnarFileSink](
          "open",
          cb.emb.getFS,
          const(s"$root/").concat(relPath),
        )
    )

  override def consumeStream(
    ctx: ExecuteContext,
    cb: EmitCodeBuilder,
    stream: StreamProducer,
    context: EmitCode,
    region: Value[Region],
  ): IEmitCode =
    context.toI(cb).map(cb) { ctxValue =>
      val ctx = ctxValue.asBaseStruct
      def ctxString(name: String): Value[String] =
        cb.memoize(ctx.loadField(cb, name).getOrAssert(cb).asString.loadString(cb))
      def ctxInt(name: String): Value[Int] =
        cb.memoize(ctx.loadField(cb, name).getOrAssert(cb).asInt.value)

      val relPath = ctxString("path")
      val partitionIndex = ctxInt("partitionIndex")
      val rowGroups = cb.memoize(
        Code.newInstance[ColumnarRowGroups, Long](const(ColumnarRowGroups.TargetBytes))
      )
      val sink = open(cb, arrowSchema, relPath)
      cb += rowGroups.invoke[ColumnarFileSink, Unit]("add", sink)
      val entryFile = entries.map { e =>
        val entryPath = ctxString("entryPath")
        val entrySink = open(cb, entriesArrowSchema.get, entryPath)
        cb += rowGroups.invoke[ColumnarFileSink, Unit]("add", entrySink)
        new EntryFileWriter(cb, e, entrySink, entryPath, ctxInt("columnCount"))
      }

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
        val rowId = cb.memoize((partitionIndex.toL << RowIdPositionBits) | count)

        val entryRow = entryFile.map(_.load(cb, row))
        cb.if_(
          rowGroups.invoke[Long]("recordCount") > 0L && (
            entryRow.fold[Code[Boolean]](const(false))(_.exceedsCaps(cb)) ||
              rowGroups.invoke[Boolean]("checkFull")
          ), {
            cb += rowGroups.invoke[Unit]("endRowGroup")
            entryFile.foreach(_.startRowGroup(cb))
          },
        )

        cb += sink.invoke[Unit]("startRecord")
        val offset = rowIdField match {
          case Some(name) =>
            field(cb, sink, name, 0)(cb += sink.invoke[Long, Unit]("addLong", rowId))
            1
          case None => 0
        }
        schema.fields.foreach { f =>
          writeField(cb, sink, f.name, f.index + offset, f.typ, row.loadField(cb, f.name), f.name)
        }
        cb += sink.invoke[Unit]("endRecord")
        entryRow.foreach(_.write(cb, rowId))
        cb += rowGroups.invoke[Unit]("endRecord")

        keys.observe(cb, row, isFirst = count ceq 0L)
        cb.assign(count, count + 1L)
      }

      cb += rowGroups.invoke[Unit]("close")

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
        ) ++ entryFile.map(f => EmitCode.present(cb.emb, SJavaString.construct(cb, f.path))),
        deepCopy = false,
      )
    }

  /** Writes each row's entries to an entry file, and tracks how many values, and how many string
    * and binary bytes, the leaf fields of the current row group hold, to keep them within the leaf
    * and byte caps (spec section 4.6.1).
    *
    * Values are counted per leaf group: the entry validity list and every entry field with no list
    * inside share one, and every other entry field has one of its own. A leaf group's count bounds
    * the count of each of its leaf fields from above, and is exact for the first leaf group. Bytes
    * are counted per entry field that holds strings or binaries, with a bound on the bytes of each
    * of its leaves.
    */
  private class EntryFileWriter(
    cb: EmitCodeBuilder,
    e: ColumnarEntries,
    sink: Value[ColumnarFileSink],
    val path: Value[String],
    columnCount: Value[Int],
  ) {
    private[this] val nestedFields = e.entryType.fields.filterNot(f => isFlat(f.typ))

    // The values each leaf of the first leaf group holds per row. An empty list is one value.
    private[this] val flatRowLeaves: Value[Long] = max(cb, columnCount.toL, 1L)

    private[this] val flatLeaves = cb.newLocal[Long]("flatLeaves", 0L)

    private[this] val nestedLeaves =
      nestedFields.map(f => cb.newLocal[Long](s"leaves_${f.name}", 0L))

    private[this] val binaryFields = e.entryType.fields.filter(f => hasBinary(f.typ))

    private[this] val binaryBytes =
      binaryFields.map(f => cb.newLocal[Long](s"bytes_${f.name}", 0L))

    def startRowGroup(cb: EmitCodeBuilder): Unit =
      (flatLeaves +: (nestedLeaves ++ binaryBytes)).foreach(cb.assign(_, 0L))

    /** Loads `row`'s entries, failing unless there is one per column. */
    def load(cb: EmitCodeBuilder, row: SBaseStructValue): EntryRow = {
      val entries = row.loadField(cb, e.field)
        .getOrFatal(cb, "columnar writer: a row's entries are missing")
        .asIndexable
      cb.if_(
        entries.loadLength cne columnCount,
        cb._fatal(
          "columnar writer: a row has ",
          entries.loadLength.toS,
          " entries, expected one per column, ",
          columnCount.toS,
        ),
      )
      cb.if_(
        flatRowLeaves > e.leafCap,
        cb._fatal(
          s"columnar writer: the entries of one row hold more than ${e.leafCap} values in each " +
            "entry field, which can't be stored: there are ",
          columnCount.toS,
          " columns",
        ),
      )
      new EntryRow(
        entries,
        nestedFields.map(f => fieldLeaves(cb, f, entries)),
        binaryFields.map(f => fieldBytes(cb, f, entries)),
      )
    }

    /** An upper bound on the bytes each string or binary leaf of entry field `f` holds for one
      * row's `entries`.
      */
    private[this] def fieldBytes(cb: EmitCodeBuilder, f: PField, entries: SIndexableValue)
      : Value[Long] = {
      val total = cb.newLocal[Long](s"rowBytes_${f.name}", 0L)
      entries.forEachDefined(cb) { (cb, _, entry) =>
        entry.asBaseStruct.loadField(cb, f.index).consume(
          cb,
          {},
          v => cb.assign(total, total + binaryByteCount(cb, f.typ, v)),
        )
      }
      cb.if_(
        total > e.byteCap,
        cb._fatal(
          s"columnar writer: the entries of one row hold more than ${e.byteCap} bytes of " +
            s"strings or binaries in entry field '${f.name}', which can't be stored"
        ),
      )
      total
    }

    /** An upper bound on the values each leaf of entry field `f` holds for one row's `entries`. */
    private[this] def fieldLeaves(cb: EmitCodeBuilder, f: PField, entries: SIndexableValue)
      : Value[Long] = {
      val total = cb.newLocal[Long](s"rowLeaves_${f.name}", 0L)
      entries.forEachDefinedOrMissing(cb)(
        (cb, _) => cb.assign(total, total + 1L),
        (cb, _, entry) =>
          entry.asBaseStruct.loadField(cb, f.index).consume(
            cb,
            cb.assign(total, total + 1L),
            v => cb.assign(total, total + leafBound(cb, f.typ, v)),
          ),
      )
      cb.if_(
        total > e.leafCap,
        cb._fatal(
          s"columnar writer: the entries of one row hold more than ${e.leafCap} values in " +
            s"entry field '${f.name}', which can't be stored"
        ),
      )
      max(cb, total, 1L)
    }

    class EntryRow(
      entries: SIndexableValue,
      rowNestedLeaves: IndexedSeq[Value[Long]],
      rowBinaryBytes: IndexedSeq[Value[Long]],
    ) {

      /** Whether adding this row to the current row group would exceed the leaf or byte cap. */
      def exceedsCaps(cb: EmitCodeBuilder): Code[Boolean] = {
        val leaves = (flatLeaves +: nestedLeaves).zip(flatRowLeaves +: rowNestedLeaves)
          .map { case (group, row) => (group + row) > e.leafCap }
        val bytes = binaryBytes.zip(rowBinaryBytes)
          .map { case (group, row) => (group + row) > e.byteCap }
        (leaves ++ bytes).reduce(_ || _)
      }

      def write(cb: EmitCodeBuilder, rowId: Value[Long]): Unit = {
        cb += sink.invoke[Unit]("startRecord")
        field(cb, sink, rowIdField.get, 0)(cb += sink.invoke[Long, Unit]("addLong", rowId))
        field(cb, sink, e.entryValidField, 1) {
          list(cb, sink, columnCount > 0) {
            entries.forEachDefinedOrMissing(cb)(
              (cb, _) =>
                listElement(cb, sink)(cb += sink.invoke[Boolean, Unit]("addBoolean", false)),
              (cb, _, _) =>
                listElement(cb, sink)(cb += sink.invoke[Boolean, Unit]("addBoolean", true)),
            )
          }
        }
        e.entryType.fields.foreach { f =>
          field(cb, sink, f.name, f.index + 2) {
            list(cb, sink, columnCount > 0) {
              entries.forEachDefinedOrMissing(cb)(
                (cb, _) => missingListElement(cb, sink),
                (cb, _, entry) =>
                  entry.asBaseStruct.loadField(cb, f.index).consume(
                    cb,
                    missingListElement(cb, sink),
                    v => listElement(cb, sink)(writeValue(cb, sink, f.typ, v, f.name)),
                  ),
              )
            }
          }
        }
        cb += sink.invoke[Unit]("endRecord")

        cb.assign(flatLeaves, flatLeaves + flatRowLeaves)
        (nestedLeaves ++ binaryBytes).zip(rowNestedLeaves ++ rowBinaryBytes).foreach {
          case (group, row) => cb.assign(group, group + row)
        }
      }
    }
  }

  /** Whether `t`'s Parquet translation has no repeated field, so that each of its leaves holds one
    * value per value of `t`.
    */
  private def isFlat(t: PType): Boolean = t match {
    case _: PContainer | _: PNDArray => false
    case t: PInterval => isFlat(t.pointType)
    case t: PBaseStruct => t.types.forall(isFlat)
    case _ => true
  }

  /** Whether `t` holds strings or binaries, at any depth. */
  private def hasBinary(t: PType): Boolean = t match {
    case _: PString | _: PBinary => true
    case t: PContainer => hasBinary(t.elementType)
    case t: PInterval => hasBinary(t.pointType)
    case t: PBaseStruct => t.types.exists(hasBinary)
    case _ => false
  }

  /** An upper bound on the bytes of strings or binaries that any one leaf of `t`'s Parquet
    * translation holds for the present value `v`. The elements of a container share its leaves, so
    * their bytes add up; the fields of a struct, and the endpoints of an interval, don't, so the
    * bound is the largest of theirs.
    */
  private def binaryByteCount(cb: EmitCodeBuilder, t: PType, v: SValue): Value[Long] = {
    def optional(t: PType, v: => IEmitCode): Value[Long] = {
      val bytes = cb.newLocal[Long]("bytes", 0L)
      v.consume(cb, {}, v => cb.assign(bytes, binaryByteCount(cb, t, v)))
      bytes
    }

    t match {
      case t if !hasBinary(t) => const(0L)
      case _: PString => cb.memoize(v.asString.toBytes(cb).loadLength(cb).toL)
      case _: PBinary => cb.memoize(v.asBinary.loadLength(cb).toL)
      case t: PContainer =>
        val total = cb.newLocal[Long]("bytes", 0L)
        v.asIndexable.forEachDefined(cb) { (cb, _, elt) =>
          cb.assign(total, total + binaryByteCount(cb, t.elementType, elt))
        }
        total
      case t: PInterval =>
        val i = v.asInterval
        max(cb, optional(t.pointType, i.loadStart(cb)), optional(t.pointType, i.loadEnd(cb)))
      case t: PBaseStruct =>
        val s = v.asBaseStruct
        t.fields.filter(f => hasBinary(f.typ)).foldLeft[Value[Long]](const(0L)) { (acc, f) =>
          max(cb, acc, optional(f.typ, s.loadField(cb, f.index)))
        }
    }
  }

  private def max(cb: EmitCodeBuilder, a: Code[Long], b: Code[Long]): Value[Long] =
    cb.memoize(Code.invokeStatic2[java.lang.Math, Long, Long, Long]("max", a, b))

  /** An upper bound on the values that any one leaf of `t`'s Parquet translation holds for the
    * present value `v`. A missing value, or an empty list, is one value in each of its leaves.
    */
  private def leafBound(cb: EmitCodeBuilder, t: PType, v: SValue): Value[Long] = {
    def optional(t: PType, v: => IEmitCode): Value[Long] =
      if (isFlat(t)) const(1L)
      else {
        val leaves = cb.newLocal[Long]("leaves", 1L)
        v.consume(cb, {}, v => cb.assign(leaves, leafBound(cb, t, v)))
        leaves
      }

    t match {
      case t if isFlat(t) => const(1L)
      case t: PNDArray =>
        // `data` holds every element, and `shape` one value per dimension.
        max(cb, SNDArray.numElements(v.asNDArray.shapes), math.max(t.nDims, 1).toLong)
      case t: PContainer =>
        val a = v.asIndexable
        val total = cb.newLocal[Long]("leaves", 0L)
        t match {
          case t: PDict =>
            a.forEachDefined(cb) { (cb, _, entry) =>
              val kv = entry.asBaseStruct
              val keyLeaves = optional(t.keyType, kv.loadField(cb, 0))
              val valueLeaves = optional(t.valueType, kv.loadField(cb, 1))
              cb.assign(total, total + max(cb, keyLeaves, valueLeaves))
            }
          case _ =>
            a.forEachDefinedOrMissing(cb)(
              (cb, _) => cb.assign(total, total + 1L),
              (cb, _, elt) => cb.assign(total, total + leafBound(cb, t.elementType, elt)),
            )
        }
        max(cb, total, 1L)
      case t: PInterval =>
        val i = v.asInterval
        max(cb, optional(t.pointType, i.loadStart(cb)), optional(t.pointType, i.loadEnd(cb)))
      case t: PBaseStruct =>
        val s = v.asBaseStruct
        t.fields.filterNot(f => isFlat(f.typ)).foldLeft[Value[Long]](const(1L)) { (acc, f) =>
          max(cb, acc, optional(f.typ, s.loadField(cb, f.index)))
        }
    }
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
