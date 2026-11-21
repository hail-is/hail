package is.hail.io.columnar

import is.hail.backend.ExecuteContext
import is.hail.collection.FastSeq
import is.hail.expr.ir._
import is.hail.expr.ir.defs._
import is.hail.expr.ir.lowering.{LowerMatrixIR, TableStage}
import is.hail.io.fs.FS
import is.hail.types.VirtualTypeWithReq
import is.hail.types.arrow.ArrowSchemaConversion
import is.hail.types.physical._
import is.hail.types.virtual._
import is.hail.utils.{fatal, Interval}
import is.hail.variant.ReferenceGenomeEnv

import org.apache.spark.sql.Row
import org.json4s.{DefaultFormats, Formats, JObject, JString, JValue}

object MatrixColumnarReader {

  /** A reader of the columnar matrix table at `path`. Its reference genomes must be equal to any of
    * the same name in `references`.
    */
  def apply(fs: FS, references: ReferenceGenomeEnv, path: String): MatrixColumnarReader = {
    val (dataset, matrix) = this.dataset(fs, references, path)
    new MatrixColumnarReader(dataset, matrix, None)
  }

  private def dataset(fs: FS, references: ReferenceGenomeEnv, path: String)
    : (ColumnarDataset, ColumnarMatrixComponents) = {
    val manifest = ColumnarManifest.read(fs, path)
    val matrix = manifest.matrix.getOrElse(
      fatal(s"columnar matrix table '$path': it is a table, not a matrix table")
    )
    val dataset = new ColumnarDataset(
      path,
      manifest,
      references.withReferences(manifest.referenceGenomes),
      MatrixReader.rowUIDFieldName,
    )
    (dataset, matrix)
  }

  /** A reader from its JSON: its `path`, and a key-range read on the row key, if any (see
    * [[ColumnarReadIntervals.fromJValue]]).
    */
  def fromJValue(fs: FS, references: ReferenceGenomeEnv, jv: JValue): MatrixColumnarReader = {
    implicit val formats: Formats = DefaultFormats
    val path = (jv \ "path").extract[String]
    val (dataset, matrix) = this.dataset(fs, references, path)
    new MatrixColumnarReader(
      dataset,
      matrix,
      ColumnarReadIntervals.fromJValue(jv, dataset.references, dataset.keyType, path),
    )
  }
}

/** Reads a matrix table in the columnar format (`dev-docs/hail-query/columnar-format-spec.md`).
  *
  * Rows are read as by [[TableColumnarReader]], and each row file is zipped by position with its
  * entry file, whose Row IDs must equal the row file's. An entry is missing when the entry validity
  * list says so. Entry files are read only when entries are requested, and then only the lists of
  * the requested entry fields and the entry validity list. The row UID is the Row ID, and the
  * column UID the column index.
  *
  * A key-range read of `intervals` reads only the rows whose row key is in them, and only the row
  * partitions whose bounds overlap them (see [[ColumnarReadIntervals]]).
  */
class MatrixColumnarReader private (
  dataset: ColumnarDataset,
  matrix: ColumnarMatrixComponents,
  val intervals: Option[ColumnarReadIntervals],
) extends MatrixHybridReader with ColumnarKeyRangeReader {

  def path: String = dataset.path

  private def manifest: ColumnarManifest = dataset.manifest

  private val colFilePType: PStruct =
    ArrowSchemaConversion.fromArrowSchema(matrix.colsSchema, dataset.references)

  // The columns file's UID is the row's position, which is the column index (spec section 4.4).
  private val colsReader = ColumnarPartitionReader(
    colFilePType,
    ColumnarSchema.serialize(matrix.colsSchema),
    None,
    colUIDFieldName,
  )

  private val entryFilePType: PStruct =
    ArrowSchemaConversion.fromArrowSchema(matrix.entriesSchema, dataset.references)

  // The Row ID as read from the entry file, under a name that is no entry field's.
  private val entryFileUIDField =
    ColumnarSchema.internalFieldName(uidFieldName, entryFilePType.fieldNames)

  private val entriesReader = ColumnarPartitionReader(
    entryFilePType,
    ColumnarSchema.serialize(matrix.entriesSchema),
    Some(manifest.rowIdField),
    entryFileUIDField,
  )

  /** The entry fields, each the element type of its list in the entry file. */
  private val entryPType: PCanonicalStruct = PCanonicalStruct(
    required = false,
    entryFilePType.fields
      .filterNot(f => f.name == manifest.rowIdField || f.name == matrix.entryValidField)
      .map(f => f.name -> f.typ.asInstanceOf[PCanonicalArray].elementType): _*
  )

  override def pathsUsed: Seq[String] = FastSeq(path)

  override def restrictedTo(intervals: IndexedSeq[Interval]): Option[MatrixColumnarReader] =
    ColumnarReadIntervals.restrict(dataset.keyType, this.intervals, intervals).map(r =>
      new MatrixColumnarReader(dataset, matrix, Some(r))
    )

  override def columnCount: Option[Int] = Some(matrix.colCount)

  override def partitionCounts: Option[IndexedSeq[Long]] = None

  override def isDistinctlyKeyed: Boolean = manifest.distinctlyKeyed

  override def rowUIDType: Type = TInt64

  override def colUIDType: Type = TInt64

  override def fullMatrixTypeWithoutUIDs: MatrixType =
    MatrixType(
      globalType = dataset.globalsPType.virtualType,
      colKey = matrix.colKey,
      colType = colFilePType.virtualType.deleteKey(matrix.colIdxField),
      rowKey = manifest.key,
      rowType = dataset.rowType,
      entryType = entryPType.virtualType,
    )

  private def entriesField: String = LowerMatrixIR.entriesFieldName

  private def colsField: String = LowerMatrixIR.colsFieldName

  private def without(t: TStruct, field: String): TStruct =
    if (t.hasField(field)) t.deleteKey(field) else t

  /** The struct type of the elements of `t`, an array of structs, such as the entries or columns.
    */
  private def elementStruct(t: Type): TStruct =
    t.asInstanceOf[TArray].elementType.asInstanceOf[TStruct]

  private def entryType(rowType: TStruct): TStruct = elementStruct(rowType.fieldType(entriesField))

  override protected def concreteRowRequiredness(ctx: ExecuteContext, requestedType: TableType)
    : VirtualTypeWithReq = {
    val rowType = requestedType.rowType
    val rowPType = dataset.rowsReader.rowPType(without(rowType, entriesField))
    VirtualTypeWithReq(PCanonicalStruct(
      required = true,
      rowType.fieldNames.map { name =>
        name -> (
          if (name == entriesField)
            PCanonicalArray(ColumnarReadPlan.subset(entryPType, entryType(rowType)), true)
          else rowPType.field(name).typ
        )
      }: _*
    ))
  }

  override protected def uidRequiredness: VirtualTypeWithReq = VirtualTypeWithReq(PInt64(true))

  override def globalRequiredness(ctx: ExecuteContext, requestedType: TableType)
    : VirtualTypeWithReq = {
    val globalType = requestedType.globalType
    val globalsPType =
      dataset.globalsRequiredness(without(globalType, colsField))
    VirtualTypeWithReq(PCanonicalStruct(
      required = true,
      globalType.fields.map { f =>
        f.name -> (
          if (f.name == colsField)
            PCanonicalArray(
              colsReader.rowPType(elementStruct(f.typ)),
              true,
            )
          else globalsPType.asInstanceOf[PStruct].field(f.name).typ
        )
      }: _*
    ))
  }

  override def toJValue: JValue =
    JObject("name" -> JString("MatrixColumnarReader"), "path" -> JString(path))
      .merge(ColumnarReadIntervals.toJValue(dataset.keyType, intervals))

  override def renderShort(): String =
    s"(MatrixColumnarReader $path${intervals.fold("")(i => s" (${i.intervals.length} intervals)")})"

  override def hashCode(): Int = (path, intervals).##

  override def equals(that: Any): Boolean = that match {
    case that: MatrixColumnarReader => path == that.path && intervals == that.intervals
    case _ => false
  }

  override def toString: String = s"MatrixColumnarReader($path)"

  /** The columns, of type `colType`, from the columns file. */
  private def cols(colType: TStruct): IR =
    bindIR(dataset.readWholeFile(matrix.colsFile, colsReader, colType)) { cols =>
      If(
        ArrayLen(cols) ceq I32(matrix.colCount),
        cols,
        Die(
          strConcat(
            s"columnar reader: the columns file has ",
            ArrayLen(cols),
            s" columns, but the manifest has ${matrix.colCount}",
          ),
          cols.typ,
        ),
      )
    }

  override def lowerGlobals(ctx: ExecuteContext, requestedGlobalsType: TStruct): IR = {
    val globals = dataset.globals(without(requestedGlobalsType, colsField))
    if (!requestedGlobalsType.hasField(colsField)) globals
    else
      bindIR(globals) { g =>
        MakeStruct(requestedGlobalsType.fields.map { f =>
          f.name -> (
            if (f.name == colsField)
              cols(elementStruct(f.typ))
            else GetField(g, f.name)
          )
        })
      }
  }

  /** The entries of type `entryType` of a row, from `entries`, its row of the entry file. */
  private def assembleEntries(entries: Atom, entryType: TStruct): IR = {
    val n = matrix.colCount
    val lists = matrix.entryValidField +: entryType.fieldNames
    bindIRs(lists.map(GetField(entries, _)): _*) { case Seq(valid, fields @ _*) =>
      val entriesIR: IR = ToArray(mapIR(rangeIR(I32(n))) { j =>
        If(
          ArrayRef(valid, j),
          MakeStruct(entryType.fieldNames.zip(fields).map { case (name, l) =>
            name -> ArrayRef(l, j)
          }),
          NA(entryType),
        )
      })
      lists.zip(valid +: fields).foldRight(entriesIR) { case ((name, l), body) =>
        If(
          ArrayLen(l) ceq I32(n),
          body,
          Die(
            strConcat(
              s"columnar reader: entry field list '$name' of the row with Row ID ",
              GetField(entries, entryFileUIDField),
              " has ",
              ArrayLen(l),
              s" elements, but the matrix table has $n columns",
            ),
            body.typ,
          ),
        )
      }
    }
  }

  override def lower(ctx: ExecuteContext, requestedType: TableType): TableStage = {
    val rowType = requestedType.rowType
    val readEntries = rowType.hasField(entriesField)
    val rowFileReader = dataset.forRead(dataset.rowsReader, intervals)
    val entryFileReader = dataset.forRead(entriesReader, intervals)
    dataset.lower(
      ctx,
      lowerGlobals(ctx, requestedType.globalType),
      intervals,
      rowType,
      TStruct(
        FastSeq("rows" -> rowFileReader.contextType) ++
          (if (readEntries) FastSeq("entries" -> entryFileReader.contextType) else FastSeq()): _*
      ),
      { r =>
        val p = r.partition
        val rows = dataset.readerContext(p.rowFile, r)
        if (readEntries) Row(rows, dataset.readerContext(p.entryFiles.head, r)) else Row(rows)
      },
      ctx => GetField(GetField(ctx, "rows"), "intervals"),
    ) { (ctx, readType) =>
      val rows = ReadPartition(GetField(ctx, "rows"), _: TStruct, rowFileReader)
      if (readEntries)
        readRowsAndEntries(
          readType,
          rows,
          ReadPartition(GetField(ctx, "entries"), _, entryFileReader),
        )
      else rows(readType)
    }
  }

  /** The rows of type `rowType`, which includes the entries, of a row partition: the rows of its
    * row file, read by `readRowFile` given a type, zipped with those of its entry file, read by
    * `readEntryFile`.
    */
  private def readRowsAndEntries(
    rowType: TStruct,
    readRowFile: TStruct => IR,
    readEntryFile: TStruct => IR,
  ): IR = {
    val entryType = this.entryType(rowType)
    val rowFileType = TStruct(
      rowType.fields.filter(f => f.name != entriesField && f.name != uidFieldName).map(f =>
        f.name -> f.typ
      ) :+ (uidFieldName -> TInt64): _*
    )
    val entryFileType = TStruct(
      (matrix.entryValidField -> TArray(TBoolean)) +:
        entryType.fields.map(f => f.name -> TArray(f.typ)) :+
        (entryFileUIDField -> TInt64): _*
    )
    zip2(
      readRowFile(rowFileType),
      readEntryFile(entryFileType),
      ArrayZipBehavior.AssertSameLength,
    ) { (row, entries) =>
      def rowId: IR = GetField(row, uidFieldName)
      If(
        rowId ceq GetField(entries, entryFileUIDField),
        MakeStruct(rowType.fieldNames.map { name =>
          name -> (
            if (name == entriesField) assembleEntries(entries, entryType)
            else GetField(row, name)
          )
        }),
        Die(
          strConcat(
            "columnar reader: an entry file's Row ID ",
            GetField(entries, entryFileUIDField),
            " differs from its row file's Row ID ",
            rowId,
            " at the same position",
          ),
          rowType,
        ),
      )
    }
  }
}
