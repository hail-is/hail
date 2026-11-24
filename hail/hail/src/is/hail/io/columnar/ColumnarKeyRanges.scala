package is.hail.io.columnar

import is.hail.collection.FastSeq
import is.hail.expr.JSONAnnotationImpex
import is.hail.expr.ir.{IRParser, TableReader}
import is.hail.rvd.{PartitionBoundOrdering, RVDPartitioner}
import is.hail.types.physical.PLocus
import is.hail.types.virtual._
import is.hail.utils.{fatal, Interval}
import is.hail.variant.{Locus, ReferenceGenomeEnv}

import scala.jdk.CollectionConverters._

import org.apache.parquet.filter2.compat.FilterCompat
import org.apache.parquet.filter2.predicate.{FilterApi, FilterPredicate, Operators}
import org.apache.parquet.filter2.statisticslevel.StatisticsFilter
import org.apache.parquet.hadoop.ParquetFileReader
import org.apache.parquet.hadoop.metadata.{BlockMetaData, ColumnPath}
import org.apache.parquet.internal.column.columnindex.{ColumnIndex, OffsetIndex}
import org.apache.parquet.internal.filter2.columnindex.{
  ColumnIndexFilter, ColumnIndexStore, RowRanges,
}
import org.apache.parquet.io.api.Binary
import org.apache.spark.sql.Row
import org.json4s.{DefaultFormats, Formats, JNothing, JValue}
import org.json4s.JsonDSL._

/** The key intervals of a key-range read, which reads only the rows whose key is in them. Their
  * points are structs that may be prefixes of the key.
  *
  * With `filterIntervals`, the read keeps the row partitions that overlap the intervals, narrowed
  * to them, and the intervals are sorted and disjoint. Otherwise, as with
  * `read_table(_intervals=...)`, each interval is a partition of the read, holding its rows from
  * every row partition it overlaps, and the intervals must be valid partition bounds.
  */
case class ColumnarReadIntervals(intervals: IndexedSeq[Interval], filterIntervals: Boolean)

object ColumnarReadIntervals {

  /** The key intervals of a key-range read of `intervals` on a key of type `keyType`. */
  def apply(
    keyType: TStruct,
    intervals: IndexedSeq[Interval],
    filterIntervals: Boolean,
    path: String,
  ): ColumnarReadIntervals =
    if (filterIntervals)
      ColumnarReadIntervals(
        Interval.union(intervals, PartitionBoundOrdering(keyType).intervalEndpointOrdering),
        filterIntervals = true,
      )
    else {
      if (!RVDPartitioner.isValid(keyType, intervals))
        fatal(s"columnar reader '$path': the intervals are not valid partition bounds: they " +
          "must be sorted and must not overlap")
      ColumnarReadIntervals(intervals, filterIntervals = false)
    }

  /** A key-range read of `read`, if any, restricted to the keys in `intervals`, on a key of type
    * `keyType`. `None` if `read` makes each interval a partition, whose bounds a restriction would
    * change.
    */
  def restrict(
    keyType: TStruct,
    read: Option[ColumnarReadIntervals],
    intervals: IndexedSeq[Interval],
  ): Option[ColumnarReadIntervals] = {
    val ord = PartitionBoundOrdering(keyType).intervalEndpointOrdering
    val restriction = Interval.union(intervals, ord)
    read match {
      case None => Some(ColumnarReadIntervals(restriction, filterIntervals = true))
      case Some(ColumnarReadIntervals(intervals, true)) =>
        Some(ColumnarReadIntervals(Interval.intersection(intervals, restriction, ord), true))
      case Some(ColumnarReadIntervals(_, false)) => None
    }
  }

  /** The key-range read in a reader's JSON: `intervals` of points of type `intervalPointType`, and
    * `filterIntervals`, false if absent. `None` if it has no intervals. Points must be prefixes of
    * a key of type `keyType`.
    */
  def fromJValue(jv: JValue, references: ReferenceGenomeEnv, keyType: TStruct, path: String)
    : Option[ColumnarReadIntervals] =
    jv \ "intervals" match {
      case JNothing => None
      case intervals =>
        implicit val formats: Formats = DefaultFormats
        val pointType = IRParser.parseType((jv \ "intervalPointType").extract[String], references)
        pointType match {
          case t: TStruct if t.isPrefixOf(keyType) =>
          case t => fatal(s"columnar reader '$path': interval point type $t is not a prefix of " +
              s"the key type $keyType")
        }
        Some(ColumnarReadIntervals(
          keyType,
          JSONAnnotationImpex.importAnnotation(
            intervals,
            TArray(TInterval(pointType)),
            padNulls = false,
          ).asInstanceOf[IndexedSeq[Interval]],
          (jv \ "filterIntervals").extractOpt[Boolean].getOrElse(false),
          path,
        ))
    }

  /** `read`, on a key of type `keyType`, as the fields of a reader's JSON. */
  def toJValue(keyType: TStruct, read: Option[ColumnarReadIntervals]): JValue =
    read match {
      case None => JNothing
      case Some(ColumnarReadIntervals(intervals, filterIntervals)) =>
        ("intervals" -> JSONAnnotationImpex.exportAnnotation(
          intervals,
          TArray(TInterval(keyType)),
        )) ~ ("intervalPointType" -> keyType.parsableString()) ~
          ("filterIntervals" -> filterIntervals)
    }
}

/** What a partition reader needs to pick, in a key-range read, the row groups and pages of a data
  * file to read: the key, of type `keyType`. The statistics and page index of the first key field
  * in the row partition's row file pick them, and the picked rows of every file of a row partition
  * are the same.
  *
  * The picked rows are a superset of the rows whose key is in the read's intervals, which the
  * caller filters to.
  */
case class ColumnarKeyRange(keyType: TStruct) {

  /** The fields this adds to a partition reader's context: the row partition's row file, and the
    * key intervals in the representation `partitionerContains` takes.
    */
  def contextFields: IndexedSeq[(String, Type)] = FastSeq(
    "rowFile" -> TString,
    "intervals" -> TArray(RVDPartitioner.intervalIRRepresentation(keyType)),
  )

  private def keyField: String = keyType.fieldNames(0)

  /** For each row group of the row file open in `reader`, the rows to read of the row groups whose
    * keys may be in `intervals`, or an empty range if none may be.
    */
  def rowRanges(reader: ParquetFileReader, intervals: IndexedSeq[Interval])
    : IndexedSeq[RowRanges] = {
    val blocks = reader.getRowGroups.asScala.toIndexedSeq
    predicate(intervals) match {
      case None => blocks.map(ColumnarKeyRange.allRows)
      case Some(Seq()) => blocks.map(_ => RowRanges.EMPTY)
      case Some(disjuncts) =>
        val p = ColumnarKeyRange.or(disjuncts)
        // `FilterCompat.get` rewrites the predicate and logs it, so call it once.
        lazy val filter = FilterCompat.get(p)
        val path = ColumnPath.get(keyField)
        blocks.map { b =>
          if (b.getRowCount == 0 || StatisticsFilter.canDrop(p, b.getColumns)) RowRanges.EMPTY
          else {
            val chunk = b.getColumns.asScala.find(_.getPath == path).get
            val columnIndex = reader.readColumnIndex(chunk)
            val offsetIndex = reader.readOffsetIndex(chunk)
            if (columnIndex == null || offsetIndex == null) ColumnarKeyRange.allRows(b)
            else
              ColumnIndexFilter.calculateRowRanges(
                filter,
                new ColumnIndexStore {
                  override def getColumnIndex(c: ColumnPath): ColumnIndex =
                    if (c == path) columnIndex else null

                  override def getOffsetIndex(c: ColumnPath): OffsetIndex =
                    if (c == path) offsetIndex
                    else throw new ColumnIndexStore.MissingOffsetIndexException(c)
                },
                java.util.Collections.singleton(path),
                b.getRowCount,
              )
          }
        }
    }
  }

  /** The disjuncts of a Parquet predicate on the first key field that holds for the first key field
    * of every key in `intervals`, or `None` if its statistics can't pick rows: it is not an
    * integer, call, locus or string, or its Parquet order may differ from Hail's.
    */
  private def predicate(intervals: IndexedSeq[Interval]): Option[IndexedSeq[FilterPredicate]] =
    keyType.types(0) match {
      case TInt32 | TCall =>
        Some(intervals.flatMap(disjunct(
          new TopLevelColumn(keyField, classOf[java.lang.Integer]),
          v => Int.box(v.asInstanceOf[Int]),
        )))
      case TInt64 =>
        Some(intervals.flatMap(disjunct(
          new TopLevelColumn(keyField, classOf[java.lang.Long]),
          v => Long.box(v.asInstanceOf[Long]),
        )))
      case t: TLocus =>
        Some(intervals.flatMap(disjunct(
          new TopLevelColumn(keyField, classOf[java.lang.Long]),
          { v =>
            val l = v.asInstanceOf[Locus]
            Long.box(PLocus.pack(t.rg.getContigIndex(l.contig), l.position))
          },
        )))
      // Parquet orders strings by their UTF-8 bytes, and Hail by their UTF-16 code units. The two
      // orders agree on comparisons with a string of characters below the surrogates.
      case TString if bounds(intervals).forall(_.asInstanceOf[String].forall(_ < 0xd800)) =>
        Some(intervals.flatMap(disjunct(
          new TopLevelColumn(keyField, classOf[Binary]),
          v => Binary.fromString(v.asInstanceOf[String]),
        )))
      case _ => None
    }

  /** The present values of the first key field that the endpoints of `intervals` hold. */
  private def bounds(intervals: IndexedSeq[Interval]): IndexedSeq[Any] =
    intervals.flatMap(i => FastSeq(i.start, i.end)).flatMap { p =>
      val r = p.asInstanceOf[Row]
      if (r.size == 0 || r.isNullAt(0)) None else Some(r.get(0))
    }

  /** A Parquet predicate on `column` that holds for the first key field of every key in `i`, or
    * `None` if there are no such keys. `value` is a present first key field's Parquet value.
    *
    * An endpoint that is a longer prefix of the key bounds the first key field inclusively. Missing
    * values are greater than all others.
    */
  private def disjunct[T <: Comparable[T], C <: Operators.Column[T] with Operators.SupportsLtGt](
    column: C,
    value: Any => T,
  )(
    i: Interval
  ): Option[FilterPredicate] = {
    def bound(p: Any, inclusive: Boolean): Option[(Any, Boolean)] = {
      val r = p.asInstanceOf[Row]
      if (r.size == 0) None else Some((r.get(0), r.size == 1 && !inclusive))
    }
    val lower = bound(i.start, i.includesStart)
    val upper = bound(i.end, i.includesEnd)
    val missing: Option[FilterPredicate] =
      if (
        lower.forall { case (v, strict) => v != null || !strict } &&
        upper.forall { case (v, strict) => v == null && !strict }
      )
        Some(FilterApi.eq(column, null.asInstanceOf[T]): FilterPredicate)
      else None
    val present: Option[FilterPredicate] =
      if (lower.exists(_._1 == null)) None
      else {
        val terms: Seq[FilterPredicate] =
          lower.map { case (v, strict) =>
            if (strict) FilterApi.gt(column, value(v)): FilterPredicate
            else FilterApi.gtEq(column, value(v)): FilterPredicate
          }.toSeq ++
            upper.filter(_._1 != null).map { case (v, strict) =>
              if (strict) FilterApi.lt(column, value(v)): FilterPredicate
              else FilterApi.ltEq(column, value(v)): FilterPredicate
            }
        Some(terms.reduceOption[FilterPredicate](FilterApi.and).getOrElse(
          FilterApi.notEq(column, null.asInstanceOf[T])
        ))
      }
    (present ++ missing).reduceOption[FilterPredicate](FilterApi.or)
  }
}

object ColumnarKeyRange {

  /** The disjunction of the nonempty `ps`, as a balanced tree. Parquet visits predicates
    * recursively, so a chain of thousands of `or`s overflows the stack.
    */
  private def or(ps: IndexedSeq[FilterPredicate]): FilterPredicate =
    if (ps.length == 1) ps(0)
    else {
      val (l, r) = ps.splitAt(ps.length / 2)
      FilterApi.or(or(l), or(r))
    }

  /** Every row of the row group `b`. */
  def allRows(b: BlockMetaData): RowRanges =
    if (b.getRowCount == 0) RowRanges.EMPTY else RowRanges.createSingle(b.getRowCount)
}

/** A Parquet filter column for the top-level field `name`, whose values are of class `cls`.
  * `FilterApi`'s column factories split their argument on dots, so for a field whose name has one
  * they name a different column, which the statistics filter treats as missing.
  */
final private class TopLevelColumn[T <: Comparable[T]](name: String, cls: Class[T])
    extends Operators.Column[T](ColumnPath.get(name), cls) with Operators.SupportsLtGt

/** A reader of a columnar table or matrix table that can do a key-range read. */
trait ColumnarKeyRangeReader extends TableReader {

  /** This reader, reading only the rows it reads whose key is also in `intervals`, whose points are
    * prefixes of the key, and keeping its row partitions. `None` if it can't, because its own
    * key-range read makes each of its intervals a partition.
    */
  def restrictedTo(intervals: IndexedSeq[Interval]): Option[ColumnarKeyRangeReader]
}
