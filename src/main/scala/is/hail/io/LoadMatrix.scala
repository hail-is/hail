package is.hail.io

import is.hail.HailContext
import is.hail.annotations._
import is.hail.expr._
import is.hail.expr.types._
import is.hail.rvd.{OrderedRVD, OrderedRVDPartitioner}
import is.hail.utils._
import is.hail.variant._
import org.apache.hadoop.conf.Configuration
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.Row

import scala.language.implicitConversions
import scala.io.Source

class LoadMatrixParser(rvb: RegionValueBuilder, fieldTypes: Array[Type], entryType: TStruct, nCols: Int, missingValue: String, file: String) {

  assert(entryType.size == 1)

  val sep = '\t'
  val nFields: Int = fieldTypes.length
  val cellf: (String, Long, Int, Int) => Int = entryType.fieldType(0) match {
    case TInt32(_) => getInt
    case TInt64(_) => getLong
    case TFloat32(_) => getFloat
    case TFloat64(_) => getDouble
    case TString(_) => getString
  }


  def parseLine(line: String, rowNum: Long): Unit = {
    var ii = 0
    var off = 0
    while (ii < fieldTypes.length) {
      off = fieldTypes(ii) match {
        case TInt32(_) => getInt(line, rowNum, ii, off)
        case TInt64(_) => getLong(line, rowNum, ii, off)
        case TFloat32(_) => getFloat(line, rowNum, ii, off)
        case TFloat64(_) => getDouble(line, rowNum, ii, off)
        case TString(_) => getString(line, rowNum, ii, off)
      }
      if (off > line.length) {
        fatal(
          s"""Error parsing row fields in row $rowNum:
             |    expected $nFields fields but only $ii found.
             |    in file $file""".stripMargin
        )
      }
      ii += 1
    }

    ii = 0
    rvb.startArray(nCols)
    while (ii < nCols) {
      if (off > line.length) {
        fatal(
          s"""Incorrect number of entries in row $rowNum:
             |    expected $nCols entries but only $ii entries found.
             |    in file $file""".stripMargin
        )
      }
      rvb.startStruct()
      off = cellf(line, rowNum, ii, off)
      rvb.endStruct()
      ii += 1
    }
    if (off < line.length) {
      fatal(
        s"""Incorrect number of entries in row $rowNum:
           |    expected $nCols entries but more data found.
           |    in file $file""".stripMargin
      )
    }
    rvb.endArray()
  }

  def getString(line: String, rowNum: Long, colNum: Int, off: Int): Int = {
    var newoff = line.indexOf(sep, off)
    if (newoff == -1) {
      newoff = line.length
    }
    val v = line.substring(off, newoff)
    if (v == missingValue){
      rvb.setMissing()
    } else rvb.addString(v)
    newoff + 1
  }

  def getInt(line: String, rowNum: Long, colNum: Int, off: Int): Int = {
    var newoff = off
    var v = 0
    var isNegative = false
    if (line(off) == sep) {
      fatal(s"Error parsing matrix. Invalid Int32 at column: $colNum, row: $rowNum in file: $file")
    }
    if (line(off) == '-' || line(off) == '+') {
      isNegative = line(off) == '-'
      newoff += 1
    }
    while (newoff < line.length && line(newoff) >= '0' && line(newoff) <= '9') {
      v *= 10
      v += (line(newoff) - '0')
      newoff += 1
    }
    if (newoff == off) {
      while (newoff - off < missingValue.length && missingValue(newoff - off) == line(newoff)) {
        newoff += 1
      }

      if (newoff - off == missingValue.length && (line.length == newoff || line(newoff) == sep)) {
        rvb.setMissing()
      } else {
        fatal(s"Error parsing matrix. Invalid Int32 at column: $colNum, row: $rowNum in file: $file")
      }
    } else if (line.length == newoff || line(newoff) == sep) {
      if (isNegative) rvb.addInt(-v) else rvb.addInt(v)
    } else {
      fatal(s"Error parsing matrix. $v Invalid Int32 at column: $colNum, row: $rowNum in file: $file")
    }
    newoff + 1
  }

  def getLong(line: String, rowNum: Long, colNum: Int, off: Int): Int = {
    var newoff = off
    var v = 0L
    var isNegative = false
    if (line(off) == sep) {
      fatal(s"Error parsing matrix. Invalid Int64 at column: $colNum, row: $rowNum in file: $file")
    }
    if (line(off) == '-' || line(off) == '+') {
      isNegative = line(off) == '-'
      newoff += 1
    }
    while (newoff < line.length && line(newoff) >= '0' && line(newoff) <= '9') {
      v *= 10
      v += line(newoff) - '0'
      newoff += 1
    }
    if (newoff == off) {
      while (newoff - off < missingValue.length && missingValue(newoff - off) == line(newoff)) {
        newoff += 1
      }

      if (newoff - off == missingValue.length && (line.length == newoff || line(newoff) == sep)) {
        rvb.setMissing()
      } else {
        fatal(s"Error parsing matrix. Invalid Int64 at column: $colNum, row: $rowNum in file: $file")
      }
    } else if (line.length == newoff || line(newoff) == sep) {
      if (isNegative) rvb.addLong(-v) else rvb.addLong(v)
    } else {
      fatal(s"Error parsing matrix. Invalid Int64 at column: $colNum, row: $rowNum in file: $file")
    }
    newoff + 1
  }

  def getFloat(line: String, rowNum: Long, colNum: Int, off: Int): Int = {
    var newoff = line.indexOf(sep, off)
    if (newoff == -1)
      newoff = line.length
    val v = line.substring(off, newoff)
    if (v == missingValue) {
      rvb.setMissing()
    } else {
      try {
        rvb.addFloat(v.toFloat)
      } catch {
        case _: NumberFormatException => fatal(s"Error parsing matrix: $v is not a Float32. column: $colNum, row: $rowNum in file: $file")
      }
    }
    newoff + 1
  }

  def getDouble(line: String, rowNum: Long, colNum: Int, off: Int): Int = {
    var newoff = line.indexOf(sep, off)
    if (newoff == -1)
      newoff = line.length
    val v = line.substring(off, newoff)
    if (v == missingValue) {
      rvb.setMissing()
    } else {
      try {
        rvb.addDouble(v.toDouble)
      } catch {
        case _: NumberFormatException => fatal(s"Error parsing matrix: $v is not a Float64. column: $colNum, row: $rowNum in file: $file")
      }
    }
    newoff + 1
  }
}

object LoadMatrix {

  def warnDuplicates(ids: Array[String]) {
    val duplicates = ids.counter().filter(_._2 > 1)
    if (duplicates.nonEmpty) {
      warn(s"Found ${ duplicates.size } duplicate ${ plural(duplicates.size, "sample ID") }:\n  @1",
        duplicates.toArray.sortBy(-_._2).map { case (id, count) => s"""($count) "$id"""" }.truncatable("\n  "))
    }
  }

  // this assumes that col IDs are in last line of header.
  def parseHeader(hConf: Configuration, file: String, sep: Char, nRowFields: Int,
    fieldHeaders: Option[Array[String]], noHeader: Boolean): (Array[String], Array[String]) = {
    val line = hConf.readFile(file) { s => Source.fromInputStream(s).getLines().next() }
    parseHeaderFields(line.split(sep), nRowFields, fieldHeaders, noHeader)
  }

  def parseHeaderFields(cols: Array[String], nRowFields: Int,
    annotationHeaders: Option[Array[String]], noHeader: Boolean=false): (Array[String], Array[String]) = {
    annotationHeaders match {
      case None =>
        if (cols.length < nRowFields)
          fatal(s"Expected $nRowFields annotation columns; only ${ cols.length } columns in table.")
        if (noHeader)
          (Array.tabulate(nRowFields) { i => "f"+i.toString() },
            Array.tabulate(cols.length - nRowFields) { i => "col"+i.toString() })
        else
          (cols.slice(0, nRowFields), cols.slice(nRowFields, cols.length))
      case Some(h) =>
        assert(h.length == nRowFields)
        (h.toArray, if (noHeader) Array.tabulate(cols.length - h.length) { i => "col"+i.toString() } else cols)
    }
  }

  def makePartitionerFromCounts(partitionCounts: Array[Long], pkType: TStruct): (OrderedRVDPartitioner, Array[Int]) = {
    var includeStart = true
    val keepPartitions = new ArrayBuilder[Int]()
    val rangeBoundIntervals = partitionCounts.zip(partitionCounts.tail).zipWithIndex.flatMap { case ((s, e), i) =>
      val interval = Interval(Row(if (includeStart) s else s - 1), Row(e - 1), includeStart, true)
      includeStart = false
      if (interval.isEmpty(pkType.ordering))
        None
      else {
        keepPartitions += i
        Some(interval)
      }
    }
    val ranges = UnsafeIndexedSeq(TArray(TInterval(pkType)), rangeBoundIntervals)
    (new OrderedRVDPartitioner(Array(pkType.fieldNames(0)), pkType, ranges), keepPartitions.result())
  }

  def apply(hc: HailContext,
    files: Array[String],
    annotationHeaders: Option[Array[String]],
    fieldTypes: Array[Type],
    keyFields: Option[Array[String]],
    nPartitions: Option[Int] = None,
    dropSamples: Boolean = false,
    cellType: TStruct = TStruct("x" -> TInt64()),
    missingValue: String = "NA",
    noHeader: Boolean = false): MatrixTable = {
    require(cellType.size == 1, "cellType can only have 1 field")

    val sep = '\t'
    val nAnnotations = fieldTypes.length

      assert(fieldTypes.forall { t =>
        t.isOfType(TString()) ||
          t.isOfType(TInt32()) ||
          t.isOfType(TInt64()) ||
          t.isOfType(TFloat32()) ||
          t.isOfType(TFloat64())
      })
    val sc = hc.sc
    val hConf = hc.hadoopConf

    val (annotationNames, header1) = parseHeader(hConf, files.head, sep, nAnnotations, annotationHeaders, noHeader)
    val symTab = annotationNames.zip(fieldTypes)
    val annotationType = TStruct(symTab: _*)
    val ec = EvalContext(symTab: _*)

    val header1Bc = sc.broadcast(header1)

    val sampleIds: Array[String] =
      if (dropSamples)
        Array.empty
      else
        header1

    val nCols = sampleIds.length

    LoadMatrix.warnDuplicates(sampleIds)

    val lines = sc.textFilesLines(files, nPartitions.getOrElse(sc.defaultMinPartitions))

    val fileByPartition = lines.partitions.map(p => partitionPath(p))
    val firstPartitions = fileByPartition.scanLeft(0) { case(i, file) => if (fileByPartition(i) == file) i else i + 1 }.tail

    val partitionCounts = lines.filter(l => l.value.nonEmpty)
      .mapPartitionsWithIndex { (i, it) =>
        if (firstPartitions(i) == i) {
          if (!noHeader) {
            val hd1 = header1Bc.value
            val (annotationNamesCheck, hd) = parseHeaderFields(it.next().value.split(sep), nAnnotations, annotationHeaders)
            if (!annotationNames.sameElements(annotationNamesCheck)) {
              fatal("column headers for annotations must be the same across files.")
            }
            if (!hd1.sameElements(hd)) {
              if (hd1.length != hd.length) {
                fatal(
                  s"""invalid sample ids: lengths of headers differ.
                     |    ${ hd1.length } elements in ${ files(0) }
                     |    ${ hd.length } elements in ${ fileByPartition(i) }
               """.stripMargin
                )
              }
              hd1.zip(hd).zipWithIndex.foreach { case ((s1, s2), j) =>
                if (s1 != s2) {
                  fatal(
                    s"""invalid sample ids: expected sample ids to be identical for all inputs. Found different sample ids at position $j.
                       |    ${ files(0) }: $s1
                       |    ${ fileByPartition(i) }: $s2""".
                      stripMargin)
                }
              }
            }
          }
        }
        it
      }.countPerPartition().scanLeft(0L)(_ + _)

    val useIndex = keyFields.isEmpty
    val rowKey = keyFields.getOrElse(Array("row_id"))
    val rowType = if (useIndex)
      TStruct("row_id" -> TInt64()) ++ annotationType
    else annotationType

    val matrixType = MatrixType.fromParts(
      TStruct.empty(),
      colType = TStruct("col_id" -> TString()),
      colKey = Array("col_id"),
      rowType = rowType,
      rowKey = rowKey.toFastIndexedSeq,
      rowPartitionKey = rowKey.toFastIndexedSeq,
      entryType = cellType)

    val rdd = lines.filter(l => l.value.nonEmpty)
      .mapPartitionsWithIndex { (i, it) =>
        val region = Region()
        val rvb = new RegionValueBuilder(region)
        val rv = RegionValue(region)

        if (firstPartitions(i) == i && !noHeader) { it.next() }

        val partitionStartInFile = partitionCounts(i) - partitionCounts(firstPartitions(i))
        val parser = new LoadMatrixParser(rvb, fieldTypes, cellType, nCols, missingValue, fileByPartition(i))

        it.zipWithIndex.map { case (v, row) =>
          val fileRowNum = partitionStartInFile + row
          val line = v.value

          region.clear()
          rvb.start(matrixType.rvRowType)
          rvb.startStruct()
          if (useIndex) {
            rvb.addLong(partitionCounts(i) + row)
          }
          parser.parseLine(line, fileRowNum)
          rvb.endStruct()
          rv.setOffset(rvb.end())
          rv
        }
      }

    val orderedRVD = if (useIndex) {
      val (partitioner, keepPartitions) = makePartitionerFromCounts(partitionCounts, matrixType.orvdType.pkType)
      OrderedRVD(matrixType.orvdType, partitioner, rdd.subsetPartitions(keepPartitions))
    } else
      OrderedRVD(matrixType.orvdType, rdd, None, None)

    new MatrixTable(hc,
      matrixType,
      Annotation.empty,
      sampleIds.map(x => Annotation(x)),
      orderedRVD)
  }
}
