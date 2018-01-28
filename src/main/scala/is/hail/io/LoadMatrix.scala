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

import scala.language.implicitConversions
import scala.io.Source

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
    fieldHeaders: Option[Seq[String]], noHeader: Boolean): (Array[String], Array[String]) = {
    val line = hConf.readFile(file) { s => Source.fromInputStream(s).getLines().next() }
    parseHeader(line.split(sep), nRowFields, fieldHeaders, noHeader)
  }

  def parseHeader(r: Array[String], nRowFields: Int,
    annotationHeaders: Option[Seq[String]], noHeader: Boolean=false): (Array[String], Array[String]) = {
    annotationHeaders match {
      case None =>
        if (r.length < nRowFields)
          fatal(s"Expected $nRowFields annotation columns; only ${ r.length } columns in table.")
        if (noHeader)
          (Array.tabulate(nRowFields) { i => "f"+i.toString() },
            Array.tabulate(r.length - nRowFields) { i => "col"+i.toString() })
        else
          (r.slice(0, nRowFields), r.slice(nRowFields, r.length))
      case Some(h) =>
        assert(h.length == nRowFields)
        (h.toArray, if (noHeader) Array.tabulate(r.length - h.length) { i => "col"+i.toString() } else r)
    }
  }

  def makePartitionerFromCounts(partitionCounts: Array[Long], kType: TStruct): OrderedRVDPartitioner = {
    val rvb = new RegionValueBuilder(Region())
    rvb.start(TArray(TStruct("pk" -> TInt64())))
    rvb.startArray(partitionCounts.length - 2)
    var c = 0L
    partitionCounts.tail.foreach { i =>
      c += i
      rvb.startStruct()
      rvb.addLong(c - 1)
      rvb.endStruct()
    }
    rvb.endArray()
    val ranges = new UnsafeIndexedSeq(TArray(TStruct("pk" -> TInt64())), rvb.region, rvb.end())
    val partitioner = new OrderedRVDPartitioner(partitionCounts.length - 1,
      Array("pk"), kType, ranges)
  }

  def apply(hc: HailContext,
    files: Array[String],
    annotationHeaders: Option[Seq[String]],
    annotationTypes: Seq[Type],
    optKeyExpr: Option[String],
    nPartitions: Option[Int] = None,
    dropSamples: Boolean = false,
    cellType: TStruct = TStruct("x" -> TInt64()),
    missingValue: String = "NA",
    noHeader: Boolean = false): MatrixTable = {
    require(cellType.size == 1, "cellType can only have 1 field")

    val sep = '\t'
    val nAnnotations = annotationTypes.length

      assert(annotationTypes.forall { t =>
        t.isOfType(TString()) ||
          t.isOfType(TInt32()) ||
          t.isOfType(TInt64()) ||
          t.isOfType(TFloat32()) ||
          t.isOfType(TFloat64())
      })
    val sc = hc.sc
    val hConf = hc.hadoopConf

    val (annotationNames, header1) = parseHeader(hConf, files.head, sep, nAnnotations, annotationHeaders, noHeader)
    val symTab = annotationNames.zip(annotationTypes)
    val annotationType = TStruct(symTab: _*)
    val ec = EvalContext(symTab: _*)

    val header1Bc = sc.broadcast(header1)

    val sampleIds: Array[String] =
      if (dropSamples)
        Array.empty
      else
        header1

    val nSamples = sampleIds.length

    LoadMatrix.warnDuplicates(sampleIds)

    val lines = sc.textFilesLines(files, nPartitions.getOrElse(sc.defaultMinPartitions))

    val fileByPartition = lines.partitions.map(p => partitionPath(p))
    val firstPartitions = fileByPartition.zipWithIndex.filter {
      case (name, partition) => partition == 0 || fileByPartition(partition - 1) != name
    }.map { case (_, partition) => partition }.toSet

    val matrixType = MatrixType.fromParts(
      TStruct.empty(),
      colType = TStruct("col_id" -> TString()),
      colKey = Array("col_id"),
      rowType = TStruct("row_id" -> keyType) ++ annotationType,
      rowKey = Array("row_id"),
      rowPartitionKey = Array("row_id"),
      entryType = cellType)

    val partitionCounts = lines.filter(l => l.value.nonEmpty)
      .mapPartitionsWithIndex { (i, it) =>
        if (firstPartitions(i)) {
          if (!noHeader) {
            val hd1 = header1Bc.value
            val (annotationNamesCheck, hd) = parseHeader(it.next().value.split(sep), sep, nAnnotations, annotationHeaders)
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

    println(partitionCounts.foldLeft("")(_ + _.toString() + ", "))



    val (t, computeRowKey) = optKeyExpr.map(Parser.parseExpr(_, ec)) match {
      case Some((t, f)) => (t, { (rowBlock: Int, row: Long) => f() })
      case None => (TInt64(), { (rowBlock: Int, row: Long) => partitionCounts(rowBlock) + row })
    }

    val rdd = lines.filter(l => l.value.nonEmpty)
      .mapPartitionsWithIndex { (i, it) =>
        val region = Region()
        val rvb = new RegionValueBuilder(region)
        val rv = RegionValue(region)

        val at = matrixType.rowType.asInstanceOf[TStruct]
        var row = 0L
        if (firstPartitions(i) && !noHeader) {
          it.next()
        }

        it.zipWithIndex.map { case (v, row) =>
          val line = v.value
          var off = 0
          var missing = false

          def getString(file: String, rowID: Annotation, colNum: Int): String = {
            var newoff = line.indexOf(sep, off)
            if (newoff == -1) {
              newoff = line.length
            }
            val v = line.substring(off, newoff)
            off = newoff + 1
            if (v == missingValue){
              missing = true
              null
            } else v
          }

          def getInt(file: String, rowID: Annotation, colNum: Int): Int = {
            var newoff = off
            var v = 0
            var isNegative = false
            if (line(off) == sep) {
              fatal(s"Error parsing matrix. Invalid Int32 at column: $colNum, row: ${ keyType.str(rowID) } in file: $file")
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
                off = newoff + 1
                missing = true
                0
              } else {
                fatal(s"Error parsing matrix. Invalid Int32 at column: $colNum, row: ${ keyType.str(rowID) } in file: $file")
              }
            } else if (line.length == newoff || line(newoff) == sep) {
              off = newoff + 1
              if (isNegative) -v else v
            } else {
              fatal(s"Error parsing matrix. $v Invalid Int32 at column: $colNum, row: ${ keyType.str(rowID) } in file: $file")
            }
          }

          def getLong(file: String, rowID: Annotation, colNum: Int): Long = {
            var newoff = off
            var v = 0L
            var isNegative = false
            if (line(off) == sep) {
              fatal(s"Error parsing matrix. Invalid Int64 at column: $colNum, row: ${ keyType.str(rowID) } in file: $file")
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
                off = newoff + 1
                missing = true
                0L
              } else {
                fatal(s"Error parsing matrix. Invalid Int64 at column: $colNum, row: ${ keyType.str(rowID) } in file: $file")
              }
            } else if (line.length == newoff || line(newoff) == sep) {
              off = newoff + 1
              if (isNegative) -v else v
            } else {
              fatal(s"Error parsing matrix. Invalid Int64 at column: $colNum, row: ${ keyType.str(rowID) } in file: $file")
            }
          }

          def getFloat(file: String, rowID: Annotation, colNum: Int): Float = {
            var newoff = line.indexOf(sep, off)
            if (newoff == -1)
              newoff = line.length
            val v = line.substring(off, newoff)
            off = newoff + 1
            if (v == missingValue) {
              missing = true
              0.0F
            } else {
              try {
              v.toFloat
              } catch {
                case _: NumberFormatException => fatal(s"Error parsing matrix: $v is not a Float32. column: $colNum, row: ${ keyType.str(rowID) } in file: $file")
              }
            }
          }

          def getDouble(file: String, rowID: Annotation, colNum: Int): Double = {
            var newoff = line.indexOf(sep, off)
            if (newoff == -1)
              newoff = line.length
            val v = line.substring(off, newoff)
            off = newoff + 1
            if (v == missingValue) {
              missing = true
              0.0
            } else {
              try {
                v.toDouble
              } catch {
                case _: NumberFormatException => fatal(s"Error parsing matrix: $v is not a Float64. column: $colNum, row: ${ keyType.str(rowID) } in file: $file")
              }
            }
          }

          var ii = 0
          while (ii < nAnnotations) {
            val t = at.fieldType(ii)
            ec.set(ii, t match {
              case _: TInt32 => getInt(fileByPartition(i), null, ii)
              case _: TInt64 => getLong(fileByPartition(i), null, ii)
              case _: TFloat32 => getFloat(fileByPartition(i), null, ii)
              case _: TFloat64 => getDouble(fileByPartition(i), null, ii)
              case _: TString => getString(fileByPartition(i), null, ii)
            })
            ii += 1
          }

          val rowKey = computeRowKey(i, row)

          region.clear()
          rvb.start(matrixType.rvRowType)
          rvb.startStruct()
          rvb.addAnnotation(keyType, rowKey)
          ii = 1 // start at 1 because we include the one key
          while (ii < at.size) {
            rvb.addAnnotation(at.fieldType(ii), ec.a(ii - 1)) // subtract 1 for the key
            ii += 1
          }

          rvb.startArray(nSamples)
          if (nSamples > 0) {
            var ii = 0
            while (ii < nSamples) {
              if (off > line.length) {
                fatal(
                  s"""Incorrect number of elements in line:
                     |    expected $nSamples elements in row $rowKey but only $ii elements found.
                     |    in file ${ fileByPartition(i) }""".stripMargin
                )
              }
              rvb.startStruct()
              missing = false
              cellType.fields(0).typ match {
                case _: TInt32 =>
                  val v = getInt(fileByPartition(i), rowKey, ii + nAnnotations)
                  if (missing) rvb.setMissing() else rvb.addInt(v)
                case _: TInt64 =>
                  val v = getLong(fileByPartition(i), rowKey, ii + nAnnotations)
                  if (missing) rvb.setMissing() else rvb.addLong(v)
                case _: TFloat32 =>
                  val v = getFloat(fileByPartition(i), rowKey, ii + nAnnotations)
                  if (missing) rvb.setMissing() else rvb.addFloat(v)
                case _: TFloat64 =>
                  val v = getDouble(fileByPartition(i), rowKey, ii + nAnnotations)
                  if (missing) rvb.setMissing() else rvb.addDouble(v)
                case _: TString =>
                  val v = getString(fileByPartition(i), rowKey, ii + nAnnotations)
                  if (missing) rvb.setMissing() else rvb.addString(v)
              }
              rvb.endStruct()
              ii += 1
            }
            if (off < line.length) {
              fatal(
                s"""Incorrect number of elements in line:
                   |    expected $nSamples elements in row but more data found.
                   |    in file ${ fileByPartition(i) }""".stripMargin
              )
            }
          }
          rvb.endArray()
          rvb.endStruct()
          rv.setOffset(rvb.end())
          rv
        }
      }

    val orderedRVD = if (optKeyExpr.isEmpty) {
      val partitioner = makePartitionerFromCounts(partitionCounts, matrixType.orvdType.kType)
      OrderedRVD(matrixType.orvdType, partitioner, rdd)
    } else
      OrderedRVD(matrixType.orvdType, rdd, None, None)

    new MatrixTable(hc,
      matrixType,
      Annotation.empty,
      sampleIds.map(x => Annotation(x)),
      orderedRVD)
  }
}
