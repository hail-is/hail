package is.hail.io

import is.hail.HailContext
import is.hail.annotations._
import is.hail.expr._
import is.hail.expr.types._
import is.hail.rvd.OrderedRVD
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
  def parseHeader(line: String, sep: Char = '\t', nAnnotations: Int, annotationHeaders: Option[Seq[String]]): (Array[String], Array[String]) =
    annotationHeaders match {
      case None =>
        val r = line.split(sep)
        if (r.length < nAnnotations)
          fatal(s"Expected $nAnnotations annotation columns; only ${ r.length } columns in table.")
        (r.slice(0, nAnnotations), r.slice(nAnnotations, r.length))
      case Some(h) =>
        assert(h.length == nAnnotations)
        (h.toArray, line.split(sep))
    }

  def getHeaderLines[T](hConf: Configuration, file: String, nLines: Int = 1): String = hConf.readFile(file) { s =>
    Source.fromInputStream(s).getLines().next()
  }

  def apply(hc: HailContext,
    files: Array[String],
    annotationHeaders: Option[Seq[String]],
    annotationTypes: Seq[Type],
    keyExpr: String,
    nPartitions: Option[Int] = None,
    dropSamples: Boolean = false,
    cellType: TStruct = TStruct("x" -> TInt64()),
    missingValue: String = "NA"): MatrixTable = {
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

    val (annotationNames, header1) = parseHeader(getHeaderLines(hConf, files.head), sep, nAnnotations, annotationHeaders)
    val symTab = annotationNames.zip(annotationTypes)
    val annotationType = TStruct(symTab: _*)
    val ec = EvalContext(symTab: _*)
    val (t, f) = Parser.parseExpr(keyExpr, ec)

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

    val matrixType = MatrixType(
      colType = TStruct("s" -> TString()),
      colKey = Array("s"),
      vType = t,
      vaType = annotationType,
      genotypeType = cellType)

    val keyType = matrixType.orderedRVType.kType

    val rdd = lines.filter(l => l.value.nonEmpty)
      .mapPartitionsWithIndex { (i, it) =>
        val region = Region()
        val rvb = new RegionValueBuilder(region)
        val rv = RegionValue(region)

        if (firstPartitions(i)) {
          val hd1 = header1Bc.value
          val (annotationNamesCheck, hd) = parseHeader(it.next().value, sep, nAnnotations, annotationHeaders)
          if (!annotationNames.sameElements(annotationNamesCheck)) {
            fatal("column headers for annotations must be the same accross files.")
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

        val at = matrixType.vaType.asInstanceOf[TStruct]

        it.map { v =>
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
              fatal(s"Error parsing matrix. Invalid Int32 at column: $colNum, row: ${ t.str(rowID) } in file: $file")
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
                fatal(s"Error parsing matrix. Invalid Int32 at column: $colNum, row: ${ t.str(rowID) } in file: $file")
              }
            } else if (line.length == newoff || line(newoff) == sep) {
              off = newoff + 1
              if (isNegative) -v else v
            } else {
              fatal(s"Error parsing matrix. $v Invalid Int32 at column: $colNum, row: ${ t.str(rowID) } in file: $file")
            }
          }

          def getLong(file: String, rowID: Annotation, colNum: Int): Long = {
            var newoff = off
            var v = 0L
            var isNegative = false
            if (line(off) == sep) {
              fatal(s"Error parsing matrix. Invalid Int64 at column: $colNum, row: ${ t.str(rowID) } in file: $file")
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
                fatal(s"Error parsing matrix. Invalid Int64 at column: $colNum, row: ${ t.str(rowID) } in file: $file")
              }
            } else if (line.length == newoff || line(newoff) == sep) {
              off = newoff + 1
              if (isNegative) -v else v
            } else {
              fatal(s"Error parsing matrix. Invalid Int64 at column: $colNum, row: ${ t.str(rowID) } in file: $file")
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
                case _: NumberFormatException => fatal(s"Error parsing matrix: $v is not a Float32. column: $colNum, row: ${ t.str(rowID) } in file: $file")
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
                case _: NumberFormatException => fatal(s"Error parsing matrix: $v is not a Float64. column: $colNum, row: ${ t.str(rowID) } in file: $file")
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

          val rowKey = f()

          region.clear()
          rvb.start(matrixType.rvRowType)
          rvb.startStruct()
          rvb.addAnnotation(t, rowKey)
          rvb.addAnnotation(t, rowKey)
          rvb.startStruct()
          ii = 0
          while (ii < at.size) {
            rvb.addAnnotation(at.fieldType(ii), ec.a(ii))
            ii += 1
          }
          rvb.endStruct()

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

    new MatrixTable(hc,
      matrixType,
      Annotation.empty,
      sampleIds.map(x => Annotation(x)),
      OrderedRVD(matrixType.orderedRVType, rdd, None, None))
  }
}
