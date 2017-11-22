package is.hail.io

import is.hail.HailContext
import is.hail.annotations._
import is.hail.expr._
import is.hail.sparkextras.OrderedRDD2
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
  def parseHeader(line: String, sep: String = "\t", hasRowIDName: Boolean = false): Array[String] =
    if (hasRowIDName)
      line.split(sep).tail
    else
      line.split(sep)

  def getHeaderLines[T](hConf: Configuration, file: String, nLines: Int = 1): String = hConf.readFile(file) { s =>
    Source.fromInputStream(s).getLines().next()
  }

  def setString(string: String, off: Int, rvb: RegionValueBuilder, sep: Char, missingValue: String, file: String, rowID: String, colNum: Int): Int = {
    var newoff = string.indexOf(sep, off)
    if (newoff == -1) {
      newoff = string.length
    }
    val v = string.substring(off, newoff)
    if (v == missingValue) {
      rvb.setMissing()
    } else {
      rvb.addString(v)
    }
    newoff
  }

  def setInt(string: String, off: Int, rvb: RegionValueBuilder, sep: Char, missingValue: String, file: String, rowID: String, colNum: Int): Int = {
    var newoff = off
    var v = 0
    var isNegative = false
    if (string(off) == sep) {
      fatal(s"Error parsing matrix. Invalid Int32 at column: $colNum, row: $rowID in file: $file")
    }
    if (string(off) == '-' || string(off) == '+') {
      isNegative = string(off) == '-'
      newoff += 1
    }
    while (newoff < string.length && string(newoff) >= '0' && string(newoff) <= '9') {
      v *= 10
      v += string(newoff) - '0'
      newoff += 1
    }
    if (newoff == off) {
      while (newoff - off < missingValue.length && missingValue(newoff - off) == string(newoff)) {
        newoff += 1
      }
      if (newoff - off == missingValue.length && (string.length == newoff || string(newoff) == sep)) {
        rvb.setMissing()
        newoff
      } else {
        fatal(s"Error parsing matrix. Invalid Int32 at column: $colNum, row: $rowID in file: $file")
      }
    } else if (string.length == newoff || string(newoff) == sep) {
      rvb.addInt(if (isNegative) -v else v)
      newoff
    } else {
      fatal(s"Error parsing matrix. Invalid Int32 at column: $colNum, row: $rowID in file: $file")
    }
  }

  def setLong(string: String, off: Int, rvb: RegionValueBuilder, sep: Char, missingValue: String, file: String, rowID: String, colNum: Int): Int = {
    var newoff = off
    var v = 0L
    var isNegative = false
    if (string(off) == sep) {
      fatal(s"Error parsing matrix. Invalid Int64 at column: $colNum, row: $rowID in file: $file")
    }
    if (string(off) == '-' || string(off) == '+') {
      isNegative = string(off) == '-'
      newoff += 1
    }
    while (newoff < string.length && string(newoff) >= '0' && string(newoff) <= '9') {
      v *= 10
      v += string(newoff) - '0'
      newoff += 1
    }
    if (newoff == off) {
      while (newoff - off < missingValue.length && missingValue(newoff - off) == string(newoff)) {
        newoff += 1
      }
      if (newoff - off == missingValue.length && (string.length == newoff || string(newoff) == sep)) {
        rvb.setMissing()
        newoff
      } else {
        fatal(s"Error parsing matrix. Invalid Int64 at column: $colNum, row: $rowID in file: $file")
      }
    } else if (string.length == newoff || string(newoff) == sep) {
      rvb.addLong(if (isNegative) -v else v)
      newoff
    } else {
      fatal(s"Error parsing matrix. Invalid Int64 at column: $colNum, row: $rowID in file: $file")
    }
  }

  def setFloat(string: String, off: Int, rvb: RegionValueBuilder, sep: Char, missingValue: String, file: String, rowID: String, colNum: Int): Int = {
    var newoff = string.indexOf(sep, off)
    if (newoff == -1) {
      newoff = string.length
    }
    val v = string.substring(off, newoff)
    if (v == missingValue) {
      rvb.setMissing()
    } else {
      try {
        rvb.addFloat(v.toFloat)
      } catch {
        case _: NumberFormatException => fatal(s"Error parsing matrix: $v is not a Float32. column: $colNum, row: $rowID in file: $file")
      }
    }
    newoff
  }

  def setDouble(string: String, off: Int, rvb: RegionValueBuilder, sep: Char, missingValue: String, file: String, rowID: String, colNum: Int): Int = {
    var newoff = string.indexOf(sep, off)
    if (newoff == -1) {
      newoff = string.length
    }
    val v = string.substring(off, newoff)
    if (v == missingValue) {
      rvb.setMissing()
    } else {
      try {
        rvb.addDouble(v.toDouble)
      } catch {
        case _: NumberFormatException => fatal(s"Error parsing matrix: $v is not a Float64! column: $colNum, row: $rowID in file: $file")
      }
    }
    newoff
  }

  def apply(hc: HailContext,
    files: Array[String],
    nPartitions: Option[Int] = None,
    dropSamples: Boolean = false,
    cellType: Type = TInt64(),
    sep: String = "\t",
    missingValue: String = "NA",
    hasRowIDName: Boolean = false): VariantSampleMatrix = {

    val cellParser: (String, Int, RegionValueBuilder, Char, String, String, String, Int) => Int = cellType match {
      case _: TInt32 => setInt
      case _: TInt64 => setLong
      case _: TFloat32 => setFloat
      case _: TFloat64 => setDouble
      case _: TString => setString
      case _ =>
      fatal(
        s"""expected cell type Int32, Int64, Float32, Float64, or String but got:
           |    ${ cellType.toPrettyString() }
         """.stripMargin)
    }

    val sc = hc.sc
    val hConf = hc.hadoopConf

    val header1 = parseHeader(getHeaderLines(hConf, files.head), sep, hasRowIDName)
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

    val matrixType = MatrixType(VSMMetadata(
      sSignature = TString(),
      vSignature = TString(),
      genotypeSignature = cellType
    ))

    val keyType = matrixType.kType
    val rowKeys: RDD[RegionValue] = lines.mapPartitionsWithIndex { (i, it) =>
      if (firstPartitions(i)) {
        val hd1 = header1Bc.value
        val hd = parseHeader(it.next().value, sep, hasRowIDName)
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

      val region = MemoryBuffer()
      val rvb = new RegionValueBuilder(region)
      val rv = RegionValue(region)
      it.map { v =>
        val line = v.value
        if (line.nonEmpty) {
          var newoff = line.indexOf(sep)
          if (newoff == -1)
            newoff = line.length
          val k = line.substring(0, newoff)
          region.clear()
          rvb.start(keyType)
          rvb.startStruct()
          rvb.addString(k)
          rvb.addString(k)
          rvb.endStruct()
          rv.setOffset(rvb.end())
        }
        rv
      }
    }

    val rdd = lines.filter(l => l.value.nonEmpty)
      .mapPartitionsWithIndex { (i, it) =>
        val region = MemoryBuffer()
        val rvb = new RegionValueBuilder(region)
        val rv = RegionValue(region)

        if (firstPartitions(i))
          it.next()

        it.map { v =>
          val line = v.value
          var newoff = line.indexOf(sep)
          if (newoff == -1)
            newoff = line.length
          val rowKey = line.substring(0, newoff)

          region.clear()
          rvb.start(matrixType.rowType)
          rvb.startStruct()

          rvb.addString(rowKey)
          rvb.addString(rowKey)
          rvb.startStruct()
          rvb.endStruct()

          rvb.startArray(nSamples)
          if (nSamples > 0) {
            var off = rowKey.length() + 1
            var ii = 0
            while (ii < nSamples) {
              if (off > line.length) {
                fatal(
                  s"""Incorrect number of elements in line:
                     |    expected $nSamples elements in row $rowKey but only $ii elements found.
                     |    in file ${ fileByPartition(i) }""".stripMargin
                )
              }
              off = cellParser(line, off, rvb, sep(0), missingValue, fileByPartition(i), rowKey, ii)
              ii += 1
              off += 1
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

    new VariantSampleMatrix(hc,
      VSMMetadata(TString(), vSignature = TString(), genotypeSignature = cellType),
      VSMLocalValue(Annotation.empty,
        sampleIds,
        Annotation.emptyIndexedSeq(sampleIds.length)),
      OrderedRDD2(matrixType.orderedRDD2Type, rdd, Some(rowKeys), None))
  }
}
