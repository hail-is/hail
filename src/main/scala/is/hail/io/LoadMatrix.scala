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
  def parseHeader(lines: Array[String], sep: String = "\t"): Array[String] = {
    lines.last.toString.split(sep)
  }

  def getHeaderLines[T](hConf: Configuration, file: String, nLines: Int = 1): Array[String] = hConf.readFile(file) { s =>
    Source.fromInputStream(s)
      .getLines()
      .take(nLines)
      .toArray
  }

  def setString(string: String, off: Int, rvb: RegionValueBuilder, sep: String = "\t", missingValue: String = "NA"): Int = {
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

  def setInt32(string: String, off: Int, rvb: RegionValueBuilder, sep: String = "\t", missingValue: String = "NA"): Int = {
    var newoff = off
    var v = 0
    var isNegative = false
    if (string(off) == sep(0)) {
      return -1
    }
    if (string(off) == '-' || string(off) == '+') {
      isNegative = string(off) == '-'
      newoff += 1
    }
    while (newoff < string.length && string(newoff) >= '0' && string(newoff) <= '9') {
      v *= 10
      if (isNegative) {
        v -= string(newoff) - '0'
      } else {
        v += string(newoff) - '0'
      }
      newoff += 1
    }
    if (newoff == off) {
      while (newoff - off < missingValue.length && missingValue(newoff - off) == string(newoff)) {
        newoff += 1
      }
      if (newoff - off == missingValue.length && (string.length == newoff || string(newoff) == sep(0))) {
        rvb.setMissing()
        newoff
      } else {
        -1
      }
    } else if (string.length == newoff || string(newoff) == sep(0)) {
      rvb.addInt(v)
      newoff
    } else {
      -1
    }
  }

  def setInt64(string: String, off: Int, rvb: RegionValueBuilder, sep: String = "\t", missingValue: String = "NA"): Int = {
    var newoff = off
    var v = 0L
    var isNegative = false
    if (string(off) == sep(0)) {
      return -1
    }
    if (string(off) == '-' || string(off) == '+') {
      isNegative = string(off) == '-'
      newoff += 1
    }
    while (newoff < string.length && string(newoff) >= '0' && string(newoff) <= '9') {
      v *= 10
      if (isNegative) {
        v -= string(newoff) - '0'
      } else {
        v += string(newoff) - '0'
      }
      newoff += 1
    }
    if (newoff == off) {
      while (newoff - off < missingValue.length && missingValue(newoff - off) == string(newoff)) {
        newoff += 1
      }
      if (newoff - off == missingValue.length && (string.length == newoff || string(newoff) == sep(0))) {
        rvb.setMissing()
        newoff
      } else {
        -1
      }
    } else if (string.length == newoff || string(newoff) == sep(0)) {
      rvb.addLong(v)
      newoff
    } else {
      -1
    }
  }

  def setFloat32(string: String, off: Int, rvb: RegionValueBuilder, sep: String = "\t", missingValue: String = "NA"): Int = {
    var newoff = string.indexOf(sep, off)
    if (newoff == -1) {
      newoff = string.length
    }
    val v = string.substring(off, newoff)
    if (v == missingValue) {
      rvb.setMissing()
    } else {
      try {
        v.toFloat
      } catch {
        case _: NumberFormatException => return -1
      }
      rvb.addFloat(v.toFloat)

    }
    newoff
  }

  def setFloat64(string: String, off: Int, rvb: RegionValueBuilder, sep: String = "\t", missingValue: String = "NA"): Int = {
    var newoff = string.indexOf(sep, off)
    if (newoff == -1) {
      newoff = string.length
    }
    val v = string.substring(off, newoff)
    if (v == missingValue) {
      rvb.setMissing()
    } else {
      try {
        v.toDouble
      } catch {
        case _: NumberFormatException => return -1
      }
      rvb.addDouble(v.toDouble)

    }
    newoff
  }

  def apply(hc: HailContext,
    files: Array[String],
    nPartitions: Option[Int] = None,
    dropSamples: Boolean = false,
    cellType: Type = TInt64(),
    sep: String = "\t",
    missingValue: String = "NA"): VariantSampleMatrix[Annotation, Annotation, Annotation] = {

    if (!Set[Type](TInt64(), TInt32(), TFloat32(), TFloat64(), TString()).contains(cellType)) {
      fatal(
        s"""expected cell type Int32, Int64, Float32, Float64, or String but got:
           |    ${ cellType.toPrettyString() }
         """.stripMargin)
    }

    val sc = hc.sc
    val hConf = hc.hadoopConf

    val header1 = parseHeader(getHeaderLines(hConf, files.head), sep)
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
        val hd = it.next().value.split(sep)
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
          val k = line.substring(0, line.indexOf(sep))
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

    val rdd = lines
      .mapPartitionsWithIndex { (i, it) =>
        val region = MemoryBuffer()
        val rvb = new RegionValueBuilder(region)
        val rv = RegionValue(region, 0)

        if (firstPartitions(i))
          it.next()

        it.map { v =>
          val line = v.value
          if (line.nonEmpty) {
            val firstsep = line.indexOf(sep)

            region.clear()
            rvb.start(matrixType.orderedRDD2Type.rowType)
            rvb.startStruct()

            rvb.addString(line.substring(0, firstsep))
            rvb.addString(line.substring(0, firstsep))
            rvb.startStruct()
            rvb.endStruct()

            rvb.startArray(nSamples)
            if (nSamples > 0) {
              var off = firstsep + 1
              var v = 0L
              var ii = 0
              while (ii < nSamples) {
                if (off > line.length) {
                  fatal(
                    s"""Incorrect number of elements in line:
                       |    expected $nSamples elements in row but only $i elements found.
                       |    in file ${ fileByPartition(i) }""".stripMargin
                  )
                }
                off = cellType match {
                  case TString(_) => setString(line, off, rvb, sep, missingValue)
                  case TInt64(_) => setInt64(line, off, rvb, sep, missingValue)
                  case TInt32(_) => setInt32(line, off, rvb, sep, missingValue)
                  case TFloat64(_) => setFloat64(line, off, rvb, sep, missingValue)
                  case TFloat32(_) => setFloat32(line, off, rvb, sep, missingValue)
                }
                ii += 1
                if (off == -1 || (off != line.length && line(off) != sep(0))) {
                  fatal(
                    s"""found a bad input in file
                       |    ${ fileByPartition(i) }""".stripMargin
                  )
                }
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
          }
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
