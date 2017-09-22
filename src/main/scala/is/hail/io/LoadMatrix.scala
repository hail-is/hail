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
      warn(s"Found ${duplicates.size} duplicate ${plural(duplicates.size, "sample ID")}:\n  @1",
        duplicates.toArray.sortBy(-_._2).map { case (id, count) => s"""($count) "$id"""" }.truncatable("\n  "))
    }
  }

  def placeholderCorrectnessFunction(str: String): Boolean = {
    // if there is anything that we want to check for on a per-line basis
    // check for number of elements in row? Do something else?
    true
  }

  // this assumes that col IDs are in last line of header.
  /// FIXME: Is the toString.split call too slow?
  def parseHeader(lines: Array[String], sep: String = "\t"): Array[String] = {
    lines.last.toString.split(sep)
  }

  def getHeaderLines[T](hConf: Configuration, file: String, nLines: Int = 1): Array[String] = hConf.readFile(file) { s =>
    Source.fromInputStream(s)
      .getLines()
      .take(nLines)
      .toArray
  }

  def apply(hc: HailContext,
            file1: String,
            files: Array[String],
            nPartitions: Option[Int] = None,
            dropSamples: Boolean = false,
            sep: String = "\t"):
  VariantSampleMatrix[Annotation, Annotation, Annotation] = {
    val sc = hc.sc
    val hConf = hc.hadoopConf

    // This imports header information from first file
    val headerLines1 = getHeaderLines(hConf, file1)
    val header1 = parseHeader(headerLines1, sep)
    val header1Bc = sc.broadcast(header1)

    val confBc = sc.broadcast(new SerializableHadoopConfiguration(hConf))

    sc.parallelize(files.tail, math.max(1, files.length - 1)).foreach { file =>

      //This imports header info from files. use reimplementation for col ID
      val hConf = confBc.value.value
      val hd = parseHeader(getHeaderLines(hConf, file), sep)
      val hd1 = header1Bc.value

      if (!hd1.sameElements(hd)) {
        hd1.zipAll(hd, None, None) //this compares sample id info from all files to first file for consistency
          .zipWithIndex.dropWhile { case ((s1, s2), i) => s1 == s2 }.toArray.headOption match {
          case Some(((s1, s2), i)) => fatal(
            s"""invalid sample ids: expected sample ids to be identical for all inputs. Found different sample ids at position $i
.
             |    ${files(0)}: $s1

             |    $file:
          $s2""".
              stripMargin)
        case None =>
        }
      }
    }

    //val VCFHeaderInfo(sampleIdsHeader, infoSignature, vaSignature, genotypeSignature, canonicalFlags) = header1

    val sampleIds: Array[String] = //dropSamples controls whether Sample IDs are stored? or whether samples are imported instead of just variant metadata?
      if (dropSamples)
        Array.empty
      else
        header1

    val nSamples = sampleIds.length

    LoadMatrix.warnDuplicates(sampleIds)

    val headerLinesBc = sc.broadcast(headerLines1) // ???

    val lines = sc.textFilesLines(files, nPartitions.getOrElse(sc.defaultMinPartitions)).filter(line => !headerLinesBc.value.exists(_.equals(line.value)))
    // this creates an RDD containing all lines from text files?
    // and also filters out anything matching the header line(s).
    // would subtract be better?
    /// FIXME: at some point, should probably become more sophisticated.

    val vsmMetadata = VSMMetadata(
      sSignature = TString,
      vSignature = TString,
      genotypeSignature = TInt64
    )
    val matrixType = MatrixType(vsmMetadata)

    val keyType = matrixType.kType

    val rowKeys: RDD[RegionValue] = lines.mapPartitions { it => //this will store just the row key.

      val region = MemoryBuffer()
      val rvb = new RegionValueBuilder(region)
      val rv = RegionValue(region)

      new Iterator[RegionValue] {
        var present = false

        def advance() {
          while (!present && it.hasNext) {
            it.next().foreach { line =>
              if (line.nonEmpty && placeholderCorrectnessFunction(line)) {
                val k = line.substring(0,line.indexOf(sep))
                region.clear()
                rvb.start(keyType)
                rvb.startStruct() // fk
                rvb.addString(k)
                rvb.addString(k)
                rvb.endStruct() // fk
                rv.setOffset(rvb.end())

                present = true
              }
            }
          }
        }

        def hasNext: Boolean = {
          if (!present)
            advance()
          present
        }

        def next(): RegionValue = {
          hasNext
          assert(present)
          present = false
          rv
        }
      }
    }

    val rdd = lines
      .mapPartitions { it => //this is the rdd that stores the data!

        val region = MemoryBuffer()
        val rvb = new RegionValueBuilder(region)
        val rv = RegionValue(region, 0) //this creates the RV block based for a partition in the rdd.

        new Iterator[RegionValue] {
          var present = false

          def advance() {
            while (!present && it.hasNext) {
              it.next().foreach { line =>
                if (line.nonEmpty && placeholderCorrectnessFunction(line)) {

                  val row = line.split(sep)
                  if (nSamples != 0 && row.length != (nSamples + 1)) {
                    fatal(
                      s"""Incorrect number of elements in line:
                         |     There are $nSamples column IDs but ${row.length} elements in line, including row ID.""".stripMargin)
                  }

                  region.clear()
                  rvb.start(matrixType.orderedRDD2Type.rowType.fundamentalType)
                  rvb.startStruct()
                  rvb.addString(row.head)
                  rvb.addString(row.head)
                  rvb.startStruct()
                  rvb.endStruct()

                  rvb.startArray(nSamples) // gs

                  if (nSamples > 0) {
                    for (v <- row.tail) {
                      rvb.addLong(v.trim.toLong)
                      /// FIXME: should we throw error if line does not have n+1 cells? If so, rewrite.
                    }
                  }
                  rvb.endArray()
                  rvb.endStruct() // row
                  rv.setOffset(rvb.end())

                  present = true
                }
              }
            }
          }

          def hasNext: Boolean = {
            if (!present)
              advance()
            present
          }

          def next(): RegionValue = {
            if (!present)
              advance()
            assert(present)
            present = false
            rv
          }
        }
      }

    // OrderedRDD2(rpk, rk, rowType, rdd, fastkeys, hintPartitioner??)
    val ordd = OrderedRDD2(matrixType.orderedRDD2Type, rdd, Some(rowKeys), None)

    new VariantSampleMatrix(hc,
      VSMMetadata(TString, vSignature = TString, genotypeSignature = TInt64),
      VSMLocalValue(Annotation.empty,
        sampleIds,
        Annotation.emptyIndexedSeq(sampleIds.length)),
      ordd)
  }
}
