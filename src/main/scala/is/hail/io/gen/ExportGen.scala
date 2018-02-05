package is.hail.io.gen

import is.hail.annotations.{Querier, UnsafeRow}
import is.hail.expr.types.TString
import is.hail.variant.{ArrayGenotypeView, MatrixTable, RegionValueVariant, Variant}
import is.hail.utils._

object ExportGen {
  def apply(vsm: MatrixTable, path: String, precision: Int = 4) {
    def writeSampleFile() {
      // FIXME: should output all relevant sample annotations such as phenotype, gender, ...
      vsm.hc.hadoopConf.writeTable(path + ".sample",
        "ID_1 ID_2 missing" :: "0 0 0" :: vsm.stringSampleIds.map(s => s"$s $s 0").toList)
    }

    def writeGenFile() {
      val varidSignature = vsm.rowType.getOption("varid")
      val varidQuery: Querier = varidSignature match {
        case Some(_) =>
          val (t, q) = vsm.queryVA("va.varid")
          t match {
            case _: TString => q
            case _ => a => null
          }
        case None => a => null
      }

      val rsidSignature = vsm.rowType.getOption("rsid")
      val rsidQuery: Querier = rsidSignature match {
        case Some(_) =>
          val (t, q) = vsm.queryVA("va.rsid")
          t match {
            case _: TString => q
            case _ => a => null
          }
        case None => a => null
      }

      val localNSamples = vsm.numCols
      val localRVType = vsm.rvRowType
      val localRowType = vsm.rowType
      val localEntriesIndex = vsm.entriesIndex

      vsm.rvd.mapPartitions { it =>
        val sb = new StringBuilder
        val view = new ArrayGenotypeView(localRVType)
        val rvv = new RegionValueVariant(localRVType)
        val row = new UnsafeRow(localRowType)

        it.map { rv =>
          view.setRegion(rv)

          row.set(rv)
          rvv.setRegion(rv)

          sb.clear()
          sb.append(rvv.contig())
          sb += ' '
          sb.append(Option(varidQuery(row)).getOrElse(rvv.variantObject().toString))
          sb += ' '
          sb.append(Option(rsidQuery(row)).getOrElse("."))
          sb += ' '
          sb.append(rvv.position())
          sb += ' '
          sb.append(rvv.alleles()(0))
          sb += ' '
          sb.append(rvv.alleles()(1))

          var i = 0
          while (i < localNSamples) {
            view.setGenotype(i)
            if (view.hasGP) {
              sb += ' '
              sb.append(formatDouble(view.getGP(0), precision))
              sb += ' '
              sb.append(formatDouble(view.getGP(1), precision))
              sb += ' '
              sb.append(formatDouble(view.getGP(2), precision))
            } else
              sb.append(" 0 0 0")
            i += 1
          }
          sb.result()
        }
      }.writeTable(path + ".gen", vsm.hc.tmpDir, None)
    }

    writeSampleFile()
    writeGenFile()
  }
}
