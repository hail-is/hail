package org.broadinstitute.hail.io.annotators

import org.apache.hadoop
import org.broadinstitute.hail.Utils._
import org.broadinstitute.hail.annotations._
import org.broadinstitute.hail.expr._

import scala.collection.mutable

object SampleFamAnnotator {
  def apply(filename: String, delimiter: String, isQuantitative: Boolean,
    hConf: hadoop.conf.Configuration): (Map[String, Annotation], Type) = {
    readLines(filename, hConf) { lines =>
      fatalIf(lines.isEmpty, "Empty .fam file")

      val phenoSig = if (isQuantitative) ("qPheno", TDouble) else ("isCase", TBoolean)

      val signature = TStruct(("famID", TString), ("patID", TString), ("matID", TString), ("isMale", TBoolean), phenoSig)

      val kidSet = mutable.Set[String]()

      val m = lines.map {
        _.transform { l =>
          val split = l.value.split(delimiter)
          fatalIf(split.length != 6, s"Malformed .fam file: expected 6 fields in line ${l.position}, got ${split.length}")
          val Array(fam, kid, dad, mom, sex, pheno) = split

          if (kidSet(kid))
            fatal(s".fam sample name is not unique: $kid")
          else
            kidSet += kid

          val fam1 = if (fam != "0") fam else null
          val dad1 = if (dad != "0") dad else null
          val mom1 = if (mom != "0") mom else null
          val sex1 = sex match {
            case "0" => null
            case "1" => true
            case "2" => false
            case other => fatal(s"Invalid sex: `$other'. Legal values are `0', `1', or `2'.")
          }
          val pheno1 =
            if (isQuantitative)
              pheno.toDouble
            else
              pheno match {
                case "0" => null
                case "1" => false
                case "2" => true
                case "-9" => null
                case other =>
                  try {
                    other.toDouble
                    fatal(s"Invalid case-control phenotype: `$other'. Legal values are `0', `1', `2', `-9', and non-numeric.")
                  } catch {
                    case e: NumberFormatException => null
                  }
              }

          (kid, Annotation(fam1, dad1, mom1, sex1, pheno1))
        }
      }.toMap
      (m, signature)
    }
  }
}