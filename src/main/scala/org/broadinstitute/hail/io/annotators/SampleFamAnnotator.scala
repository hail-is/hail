package org.broadinstitute.hail.io.annotators

import org.apache.hadoop
import org.broadinstitute.hail.Utils._
import org.broadinstitute.hail.annotations._
import org.broadinstitute.hail.expr._

import scala.collection.mutable

object SampleFamAnnotator {
  //Matches decimal numbers, including scientific notation
  val numericRegex = """^-?(?:\d+|\d*\.\d+)(?:[eE]-?\d+)?$""".r

  def apply(filename: String, delimiter: String, isQuantitative: Boolean, missing: String,
    hConf: hadoop.conf.Configuration): (Map[String, Annotation], Type) = {
    readLines(filename, hConf) { lines =>
      fatalIf(lines.isEmpty, "Empty .fam file")

      val phenoSig = if (isQuantitative) ("qPheno", TDouble) else ("isCase", TBoolean)

      val signature = TStruct(("famID", TString), ("patID", TString), ("matID", TString), ("isMale", TBoolean), phenoSig)

      val kidSet = mutable.Set[String]()

      val m = lines.map {
        _.transform { line =>
          val split = line.value.split(delimiter)
          fatalIf(split.length != 6, s"Malformed .fam file: expected 6 fields in line ${line.position}, got ${split.length}")
          val Array(fam, kid, dad, mom, isMale, pheno) = split

          if (kidSet(kid))
            fatal(s".fam sample name is not unique: $kid")
          else
            kidSet += kid

          val fam1 = if (fam != "0") fam else null
          val dad1 = if (dad != "0") dad else null
          val mom1 = if (mom != "0") mom else null
          val isMale1 = isMale match {
            case "0" => null
            case "1" => true
            case "2" => false
            case _ => line.fatal(s"Invalid sex: `$isMale'. Male is `1', female is `2', unkown is `0'")
          }
          val pheno1 =
            if (isQuantitative)
              pheno match {
                case `missing` => null
                case numericRegex() => pheno.toDouble
                case _ => line.fatal(s"Invalid quantitative phenotype: `$pheno'. Value must be numeric or `$missing'")
              }
            else
              pheno match {
                case `missing` => null
                case "1" => false
                case "2" => true
                case "0" => null
                case "-9" => null
                case numericRegex() => line.fatal(s"Invalid case-control phenotype: `$pheno'. Control is `1', case is `2', missing is `0', `-9', `$missing', or non-numeric.")
                case _ => null
              }

          (kid, Annotation(fam1, dad1, mom1, isMale1, pheno1))
        }
      }.toMap
      (m, signature)
    }
  }
}