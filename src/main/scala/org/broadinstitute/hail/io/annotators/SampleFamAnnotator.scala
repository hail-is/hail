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

      val signature = TStruct(("famID", TString), ("patID", TString), ("matID", TString), ("sex", TInt), phenoSig)

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

          val fam2 = if (fam != "0") fam else null
          val dad2 = if (dad != "0") dad else null
          val mom2 = if (mom != "0") mom else null
          val sex2 =
            if (sex == "1" || sex == "2")
              sex.toInt
            else if (sex == "0")
              null
            else
              fatal(s"Sex value must be `0', `1', or `2'") // FIXME
          val pheno2 =
            if (isQuantitative)
              pheno.toDouble
            else if (pheno == "1")
              false
            else if (pheno == "2")
              true
            else if (pheno == "0" || pheno == "-9")
              null
            else
              fatal(s"Blah")

          (kid, Annotation(fam2, dad2, mom2, sex2, pheno2))
        }
      }.toMap
      (m, signature)
    }
  }
}