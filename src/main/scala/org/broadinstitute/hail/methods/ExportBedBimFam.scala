package org.broadinstitute.hail.methods

import org.broadinstitute.hail.variant._

import scala.collection.mutable

object ExportBedBimFam {

  def makeBedRow(gs: Iterable[Genotype]): Array[Byte] = {
    val ab = new mutable.ArrayBuilder.ofByte()
    var j = 0
    var b = 0
    for (g <- gs) {
      val i = g.gt match {
        case Some(0) => 3
        case Some(1) => 2
        case Some(2) => 0
        case _ => 1
      }
      b |= i << (j * 2)
      if (j == 3) {
        ab += b.toByte
        b = 0
        j = 0
      } else
        j += 1
    }
    if (j > 0)
      ab += b.toByte

    ab.result()
  }

  def makeBimRow(v: Variant): String = {
    val id = s"${v.contig}:${v.start}:${v.ref}:${v.alt}"
    s"""${v.contig}\t$id\t0\t${v.start}\t${v.alt}\t${v.ref}"""
  }

  def makeFamRow(s: String): String = {
    s"0\t$s\t0\t0\t0\t-9"
  }

  def makeFamRow(famID : String, s: String, patID: String, matID: String, sex: String, pheno: String): String = {
    s"$famID\t$s\t$patID\t$matID\t$sex\t$pheno"
  }

/**
  //Create family row for case/control
  def makeFamRow(famID: Option[String], sID: String, patID: Option[String], matID: Option[String], isFemale: Option[Boolean], isCase: Option[Boolean]): String = {
    val phenoString = isCase match{
      case Some(c) => if(c) "2" else "1"
      case None => defaultPheno
    }
    makeFamRow(famID,sID,patID,matID,isFemale,phenoString)
  }

  //Create family row for quantitative phenotype
  /**def makeFamRow(famID: Option[String], sID: String, patID: Option[String], matID: Option[String], isFemale: Option[Boolean], qPheno: Option[Double]): String = {
    val phenoString = qPheno match{
      case Some(pheno) => pheno.toString //Todo: Not sure what precision should be used here
      case None => defaultPheno
    }
    makeFamRow(famID,sID,patID,matID,isFemale,phenoString)
  }**/

  private def makeFamRow(famID: Option[Any], sID: String, patID: Option[Any], matID: Option[Any], isFemale: Option[Boolean], phenoString: Option[Any]): String = {
    "%s\t%s\t%s\t%s\t%d\t%s".format(
      famID.getOrElse(defaultFam).toString,
      sID,
      patID.getOrElse(defaultPatID).toString,
      matID.getOrElse(defaultMatID).toString,
      isFemale match{
        case Some(female) => if(female) 2 else 1
        case None => defaultSex
      },
      phenoString
    )
  }
**/
}
