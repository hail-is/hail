package is.hail.variant

import java.io.InputStream

import is.hail.expr.JSONExtractGenomeReference
import is.hail.utils._
import org.json4s._
import org.json4s.jackson.JsonMethods

case class GenomeReference(name: String, contigs: Array[Contig], xContigs: Set[String],
  yContigs: Set[String], mtContigs: Set[String], par: Array[Interval[Locus]]) extends Serializable {

  def inX(contig: String): Boolean = xContigs.contains(contig)

  def inY(contig: String): Boolean = yContigs.contains(contig)

  def isMitochondrial(contig: String): Boolean = mtContigs.contains(contig)

  def inXPar(locus: Locus): Boolean = inX(locus.contig) && par.exists(_.contains(locus))

  def inYPar(locus: Locus): Boolean = inY(locus.contig) && par.exists(_.contains(locus))
}

object GenomeReference {
  def GRCh37 = fromResource("reference/human_g1k_v37.json")

  def fromJSON(json: JValue): GenomeReference = json.extract[JSONExtractGenomeReference].toGenomeReference

  def fromResource(file: String): GenomeReference = loadFromResource[GenomeReference](file) {
    (is: InputStream) => fromJSON(JsonMethods.parse(is))
  }
}

