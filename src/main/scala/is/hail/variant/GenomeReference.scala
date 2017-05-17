package is.hail.variant

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

  def fromResource(file: String): GenomeReference = {
    val resourceStream = Thread.currentThread().getContextClassLoader.getResourceAsStream(file)

    try {
      if (resourceStream == null) {
        throw new RuntimeException(s"Could not read genome reference file `$file'.")
      }

      val json = JsonMethods.parse(resourceStream)

      json.extract[JSONExtractGenomeReference].toGenomeReference

    } catch {
      case npe: NullPointerException =>
        throw new RuntimeException(s"Error while locating file $file", npe)
      case e: Exception =>
        throw new RuntimeException(s"Error loading data from $file", e)
    } finally {
      if (resourceStream != null) {
        try {
          resourceStream.close()
        } catch {
          case e: Exception =>
            throw new RuntimeException("Error closing hail genome reference resource stream", e)
        }
      }
    }
  }
}

