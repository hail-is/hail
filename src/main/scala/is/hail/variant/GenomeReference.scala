package is.hail.variant

import is.hail.expr.JSONAnnotationImpex
import is.hail.utils._
import org.apache.spark.sql.Row
import org.json4s._
import org.json4s.jackson.JsonMethods

import scala.reflect.ClassTag

case class GenomeReference(name: String, contigs: Array[Contig]) extends Serializable {
  val xContigs = contigs.filter(_.isX)
  val yContigs = contigs.filter(_.isY)
  val mtContigs = contigs.filter(_.isMT)
  val autosomalContigs = contigs.filterNot(c => c.isX && c.isY && c.isMT)

  def inX(contig: String): Boolean = xContigs.exists(_.name == contig)

  def inY(contig: String): Boolean = yContigs.exists(_.name == contig)

  def isMitochondrial(contig: String): Boolean = mtContigs.exists(_.name == contig)

  def inXPar(locus: Locus): Boolean = xContigs.exists(_.par.exists(_.contains(locus)))

  def inYPar(locus: Locus): Boolean = yContigs.exists(_.par.exists(_.contains(locus)))
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

      val fields = json.asInstanceOf[JObject].obj.toMap

      def getAndCastJSON[T <: JValue](fields: Map[String, JValue], fname: String)(implicit tct: ClassTag[T]): T =
        fields.get(fname) match {
          case Some(t: T) => t
          case Some(other) =>
            fatal(
              s"""corrupt json: invalid metadata
               |  Expected `${ tct.runtimeClass.getName }' in field `$fname', but got `${ other.getClass.getName }'""".stripMargin)
          case None =>
            fatal(
              s"""corrupt json: invalid metadata
               |  Missing field `$fname'""".stripMargin)
        }

      val name = getAndCastJSON[JString](fields, "name").s

      val contigs = getAndCastJSON[JArray](fields, "contigs")
        .arr
        .map { jv => Contig.fromRow(JSONAnnotationImpex.importAnnotation(jv, Contig.t).asInstanceOf[Row]) }
        .toArray

      GenomeReference(name, contigs)

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

