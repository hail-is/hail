package is.hail.variant

import is.hail.expr.{JSONAnnotationImpex, TInterval}
import is.hail.utils._
import org.json4s._
import org.json4s.jackson.JsonMethods

import scala.reflect.ClassTag

case class GenomeReference(name: String, contigs: Array[Contig], xContigs: Array[String],
  yContigs: Array[String], mtContigs: Array[String], par: Interval[Locus]*) extends Serializable {

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

      val fields = json.asInstanceOf[JObject].obj.toMap

      def getAndCastJSON[T <: JValue](fname: String)(implicit tct: ClassTag[T]): T =
        fields.get(fname) match {
          case Some(t: T) => t
          case Some(other) =>
            fatal(
              s"""corrupt json: invalid metadata
               |  Expected `${ tct.runtimeClass.getName }' in field `$fname', but got `${ other.getClass.getName }'.""".stripMargin)
          case None =>
            fatal(
              s"""corrupt json: invalid metadata
               |  Missing field `$fname'.""".stripMargin)
        }

      val referenceName = getAndCastJSON[JString]("name").s

      val contigs = getAndCastJSON[JArray]("contigs")
        .arr
        .map {
          case JObject(List(("name", JString(name)), ("length", JInt(length)))) => Contig(name, length.toInt)
          case other => fatal(s"Contig schema did not match {'name': String, 'length': Int}. Found $other.")
        }.toArray

      val xContigs = getAndCastJSON[JArray]("x_contigs").arr.map { case JString(s) => s }.toArray
      val yContigs = getAndCastJSON[JArray]("y_contigs").arr.map { case JString(s) => s }.toArray
      val mtContigs = getAndCastJSON[JArray]("mt_contigs").arr.map { case JString(s) => s }.toArray

      val par = getAndCastJSON[JArray]("par")
        .arr
        .map { jv => JSONAnnotationImpex.importAnnotation(jv, TInterval).asInstanceOf[Interval[Locus]] }
        .toArray

      GenomeReference(referenceName, contigs, xContigs, yContigs, mtContigs, par: _*)

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

