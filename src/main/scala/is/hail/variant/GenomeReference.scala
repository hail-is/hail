package is.hail.variant

import java.io.InputStream

import is.hail.check.Gen
import is.hail.expr.JSONExtractGenomeReference
import is.hail.utils._
import org.json4s._
import org.json4s.jackson.JsonMethods

import scala.collection.JavaConverters._
import scala.language.implicitConversions

abstract class GRBase extends Serializable {
  def isValidContig(contig: String): Boolean

  def contigLength(contig: String): Int

  def contigLength(contigIdx: Int): Int

  def inX(contigIdx: Int): Boolean

  def inX(contig: String): Boolean

  def inY(contigIdx: Int): Boolean

  def inY(contig: String): Boolean

  def isMitochondrial(contigIdx: Int): Boolean

  def isMitochondrial(contig: String): Boolean

  def inXPar(locus: Locus): Boolean

  def inYPar(locus: Locus): Boolean

  def toJSON: JValue

  def unify(concrete: GenomeReference): Boolean

  def isBound: Boolean

  def clear(): Unit

  def subst(): GenomeReference
}

case class GenomeReference(name: String, contigs: Array[String], lengths: Map[String, Int], xContigs: Set[String],
  yContigs: Set[String], mtContigs: Set[String], par: Array[Interval[Locus]]) extends GRBase {

  val nContigs = contigs.length

  if (nContigs <= 0)
    fatal("Must have at least one contig in the reference genome.")

  if (!contigs.areDistinct())
    fatal("Repeated contig names are not allowed.")

  val missingLengths = contigs.toSet.diff(lengths.keySet)
  val extraLengths = lengths.keySet.diff(contigs.toSet)

  if (missingLengths.nonEmpty)
    fatal(s"No lengths given for the following contigs: ${ missingLengths.mkString(", ")}")

  if (extraLengths.nonEmpty)
    fatal(s"Contigs found in `lengths' that are not present in `contigs': ${ extraLengths.mkString(", ")}")

  if (xContigs.intersect(yContigs).nonEmpty)
    fatal(s"Found the contigs `${ xContigs.intersect(yContigs).mkString(", ") }' in both X and Y contigs.")

  if (xContigs.intersect(mtContigs).nonEmpty)
    fatal(s"Found the contigs `${ xContigs.intersect(mtContigs).mkString(", ") }' in both X and MT contigs.")

  if (yContigs.intersect(mtContigs).nonEmpty)
    fatal(s"Found the contigs `${ yContigs.intersect(mtContigs).mkString(", ") }' in both Y and MT contigs.")

  par.foreach { i =>
    if ((!xContigs.contains(i.start.contig) && !yContigs.contains(i.start.contig)) ||
      (!xContigs.contains(i.end.contig) && !yContigs.contains(i.end.contig)))
      fatal(s"The contig name for PAR interval `$i' was not found in xContigs `$xContigs' or in yContigs `$yContigs'.")
  }

  val contigsIndex: Map[String, Int] = contigs.zipWithIndex.toMap
  val contigsSet: Set[String] = contigs.toSet
  val lengthsByIndex: Array[Int] = contigs.map(lengths)

  lengths.foreach { case (n, l) =>
    if (l <= 0)
      fatal(s"Contig length must be positive. Contig `$n' has length equal to $l.")
  }

  val xNotInRef = xContigs.diff(contigsSet)
  val yNotInRef = yContigs.diff(contigsSet)
  val mtNotInRef = mtContigs.diff(contigsSet)

  if (xNotInRef.nonEmpty)
    fatal(s"The following X contig names are absent from the reference: `${ xNotInRef.mkString(", ") }'.")

  if (yNotInRef.nonEmpty)
    fatal(s"The following Y contig names are absent from the reference: `${ yNotInRef.mkString(", ") }'.")

  if (mtNotInRef.nonEmpty)
    fatal(s"The following mitochondrial contig names are absent from the reference: `${ mtNotInRef.mkString(", ") }'.")

  val xContigIndices = xContigs.map(contigsIndex)
  val yContigIndices = yContigs.map(contigsIndex)
  val mtContigIndices = mtContigs.map(contigsIndex)

  def contigLength(contig: String): Int = lengths.get(contig) match {
    case Some(l) => l
    case None => fatal(s"Invalid contig name: `$contig'.")
  }

  def contigLength(contigIdx: Int): Int = {
    require(contigIdx >= 0 && contigIdx < nContigs)
    lengthsByIndex(contigIdx)
  }

  def isValidContig(contig: String): Boolean = contigsSet.contains(contig)

  def inX(contigIdx: Int): Boolean = xContigIndices.contains(contigIdx)

  def inX(contig: String): Boolean = xContigs.contains(contig)

  def inY(contigIdx: Int): Boolean = yContigIndices.contains(contigIdx)

  def inY(contig: String): Boolean = yContigs.contains(contig)

  def isMitochondrial(contigIdx: Int): Boolean = mtContigIndices.contains(contigIdx)

  def isMitochondrial(contig: String): Boolean = mtContigs.contains(contig)

  def inXPar(locus: Locus): Boolean = inX(locus.contig) && par.exists(_.contains(locus))

  def inYPar(locus: Locus): Boolean = inY(locus.contig) && par.exists(_.contains(locus))

  def toJSON: JValue = JObject(
    ("name", JString(name)),
    ("contigs", JArray(contigs.map(c => JObject(("name", JString(c)), ("length", JInt(lengths(c))))).toList)),
    ("xContigs", JArray(xContigs.map(JString(_)).toList)),
    ("yContigs", JArray(yContigs.map(JString(_)).toList)),
    ("mtContigs", JArray(mtContigs.map(JString(_)).toList)),
    ("par", JArray(par.map(_.toJSON(_.toJSON)).toList))
  )

  override def equals(other: Any): Boolean = {
    other match {
      case gr: GenomeReference =>
        name == gr.name &&
          contigs.sameElements(gr.contigs) &&
          lengths == gr.lengths &&
          xContigs == gr.xContigs &&
          yContigs == gr.yContigs &&
          mtContigs == gr.mtContigs &&
          par.sameElements(gr.par)
      case _ => false
    }
  }

  def unify(concrete: GenomeReference): Boolean = this == concrete

  def isBound: Boolean = true

  def clear() {}

  def subst(): GenomeReference = this
}

object GenomeReference {

  def GRCh37: GenomeReference = fromResource("reference/grch37.json")

  def GRCh38: GenomeReference = fromResource("reference/grch38.json")

  def fromJSON(json: JValue): GenomeReference = json.extract[JSONExtractGenomeReference].toGenomeReference

  def fromResource(file: String): GenomeReference = loadFromResource[GenomeReference](file) {
    (is: InputStream) => fromJSON(JsonMethods.parse(is))
  }

  def gen: Gen[GenomeReference] = for {
    name <- Gen.identifier
    nContigs <- Gen.choose(3, 50)
    contigs <- Gen.distinctBuildableOfN[Array, String](nContigs, Gen.identifier)
    lengths <- Gen.distinctBuildableOfN[Array, Int](nContigs, Gen.choose(1000000, 500000000))
    xContig <- Gen.oneOfSeq(contigs)
    yContig <- Gen.oneOfSeq((contigs.toSet - xContig).toSeq)
    mtContig <- Gen.oneOfSeq((contigs.toSet - xContig - yContig).toSeq)
    parX <- Gen.distinctBuildableOfN[Array, Interval[Locus]](2, Interval.gen(Locus.gen(Seq(xContig))))
    parY <- Gen.distinctBuildableOfN[Array, Interval[Locus]](2, Interval.gen(Locus.gen(Seq(yContig))))
  } yield GenomeReference(name, contigs, contigs.zip(lengths).toMap, Set(xContig), Set(yContig), Set(mtContig), parX ++ parY)

  def apply(name: java.lang.String, contigs: java.util.ArrayList[String], lengths: java.util.HashMap[String, Int],
    xContigs: java.util.ArrayList[String], yContigs: java.util.ArrayList[String],
    mtContigs: java.util.ArrayList[String], par: java.util.ArrayList[Interval[Locus]]): GenomeReference =
    GenomeReference(name, contigs.asScala.toArray, lengths.asScala.toMap, xContigs.asScala.toSet, yContigs.asScala.toSet, mtContigs.asScala.toSet,
      par.asScala.toArray)
}

case class GRVariable(var gr: GenomeReference = null) extends GRBase {

  override def toString = "GenomeReference"

  def unify(concrete: GenomeReference): Boolean = {
    if (gr == null) {
      gr = concrete
      true
    } else
      gr == concrete
  }

  def isBound: Boolean = gr != null

  def clear() {
    gr = null
  }

  def subst(): GenomeReference = {
    assert(gr != null)
    gr
  }

  def isValidContig(contig: String): Boolean = ???

  def contigLength(contig: String): Int = ???

  def contigLength(contigIdx: Int): Int = ???

  def inX(contigIdx: Int): Boolean = ???

  def inX(contig: String): Boolean = ???

  def inY(contigIdx: Int): Boolean = ???

  def inY(contig: String): Boolean = ???

  def isMitochondrial(contigIdx: Int): Boolean = ???

  def isMitochondrial(contig: String): Boolean = ???

  def inXPar(locus: Locus): Boolean = ???

  def inYPar(locus: Locus): Boolean = ???

  def toJSON: JValue = ???
}

