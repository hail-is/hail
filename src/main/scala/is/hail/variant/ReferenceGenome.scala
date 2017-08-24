package is.hail.variant

import java.io.InputStream

import is.hail.check.Gen
import is.hail.expr.JSONExtractReferenceGenome
import is.hail.utils._
import org.json4s._
import org.json4s.jackson.JsonMethods

import scala.collection.JavaConverters._
import scala.language.implicitConversions

abstract class RGBase extends Serializable {
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

  def unify(concrete: ReferenceGenome): Boolean

  def isBound: Boolean

  def clear(): Unit

  def subst(): ReferenceGenome
}

case class ReferenceGenome(name: String, contigs: Array[String], lengths: Map[String, Int], xContigs: Set[String],
  yContigs: Set[String], mtContigs: Set[String], par: Array[Interval[Locus]]) extends RGBase {

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
    fatal(s"The following contig names were found in both X and Y contigs: `${ xContigs.intersect(yContigs).mkString(", ") }'")

  if (xContigs.intersect(mtContigs).nonEmpty)
    fatal(s"The following contig names were found in both X and MT contigs: `${ xContigs.intersect(mtContigs).mkString(", ") }'")

  if (yContigs.intersect(mtContigs).nonEmpty)
    fatal(s"The following contig names were found in both Y and MT contigs: `${ yContigs.intersect(mtContigs).mkString(", ") }' in both yContigs and mtContigs.")

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
      case rg: ReferenceGenome =>
        name == rg.name &&
          contigs.sameElements(rg.contigs) &&
          lengths == rg.lengths &&
          xContigs == rg.xContigs &&
          yContigs == rg.yContigs &&
          mtContigs == rg.mtContigs &&
          par.sameElements(rg.par)
      case _ => false
    }
  }

  def unify(concrete: ReferenceGenome): Boolean = this == concrete

  def isBound: Boolean = true

  def clear() {}

  def subst(): ReferenceGenome = this
}

object ReferenceGenome {

  def GRCh37: ReferenceGenome = fromResource("reference/grch37.json")

  def GRCh38: ReferenceGenome = fromResource("reference/grch38.json")

  def fromJSON(json: JValue): ReferenceGenome = json.extract[JSONExtractReferenceGenome].toReferenceGenome

  def fromResource(file: String): ReferenceGenome = loadFromResource[ReferenceGenome](file) {
    (is: InputStream) => fromJSON(JsonMethods.parse(is))
  }

  def gen: Gen[ReferenceGenome] = for {
    name <- Gen.identifier
    nContigs <- Gen.choose(3, 50)
    contigs <- Gen.distinctBuildableOfN[Array, String](nContigs, Gen.identifier)
    lengths <- Gen.distinctBuildableOfN[Array, Int](nContigs, Gen.choose(1000000, 500000000))
    xContig <- Gen.oneOfSeq(contigs)
    yContig <- Gen.oneOfSeq((contigs.toSet - xContig).toSeq)
    mtContig <- Gen.oneOfSeq((contigs.toSet - xContig - yContig).toSeq)
    parX <- Gen.distinctBuildableOfN[Array, Interval[Locus]](2, Interval.gen(Locus.gen(Seq(xContig))))
    parY <- Gen.distinctBuildableOfN[Array, Interval[Locus]](2, Interval.gen(Locus.gen(Seq(yContig))))
  } yield ReferenceGenome(name, contigs, contigs.zip(lengths).toMap, Set(xContig), Set(yContig), Set(mtContig), parX ++ parY)

  def apply(name: java.lang.String, contigs: java.util.ArrayList[String], lengths: java.util.HashMap[String, Int],
    xContigs: java.util.ArrayList[String], yContigs: java.util.ArrayList[String],
    mtContigs: java.util.ArrayList[String], par: java.util.ArrayList[Interval[Locus]]): ReferenceGenome =
    ReferenceGenome(name, contigs.asScala.toArray, lengths.asScala.toMap, xContigs.asScala.toSet, yContigs.asScala.toSet, mtContigs.asScala.toSet,
      par.asScala.toArray)
}

case class RGVariable(var rg: ReferenceGenome = null) extends RGBase {

  override def toString = "ReferenceGenome"

  def unify(concrete: ReferenceGenome): Boolean = {
    if (rg == null) {
      rg = concrete
      true
    } else
      rg == concrete
  }

  def isBound: Boolean = rg != null

  def clear() {
    rg = null
  }

  def subst(): ReferenceGenome = {
    assert(rg != null)
    rg
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

