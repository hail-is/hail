package is.hail.variant

import java.io.InputStream

import is.hail.check.Gen
import is.hail.expr.JSONExtractGenomeReference
import is.hail.utils._
import org.json4s._
import org.json4s.jackson.JsonMethods

case class GenomeReference(name: String, contigs: Array[Contig], xContigs: Set[String],
  yContigs: Set[String], mtContigs: Set[String], par: Array[Interval[Locus]]) extends Serializable {

  require(contigs.length > 0, "Must have at least one contig in the genome reference.")

  require(xContigs.intersect(yContigs).isEmpty,
    s"Found the contigs `${ xContigs.intersect(yContigs).mkString(", ") }' in both xContigs and yContigs.")
  require(xContigs.intersect(mtContigs).isEmpty,
    s"Found the contigs `${ xContigs.intersect(mtContigs).mkString(", ") }' in both xContigs and mtContigs.")
  require(yContigs.intersect(mtContigs).isEmpty,
    s"Found the contigs `${ yContigs.intersect(mtContigs).mkString(", ") }' in both yContigs and mtContigs.")

  par.foreach { i =>
    require((xContigs.contains(i.start.contig) || yContigs.contains(i.start.contig)) &&
      (xContigs.contains(i.end.contig) || yContigs.contains(i.end.contig)),
      s"The contig name for PAR interval `$i' was not found in xContigs `$xContigs' or in yContigs `$yContigs'.")
  }

  val contigIndex: Map[String, Int] = contigs.map(_.name).zipWithIndex.toMap
  val contigNames: Set[String] = contigs.map(_.name).toSet

  val xNotInRef = xContigs.diff(contigNames)
  val yNotInRef = yContigs.diff(contigNames)
  val mtNotInRef = mtContigs.diff(contigNames)

  require(xNotInRef.isEmpty, s"The following X contig names were not found in the reference: `${ xNotInRef.mkString(", ") }'.")
  require(yNotInRef.isEmpty, s"The following Y contig names were not found in the reference: `${ yNotInRef.mkString(", ") }'.")
  require(mtNotInRef.isEmpty, s"The following MT contig names were not found in the reference: `${ mtNotInRef.mkString(", ") }'.")

  val xContigIndices = xContigs.map(contigIndex)
  val yContigIndices = yContigs.map(contigIndex)
  val mtContigIndices = mtContigs.map(contigIndex)

  def isValidContig(contig: String): Boolean = contigNames.contains(contig)

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
    ("contigs", JArray(contigs.map(_.toJSON).toList)),
    ("xContigs", JArray(xContigs.map(JString(_)).toList)),
    ("yContigs", JArray(yContigs.map(JString(_)).toList)),
    ("mtContigs", JArray(mtContigs.map(JString(_)).toList)),
    ("par", JArray(par.map(_.toJSON(_.toJSON)).toList))
  )

  def same(other: GenomeReference): Boolean = {
    name == other.name &&
      contigs.sameElements(other.contigs) &&
      xContigs == other.xContigs &&
      yContigs == other.yContigs &&
      mtContigs == other.mtContigs &&
      par.sameElements(other.par)
  }
}

object GenomeReference {

  def GRCh37 = fromResource("reference/human_g1k_v37.json")

  def GRCh38 = fromResource("reference/Homo_sapiens_assembly38.json")

  def fromJSON(json: JValue): GenomeReference = json.extract[JSONExtractGenomeReference].toGenomeReference

  def fromResource(file: String): GenomeReference = loadFromResource[GenomeReference](file) {
    (is: InputStream) => fromJSON(JsonMethods.parse(is))
  }

  def gen: Gen[GenomeReference] = for {
    name <- Gen.identifier
    nContigs <- Gen.choose(3, 50)
    contigs <- Gen.distinctBuildableOfN[Array, Contig](nContigs, Contig.gen)
    contigNames = contigs.map(_.name).toSet
    xContig <- Gen.oneOfSeq(contigNames.toSeq)
    yContig <- Gen.oneOfSeq((contigNames - xContig).toSeq)
    mtContig <- Gen.oneOfSeq((contigNames - xContig - yContig).toSeq)
    parX <- Gen.distinctBuildableOfN[Array, Interval[Locus]](2, Interval.gen(Locus.gen(Seq(xContig))))
    parY <- Gen.distinctBuildableOfN[Array, Interval[Locus]](2, Interval.gen(Locus.gen(Seq(yContig))))
  } yield GenomeReference(name, contigs, Set(xContig), Set(yContig), Set(mtContig), parX ++ parY)
}

