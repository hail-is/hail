package is.hail.variant

import java.io.InputStream

import is.hail.check.Gen
import is.hail.expr.JSONExtractGenomeReference
import is.hail.utils._
import org.json4s._
import org.json4s.jackson.JsonMethods

case class GenomeReference(name: String, contigs: Array[Contig], xContigs: Set[String],
  yContigs: Set[String], mtContigs: Set[String]) extends Serializable {

  assert(contigs.length > 0, "Must have at least one contig in genome reference.")

  var par: Array[Interval[Locus]] = _

  val contigIndex: Map[String, Int] = contigs.map(_.name).zipWithIndex.toMap
  val contigNames: Array[String] = contigs.map(_.name)

  assert(xContigs.forall(contigNames.contains) && yContigs.forall(contigNames.contains) && mtContigs.forall(contigNames.contains))

  val xContigIndices = xContigs.map(contigIndex)
  val yContigIndices = yContigs.map(contigIndex)
  val mtContigIndices = mtContigs.map(contigIndex)

  def inX(contig: Int): Boolean = xContigIndices.contains(contig)
  def inX(contig: String): Boolean = xContigs.contains(contig)

  def inY(contig: Int): Boolean = yContigIndices.contains(contig)
  def inY(contig: String): Boolean = yContigs.contains(contig)

  def isMitochondrial(contig: Int): Boolean = mtContigIndices.contains(contig)
  def isMitochondrial(contig: String): Boolean = mtContigs.contains(contig)

  def inXPar(locus: Locus): Boolean = inX(locus.contig) && par.exists(_.contains(locus))

  def inYPar(locus: Locus): Boolean = inY(locus.contig) && par.exists(_.contains(locus))

  def toJSON: JValue = JObject(
    ("name", JString(name)),
    ("contigs", JArray(contigs.map(_.toJSON).toList)),
    ("xContigs", JArray(xContigs.map(JString(_)).toList)),
    ("yContigs", JArray(yContigs.map(JString(_)).toList)),
    ("mtContigs", JArray(mtContigs.map(JString(_)).toList)),
    ("par", JArray(par.map(_.toJSON(_.toJSON(this))).toList))
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
  @volatile implicit var genomeReference: GenomeReference = _

  def setReference(gr: GenomeReference) {
    if (genomeReference == null) {
      synchronized {
        genomeReference = gr
      }
    }
  }

  def GRCh37 = fromResource("reference/human_g1k_v37.json")

  def GRCh38 = fromResource("reference/Homo_sapiens_assembly38.json")

  def fromJSON(json: JValue): GenomeReference = json.extract[JSONExtractGenomeReference].toGenomeReference

  def fromResource(file: String): GenomeReference = loadFromResource[GenomeReference](file) {
    (is: InputStream) => fromJSON(JsonMethods.parse(is))
  }

  def gen: Gen[GenomeReference] = for {
    name <- Gen.identifier
    contigs <- Gen.distinctBuildableOfAtLeast[Array, Contig](1, Contig.gen)
  } yield GenomeReference(name, contigs, Set.empty[String], Set.empty[String], Set.empty[String])
}

