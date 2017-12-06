package is.hail.variant

import java.io.InputStream

import is.hail.HailContext
import is.hail.check.Gen
import is.hail.expr.{JSONExtractGenomeReference, TInterval, TLocus, TVariant}
import is.hail.utils._
import org.json4s._
import org.json4s.jackson.JsonMethods

import scala.collection.JavaConverters._
import scala.collection.mutable
import scala.language.implicitConversions

abstract class GRBase extends Serializable {
  val variant: TVariant = TVariant(this)
  val locus: TLocus = TLocus(this)
  val interval: TInterval = TInterval(this)

  def name: String

  def isValidContig(contig: String): Boolean

  def isValidPosition(contig: String, start: Int): Boolean

  def checkVariant(v: Variant): Unit

  def checkLocus(l: Locus): Unit

  def checkLocus(contig: String, pos: Int): Unit

  def checkInterval(i: Interval[Locus]): Unit

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

  def unify(concrete: GRBase): Boolean

  def isBound: Boolean

  def clear(): Unit

  def subst(): GRBase
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

    if (!isValidPosition(i.start.contig, i.start.position))
      fatal(s"Invalid PAR interval `$i' found. Start `${ i.start }' is not within the range [1-${ contigLength(i.start.contig) }] for reference genome `$name'.")
    if (!isValidPosition(i.end.contig, i.end.position))
      fatal(s"Invalid PAR interval `$i' found. End `${ i.end }' is not within the range [1-${ contigLength(i.end.contig) }] for reference genome `$name'.")
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

  def isValidPosition(contig: String, pos: Int): Boolean = pos > 0 && pos <= contigLength(contig)

  def checkLocus(l: Locus): Unit = checkLocus(l.contig, l.position)

  def checkLocus(contig: String, pos: Int): Unit = {
    if (!isValidContig(contig))
      fatal(s"Invalid locus `$contig:$pos' found. Contig `$contig' is not in the reference genome `$name'.")
    if (!isValidPosition(contig, pos))
      fatal(s"Invalid locus `$contig:$pos' found. Position `$pos' is not within the range [1-${ contigLength(contig) }] for reference genome `$name'.")
  }

  def checkVariant(v: Variant): Unit = {
    if (!isValidContig(v.contig))
      fatal(s"Invalid variant `$v' found. Contig `${ v.contig }' is not in the reference genome `$name'.")
    if (!isValidPosition(v.contig, v.start))
      fatal(s"Invalid variant `$v' found. Start `${ v.start }' is not within the range [1-${ contigLength(v.contig) }] for reference genome `$name'.")
  }

  def checkVariant(contig: String, start: Int, ref: String, alt: String): Unit = {
    val v = s"$contig:$start:$ref:$alt"
    if (!isValidContig(contig))
      fatal(s"Invalid variant `$v' found. Contig `$contig' is not in the reference genome `$name'.")
    if (!isValidPosition(contig, start))
      fatal(s"Invalid variant `$v' found. Start `$start' is not within the range [1-${ contigLength(contig) }] for reference genome `$name'.")
  }

  def checkVariant(contig: String, start: Int, ref: String, alts: Array[String]): Unit = checkVariant(contig, start, ref, alts.mkString(","))

  def checkVariant(contig: String, start: Int, ref: String, alts: java.util.ArrayList[String]): Unit = checkVariant(contig, start, ref, alts.asScala.toArray)

  def checkInterval(i: Interval[Locus]): Unit = {
    if (!isValidContig(i.start.contig))
      fatal(s"Invalid interval `$i' found. Contig `${ i.start.contig }' is not in the reference genome `$name'.")
    if (!isValidContig(i.end.contig))
      fatal(s"Invalid interval `$i' found. Contig `${ i.end.contig }' is not in the reference genome `$name'.")
    if (!isValidPosition(i.start.contig, i.start.position))
      fatal(s"Invalid interval `$i' found. Start `${ i.start }' is not within the range [1-${ contigLength(i.start.contig) }] for reference genome `$name'.")
    if (!isValidPosition(i.end.contig, i.end.position))
      fatal(s"Invalid interval `$i' found. End `${ i.end }' is not within the range [1-${ contigLength(i.end.contig) }] for reference genome `$name'.")
  }

  def inX(contigIdx: Int): Boolean = xContigIndices.contains(contigIdx)

  def inX(contig: String): Boolean = xContigs.contains(contig)

  def inY(contigIdx: Int): Boolean = yContigIndices.contains(contigIdx)

  def inY(contig: String): Boolean = yContigs.contains(contig)

  def isMitochondrial(contigIdx: Int): Boolean = mtContigIndices.contains(contigIdx)

  def isMitochondrial(contig: String): Boolean = mtContigs.contains(contig)

  def inXPar(l: Locus): Boolean = inX(l.contig) && par.exists(_.contains(l))

  def inYPar(l: Locus): Boolean = inY(l.contig) && par.exists(_.contains(l))

  def toJSON: JValue = JObject(
    ("name", JString(name)),
    ("contigs", JArray(contigs.map(c => JObject(("name", JString(c)), ("length", JInt(lengths(c))))).toList)),
    ("xContigs", JArray(xContigs.map(JString(_)).toList)),
    ("yContigs", JArray(yContigs.map(JString(_)).toList)),
    ("mtContigs", JArray(mtContigs.map(JString(_)).toList)),
    ("par", JArray(par.map(_.toJSON(_.toJSON)).toList))
  )

  def validateContigRemap(contigMapping: Map[String, String]) {
    val badContigs = mutable.Set[(String, String)]()

    contigMapping.foreach { case (oldName, newName) =>
      if (!contigsSet.contains(newName))
        badContigs += ((oldName, newName))
    }

    if (badContigs.nonEmpty)
      fatal(s"Found ${ badContigs.size } ${ plural(badContigs.size, "contig mapping") } that do not have remapped contigs in the reference genome `$name':\n  " +
        s"@1", contigMapping.truncatable("\n  "))
  }

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

  def unify(concrete: GRBase): Boolean = this eq concrete

  def isBound: Boolean = true

  def clear() {}

  def subst(): GenomeReference = this

  override def toString: String = name
}

object GenomeReference {
  var references: Map[String, GenomeReference] = Map()
  val GRCh37: GenomeReference = fromResource("reference/grch37.json")
  val GRCh38: GenomeReference = fromResource("reference/grch38.json")
  var defaultReference = GRCh37

  def addReference(gr: GenomeReference) {
    if (references.contains(gr.name))
      fatal(s"Cannot add reference genome. Reference genome `${ gr.name }' already exists.")
    references += (gr.name -> gr)
  }

  def getReference(name: String): GenomeReference = {
    references.get(name) match {
      case Some(gr) => gr
      case None => fatal(s"No genome reference with name `$name' exists. Available references: `${ references.keys.mkString(", ") }'.")
    }
  }

  def hasReference(name: String): Boolean = references.contains(name)

  def setDefaultReference(gr: GenomeReference) {
    assert(references.contains(gr.name))
    defaultReference = gr
  }

  def setDefaultReference(hc: HailContext, grSource: String) {
    defaultReference =
      if (hasReference(grSource))
        getReference(grSource)
      else
        fromFile(hc, grSource)
  }

  def fromJSON(json: JValue): GenomeReference = json.extract[JSONExtractGenomeReference].toGenomeReference

  def fromResource(file: String): GenomeReference = {
    val gr = loadFromResource[GenomeReference](file) {
      (is: InputStream) => fromJSON(JsonMethods.parse(is))
    }
    addReference(gr)
    gr
  }

  def fromFile(hc: HailContext, file: String): GenomeReference = {
    val gr = hc.hadoopConf.readFile(file) { (is: InputStream) => fromJSON(JsonMethods.parse(is)) }
    addReference(gr)
    gr
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
    mtContigs: java.util.ArrayList[String], par: java.util.ArrayList[Interval[Locus]]): GenomeReference = {
    val gr = GenomeReference(name, contigs.asScala.toArray, lengths.asScala.toMap, xContigs.asScala.toSet, yContigs.asScala.toSet, mtContigs.asScala.toSet, par.asScala.toArray)
    addReference(gr)
    gr
  }
}

case class GRVariable(var gr: GRBase = null) extends GRBase {

  override def toString = "?GR"

  def unify(concrete: GRBase): Boolean = {
    if (gr == null) {
      gr = concrete
      true
    } else
      gr eq concrete
  }

  def isBound: Boolean = gr != null

  def clear() {
    gr = null
  }

  def subst(): GRBase = {
    assert(gr != null)
    gr
  }

  def name: String = ???

  def isValidContig(contig: String): Boolean = ???

  def isValidPosition(contig: String, start: Int): Boolean = ???

  def checkVariant(v: Variant): Unit = ???

  def checkLocus(l: Locus): Unit = ???

  def checkLocus(contig: String, pos: Int): Unit = ???

  def checkInterval(i: Interval[Locus]): Unit = ???
  
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

