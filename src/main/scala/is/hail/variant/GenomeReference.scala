package is.hail.variant

import java.io.InputStream

import is.hail.HailContext
import is.hail.check.Gen
import is.hail.expr.types._
import is.hail.expr.{JSONAnnotationImpex, JSONExtractContig, JSONExtractGenomeReference, JSONExtractIntervalLocus, Parser}
import is.hail.utils._
import org.json4s._
import org.json4s.jackson.{JsonMethods, Serialization}

import scala.collection.JavaConverters._
import scala.collection.mutable
import scala.language.implicitConversions
import is.hail.expr.Parser._
import is.hail.variant.CopyState.CopyState
import is.hail.variant.Sex.Sex
import org.apache.hadoop.conf.Configuration

abstract class GRBase extends Serializable {
  val variantType: TVariant
  val locusType: TLocus
  val intervalType: TInterval

  def name: String

  def variantOrdering: Ordering[Variant]

  def locusOrdering: Ordering[Locus]

  def contigParser: Parser[String]

  def isValidContig(contig: String): Boolean

  def checkVariant(v: Variant): Unit

  def checkVariant(contig: String, pos: Int, ref: String, alts: Array[String]): Unit

  def checkVariant(contig: String, start: Int, ref: String, alts: java.util.ArrayList[String]): Unit

  def checkVariant(contig: String, pos: Int, ref: String, alt: String): Unit

  def checkLocus(l: Locus): Unit

  def checkLocus(contig: String, pos: Int): Unit

  def checkInterval(i: Interval): Unit

  def checkInterval(l1: Locus, l2: Locus): Unit

  def checkInterval(contig: String, start: Int, end: Int): Unit

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

  def compare(c1: String, c2: String): Int

  def compare(v1: IVariant, v2: IVariant): Int

  def compare(l1: Locus, l2: Locus): Int

  def unify(concrete: GRBase): Boolean

  def isBound: Boolean

  def clear(): Unit

  def subst(): GRBase
}

case class GenomeReference(name: String, contigs: Array[String], lengths: Map[String, Int],
  xContigs: Set[String] = Set.empty[String], yContigs: Set[String] = Set.empty[String],
  mtContigs: Set[String] = Set.empty[String], parInput: Array[(Locus, Locus)] = Array.empty[(Locus, Locus)]) extends GRBase {

  val nContigs = contigs.length

  if (nContigs <= 0)
    fatal("Must have at least one contig in the reference genome.")

  if (!contigs.areDistinct())
    fatal("Repeated contig names are not allowed.")

  val missingLengths = contigs.toSet.diff(lengths.keySet)
  val extraLengths = lengths.keySet.diff(contigs.toSet)

  if (missingLengths.nonEmpty)
    fatal(s"No lengths given for the following contigs: ${ missingLengths.mkString(", ") }")

  if (extraLengths.nonEmpty)
    fatal(s"Contigs found in `lengths' that are not present in `contigs': ${ extraLengths.mkString(", ") }")

  if (xContigs.intersect(yContigs).nonEmpty)
    fatal(s"Found the contigs `${ xContigs.intersect(yContigs).mkString(", ") }' in both X and Y contigs.")

  if (xContigs.intersect(mtContigs).nonEmpty)
    fatal(s"Found the contigs `${ xContigs.intersect(mtContigs).mkString(", ") }' in both X and MT contigs.")

  if (yContigs.intersect(mtContigs).nonEmpty)
    fatal(s"Found the contigs `${ yContigs.intersect(mtContigs).mkString(", ") }' in both Y and MT contigs.")

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

  val variantOrdering = new Ordering[Variant] {
    def compare(x: Variant, y: Variant): Int = GenomeReference.compare(contigsIndex, x, y)
  }

  val locusOrdering = new Ordering[Locus] {
    def compare(x: Locus, y: Locus): Int = GenomeReference.compare(contigsIndex, x, y)
  }

  // must be constructed after orderings
  val variantType: TVariant = TVariant(this)
  val locusType: TLocus = TLocus(this)
  val intervalType: TInterval = TInterval(locusType)

  val par = parInput.map { case (start, end) =>
    if (start.contig != end.contig)
      fatal(s"The contigs for the `start' and `end' of a PAR interval must be the same. Found `$start-$end'.")

    if ((!xContigs.contains(start.contig) && !yContigs.contains(start.contig)) ||
      (!xContigs.contains(end.contig) && !yContigs.contains(end.contig)))
      fatal(s"The contig name for PAR interval `$start-$end' was not found in xContigs `${ xContigs.mkString(",") }' or in yContigs `${ yContigs.mkString(",") }'.")

    Interval(start, end, true, false)
  }

  def contigParser = Parser.oneOfLiteral(contigs)

  def contigLength(contig: String): Int = lengths.get(contig) match {
    case Some(l) => l
    case None => fatal(s"Invalid contig name: `$contig'.")
  }

  def contigLength(contigIdx: Int): Int = {
    require(contigIdx >= 0 && contigIdx < nContigs)
    lengthsByIndex(contigIdx)
  }

  def isValidContig(contig: String): Boolean = contigsSet.contains(contig)

  private def isValidPosition(contig: String, pos: Int): Boolean = pos > 0 && pos <= contigLength(contig)

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

  def checkInterval(i: Interval): Unit = {
    val start = i.start.asInstanceOf[Locus]
    val end = i.end.asInstanceOf[Locus]
    if (!isValidContig(start.contig))
      fatal(s"Invalid interval `$i' found. Contig `${ start.contig }' is not in the reference genome `$name'.")
    if (!isValidContig(end.contig))
      fatal(s"Invalid interval `$i' found. Contig `${ end.contig }' is not in the reference genome `$name'.")
    if (!isValidPosition(start.contig, start.position))
      fatal(s"Invalid interval `$i' found. Start `$start' is not within the range [1-${ contigLength(start.contig) }] for reference genome `$name'.")
    if (!isValidPosition(end.contig, end.position))
      fatal(s"Invalid interval `$i' found. End `$end' is not within the range [1-${ contigLength(end.contig) }] for reference genome `$name'.")
  }

  def checkInterval(l1: Locus, l2: Locus): Unit = {
    val i = s"$l1-$l2"
    if (!isValidPosition(l1.contig, l1.position))
      fatal(s"Invalid interval `$i' found. Locus `$l1' is not in the reference genome `$name'.")
    if (!isValidPosition(l2.contig, l2.position))
      fatal(s"Invalid interval `$i' found. Locus `$l2' is not in the reference genome `$name'.")
  }

  def checkInterval(contig: String, start: Int, end: Int): Unit = {
    val i = s"$contig:$start-$end"
    if (!isValidContig(contig))
      fatal(s"Invalid interval `$i' found. Contig `$contig' is not in the reference genome `$name'.")
    if (!isValidPosition(contig, start))
      fatal(s"Invalid interval `$i' found. Start `$start' is not within the range [1-${ contigLength(contig) }] for reference genome `$name'.")
    if (!isValidPosition(contig, end))
      fatal(s"Invalid interval `$i' found. End `$end' is not within the range [1-${ contigLength(contig) }] for reference genome `$name'.")
  }

  def inX(contigIdx: Int): Boolean = xContigIndices.contains(contigIdx)

  def inX(contig: String): Boolean = xContigs.contains(contig)

  def inY(contigIdx: Int): Boolean = yContigIndices.contains(contigIdx)

  def inY(contig: String): Boolean = yContigs.contains(contig)

  def copyState(sex: Sex, locus: Locus): CopyState = {
      // FIXME this seems wrong (no MT); I copied it from Variant
      if (sex == Sex.Male)
        if (inX(locus.contig) && !inXPar(locus))
          CopyState.HemiX
        else if (inY(locus.contig) && !inYPar(locus))
          CopyState.HemiY
        else
          CopyState.Auto
      else
        CopyState.Auto
  }

  def isMitochondrial(contigIdx: Int): Boolean = mtContigIndices.contains(contigIdx)

  def isMitochondrial(contig: String): Boolean = mtContigs.contains(contig)

  def inXPar(l: Locus): Boolean = inX(l.contig) && par.exists(_.contains(locusType.ordering, l))

  def inYPar(l: Locus): Boolean = inY(l.contig) && par.exists(_.contains(locusType.ordering, l))

  def compare(contig1: String, contig2: String): Int = GenomeReference.compare(contigsIndex, contig1, contig2)

  def compare(v1: IVariant, v2: IVariant): Int = GenomeReference.compare(contigsIndex, v1, v2)

  def compare(l1: Locus, l2: Locus): Int = GenomeReference.compare(contigsIndex, l1, l2)

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

  def write(hc: HailContext, file: String): Unit =
    hc.hadoopConf.writeTextFile(file) { out =>
      val jgr = JSONExtractGenomeReference(name,
        contigs.map(contig => JSONExtractContig(contig, contigLength(contig))),
        xContigs, yContigs, mtContigs,
        par.map(i => JSONExtractIntervalLocus(i.start.asInstanceOf[Locus], i.end.asInstanceOf[Locus])))
      implicit val formats = defaultJSONFormats
      Serialization.write(jgr, out)
    }
}

object GenomeReference {
  var references: Map[String, GenomeReference] = Map()
  val GRCh37: GenomeReference = fromResource("reference/grch37.json")
  val GRCh38: GenomeReference = fromResource("reference/grch38.json")
  var defaultReference = GRCh37
  val hailReferences = references.keySet

  def addReference(gr: GenomeReference) {
    if (hasReference(gr.name))
      fatal(s"Cannot add reference genome. `${ gr.name }' already exists. Choose a reference name NOT in the following list:\n  " +
        s"@1", references.keys.truncatable("\n  "))

    references += (gr.name -> gr)
  }

  def getReference(name: String): GenomeReference = {
    references.get(name) match {
      case Some(gr) => gr
      case None => fatal(s"Cannot get reference genome. `$name' does not exist. Choose a reference name from the following list:\n  " +
        s"@1", references.keys.truncatable("\n  "))
    }
  }

  def hasReference(name: String): Boolean = references.contains(name)

  def removeReference(name: String): Unit = {
    val nonBuiltInReferences = references.keySet -- hailReferences

    if (hailReferences.contains(name))
      fatal(s"Cannot remove reference genome. `$name' is a built-in Hail reference. Choose a reference name from the following list:\n  " +
        s"@1", nonBuiltInReferences.truncatable("\n  "))
    if (!hasReference(name))
      fatal(s"Cannot remove reference genome. `$name' does not exist. Choose a reference name from the following list:\n  " +
        s"@1", nonBuiltInReferences.truncatable("\n  "))

    references -= name
  }

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

  def read(is: InputStream): GenomeReference = {
    implicit val formats = defaultJSONFormats
    JsonMethods.parse(is).extract[JSONExtractGenomeReference].toGenomeReference
  }

  def fromResource(file: String): GenomeReference = {
    val gr = loadFromResource[GenomeReference](file)(read)
    addReference(gr)
    gr
  }

  def fromFile(hc: HailContext, file: String): GenomeReference = {
    val gr = hc.hadoopConf.readFile(file)(read)
    addReference(gr)
    gr
  }

  def importReferences(hConf: Configuration, path: String) {
    if (hConf.exists(path)) {
      val refs = hConf.listStatus(path)
      refs.foreach { fs =>
        val grPath = fs.getPath.toString
        val gr = hConf.readFile(grPath)(read)
        val name = gr.name
        if (!GenomeReference.hasReference(name))
          addReference(gr)
        else {
          if (GenomeReference.getReference(name) != gr)
            fatal(s"`$name' already exists and is not identical to the imported reference from `$grPath'.")
        }
      }
    }
  }

  private def writeReference(hc: HailContext, path: String, gr: GRBase) {
    val grPath = path + "/" + gr.name + ".json.gz"
    if (!hailReferences.contains(gr.name) && !hc.hadoopConf.exists(grPath))
      gr.asInstanceOf[GenomeReference].write(hc, grPath)
  }

  def exportReferences(hc: HailContext, path: String, t: Type) { (t: @unchecked) match {
      case TArray(elementType, _) => exportReferences(hc, path, elementType)
      case TSet(elementType, req) => exportReferences(hc, path, elementType)
      case TDict(keyType, valueType, _) =>
        exportReferences(hc, path, keyType)
        exportReferences(hc, path, valueType)
      case TStruct(fields, _) => fields.foreach(fd => exportReferences(hc, path, fd.typ))
      case TVariant(gr, _) => writeReference(hc, path, gr)
      case TLocus(gr, _) => writeReference(hc, path, gr)
      case TInterval(TLocus(gr, _), _) => writeReference(hc, path, gr)
      case _ =>
  }}

  def compare(contigsIndex: Map[String, Int], c1: String, c2: String): Int = {
    (contigsIndex.get(c1), contigsIndex.get(c2)) match {
      case (Some(i), Some(j)) => i.compare(j)
      case (Some(_), None) => -1
      case (None, Some(_)) => 1
      case (None, None) => c1.compare(c2)
    }
  }

  def compare(contigsIndex: Map[String, Int], v1: IVariant, v2: IVariant): Int = {
    var c = compare(contigsIndex, v1.contig(), v2.contig())
    if (c != 0)
      return c

    c = v1.start().compare(v2.start())
    if (c != 0)
      return c

    c = v1.ref().compare(v2.ref())
    if (c != 0)
      return c

    Ordering.Iterable[AltAllele].compare(v1.altAlleles(), v2.altAlleles())
  }

  def compare(contigsIndex: Map[String, Int], l1: Locus, l2: Locus): Int = {
    val c = compare(contigsIndex, l1.contig, l2.contig)
    if (c != 0)
      return c

    Integer.compare(l1.position, l2.position)
  }

  def gen: Gen[GenomeReference] = for {
    name <- Gen.identifier.filter(!GenomeReference.hasReference(_))
    nContigs <- Gen.choose(3, 10)
    contigs <- Gen.distinctBuildableOfN[Array](nContigs, Gen.identifier)
    lengths <- Gen.buildableOfN[Array](nContigs, Gen.choose(1000000, 500000000))
    contigsIndex = contigs.zip(lengths).toMap
    xContig <- Gen.oneOfSeq(contigs)
    parXA <- Gen.choose(0, contigsIndex(xContig))
    parXB <- Gen.choose(0, contigsIndex(xContig))
    yContig <- Gen.oneOfSeq(contigs) if yContig != xContig
    parYA <- Gen.choose(0, contigsIndex(yContig))
    parYB <- Gen.choose(0, contigsIndex(yContig))
    mtContig <- Gen.oneOfSeq(contigs) if mtContig != xContig && mtContig != yContig
  } yield GenomeReference(name, contigs, contigs.zip(lengths).toMap, Set(xContig), Set(yContig), Set(mtContig),
    Array(
      (Locus(xContig, math.min(parXA, parXB)),
        Locus(xContig, math.max(parXA, parXB))),
      (Locus(yContig, math.min(parYA, parYB)),
        Locus(yContig, math.max(parYA, parYB)))))

  def apply(name: java.lang.String, contigs: java.util.ArrayList[String], lengths: java.util.HashMap[String, Int],
    xContigs: java.util.ArrayList[String], yContigs: java.util.ArrayList[String],
    mtContigs: java.util.ArrayList[String], parInput: java.util.ArrayList[String]): GenomeReference = {
    val parRegex = """(\w+):(\d+)-(\d+)""".r

    val par = parInput.asScala.toArray.map {
      case parRegex(contig, start, end) => (Locus(contig.toString, start.toInt), Locus(contig.toString, end.toInt))
      case _ => fatal("expected PAR input of form contig:start-end")
    }

    val gr = GenomeReference(name, contigs.asScala.toArray, lengths.asScala.toMap, xContigs.asScala.toSet,
      yContigs.asScala.toSet, mtContigs.asScala.toSet, par)
    addReference(gr)
    gr
  }
}

case class GRVariable(var gr: GRBase = null) extends GRBase {
  val variantType: TVariant = TVariant(this)
  val locusType: TLocus = TLocus(this)
  val intervalType: TInterval = TInterval(locusType)

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

  def variantOrdering: Ordering[Variant] =
    new Ordering[Variant] {
      def compare(x: Variant, y: Variant): Int = throw new UnsupportedOperationException("GRVariable.variantOrdering unimplemented")
    }

  def locusOrdering: Ordering[Locus] =
    new Ordering[Locus] {
      def compare(x: Locus, y: Locus): Int = throw new UnsupportedOperationException("GRVariable.locusOrdering unimplemented")
    }

  def contigParser: Parser[String] = ???

  def isValidContig(contig: String): Boolean = ???

  def checkVariant(v: Variant): Unit = ???

  def checkVariant(contig: String, pos: Int, ref: String, alts: Array[String]): Unit = ???

  def checkVariant(contig: String, pos: Int, ref: String, alt: String): Unit = ???

  def checkVariant(contig: String, start: Int, ref: String, alts: java.util.ArrayList[String]): Unit = ???

  def checkLocus(l: Locus): Unit = ???

  def checkLocus(contig: String, pos: Int): Unit = ???

  def checkInterval(i: Interval): Unit = ???

  def checkInterval(l1: Locus, l2: Locus): Unit = ???

  def checkInterval(contig: String, start: Int, end: Int): Unit = ???
  
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

  def compare(c1: String, c2: String): Int = ???

  def compare(v1: IVariant, v2: IVariant): Int = ???

  def compare(l1: Locus, l2: Locus): Int = ???
}

