package is.hail.variant

import java.io.InputStream

import htsjdk.samtools.reference.FastaSequenceIndex
import is.hail.HailContext
import is.hail.check.Gen
import is.hail.expr.types._
import is.hail.expr.{JSONExtractContig, JSONExtractReferenceGenome, JSONExtractIntervalLocus, Parser}
import is.hail.io.reference.FASTAReader
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

abstract class RGBase extends Serializable {
  val locusType: TLocus
  val intervalType: TInterval

  def name: String

  def variantOrdering: Ordering[Variant]

  def locusOrdering: Ordering[Locus]

  def contigParser: Parser[String]

  def isValidContig(contig: String): Boolean

  def checkVariant(contig: String, pos: Int, ref: String, alts: Array[String]): Unit

  def checkVariant(contig: String, pos: Int, ref: String, alt: String): Unit

  def checkLocus(l: Locus): Unit

  def checkLocus(contig: String, pos: Int): Unit

  def checkLocusInterval(i: Interval): Unit

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

  def unify(concrete: RGBase): Boolean

  def isBound: Boolean

  def clear(): Unit

  def subst(): RGBase
}

case class ReferenceGenome(name: String, contigs: Array[String], lengths: Map[String, Int],
  xContigs: Set[String] = Set.empty[String], yContigs: Set[String] = Set.empty[String],
  mtContigs: Set[String] = Set.empty[String], parInput: Array[(Locus, Locus)] = Array.empty[(Locus, Locus)]) extends RGBase {

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
    def compare(x: Variant, y: Variant): Int = ReferenceGenome.compare(contigsIndex, x, y)
  }

  val locusOrdering = new Ordering[Locus] {
    def compare(x: Locus, y: Locus): Int = ReferenceGenome.compare(contigsIndex, x, y)
  }

  // must be constructed after orderings
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

  private var fastaReader: FASTAReader = _

  def contigParser = Parser.oneOfLiteral(contigs)

  val globalPosContigStarts = {
    var pos = 0L
    contigs.map { c =>
      val x = (c, pos)
      pos += contigLength(c)
      x
    }.toMap
  }

  val nBases = lengths.map(_._2.toLong).sum

  private val globalPosOrd = TInt64().ordering

  @transient private var globalPosTree: IntervalTree[String] = _

  def getGlobalPosTree = IntervalTree.annotationTree[String](globalPosOrd, {
    var pos = 0L
    contigs.map { c =>
      val x = Interval(pos, pos + contigLength(c), includesStart = true, includesEnd = false)
      pos += contigLength(c)
      (x, c)
    }
  })

  def locusToGlobalPos(contig: String, pos: Int): Long =
    globalPosContigStarts(contig) + (pos - 1)

  def locusToGlobalPos(l: Locus): Long = locusToGlobalPos(l.contig, l.position)

  def globalPosToContig(idx: Long): String = {
    if (globalPosTree == null)
      globalPosTree = getGlobalPosTree
    val result = globalPosTree.queryValues(globalPosOrd, idx)
    assert(result.length == 1)
    result(0)
  }

  def globalPosToLocus(idx: Long): Locus = {
    val contig = globalPosToContig(idx)
    Locus(contig, (idx - globalPosContigStarts(contig) + 1).toInt)
  }

  def contigLength(contig: String): Int = lengths.get(contig) match {
    case Some(l) => l
    case None => fatal(s"Invalid contig name: `$contig'.")
  }

  def contigLength(contigIdx: Int): Int = {
    require(contigIdx >= 0 && contigIdx < nContigs)
    lengthsByIndex(contigIdx)
  }

  def isValidContig(contig: String): Boolean = contigsSet.contains(contig)

  def isValidLocus(contig: String, pos: Int): Boolean = pos > 0 && pos <= contigLength(contig)

  def isValidLocus(l: Locus): Boolean = isValidLocus(l.contig, l.position)

  def isValidLocusInterval(startContig: String, startPos: Int, endContig: String, endPos: Int, includesStart: Boolean, includesEnd: Boolean): Boolean = {
    isValidContig(startContig) &&
      isValidContig(endContig) &&
      isValidLocus(startContig, if (includesStart) startPos else startPos + 1) &&
      isValidLocus(endContig, if (includesEnd) endPos else endPos - 1)
  }

  def checkLocus(l: Locus): Unit = checkLocus(l.contig, l.position)

  def checkLocus(contig: String, pos: Int): Unit = {
    if (!isValidContig(contig))
      fatal(s"Invalid locus `$contig:$pos' found. Contig `$contig' is not in the reference genome `$name'.")
    if (!isValidLocus(contig, pos))
      fatal(s"Invalid locus `$contig:$pos' found. Position `$pos' is not within the range [1-${ contigLength(contig) }] for reference genome `$name'.")
  }

  def checkVariant(contig: String, start: Int, ref: String, alt: String): Unit = {
    val v = s"$contig:$start:$ref:$alt"
    if (!isValidContig(contig))
      fatal(s"Invalid variant `$v' found. Contig `$contig' is not in the reference genome `$name'.")
    if (!isValidLocus(contig, start))
      fatal(s"Invalid variant `$v' found. Start `$start' is not within the range [1-${ contigLength(contig) }] for reference genome `$name'.")
  }

  def checkVariant(contig: String, start: Int, ref: String, alts: Array[String]): Unit = checkVariant(contig, start, ref, alts.mkString(","))

  def checkLocusInterval(i: Interval): Unit = {
    val start = i.start.asInstanceOf[Locus]
    val end = i.end.asInstanceOf[Locus]
    val includesStart = i.includesStart
    val includesEnd = i.includesEnd

    if (!isValidContig(start.contig))
      fatal(s"Invalid interval `$i' found. Contig `${ start.contig }' is not in the reference genome `$name'.")
    if (!isValidContig(end.contig))
      fatal(s"Invalid interval `$i' found. Contig `${ end.contig }' is not in the reference genome `$name'.")
    if (!isValidLocus(start.contig, if (includesStart) start.position else start.position + 1))
      fatal(s"Invalid interval `$i' found. Start `$start' is not within the range [1-${ contigLength(start.contig) }] for reference genome `$name'.")
    if (!isValidLocus(end.contig, if (includesEnd) end.position else end.position - 1))
      fatal(s"Invalid interval `$i' found. End `$end' is not within the range [1-${ contigLength(end.contig) }] for reference genome `$name'.")
  }

  def normalizeLocusInterval(i: Interval): Interval = {
    var start = i.start.asInstanceOf[Locus]
    var end = i.end.asInstanceOf[Locus]
    var includesStart = i.includesStart
    var includesEnd = i.includesEnd

    if (!includesStart && start.position == 0) {
      start = start.copyChecked(this, position = 1)
      includesStart = true
    }
    if (!includesEnd && end.position == contigLength(end.contig) + 1) {
      end = end.copyChecked(this, position = contigLength(end.contig))
      includesEnd = true
    }

    Interval(start, end, includesStart, includesEnd)
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

  def compare(contig1: String, contig2: String): Int = ReferenceGenome.compare(contigsIndex, contig1, contig2)

  def compare(v1: IVariant, v2: IVariant): Int = ReferenceGenome.compare(contigsIndex, v1, v2)

  def compare(l1: Locus, l2: Locus): Int = ReferenceGenome.compare(contigsIndex, l1, l2)

  def validateContigRemap(contigMapping: Map[String, String]) {
    val badContigs = mutable.Set[(String, String)]()

    contigMapping.foreach { case (oldName, newName) =>
      if (!contigsSet.contains(newName))
        badContigs += ((oldName, newName))
    }

    if (badContigs.nonEmpty)
      fatal(s"Found ${ badContigs.size } ${ plural(badContigs.size, "contig mapping that does", "contigs mapping that do") }" +
        s" not have remapped contigs in reference genome '$name':\n  " +
        s"@1", badContigs.truncatable("\n  "))
  }

  def hasSequence: Boolean = fastaReader != null

  def addSequence(hc: HailContext, fastaFile: String, indexFile: String) {
    if (hasSequence)
      fatal(s"FASTA sequence has already been loaded for reference genome `$name'.")

    val hConf = hc.hadoopConf
    if (!hConf.exists(fastaFile))
      fatal(s"FASTA file '$fastaFile' does not exist.")
    if (!hConf.exists(indexFile))
      fatal(s"FASTA index file '$indexFile' does not exist.")

    val localIndexFile = FASTAReader.getUriLocalIndexFile(hConf, indexFile)
    val index = new FastaSequenceIndex(new java.io.File(localIndexFile))

    val missingContigs = contigs.filterNot(index.hasIndexEntry)
    if (missingContigs.nonEmpty)
      fatal(s"Contigs missing in FASTA `$fastaFile' that are present in reference genome `$name':\n  " +
        s"@1", missingContigs.truncatable("\n  "))

    val invalidLengths = lengths.flatMap { case (c, l) =>
      val fastaLength = index.getIndexEntry(c).getSize
      if (fastaLength != l)
        Some((c, l, fastaLength))
      else
        None
    }.map { case (c, e, f) => s"$c\texpected:$e\tfound:$f"}

    if (invalidLengths.nonEmpty)
      fatal(s"Contig sizes in FASTA `$fastaFile' do not match expected sizes for reference genome `$name':\n  " +
        s"@1", invalidLengths.truncatable("\n  "))

    fastaReader = FASTAReader(hc, this, fastaFile, indexFile)
  }

  def getSequence(contig: String, position: Int, before: Int = 0, after: Int = 0): String = {
    if (!hasSequence)
      fatal(s"FASTA file has not been loaded for reference genome '$name'.")
    fastaReader.lookup(contig, position, before, after)
  }

  def getSequence(l: Locus, before: Int, after: Int): String =
    getSequence(l.contig, l.position, before, after)

  def getSequence(i: Interval): String = {
    if (!hasSequence)
      fatal(s"FASTA file has not been loaded for reference genome '$name'.")
    fastaReader.lookup(i)
  }

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

  def unify(concrete: RGBase): Boolean = this eq concrete

  def isBound: Boolean = true

  def clear() {}

  def subst(): ReferenceGenome = this

  override def toString: String = name

  def write(hc: HailContext, file: String): Unit =
    hc.hadoopConf.writeTextFile(file) { out =>
      val jrg = JSONExtractReferenceGenome(name,
        contigs.map(contig => JSONExtractContig(contig, contigLength(contig))),
        xContigs, yContigs, mtContigs,
        par.map(i => JSONExtractIntervalLocus(i.start.asInstanceOf[Locus], i.end.asInstanceOf[Locus])))
      implicit val formats = defaultJSONFormats
      Serialization.write(jrg, out)
    }
}

object ReferenceGenome {
  var references: Map[String, ReferenceGenome] = Map()
  val GRCh37: ReferenceGenome = fromResource("reference/grch37.json")
  val GRCh38: ReferenceGenome = fromResource("reference/grch38.json")
  var defaultReference = GRCh37
  references += ("default" -> defaultReference)
  val hailReferences = references.keySet

  def addReference(rg: ReferenceGenome) {
    if (hasReference(rg.name))
      fatal(s"Cannot add reference genome. `${ rg.name }' already exists. Choose a reference name NOT in the following list:\n  " +
        s"@1", references.keys.truncatable("\n  "))

    references += (rg.name -> rg)
  }

  def getReference(name: String): ReferenceGenome = {
    references.get(name) match {
      case Some(rg) => rg
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

  def setDefaultReference(rg: ReferenceGenome) {
    assert(references.contains(rg.name))
    defaultReference = rg
  }

  def setDefaultReference(hc: HailContext, rgSource: String) {
    defaultReference =
      if (hasReference(rgSource))
        getReference(rgSource)
      else
        fromFile(hc, rgSource)
  }

  def read(is: InputStream): ReferenceGenome = {
    implicit val formats = defaultJSONFormats
    JsonMethods.parse(is).extract[JSONExtractReferenceGenome].toReferenceGenome
  }

  def fromResource(file: String): ReferenceGenome = {
    val rg = loadFromResource[ReferenceGenome](file)(read)
    addReference(rg)
    rg
  }

  def fromFile(hc: HailContext, file: String): ReferenceGenome = {
    val rg = hc.hadoopConf.readFile(file)(read)
    addReference(rg)
    rg
  }

  def fromFASTAFile(hc: HailContext, name: String, fastaFile: String, indexFile: String,
    xContigs: java.util.ArrayList[String], yContigs: java.util.ArrayList[String],
    mtContigs: java.util.ArrayList[String], parInput: java.util.ArrayList[String]): ReferenceGenome =
    fromFASTAFile(hc, name, fastaFile, indexFile, xContigs.asScala.toArray, yContigs.asScala.toArray,
      mtContigs.asScala.toArray, parInput.asScala.toArray)

  def fromFASTAFile(hc: HailContext, name: String, fastaFile: String, indexFile: String,
    xContigs: Array[String] = Array.empty[String], yContigs: Array[String] = Array.empty[String],
    mtContigs: Array[String] = Array.empty[String], parInput: Array[String] = Array.empty[String]): ReferenceGenome = {
    val hConf = hc.hadoopConf
    if (!hConf.exists(fastaFile))
      fatal(s"FASTA file '$fastaFile' does not exist.")
    if (!hConf.exists(indexFile))
      fatal(s"FASTA index file '$indexFile' does not exist.")

    val localIndexFile = FASTAReader.getUriLocalIndexFile(hConf, indexFile)
    val index = new FastaSequenceIndex(new java.io.File(localIndexFile))

    val contigs = new ArrayBuilder[String]
    val lengths = new ArrayBuilder[(String, Int)]

    index.iterator().asScala.foreach { entry =>
      val contig = entry.getContig
      val length = entry.getSize
      contigs += contig
      lengths += (contig, length.toInt)
    }

    val rg = ReferenceGenome(name, contigs.result(), lengths.result().toMap, xContigs, yContigs, mtContigs, parInput)
    rg.fastaReader = FASTAReader(hc, rg, fastaFile, indexFile)
    rg
  }

  def importReferences(hConf: Configuration, path: String) {
    if (hConf.exists(path)) {
      val refs = hConf.listStatus(path)
      refs.foreach { fs =>
        val rgPath = fs.getPath.toString
        val rg = hConf.readFile(rgPath)(read)
        val name = rg.name
        if (!ReferenceGenome.hasReference(name))
          addReference(rg)
        else {
          if (ReferenceGenome.getReference(name) != rg)
            fatal(s"`$name' already exists and is not identical to the imported reference from `$rgPath'.")
        }
      }
    }
  }

  private def writeReference(hc: HailContext, path: String, rg: RGBase) {
    val rgPath = path + "/" + rg.name + ".json.gz"
    if (!hailReferences.contains(rg.name) && !hc.hadoopConf.exists(rgPath))
      rg.asInstanceOf[ReferenceGenome].write(hc, rgPath)
  }

  def exportReferences(hc: HailContext, path: String, t: Type) {
    (t: @unchecked) match {
      case TArray(elementType, _) => exportReferences(hc, path, elementType)
      case TSet(elementType, req) => exportReferences(hc, path, elementType)
      case TDict(keyType, valueType, _) =>
        exportReferences(hc, path, keyType)
        exportReferences(hc, path, valueType)
      case TStruct(fields, _) => fields.foreach(fd => exportReferences(hc, path, fd.typ))
      case TLocus(rg, _) => writeReference(hc, path, rg)
      case TInterval(TLocus(rg, _), _) => writeReference(hc, path, rg)
      case _ =>
    }
  }

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

  def gen: Gen[ReferenceGenome] = for {
    name <- Gen.identifier.filter(!ReferenceGenome.hasReference(_))
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
  } yield ReferenceGenome(name, contigs, contigs.zip(lengths).toMap, Set(xContig), Set(yContig), Set(mtContig),
    Array(
      (Locus(xContig, math.min(parXA, parXB)),
        Locus(xContig, math.max(parXA, parXB))),
      (Locus(yContig, math.min(parYA, parYB)),
        Locus(yContig, math.max(parYA, parYB)))))

  def apply(name: String, contigs: Array[String], lengths: Map[String, Int], xContigs: Array[String], yContigs: Array[String],
    mtContigs: Array[String], parInput: Array[String]): ReferenceGenome = {
    val parRegex = """(\w+):(\d+)-(\d+)""".r

    val par = parInput.map {
      case parRegex(contig, start, end) => (Locus(contig.toString, start.toInt), Locus(contig.toString, end.toInt))
      case _ => fatal("expected PAR input of form contig:start-end")
    }

    val rg = ReferenceGenome(name, contigs, lengths, xContigs.toSet, yContigs.toSet, mtContigs.toSet, par)
    addReference(rg)
    rg
  }

  def apply(name: java.lang.String, contigs: java.util.ArrayList[String], lengths: java.util.HashMap[String, Int],
    xContigs: java.util.ArrayList[String], yContigs: java.util.ArrayList[String],
    mtContigs: java.util.ArrayList[String], parInput: java.util.ArrayList[String]): ReferenceGenome =
    ReferenceGenome(name, contigs.asScala.toArray, lengths.asScala.toMap, xContigs.asScala.toArray, yContigs.asScala.toArray,
      mtContigs.asScala.toArray, parInput.asScala.toArray)
}

case class RGVariable(var rg: RGBase = null) extends RGBase {
  val locusType: TLocus = TLocus(this)
  val intervalType: TInterval = TInterval(locusType)

  override def toString = "?RG"

  def unify(concrete: RGBase): Boolean = {
    if (rg == null) {
      rg = concrete
      true
    } else
      rg eq concrete
  }

  def isBound: Boolean = rg != null

  def clear() {
    rg = null
  }

  def subst(): RGBase = {
    assert(rg != null)
    rg
  }

  def name: String = ???

  def variantOrdering: Ordering[Variant] =
    new Ordering[Variant] {
      def compare(x: Variant, y: Variant): Int = throw new UnsupportedOperationException("RGVariable.variantOrdering unimplemented")
    }

  def locusOrdering: Ordering[Locus] =
    new Ordering[Locus] {
      def compare(x: Locus, y: Locus): Int = throw new UnsupportedOperationException("RGVariable.locusOrdering unimplemented")
    }

  def contigParser: Parser[String] = ???

  def isValidContig(contig: String): Boolean = ???

  def checkVariant(contig: String, pos: Int, ref: String, alts: Array[String]): Unit = ???

  def checkVariant(contig: String, pos: Int, ref: String, alt: String): Unit = ???

  def checkLocus(l: Locus): Unit = ???

  def checkLocus(contig: String, pos: Int): Unit = ???

  def checkLocusInterval(i: Interval): Unit = ???

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

