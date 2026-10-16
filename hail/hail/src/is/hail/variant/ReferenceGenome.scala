package is.hail.variant

import is.hail.annotations.ExtendedOrdering
import is.hail.backend.ExecuteContext
import is.hail.collection.implicits._
import is.hail.expr.{
  JSONExtractContig, JSONExtractIntervalLocus, JSONExtractReferenceGenome, Parser,
}
import is.hail.expr.ir.RelationalSpec
import is.hail.io.fs.FS
import is.hail.io.reference.FastaSequenceIndex
import is.hail.types._
import is.hail.types.virtual.{TLocus, Type}
import is.hail.utils._

import scala.collection.immutable.ArraySeq
import scala.collection.mutable
import scala.jdk.CollectionConverters._

import java.io.{FileNotFoundException, InputStream}

import org.json4s.{DefaultFormats, Extraction, Formats, JValue}
import org.json4s.jackson.{JsonMethods, Serialization}

/** A reference genome: its defining fields, and everything derived from them.
  *
  * A reference genome is immutable, so the builtin reference genomes are cached once per JVM and
  * shared across sessions, threads and users. Nothing on a reference genome may be mutable or
  * scoped to a session, and any lazily derived field must be safe to publish.
  *
  * Equality checks identity first, then compares the defining fields. Comparing names alone is not
  * enough, since values from different sessions can meet in one reused Batch driver JVM.
  *
  * A reference genome serializes through a small proxy. A builtin serializes as its name and
  * resolves to the cached copy on read. Any other reference genome serializes only its defining
  * fields, and rebuilds the derived ones on read.
  */
final class ReferenceGenome(
  val name: String,
  val contigs: IndexedSeq[String],
  val lengths: Map[String, Int],
  val xContigs: Set[String],
  val yContigs: Set[String],
  val mtContigs: Set[String],
  val parInput: IndexedSeq[(Locus, Locus)],
) extends Serializable {

  val nContigs = contigs.length

  if (nContigs <= 0)
    fatal("Must have at least one contig in the reference genome.")

  if (!contigs.areDistinct())
    fatal("Repeated contig names are not allowed.")

  {
    val missingLengths = contigs.toSet.diff(lengths.keySet)
    val extraLengths = lengths.keySet.diff(contigs.toSet)

    if (missingLengths.nonEmpty)
      fatal(s"No lengths given for the following contigs: ${missingLengths.mkString(", ")}")

    if (extraLengths.nonEmpty)
      fatal(
        s"Contigs found in 'lengths' that are not present in 'contigs': ${extraLengths.mkString(", ")}"
      )
  }

  if (xContigs.intersect(yContigs).nonEmpty)
    fatal(
      s"Found the contigs '${xContigs.intersect(yContigs).mkString(", ")}' in both X and Y contigs."
    )

  if (xContigs.intersect(mtContigs).nonEmpty)
    fatal(
      s"Found the contigs '${xContigs.intersect(mtContigs).mkString(", ")}' in both X and MT contigs."
    )

  if (yContigs.intersect(mtContigs).nonEmpty)
    fatal(
      s"Found the contigs '${yContigs.intersect(mtContigs).mkString(", ")}' in both Y and MT contigs."
    )

  private val contigsIndex: java.util.HashMap[String, Integer] =
    makeJavaMap(contigs.iterator.zipWithIndex.map { case (c, i) => (c, box(i)) })

  private val contigsSet: java.util.HashSet[String] = makeJavaSet(contigs)

  private[this] val jLengths: java.util.HashMap[String, java.lang.Integer] =
    makeJavaMap(lengths.iterator.map { case (c, i) => (c, box(i)) })

  val lengthsByIndex: IndexedSeq[Int] = contigs.map(lengths)

  lengths.foreach { case (n, l) =>
    if (l <= 0)
      fatal(s"Contig length must be positive. Contig '$n' has length equal to $l.")
  }

  {
    val xNotInRef = xContigs.diff(contigsSet.asScala)
    val yNotInRef = yContigs.diff(contigsSet.asScala)
    val mtNotInRef = mtContigs.diff(contigsSet.asScala)

    if (xNotInRef.nonEmpty)
      fatal(
        s"The following X contig names are absent from the reference: '${xNotInRef.mkString(", ")}'."
      )

    if (yNotInRef.nonEmpty)
      fatal(
        s"The following Y contig names are absent from the reference: '${yNotInRef.mkString(", ")}'."
      )

    if (mtNotInRef.nonEmpty)
      fatal(
        s"The following mitochondrial contig names are absent from the reference: '${mtNotInRef.mkString(", ")}'."
      )
  }

  val xContigIndices = xContigs.map(contigsIndex.get)
  val yContigIndices = yContigs.map(contigsIndex.get)
  val mtContigIndices = mtContigs.map(contigsIndex.get)

  val locusOrdering = {
    val localContigsIndex = contigsIndex
    new Ordering[Locus] {
      override def compare(x: Locus, y: Locus): Int =
        ReferenceGenome.compare(localContigsIndex, x, y)
    }
  }

  val extendedLocusOrdering = ExtendedOrdering.extendToNull(locusOrdering)

  lazy val locusType: TLocus = TLocus(this)

  val par = parInput.map { case (start, end) =>
    if (start.contig != end.contig)
      fatal(
        s"The contigs for the 'start' and 'end' of a PAR interval must be the same. Found '$start-$end'."
      )

    if (
      (!xContigs.contains(start.contig) && !yContigs.contains(start.contig)) ||
      (!xContigs.contains(end.contig) && !yContigs.contains(end.contig))
    )
      fatal(
        s"The contig name for PAR interval '$start-$end' was not found in xContigs '${xContigs.mkString(",")}' or in yContigs '${yContigs.mkString(",")}'."
      )

    Interval(start, end, includesStart = true, includesEnd = false)
  }

  lazy val contigParser = Parser.oneOfLiteral(contigs)

  val globalPosContigStarts = {
    var pos = 0L
    contigs.map { c =>
      val x = (c, pos)
      pos += lengths(c)
      x
    }.toMap
  }

  val nBases = lengths.map(_._2.toLong).sum

  private[this] lazy val globalContigEnds: IndexedSeq[Long] =
    lengthsByIndex.map(_.toLong).scan(0L)(_ + _).tail

  def copy(
    name: String = name,
    contigs: IndexedSeq[String] = contigs,
    lengths: Map[String, Int] = lengths,
    xContigs: Set[String] = xContigs,
    yContigs: Set[String] = yContigs,
    mtContigs: Set[String] = mtContigs,
    parInput: IndexedSeq[(Locus, Locus)] = parInput,
  ): ReferenceGenome =
    new ReferenceGenome(name, contigs, lengths, xContigs, yContigs, mtContigs, parInput)

  def locusToGlobalPos(contig: String, pos: Int): Long =
    globalPosContigStarts(contig) + (pos - 1)

  def locusToGlobalPos(l: Locus): Long = locusToGlobalPos(l.contig, l.position)

  def globalPosToContig(idx: Long): String =
    contigs(globalContigEnds.partitionPoint(_ > idx))

  def globalPosToLocus(idx: Long): Locus = {
    val contig = globalPosToContig(idx)
    Locus(contig, (idx - globalPosContigStarts(contig) + 1).toInt)
  }

  def getContigIndex(contig: String): Int = contigsIndex.get(contig)

  /** The index of `contig`, or `None` if it is null or not in this reference genome. */
  def contigIndexOption(contig: String): Option[Int] =
    Option(contigsIndex.get(contig)).map(_.intValue())

  def contigLength(contig: String): Int = {
    val r = jLengths.get(contig)
    if (r == null)
      fatal(s"Invalid contig name: '$contig'.")
    r.intValue()
  }

  def contigLength(contigIdx: Int): Int = {
    require(contigIdx >= 0 && contigIdx < nContigs)
    lengthsByIndex(contigIdx)
  }

  def isValidContig(contig: String): Boolean =
    contigsSet.contains(contig)

  def isValidLocus(contig: String, pos: Int): Boolean =
    isValidContig(contig) && pos > 0 && pos <= contigLength(contig)

  def isValidLocus(l: Locus): Boolean = isValidLocus(l.contig, l.position)

  def checkContig(contig: String): Unit =
    if (!isValidContig(contig))
      fatal(s"Contig '$contig' is not in the reference genome '$name'.")

  def checkLocus(l: Locus): Unit = checkLocus(l.contig, l.position)

  def checkLocus(contig: String, pos: Int): Unit = {
    if (!isValidLocus(contig, pos)) {
      if (!isValidContig(contig))
        fatal(
          s"Invalid locus '$contig:$pos' found. Contig '$contig' is not in the reference genome '$name'."
        )
      else
        fatal(
          s"Invalid locus '$contig:$pos' found. Position '$pos' is not within the range [1-${contigLength(contig)}] for reference genome '$name'."
        )
    }
  }

  def toLocusInterval(i: Interval, invalidMissing: Boolean): Interval = {
    var start = i.start.asInstanceOf[Locus]
    var end = i.end.asInstanceOf[Locus]
    var includesStart = i.includesStart
    var includesEnd = i.includesEnd

    if (!isValidLocus(start.contig, if (includesStart) start.position else start.position + 1)) {
      if (invalidMissing)
        return null
      else {
        if (!isValidContig(start.contig))
          fatal(
            s"Invalid interval '$i' found. Contig '${start.contig}' is not in the reference genome '$name'."
          )
        else
          fatal(
            s"Invalid interval '$i' found. Start '$start' is not within the range [1-${contigLength(start.contig)}] for reference genome '$name'."
          )
      }
    }

    if (!isValidLocus(end.contig, if (includesEnd) end.position else end.position - 1)) {
      if (invalidMissing)
        return null
      else {
        if (!isValidContig(end.contig))
          fatal(
            s"Invalid interval '$i' found. Contig '${end.contig}' is not in the reference genome '$name'."
          )
        else
          fatal(
            s"Invalid interval '$i' found. End '$end' is not within the range [1-${contigLength(end.contig)}] for reference genome '$name'."
          )
      }
    }

    val contigEnd = contigLength(end.contig)

    if (!includesStart && start.position == 0) {
      start = start.copy(position = 1)
      includesStart = true
    }

    if (!includesEnd && end.position == contigEnd + 1) {
      end = end.copy(position = contigEnd)
      includesEnd = true
    }

    if (start.contig == end.contig && start.position == end.position) {
      (includesStart, includesEnd) match {
        case (true, true) =>
        case (true, false) =>
          if (start.position != 1) {
            start = start.copy(position = start.position - 1)
            includesStart = false
          }
        case (false, true) =>
          if (end.position != contigEnd) {
            end = end.copy(position = end.position + 1)
            includesEnd = false
          }
        case (false, false) =>
      }
    }

    if (!Interval.isValid(extendedLocusOrdering, start, end, includesStart, includesEnd))
      if (invalidMissing)
        return null
      else
        fatal(s"Invalid interval `$i' found. ")

    Interval(start, end, includesStart, includesEnd)
  }

  def inX(contigIdx: Int): Boolean = xContigIndices.contains(contigIdx)

  def inX(contig: String): Boolean = xContigs.contains(contig)

  def inY(contigIdx: Int): Boolean = yContigIndices.contains(contigIdx)

  def inY(contig: String): Boolean = yContigs.contains(contig)

  def isMitochondrial(contigIdx: Int): Boolean = mtContigIndices.contains(contigIdx)

  def isMitochondrial(contig: String): Boolean = mtContigs.contains(contig)

  def isAutosomal(contig: String): Boolean =
    !(inX(contig) || inY(contig) || isMitochondrial(contig))

  def inPar(l: Locus): Boolean = par.exists(_.contains(extendedLocusOrdering, l))

  def inXPar(l: Locus): Boolean = inX(l.contig) && inPar(l)

  def inYPar(l: Locus): Boolean = inY(l.contig) && inPar(l)

  def inXNonPar(l: Locus): Boolean = inX(l.contig) && !inPar(l)

  def inYNonPar(l: Locus): Boolean = inY(l.contig) && !inPar(l)

  def isAutosomalOrPseudoAutosomal(l: Locus): Boolean =
    isAutosomal(l.contig) || ((inX(l.contig) || inY(l.contig)) && inPar(l))

  def compare(contig1: String, contig2: String): Int =
    ReferenceGenome.compare(contigsIndex, contig1, contig2)

  def validateContigRemap(contigMapping: Map[String, String]): Unit = {
    val badContigs = mutable.Set[(String, String)]()

    contigMapping.foreach { case (oldName, newName) =>
      if (!contigsSet.contains(newName))
        badContigs += ((oldName, newName))
    }

    if (badContigs.nonEmpty)
      fatal(
        s"Found ${badContigs.size} ${plural(badContigs.size, "contig mapping that does", "contigs mapping that do")}" +
          s" not have remapped contigs in reference genome '$name':\n  " +
          s"@1",
        badContigs.truncatable("\n  "),
      )
  }

  override val hashCode: Int =
    (name, contigs, lengths, xContigs, yContigs, mtContigs, parInput).##

  override def equals(other: Any): Boolean =
    other match {
      case that: ReferenceGenome =>
        (this eq that) || (
          hashCode == that.hashCode &&
            name == that.name &&
            contigs == that.contigs &&
            lengths == that.lengths &&
            xContigs == that.xContigs &&
            yContigs == that.yContigs &&
            mtContigs == that.mtContigs &&
            parInput == that.parInput
        )
      case _ => false
    }

  def isBound: Boolean = true

  override def toString: String = name

  @transient implicit private[this] val fmts: Formats = DefaultFormats

  def write(fs: is.hail.io.fs.FS, file: String): Unit =
    using(fs.create(file))(out => Serialization.write(toJSON, out))

  def toJSON: JValue =
    Extraction.decompose(
      JSONExtractReferenceGenome(
        name,
        contigs.map(contig => JSONExtractContig(contig, contigLength(contig))),
        xContigs,
        yContigs,
        mtContigs,
        par.map(i =>
          JSONExtractIntervalLocus(i.start.asInstanceOf[Locus], i.end.asInstanceOf[Locus])
        ),
      )
    )

  private def writeReplace(): AnyRef =
    if (ReferenceGenome.builtins.get(name).exists(_ eq this))
      ReferenceGenome.Builtin(name)
    else
      new ReferenceGenome.Inline(
        name,
        contigs.toArray,
        lengthsByIndex.toArray,
        xContigs.toArray,
        yContigs.toArray,
        mtContigs.toArray,
        parInput.map(_._1).toArray,
        parInput.map(_._2).toArray,
      )
}

object ReferenceGenome {
  var GRCh37: String = "GRCh37"
  var GRCh38: String = "GRCh38"
  var GRCm38: String = "GRCm38"
  var CanFam3: String = "CanFam3"
  val hailReferences: Set[String] = Set(GRCh37, GRCh38, GRCm38, CanFam3)

  private[this] val builtinResources = ArraySeq(
    "reference/grch37.json",
    "reference/grch38.json",
    "reference/grcm38.json",
    "reference/canfam3.json",
  )

  /** The builtin reference genomes, parsed from the Hail jar on first use and cached for the life
    * of the Hail-jar classloader. Sharing them is safe because a reference genome is immutable.
    */
  private[variant] lazy val builtins: Map[String, ReferenceGenome] =
    builtinResources.map { file =>
      val rg = loadFromResource(file)(read)
      rg.name -> rg
    }.toMap

  // A builtin is identified by being the cached reference genome itself, never by its name: a
  // user-defined genome may reuse a builtin's name, and must serialize in full.
  final private case class Builtin(name: String) {
    private def readResolve(): AnyRef = builtins(name)
  }

  final private class Inline(
    val name: String,
    val contigs: Array[String],
    val lengthsByIndex: Array[Int],
    val xContigs: Array[String],
    val yContigs: Array[String],
    val mtContigs: Array[String],
    val parStarts: Array[Locus],
    val parEnds: Array[Locus],
  ) extends Serializable {
    private def readResolve(): AnyRef =
      new ReferenceGenome(
        name,
        ArraySeq.unsafeWrapArray(contigs),
        contigs.iterator.zip(lengthsByIndex.iterator).toMap,
        xContigs.toSet,
        yContigs.toSet,
        mtContigs.toSet,
        ArraySeq.unsafeWrapArray(parStarts.zip(parEnds)),
      )
  }

  def builtinReferences(): Map[String, ReferenceGenome] = builtins

  def read(is: InputStream): ReferenceGenome = {
    implicit val formats = DefaultFormats
    JsonMethods.parse(is).extract[JSONExtractReferenceGenome].toReferenceGenome
  }

  def parse(str: String): ReferenceGenome = {
    implicit val formats = DefaultFormats
    JsonMethods.parse(str).extract[JSONExtractReferenceGenome].toReferenceGenome
  }

  def fromResource(file: String): ReferenceGenome =
    loadFromResource[ReferenceGenome](file)(read)

  def fromFile(fs: FS, file: String): ReferenceGenome =
    using(fs.open(file))(read)

  def fromHailDataset(fs: FS, path: String): Array[ReferenceGenome] =
    RelationalSpec.readReferences(fs, path)

  def fromJSON(config: String): ReferenceGenome =
    parse(config)

  def fromFASTAFile(
    ctx: ExecuteContext,
    name: String,
    fastaFile: String,
    indexFile: String,
    xContigs: IndexedSeq[String] = ArraySeq.empty[String],
    yContigs: IndexedSeq[String] = ArraySeq.empty[String],
    mtContigs: IndexedSeq[String] = ArraySeq.empty[String],
    parInput: IndexedSeq[String] = ArraySeq.empty[String],
  ): ReferenceGenome = {
    val fs = ctx.fs

    if (!fs.isFile(fastaFile))
      fatal(s"FASTA file '$fastaFile' does not exist, is not a file, or you do not have access.")

    val contigs = ArraySeq.newBuilder[String]
    val lengths = ArraySeq.newBuilder[(String, Int)]

    FastaSequenceIndex(fs, indexFile).foreach { entry =>
      val contig = entry.getContig
      val length = entry.getSize
      contigs += contig
      lengths += (contig -> length.toInt)
    }

    ReferenceGenome(
      name,
      contigs.result(),
      lengths.result().toMap,
      xContigs,
      yContigs,
      mtContigs,
      parInput,
    )
  }

  def readReferences(fs: FS, path: String): Array[ReferenceGenome] = {
    val refs =
      try
        fs.listDirectory(path)
      catch {
        case _: FileNotFoundException =>
          return Array()
      }

    val rgs = mutable.Set[ReferenceGenome]()
    refs.foreach { fileSystem =>
      val rgPath = fileSystem.getPath.toString
      val rg = using(fs.open(rgPath))(read)
      val name = rg.name
      if (!rgs.contains(rg) && !hailReferences.contains(name))
        rgs += rg
    }
    rgs.toArray
  }

  def writeReference(fs: FS, path: String, rg: ReferenceGenome): Unit = {
    val rgPath = path + "/" + rg.name + ".json.gz"
    if (!hailReferences.contains(rg.name) && !fs.isFile(rgPath))
      rg.write(fs, rgPath)
  }

  def getReferences(t: Type): Set[ReferenceGenome] = {
    var rgs = Set[ReferenceGenome]()
    MapTypes.foreach {
      case tl: TLocus =>
        rgs += tl.rg
      case _ =>
    }(t)
    rgs
  }

  def exportReferences(fs: FS, path: String, rgs: Set[ReferenceGenome]): Unit =
    rgs.foreach(writeReference(fs, path, _))

  def compare(contigsIndex: java.util.HashMap[String, Integer], c1: String, c2: String): Int = {
    val i1 = contigsIndex.get(c1)
    assert(i1 != null)
    val i2 = contigsIndex.get(c2)
    assert(i2 != null)
    Integer.compare(i1.intValue(), i2.intValue())
  }

  def compare(contigsIndex: java.util.HashMap[String, Integer], l1: Locus, l2: Locus): Int = {
    val c = compare(contigsIndex, l1.contig, l2.contig)
    if (c != 0)
      return c

    Integer.compare(l1.position, l2.position)
  }

  def apply(
    name: String,
    contigs: IndexedSeq[String],
    lengths: Map[String, Int],
    xContigs: Set[String] = Set.empty[String],
    yContigs: Set[String] = Set.empty[String],
    mtContigs: Set[String] = Set.empty[String],
    parInput: IndexedSeq[(Locus, Locus)] = IndexedSeq.empty[(Locus, Locus)],
  ): ReferenceGenome =
    new ReferenceGenome(name, contigs, lengths, xContigs, yContigs, mtContigs, parInput)

  def apply(
    name: String,
    contigs: IndexedSeq[String],
    lengths: Map[String, Int],
    xContigs: IndexedSeq[String],
    yContigs: IndexedSeq[String],
    mtContigs: IndexedSeq[String],
    parInput: IndexedSeq[String],
  ): ReferenceGenome = {
    val parRegex = """(\w+):(\d+)-(\d+)""".r

    val par = parInput.map {
      case parRegex(contig, start, end) =>
        (Locus(contig, start.toInt), Locus(contig, end.toInt))
      case _ => fatal("expected PAR input of form contig:start-end")
    }

    ReferenceGenome(name, contigs, lengths, xContigs.toSet, yContigs.toSet, mtContigs.toSet, par)
  }

  def apply(
    name: java.lang.String,
    contigs: java.util.List[String],
    lengths: java.util.Map[String, Int],
    xContigs: java.util.List[String],
    yContigs: java.util.List[String],
    mtContigs: java.util.List[String],
    parInput: java.util.List[String],
  ): ReferenceGenome =
    ReferenceGenome(
      name,
      contigs.asScala.to(ArraySeq),
      lengths.asScala.toMap,
      xContigs.asScala.to(ArraySeq),
      yContigs.asScala.to(ArraySeq),
      mtContigs.asScala.to(ArraySeq),
      parInput.asScala.to(ArraySeq),
    )

}
