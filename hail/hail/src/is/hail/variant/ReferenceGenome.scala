package is.hail.variant

import is.hail.annotations.ExtendedOrdering
import is.hail.backend.ExecuteContext
import is.hail.collection.compat.immutable.ArraySeq
import is.hail.collection.implicits.{toRichIndexedSeq, toRichIterable}
import is.hail.expr.{
  JSONExtractContig, JSONExtractIntervalLocus, JSONExtractReferenceGenome, Parser,
}
import is.hail.expr.ir.RelationalSpec
import is.hail.io.fs.FS
import is.hail.io.reference.{
  FASTAReader, FASTAReaderConfig, FastaSequenceIndex, IndexedFastaSequenceFile, LiftOver,
}
import is.hail.types._
import is.hail.types.virtual.{TLocus, Type}
import is.hail.utils._

import scala.collection.mutable
import scala.jdk.CollectionConverters._

import java.io.{FileNotFoundException, InputStream}

import org.json4s.{DefaultFormats, Extraction, Formats, JValue}
import org.json4s.jackson.{JsonMethods, Serialization}

case class ReferenceGenome(
  name: String,
  contigs: IndexedSeq[String],
  lengths: Map[String, Int],
  xContigs: Set[String] = Set.empty[String],
  yContigs: Set[String] = Set.empty[String],
  mtContigs: Set[String] = Set.empty[String],
  parInput: IndexedSeq[(Locus, Locus)] = IndexedSeq.empty[(Locus, Locus)],
) extends Serializable {

  val nContigs = contigs.length

  if (nContigs <= 0)
    fatal("Must have at least one contig in the reference genome.")

  if (!contigs.areDistinct())
    fatal("Repeated contig names are not allowed.")

  val missingLengths = contigs.toSet.diff(lengths.keySet)
  val extraLengths = lengths.keySet.diff(contigs.toSet)

  if (missingLengths.nonEmpty)
    fatal(s"No lengths given for the following contigs: ${missingLengths.mkString(", ")}")

  if (extraLengths.nonEmpty)
    fatal(
      s"Contigs found in 'lengths' that are not present in 'contigs': ${extraLengths.mkString(", ")}"
    )

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

  val contigsIndex: java.util.HashMap[String, Integer] =
    makeJavaMap(contigs.iterator.zipWithIndex.map { case (c, i) => (c, box(i)) })

  val contigsSet: java.util.HashSet[String] = makeJavaSet(contigs)

  private val jLengths: java.util.HashMap[String, java.lang.Integer] =
    makeJavaMap(lengths.iterator.map { case (c, i) => (c, box(i)) })

  val lengthsByIndex: IndexedSeq[Int] = contigs.map(lengths)

  lengths.foreach { case (n, l) =>
    if (l <= 0)
      fatal(s"Contig length must be positive. Contig '$n' has length equal to $l.")
  }

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

  // must be constructed after orderings
  @transient @volatile var _locusType: TLocus = _

  def locusType: TLocus = {
    if (_locusType == null) {
      synchronized {
        if (_locusType == null)
          _locusType = TLocus(this.name)
      }
    }
    _locusType
  }

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

  private[this] var fastaFile: IndexedFastaSequenceFile = _
  @transient private[this] var fastaReaderCfg: FASTAReaderConfig = _

  @transient lazy val contigParser = Parser.oneOfLiteral(contigs)

  val globalPosContigStarts = {
    var pos = 0L
    contigs.map { c =>
      val x = (c, pos)
      pos += contigLength(c)
      x
    }.toMap
  }

  val nBases = lengths.map(_._2.toLong).sum

  @transient private var globalContigEnds: IndexedSeq[Long] = _

  def getGlobalContigEnds: IndexedSeq[Long] =
    contigs.map(contigLength(_).toLong).scan(0L)(_ + _).tail

  def locusToGlobalPos(contig: String, pos: Int): Long =
    globalPosContigStarts(contig) + (pos - 1)

  def locusToGlobalPos(l: Locus): Long = locusToGlobalPos(l.contig, l.position)

  def globalPosToContig(idx: Long): String = {
    if (globalContigEnds == null)
      globalContigEnds = getGlobalContigEnds
    contigs(globalContigEnds.partitionPoint(_ > idx))
  }

  def globalPosToLocus(idx: Long): Locus = {
    val contig = globalPosToContig(idx)
    Locus(contig, (idx - globalPosContigStarts(contig) + 1).toInt)
  }

  def getContigIndex(contig: String): Int = contigsIndex.get(contig)

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

  private def hasSequence: Boolean = fastaFile != null

  def addSequence(fasta: IndexedFastaSequenceFile): Unit = {
    if (hasSequence)
      fatal(s"FASTA sequence has already been loaded for reference genome '$name'.")

    fasta.raiseIfIncompatible(this)
    fastaFile = fasta
  }

  @transient private lazy val realFastaReader: ThreadLocal[FASTAReader] =
    new ThreadLocal[FASTAReader]

  private def fastaReader(): FASTAReader = {
    if (!hasSequence)
      fatal(s"FASTA file has not been loaded for reference genome '$name'.")

    if (realFastaReader.get() == null)
      realFastaReader.set(fastaReaderCfg.reader)
    if (realFastaReader.get().cfg != fastaReaderCfg)
      realFastaReader.set(fastaReaderCfg.reader)
    realFastaReader.get()
  }

  def getSequence(contig: String, position: Int, before: Int = 0, after: Int = 0): String =
    fastaReader().lookup(contig, position, before, after)

  def getSequence(l: Locus, before: Int, after: Int): String =
    getSequence(l.contig, l.position, before, after)

  def getSequence(i: Interval): String =
    fastaReader().lookup(i)

  def removeSequence(): Unit = {
    if (!hasSequence)
      fatal(s"Reference genome '$name' does not have sequence loaded.")
    fastaFile = null
    fastaReaderCfg = null
  }

  private[this] val liftovers: mutable.Map[String, LiftOver] =
    mutable.Map.empty

  def addLiftover(destRef: ReferenceGenome, liftOver: LiftOver): Unit = {
    if (name == destRef.name)
      fatal(s"Destination reference genome cannot have the same name as this reference '$name'")
    if (liftovers.contains(destRef.name))
      fatal(
        s"LiftOver already exists for source reference '$name' and destination reference '${destRef.name}'."
      )

    liftOver.checkChainFile(this, destRef)
    liftovers += destRef.name -> liftOver
  }

  def removeLiftover(destRGName: String): Unit =
    liftovers -= destRGName

  def liftoverLocus(destRGName: String, l: Locus, minMatch: Double): (Locus, Boolean) =
    liftovers(destRGName).queryLocus(l, minMatch)

  def liftoverLocusInterval(destRGName: String, interval: Interval, minMatch: Double)
    : (Interval, Boolean) =
    liftovers(destRGName).queryInterval(interval, minMatch)

  def heal(tmpdir: String, fs: FS): Unit = synchronized {
    liftovers.values.foreach(_.restore(fs))
    // add sequence
    if (fastaFile != null) {
      fastaFile.restore(fs)
      val fastaPath = fs.parseUrl(fastaFile.path).toString
      val indexPath = fs.parseUrl(fastaFile.index.path).toString
      fastaReaderCfg = FASTAReaderConfig(tmpdir, fs, this, fastaPath, indexPath)
    }
  }

  override def hashCode: Int = {
    import org.apache.commons.lang3.builder.HashCodeBuilder

    val b = new HashCodeBuilder()
      .append(name)
      .append(contigs)
      .append(lengths)
      .append(xContigs)
      .append(yContigs)
      .append(mtContigs)
      .append(par)
    b.toHashCode
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

}

object ReferenceGenome {
  var GRCh37: String = "GRCh37"
  var GRCh38: String = "GRCh38"
  var GRCm38: String = "GRCm38"
  var CanFam3: String = "CanFam3"
  val hailReferences: Set[String] = Set(GRCh37, GRCh38, GRCm38, CanFam3)

  def builtinReferences(): Map[String, ReferenceGenome] = {
    var builtin: Map[String, ReferenceGenome] = Map()
    val files = Array(
      "reference/grch37.json",
      "reference/grch38.json",
      "reference/grcm38.json",
      "reference/canfam3.json",
    )
    for (filename <- files) {
      val rg = loadFromResource[ReferenceGenome](filename)(read)
      builtin += (rg.name -> rg)
    }
    builtin
  }

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
    xContigs: Array[String] = Array.empty[String],
    yContigs: Array[String] = Array.empty[String],
    mtContigs: Array[String] = Array.empty[String],
    parInput: Array[String] = Array.empty[String],
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

  def getReferences(t: Type): Set[String] = {
    var rgs = Set[String]()
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
      contigs.asScala.toArray,
      lengths.asScala.toMap,
      xContigs.asScala.toArray,
      yContigs.asScala.toArray,
      mtContigs.asScala.toArray,
      parInput.asScala.toArray,
    )

  def getMapFromArray(arr: Array[ReferenceGenome]): Map[String, ReferenceGenome] =
    arr.map(rg => (rg.name, rg)).toMap

  def addFatalOnCollision(
    existing: mutable.Map[String, ReferenceGenome],
    newReferences: Iterable[ReferenceGenome],
  ): Unit =
    for (rg <- newReferences) {
      if (existing.get(rg.name).exists(_ != rg))
        fatal(
          s"Cannot add reference genome '${rg.name}', a different reference with that name already exists. " ++
            "Choose a reference name NOT in the following list:" ++
            existing.keys.toFastSeq.sorted.mkString(start = "\n  ", sep = "\n  ", end = "")
        )

      existing += (rg.name -> rg)
    }

  def addFatalOnCollision(
    existing: Map[String, ReferenceGenome],
    newReferences: Iterable[ReferenceGenome],
  ): Map[String, ReferenceGenome] = {
    val mut = mutable.Map(existing.toSeq: _*)
    addFatalOnCollision(mut, newReferences)
    mut.toMap
  }

}
