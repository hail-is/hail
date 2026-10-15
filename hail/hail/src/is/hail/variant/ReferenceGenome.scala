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

/** A reference genome, given by its [[ReferenceGenomeLayout]]. */
class ReferenceGenome(val layout: ReferenceGenomeLayout) extends Serializable {

  def name: String = layout.name
  def contigs: IndexedSeq[String] = layout.contigs
  def lengths: Map[String, Int] = layout.lengths
  def xContigs: Set[String] = layout.xContigs
  def yContigs: Set[String] = layout.yContigs
  def mtContigs: Set[String] = layout.mtContigs
  def parInput: IndexedSeq[(Locus, Locus)] = layout.parInput

  def nContigs: Int = layout.nContigs
  def contigsIndex: java.util.HashMap[String, Integer] = layout.contigsIndex
  def contigsSet: java.util.HashSet[String] = layout.contigsSet
  def lengthsByIndex: IndexedSeq[Int] = layout.lengthsByIndex
  def xContigIndices: Set[Integer] = layout.xContigIndices
  def yContigIndices: Set[Integer] = layout.yContigIndices
  def mtContigIndices: Set[Integer] = layout.mtContigIndices
  def locusOrdering: Ordering[Locus] = layout.locusOrdering
  def extendedLocusOrdering: ExtendedOrdering = layout.extendedLocusOrdering
  def locusType: TLocus = layout.locusType
  def par: IndexedSeq[Interval] = layout.par
  def contigParser: Parser.Parser[String] = layout.contigParser
  def globalPosContigStarts: Map[String, Long] = layout.globalPosContigStarts
  def nBases: Long = layout.nBases

  def copy(
    name: String = name,
    contigs: IndexedSeq[String] = contigs,
    lengths: Map[String, Int] = lengths,
    xContigs: Set[String] = xContigs,
    yContigs: Set[String] = yContigs,
    mtContigs: Set[String] = mtContigs,
    parInput: IndexedSeq[(Locus, Locus)] = parInput,
  ): ReferenceGenome =
    new ReferenceGenome(
      layout.copy(name, contigs, lengths, xContigs, yContigs, mtContigs, parInput)
    )

  def getGlobalContigEnds: IndexedSeq[Long] = layout.globalContigEnds

  def locusToGlobalPos(contig: String, pos: Int): Long =
    globalPosContigStarts(contig) + (pos - 1)

  def locusToGlobalPos(l: Locus): Long = locusToGlobalPos(l.contig, l.position)

  def globalPosToContig(idx: Long): String =
    contigs(layout.globalContigEnds.partitionPoint(_ > idx))

  def globalPosToLocus(idx: Long): Locus = {
    val contig = globalPosToContig(idx)
    Locus(contig, (idx - globalPosContigStarts(contig) + 1).toInt)
  }

  def getContigIndex(contig: String): Int = {
    val idx = contigsIndex.get(contig)
    if (idx == null)
      fatal(s"Contig '$contig' is not in the reference genome '$name'.")
    idx.intValue()
  }

  def getContig(contigIdx: Int): String = contigs(contigIdx)

  def contigLength(contig: String): Int = {
    val r = layout.jLengths.get(contig)
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

  override def hashCode: Int = layout.hashCode

  override def equals(other: Any): Boolean =
    other match {
      case rg: ReferenceGenome => layout == rg.layout
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

}

object ReferenceGenome {
  var GRCh37: String = "GRCh37"
  var GRCh38: String = "GRCh38"
  var GRCm38: String = "GRCm38"
  var CanFam3: String = "CanFam3"
  val hailReferences: Set[String] = Set(GRCh37, GRCh38, GRCm38, CanFam3)

  /** The builtin reference genomes, one per name, over the cached builtin layouts. Shared for the
    * life of the Hail-jar classloader, which is safe because a reference genome is immutable.
    */
  private[variant] lazy val builtins: Map[String, ReferenceGenome] =
    ReferenceGenomeLayout.builtins.map { case (name, layout) =>
      name -> new ReferenceGenome(layout)
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
    xContigs: Set[String] = Set.empty[String],
    yContigs: Set[String] = Set.empty[String],
    mtContigs: Set[String] = Set.empty[String],
    parInput: IndexedSeq[(Locus, Locus)] = IndexedSeq.empty[(Locus, Locus)],
  ): ReferenceGenome =
    new ReferenceGenome(
      ReferenceGenomeLayout(name, contigs, lengths, xContigs, yContigs, mtContigs, parInput)
    )

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
            ArraySeq.sorted(existing.keys).mkString(start = "\n  ", sep = "\n  ", end = "")
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
