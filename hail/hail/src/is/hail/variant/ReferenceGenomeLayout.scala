package is.hail.variant

import is.hail.annotations.ExtendedOrdering
import is.hail.collection.implicits._
import is.hail.expr.Parser
import is.hail.types.virtual.TLocus
import is.hail.utils._

import scala.collection.immutable.ArraySeq
import scala.jdk.CollectionConverters._

/** The contig layout of a reference genome: its defining fields, and everything derived from them.
  *
  * A layout is immutable, so the builtin layouts are cached once per JVM and shared across
  * sessions, threads and users. Nothing on a layout may be mutable or scoped to a session.
  *
  * A layout serializes through a small proxy. A builtin layout serializes as its name and resolves
  * to the cached layout on read. Any other layout serializes only its defining fields, and rebuilds
  * the derived ones on read.
  */
final case class ReferenceGenomeLayout(
  name: String,
  contigs: IndexedSeq[String],
  lengths: Map[String, Int],
  xContigs: Set[String] = Set.empty[String],
  yContigs: Set[String] = Set.empty[String],
  mtContigs: Set[String] = Set.empty[String],
  parInput: IndexedSeq[(Locus, Locus)] = IndexedSeq.empty[(Locus, Locus)],
) {

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

  val contigsIndex: java.util.HashMap[String, Integer] =
    makeJavaMap(contigs.iterator.zipWithIndex.map { case (c, i) => (c, box(i)) })

  val contigsSet: java.util.HashSet[String] = makeJavaSet(contigs)

  private[variant] val jLengths: java.util.HashMap[String, java.lang.Integer] =
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

  lazy val locusType: TLocus = TLocus(name)

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

  lazy val globalContigEnds: IndexedSeq[Long] =
    lengthsByIndex.map(_.toLong).scan(0L)(_ + _).tail

  private def writeReplace(): AnyRef =
    if (ReferenceGenomeLayout.builtins.get(name).exists(_ eq this))
      ReferenceGenomeLayout.Builtin(name)
    else
      new ReferenceGenomeLayout.Inline(
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

object ReferenceGenomeLayout {

  private[this] val builtinResources = ArraySeq(
    "reference/grch37.json",
    "reference/grch38.json",
    "reference/grcm38.json",
    "reference/canfam3.json",
  )

  /** The builtin layouts, parsed from the Hail jar on first use and cached for the life of the
    * Hail-jar classloader.
    */
  lazy val builtins: Map[String, ReferenceGenomeLayout] =
    builtinResources.map { file =>
      val layout = loadFromResource(file)(ReferenceGenome.read).layout
      layout.name -> layout
    }.toMap

  // A builtin is identified by being the cached layout itself, never by its name: a user-defined
  // genome may reuse a builtin's name, and must serialize in full.
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
      ReferenceGenomeLayout(
        name,
        ArraySeq.unsafeWrapArray(contigs),
        contigs.iterator.zip(lengthsByIndex.iterator).toMap,
        xContigs.toSet,
        yContigs.toSet,
        mtContigs.toSet,
        ArraySeq.unsafeWrapArray(parStarts.zip(parEnds)),
      )
  }
}
