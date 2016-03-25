package org.broadinstitute.hail.driver

import org.broadinstitute.hail.Utils._
import org.broadinstitute.hail.annotations._
import org.broadinstitute.hail.expr
import org.broadinstitute.hail.expr._
import org.broadinstitute.hail.io.annotators._
import org.broadinstitute.hail.methods.ProgrammaticAnnotation
import org.broadinstitute.hail.variant.Sample
import org.kohsuke.args4j.{Option => Args4jOption}

import scala.collection.mutable
import scala.collection.mutable.ArrayBuffer

object AnnotateVariants extends Command {

  class Options extends BaseOptions {
    @Args4jOption(required = true, name = "-c", aliases = Array("--condition"),
      usage = "Annotation file path")
    var condition: String = _

    @Args4jOption(required = false, name = "-t", aliases = Array("--types"),
      usage = "Define types of fields in annotations files")
    var types: String = _

    @Args4jOption(required = false, name = "-r", aliases = Array("--root"),
      usage = "Place annotations in the path 'va.<root>.<field>'")
    var root: String = _

    @Args4jOption(required = false, name = "-m", aliases = Array("--missing"),
      usage = "Specify additional identifiers to be treated as missing (default: 'NA')")
    var missingIdentifiers: String = _

    @Args4jOption(required = false, name = "-v", aliases = Array("--vcolumns"),
      usage = "Specify the column identifiers for chromosome, position, ref, and alt (in that order)" +
        " (default: 'Chromosome,Position,Ref,Alt'")
    var vCols: String = null
  }

  def newOptions = new Options

  def name = "annotatevariants"

  def description = "Annotate variants in current dataset"

  def parseTypeMap(s: String): Map[String, String] = {
    s.split(",")
      .map(_.trim())
      .map(s => s.split(":").map(_.trim()))
      .map {
        case Array(f, t) => (f, t)
        case arr => fatal("parse error in type declaration")
      }
      .toMap
  }

  def parseMissing(s: String): Set[String] = {
    s.split(",")
      .map(_.trim())
      .toSet
  }

  def parseColumns(s: String): Array[String] = {
    val split = s.split(",").map(_.trim)
    fatalIf(split.length != 4 && split.length != 1,
      "Cannot read chr, pos, ref, alt columns from '" + s +
        "': enter 4 comma-separated column identifiers for separate chr/pos/ref/alt columns, " +
        "or one identifier for chr:pos:ref:alt")
    split
  }

  def parseRoot(s: String): List[String] = s match {
    case r if r.startsWith("va.") => r.substring(3).split("""\.""").toList
    case "va" => List[String]()
    case error => fatal(s"invalid root '$error': expect 'va.<path[.path2...]>'")
  }

  def run(state: State, options: Options): State = {
    val vds = state.vds

    val cond = options.condition

    val stripped = hadoopStripCodec(cond, state.sc.hadoopConfiguration)


    val conf = state.sc.hadoopConfiguration

    val annotated = stripped match {
      case intervalList if intervalList.endsWith(".interval_list") =>
        if (options.root == null)
          fatal("argument 'root' is required for '.interval_list' annotation")
        if (options.types != null)
          warn("argument 'types' is unnecessary for '.interval_list' annotation, ignoring it")
        if (options.missingIdentifiers != null)
          warn("argument 'missing' is unnecessary for '.interval_list' annotation, ignoring it")
        if (options.vCols != null)
          warn("argument 'vcolumns' is unnecessary for '.interval_list' annotation, ignoring it")
        val (iList, signature) = IntervalListAnnotator(cond, conf)
        vds.annotateInvervals(iList, signature, parseRoot(options.root))
      case bed if bed.endsWith(".bed") =>
        if (options.root == null)
          fatal("argument 'root' is required for '.bed' annotation")
        if (options.types != null)
          warn("argument 'types' is unnecessary for '.bed' annotation, ignoring it")
        if (options.missingIdentifiers != null)
          warn("argument 'missing' is unnecessary for '.bed' annotation, ignoring it")
        if (options.vCols != null)
          warn("argument 'vcolumns' is unnecessary for '.bed' annotation, ignoring it")
        val (iList, signature) = BedAnnotator(cond, conf)
        vds.annotateInvervals(iList, signature, parseRoot(options.root))
      case tsv if tsv.endsWith(".tsv") =>
        if (options.root == null)
          fatal("argument 'root' is required for '.tsv' annotation")
        val (rdd, signature) = VariantTSVAnnotator(vds.sparkContext, cond,
          parseColumns(Option(options.vCols).getOrElse("Chromosome,Position,Ref,Alt")),
          parseTypeMap(Option(options.types).getOrElse("")),
          parseMissing(Option(options.missingIdentifiers).getOrElse("NA")))
        vds.annotateVariants(rdd, signature, parseRoot(options.root))
      case vcf if vcf.endsWith(".vcf") =>
        if (options.root == null)
          fatal("argument 'root' is required for '.vcf' annotation")
        if (options.types != null)
          warn("argument 'types' is unnecessary for '.vcf' annotation, ignoring it")
        if (options.missingIdentifiers != null)
          warn("argument 'missing' is unnecessary for '.vcf' annotation, ignoring it")
        if (options.vCols != null)
          warn("argument 'vcolumns' is unnecessary for '.vcf' annotation, ignoring it")
        val (rdd, signature) = VCFAnnotator(vds.sparkContext, cond)
        vds.annotateVariants(rdd, signature, parseRoot(options.root))
      case otherVds if otherVds.endsWith(".vds") =>
        if (options.root == null)
          fatal("argument 'root' is required for '.vds' annotation")
        if (options.types != null)
          warn("argument 'types' is unnecessary for '.vds' annotation, ignoring it")
        if (options.missingIdentifiers != null)
          warn("argument 'missing' is unnecessary for '.vds' annotation, ignoring it")
        if (options.vCols != null)
          warn("argument 'vcolumns' is unnecessary for '.vds' annotation, ignoring it")
        val readOtherVds = {
          val s2 = Read.run(State(state.sc, state.sqlContext, null), Array("-i", cond))
          if (s2.vds.wasSplit)
            s2.vds
          else
            SplitMulti.run(s2).vds
        }
        vds.annotateVariants(readOtherVds.variantsAndAnnotations,
          readOtherVds.vaSignature, parseRoot(options.root))
      case programmatic =>
        if (options.root != null)
          warn("argument 'root' is unnecessary for programmatic annotation, ignoring it")
        if (options.types != null)
          warn("argument 'types' is unnecessary for programmatic annotation, ignoring it")
        if (options.missingIdentifiers != null)
          warn("argument 'missing' is unnecessary for programmatic annotation, ignoring it")
        if (options.vCols != null)
          warn("argument 'vcolumns' is unnecessary for programmatic annotation, ignoring it")

        val symTab = Map(
          "v" ->(0, TVariant),
          "va" ->(1, vds.vaSignature),
          "gs" ->(2, TGenotypeStream))
        val symTab2 = Map(
          "v" ->(0, TVariant),
          "va" ->(1, vds.vaSignature),
          "s" ->(2, TSample),
          "sa" ->(3, vds.saSignature),
          "g" ->(4, TGenotype)
        )
        val a = new ArrayBuffer[Any]()
        val a2 = new ArrayBuffer[Any]()
        val a3 = new ArrayBuffer[Aggregator]()

        for (_ <- symTab)
          a += null
        for (_ <- symTab2)
          a2 += null
        val parsed = expr.Parser.parseAnnotationArgs(symTab, symTab2, a, a2, a3, cond)


        val keyedSignatures = parsed.map { case (ids, t, f) =>
          if (ids.head != "va")
            fatal(s"expect 'va[.identifier]+', got ${ids.mkString(".")}")
          ProgrammaticAnnotation.checkType(ids.mkString("."), t)
          (ids.tail, t)
        }

        val inserterBuilder = mutable.ArrayBuilder.make[Inserter]

        val computations = parsed.map(_._3)

        val vdsAddedSigs = keyedSignatures.foldLeft(vds) { case (v, (ids, signature)) =>
          val (s, i) = v.insertVA(signature, ids)
          inserterBuilder += i
          v.copy(vaSignature = s)
        }

        val inserters = inserterBuilder.result()

        val sampleInfoBc = vds.sparkContext.broadcast(
          vds.localSamples.map(vds.sampleAnnotations)
            .zip(vds.localSamples.map(vds.sampleIds).map(Sample)))

        vdsAddedSigs.mapAnnotations { case (v, va, gs) =>
          a(0) = v
          a(1) = va
          a(2) = gs
          if (a3.nonEmpty) {
            val gsQueries = a3.toArray.map(_._1)
            gs.iterator
              .zip(sampleInfoBc.value.iterator)
              .foreach {
                case (g, (sa, s)) =>
                  a2(0) = v
                  a2(1) = va
                  a2(2) = s
                  a2(3) = sa
                  a2(4) = g
                  a3.iterator.zipWithIndex
                    .foreach {
                      case ((zv, so, co), i) =>
                        gsQueries(i) = so(gsQueries(i))
                    }
              }
            gsQueries.iterator.zipWithIndex
              .foreach { case (res, i) =>
                a2(5 + i) = res
              }
          }

          computations.indices.foreach { i =>
            a(1) = inserters(i).apply(a(1), Option(computations(i)()))
          }
          a(1): Annotation
        }
    }
    state.copy(vds = annotated)
  }
}
