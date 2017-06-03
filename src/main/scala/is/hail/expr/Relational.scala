package is.hail.expr

import java.io.FileNotFoundException

import is.hail.HailContext
import is.hail.annotations.Annotation
import is.hail.methods.{Aggregators, Filter}
import is.hail.sparkextras.{OrderedPartitioner, OrderedRDD}
import is.hail.variant.{Genotype, Locus, VSMLocalValue, VSMMetadata, Variant, VariantDataset, VariantSampleMatrix}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.Row
import is.hail.utils._
import org.apache.spark.SparkContext
import org.apache.spark.broadcast.Broadcast
import org.json4s.jackson.JsonMethods

import scala.reflect.ClassTag

case class MatrixType(
  metadata: VSMMetadata) {
  def globalType: Type = metadata.globalSignature

  def sType: Type = metadata.sSignature

  def saType: Type = metadata.saSignature

  def vType: Type = metadata.vSignature

  def vaType: Type = metadata.vaSignature

  def genotypeType: Type = metadata.genotypeSignature

  def typ = TStruct(
    "v" -> vType,
    "va" -> vaType,
    "s" -> sType,
    "sa" -> saType,
    "g" -> genotypeType)

  def sampleEC: EvalContext = {
    val aggregationST = Map(
      "global" -> (0, globalType),
      "s" -> (1, sType),
      "sa" -> (2, saType),
      "g" -> (3, genotypeType),
      "v" -> (4, vType),
      "va" -> (5, vaType))
    EvalContext(Map(
      "global" -> (0, globalType),
      "s" -> (1, sType),
      "sa" -> (2, saType),
      "gs" -> (3, TAggregable(genotypeType, aggregationST))))
  }

  def variantEC: EvalContext = {
    val aggregationST = Map(
      "global" -> (0, globalType),
      "v" -> (1, TVariant),
      "va" -> (2, vaType),
      "g" -> (3, genotypeType),
      "s" -> (4, sType),
      "sa" -> (5, saType))
    EvalContext(Map(
      "global" -> (0, globalType),
      "v" -> (1, vType),
      "va" -> (2, vaType),
      "gs" -> (3, TAggregable(genotypeType, aggregationST))))
  }
}

object NewAST {
  def genericRewriteTopDown(ast: NewAST, rule: PartialFunction[NewAST, NewAST]): NewAST = {
    def rewrite(ast: NewAST): NewAST = {
      rule.lift(ast) match {
        case Some(newAST) if newAST != ast =>
          rewrite(newAST)
        case None =>
          val newChildren = ast.children.map(rewrite)
          if ((ast.children, newChildren).zipped.forall(_ eq _))
            ast
          else
            ast.copy(newChildren)
      }
    }

    rewrite(ast)
  }

  def rewriteTopDown[T](ast: MatrixAST[T], rule: PartialFunction[NewAST, NewAST]): MatrixAST[T] =
    genericRewriteTopDown(ast, rule).asInstanceOf[MatrixAST[T]]

  def rewriteTopDown[T](ast: KeyTableAST, rule: PartialFunction[NewAST, NewAST]): KeyTableAST =
    genericRewriteTopDown(ast, rule).asInstanceOf[KeyTableAST]

  def genericRewriteBottomUp(ast: NewAST, rule: PartialFunction[NewAST, NewAST]): NewAST = {
    def rewrite(ast: NewAST): NewAST = {
      val newChildren = ast.children.map(rewrite)

      // only recons if necessary
      val rewritten =
        if ((ast.children, newChildren).zipped.forall(_ eq _))
          ast
        else
          ast.copy(newChildren)

      rule.lift(rewritten) match {
        case Some(newAST) if newAST != rewritten =>
          rewrite(newAST)
        case None =>
          rewritten
      }
    }

    rewrite(ast)
  }
  
  def rewriteBottomUp[T](ast: MatrixAST[T], rule: PartialFunction[NewAST, NewAST]): MatrixAST[T] =
    genericRewriteBottomUp(ast, rule).asInstanceOf[MatrixAST[T]]

  def rewriteBottomUp[T](ast: KeyTableAST, rule: PartialFunction[NewAST, NewAST]): KeyTableAST =
    genericRewriteBottomUp(ast, rule).asInstanceOf[KeyTableAST]
}

abstract class NewAST {
  def children: IndexedSeq[NewAST]

  def copy(newChildren: IndexedSeq[NewAST]): NewAST

  def mapChildren(f: (NewAST) => NewAST): NewAST = {
    copy(children.map(f))
  }
}

case class MatrixValue[T](
  localValue: VSMLocalValue,
  rdd: OrderedRDD[Locus, Variant, (Any, Iterable[T])])(implicit tct: ClassTag[T]) {
  def sparkContext: SparkContext = rdd.sparkContext

  def nPartitions: Int = rdd.partitions.length

  def nSamples: Int = localValue.nSamples

  def globalAnnotation: Annotation = localValue.globalAnnotation

  def sampleIds: IndexedSeq[Annotation] = localValue.sampleIds

  def sampleAnnotations: IndexedSeq[Annotation] = localValue.sampleAnnotations

  lazy val sampleIdsBc: Broadcast[IndexedSeq[Annotation]] = sparkContext.broadcast(sampleIds)

  lazy val sampleAnnotationsBc: Broadcast[IndexedSeq[Annotation]] = sparkContext.broadcast(sampleAnnotations)

  def sampleIdsAndAnnotations: IndexedSeq[(Annotation, Annotation)] = sampleIds.zip(sampleAnnotations)

  def filterSamples(p: (Annotation, Annotation) => Boolean): MatrixValue[T] = {
    val mask = sampleIdsAndAnnotations.map { case (s, sa) => p(s, sa) }
    val maskBc = sparkContext.broadcast(mask)
    val localtct = tct
    copy(localValue =
      localValue.copy(
        sampleIds = sampleIds.zipWithIndex
          .filter { case (s, i) => mask(i) }
          .map(_._1),
        sampleAnnotations = sampleAnnotations.zipWithIndex
          .filter { case (sa, i) => mask(i) }
          .map(_._1)),
      rdd = rdd.mapValues { case (va, gs) =>
        (va, gs.lazyFilterWith(maskBc.value, (g: T, m: Boolean) => m))
      }.asOrderedRDD)
  }
}

object MatrixAST {
  def optimize[T](ast: MatrixAST[T]): MatrixAST[T] = {
    NewAST.rewriteTopDown(ast, {
      case FilterVariants(
      MatrixRead(hc, path, metadata, dropSamples, _, typ),
      Const(_, false, TBoolean)) =>
        MatrixRead(hc, path, metadata, dropSamples, dropVariants = true, typ)
      case FilterSamples(
      MatrixRead(hc, path, metadata, _, dropVariants, typ),
      Const(_, false, TBoolean)) =>
        MatrixRead(hc, path, metadata, dropSamples = true, dropVariants, typ)

      case FilterVariants(m, Const(_, true, TBoolean)) =>
        m
      case FilterSamples(m, Const(_, true, TBoolean)) =>
        m

      // minor, but push FilterVariants into FilterSamples
      case FilterVariants(FilterSamples(m, spred), vpred) =>
        FilterSamples(FilterVariants(m, vpred), spred)

      case FilterVariants(FilterVariants(m, pred1), pred2) =>
        FilterVariants(m, Apply(pred1.getPos, "&&", Array(pred1, pred2)))

      case FilterSamples(FilterSamples(m, pred1), pred2) =>
        FilterSamples(m, Apply(pred1.getPos, "&&", Array(pred1, pred2)))
    })
  }
}

abstract sealed class MatrixAST[T] extends NewAST {
  def typ: MatrixType

  def execute(hc: HailContext): MatrixValue[T]
}

case class MatrixLiteral[T](
  typ: MatrixType,
  value: MatrixValue[T]) extends MatrixAST[T] {

  def children: IndexedSeq[NewAST] = Array.empty[NewAST]

  def execute(hc: HailContext): MatrixValue[T] = value

  def copy(newChildren: IndexedSeq[NewAST]): MatrixLiteral[T] = {
    assert(newChildren.isEmpty)
    this
  }

  override def toString: String = "MatrixLiteral(...)"
}

case class MatrixRead[T](
  hc: HailContext,
  path: String,
  metadata: VSMMetadata,
  dropSamples: Boolean,
  dropVariants: Boolean,
  typ: MatrixType)(implicit ev: T =:= Genotype) extends MatrixAST[T] {

  def children: IndexedSeq[NewAST] = Array.empty[NewAST]

  def copy(newChildren: IndexedSeq[NewAST]): MatrixRead[T] = {
    assert(newChildren.isEmpty)
    this
  }

  def execute(hc: HailContext): MatrixValue[T] = {
    val sc = hc.sc
    val hConf = sc.hadoopConfiguration
    val sqlContext = hc.sqlContext

    val parquetFile = path + "/rdd.parquet"

    val vaSignature = typ.vaType
    val vaRequiresConversion = SparkAnnotationImpex.requiresConversion(vaSignature)
    val isLinearScale = metadata.isLinearScale

    val orderedRDD =
      if (dropVariants) {
        OrderedRDD.empty[Locus, Variant, (Annotation, Iterable[Genotype])](sc)
      } else {
        val rdd = if (dropSamples)
          sqlContext.readParquetSorted(parquetFile, Some(Array("variant", "annotations")))
            .map(row => (row.getVariant(0),
              (if (vaRequiresConversion) SparkAnnotationImpex.importAnnotation(row.get(1), vaSignature) else row.get(1),
                Iterable.empty[Genotype])))
        else {
          val rdd = sqlContext.readParquetSorted(parquetFile)

          rdd.map { row =>
            val v = row.getVariant(0)
            (v,
              (if (vaRequiresConversion) SparkAnnotationImpex.importAnnotation(row.get(1), vaSignature) else row.get(1),
                row.getGenotypeStream(v, 2, isLinearScale): Iterable[Genotype]))
          }
        }

        val partitioner: OrderedPartitioner[Locus, Variant] =
          try {
            val jv = hConf.readFile(path + "/partitioner.json.gz")(JsonMethods.parse(_))
            jv.fromJSON[OrderedPartitioner[Locus, Variant]]
          } catch {
            case _: FileNotFoundException =>
              fatal("missing partitioner.json.gz when loading VDS, create with HailContext.write_partitioning.")
          }

        OrderedRDD(rdd, partitioner)
      }

    val (fileMetadata, _) = VariantSampleMatrix.readFileMetadata(hConf, path)
    val localValue = fileMetadata.localValue

    MatrixValue(
      if (dropSamples)
        localValue.dropSamples()
      else
        localValue,
      orderedRDD)
      .asInstanceOf[MatrixValue[T]]
  }

  override def toString: String = s"MatrixRead($path, dropSamples = $dropSamples, dropVariants = $dropVariants)"
}

case class FilterSamples[T](
  child: MatrixAST[T],
  pred: AST) extends MatrixAST[T] {
  def children: IndexedSeq[NewAST] = Array(child)

  def copy(newChildren: IndexedSeq[NewAST]): FilterSamples[T] = {
    assert(newChildren.length == 1)
    FilterSamples(newChildren(0).asInstanceOf[MatrixAST[T]], pred)
  }

  def typ: MatrixType = child.typ

  def execute(hc: HailContext): MatrixValue[T] = {
    val prev = child.execute(hc)

    val localGlobalAnnotation = prev.localValue.globalAnnotation
    val sas = typ.metadata.saSignature
    val ec = typ.sampleEC

    val f: () => java.lang.Boolean = Parser.evalTypedExpr[java.lang.Boolean](pred, ec)

    val sampleAggregationOption = Aggregators.buildSampleAggregations(hc, prev, ec)

    val p = (s: Annotation, sa: Annotation) => {
      sampleAggregationOption.foreach(f => f.apply(s))
      ec.setAll(localGlobalAnnotation, s, sa)
      f() == true
    }
    prev.filterSamples(p)
  }
}

case class FilterVariants[T](
  child: MatrixAST[T],
  pred: AST)(implicit tct: ClassTag[T]) extends MatrixAST[T] {
  def children: IndexedSeq[NewAST] = Array(child)

  def copy(newChildren: IndexedSeq[NewAST]): FilterVariants[T] = {
    assert(newChildren.length == 1)
    FilterVariants(newChildren(0).asInstanceOf[MatrixAST[T]], pred)
  }

  def typ: MatrixType = child.typ

  def execute(hc: HailContext): MatrixValue[T] = {
    val prev = child.execute(hc)

    val localGlobalAnnotation = prev.localValue.globalAnnotation
    val ec = child.typ.variantEC

    val f: () => java.lang.Boolean = Parser.evalTypedExpr[java.lang.Boolean](pred, ec)

    val aggregatorOption = Aggregators.buildVariantAggregations(
      prev.rdd.sparkContext, prev.localValue, ec)

    val p = (v: Variant, va: Annotation, gs: Iterable[T]) => {
      aggregatorOption.foreach(f => f(v, va, gs))

      ec.setAll(localGlobalAnnotation, v, va)

      // null => false
      f() == true
    }

    prev.copy(rdd = prev.rdd.filter { case (v, (va, gs)) => p(v, va, gs) }.asOrderedRDD)
  }
}

abstract sealed class KeyTableAST extends NewAST {
  def execute(hc: HailContext): RDD[Row]
}

case class KeyTableLiteral(
  typ: TStruct,
  rdd: RDD[Row]) {

  def children: IndexedSeq[NewAST] = Array.empty[NewAST]

  def copy(newChildren: IndexedSeq[NewAST]): KeyTableLiteral = {
    assert(newChildren.isEmpty)
    this
  }

  def execute(hc: HailContext): RDD[Row] = rdd
}
