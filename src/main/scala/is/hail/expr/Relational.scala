package is.hail.expr

import java.io.FileNotFoundException

import is.hail.HailContext
import is.hail.annotations.{Annotation, ReadRowsRDD}
import is.hail.methods.Aggregators
import is.hail.sparkextras.{OrderedKey, OrderedPartitioner, OrderedRDD}
import is.hail.variant.{Genotype, Locus, VSMFileMetadata, VSMLocalValue, VSMMetadata, Variant, VariantSampleMatrix}
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

  def rowType: TStruct = TStruct(
    "v" -> vType,
    "va" -> vaType,
    "gs" -> TArray(genotypeType))

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

  def rewriteTopDown[RPK, RK, T](ast: MatrixAST[RPK, RK, T], rule: PartialFunction[NewAST, NewAST]): MatrixAST[RPK, RK, T] =
    genericRewriteTopDown(ast, rule).asInstanceOf[MatrixAST[RPK, RK, T]]

  def rewriteTopDown(ast: KeyTableAST, rule: PartialFunction[NewAST, NewAST]): KeyTableAST =
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

  def rewriteBottomUp[RPK, RK, T](ast: MatrixAST[RPK, RK, T], rule: PartialFunction[NewAST, NewAST]): MatrixAST[RPK, RK, T] =
    genericRewriteBottomUp(ast, rule).asInstanceOf[MatrixAST[RPK, RK, T]]

  def rewriteBottomUp(ast: KeyTableAST, rule: PartialFunction[NewAST, NewAST]): KeyTableAST =
    genericRewriteBottomUp(ast, rule).asInstanceOf[KeyTableAST]
}

abstract class NewAST {
  def children: IndexedSeq[NewAST]

  def copy(newChildren: IndexedSeq[NewAST]): NewAST

  def mapChildren(f: (NewAST) => NewAST): NewAST = {
    copy(children.map(f))
  }
}

case class MatrixValue[RPK, RK, T](
  localValue: VSMLocalValue,
  rdd: OrderedRDD[RPK, RK, (Any, Iterable[T])])(implicit tct: ClassTag[T]) {
  implicit val kOk: OrderedKey[RPK, RK] = rdd.kOk

  import kOk._

  def sparkContext: SparkContext = rdd.sparkContext

  def nPartitions: Int = rdd.partitions.length

  def nSamples: Int = localValue.nSamples

  def globalAnnotation: Annotation = localValue.globalAnnotation

  def sampleIds: IndexedSeq[Annotation] = localValue.sampleIds

  def sampleAnnotations: IndexedSeq[Annotation] = localValue.sampleAnnotations

  lazy val sampleIdsBc: Broadcast[IndexedSeq[Annotation]] = sparkContext.broadcast(sampleIds)

  lazy val sampleAnnotationsBc: Broadcast[IndexedSeq[Annotation]] = sparkContext.broadcast(sampleAnnotations)

  def sampleIdsAndAnnotations: IndexedSeq[(Annotation, Annotation)] = sampleIds.zip(sampleAnnotations)

  def filterSamples(p: (Annotation, Annotation) => Boolean): MatrixValue[RPK, RK, T] = {
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
  def optimize[RPK, RK, T](ast: MatrixAST[RPK, RK, T]): MatrixAST[RPK, RK, T] = {
    NewAST.rewriteTopDown(ast, {
      case FilterVariants(
      MatrixRead(hc, path, nPartitions, fileMetadata, dropSamples, _),
      Const(_, false, TBoolean)) =>
        MatrixRead(hc, path, nPartitions, fileMetadata, dropSamples, dropVariants = true)
      case FilterSamples(
      MatrixRead(hc, path, nPartitions, fileMetadata, _, dropVariants),
      Const(_, false, TBoolean)) =>
        MatrixRead(hc, path, nPartitions, fileMetadata, dropSamples = true, dropVariants)

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

abstract sealed class MatrixAST[RPK, RK, T] extends NewAST {
  implicit val kOk: OrderedKey[RPK, RK]

  def typ: MatrixType

  def execute(hc: HailContext): MatrixValue[RPK, RK, T]
}

case class MatrixLiteral[RPK, RK, T](
  typ: MatrixType,
  value: MatrixValue[RPK, RK, T]) extends MatrixAST[RPK, RK, T] {
  implicit val kOk: OrderedKey[RPK, RK] = value.kOk

  def children: IndexedSeq[NewAST] = Array.empty[NewAST]

  def execute(hc: HailContext): MatrixValue[RPK, RK, T] = value

  def copy(newChildren: IndexedSeq[NewAST]): MatrixLiteral[RPK, RK, T] = {
    assert(newChildren.isEmpty)
    this
  }

  override def toString: String = "MatrixLiteral(...)"
}

case class MatrixRead(
  hc: HailContext,
  path: String,
  nPartitions: Int,
  fileMetadata: VSMFileMetadata,
  dropSamples: Boolean,
  dropVariants: Boolean) extends MatrixAST[Annotation, Annotation, Annotation] {

  val kOk: OrderedKey[Annotation, Annotation] = fileMetadata.metadata.vSignature.orderedKey

  def typ: MatrixType = MatrixType(fileMetadata.metadata)

  def children: IndexedSeq[NewAST] = Array.empty[NewAST]

  def copy(newChildren: IndexedSeq[NewAST]): MatrixRead = {
    assert(newChildren.isEmpty)
    this
  }

  def execute(hc: HailContext): MatrixValue[Annotation, Annotation, Annotation] = {
    val metadata = fileMetadata.metadata
    val localValue =
      if (dropSamples)
        fileMetadata.localValue.dropSamples()
      else
        fileMetadata.localValue

    val parquetFile = path + "/rdd.parquet"

    implicit val localKOk = kOk

    val orderedRDD =
      if (dropVariants)
        OrderedRDD.empty[Annotation, Annotation, (Annotation, Iterable[Annotation])](hc.sc)
      else {
        val rdd = new ReadRowsRDD(hc.sc, path, typ.rowType, nPartitions)
        .map { ur =>
          val v = ur.get(0)
          val va = ur.get(1)
          val gs: Iterable[Annotation] = ur.getAs[IndexedSeq[Annotation]](2)

          (v, (va, gs))
        }

        val jv = hc.hadoopConf.readFile(path + "/partitioner.json.gz")(JsonMethods.parse(_))
        implicit val pkjr = typ.vType.partitionKey.jsonReader
        val partitioner = jv.fromJSON[OrderedPartitioner[Annotation, Annotation]]

        val columns = someIf(dropSamples, Array("variant", "annotations"))
        OrderedRDD[Annotation, Annotation, (Annotation, Iterable[Annotation])](
          rdd, partitioner)
      }

    MatrixValue(
      localValue,
      orderedRDD)
  }

  override def toString: String = s"MatrixRead($path, dropSamples = $dropSamples, dropVariants = $dropVariants)"
}

case class FilterSamples[RPK, RK, T >: Null](
  child: MatrixAST[RPK, RK, T],
  pred: AST) extends MatrixAST[RPK, RK, T] {
  implicit val kOk: OrderedKey[RPK, RK] = child.kOk

  def children: IndexedSeq[NewAST] = Array(child)

  def copy(newChildren: IndexedSeq[NewAST]): FilterSamples[RPK, RK, T] = {
    assert(newChildren.length == 1)
    FilterSamples(newChildren(0).asInstanceOf[MatrixAST[RPK, RK, T]], pred)
  }

  def typ: MatrixType = child.typ

  def execute(hc: HailContext): MatrixValue[RPK, RK, T] = {
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

case class FilterVariants[RPK, RK, T >: Null](
  child: MatrixAST[RPK, RK, T],
  pred: AST)(implicit tct: ClassTag[T]) extends MatrixAST[RPK, RK, T] {
  implicit val kOk: OrderedKey[RPK, RK] = child.kOk

  import kOk._

  def children: IndexedSeq[NewAST] = Array(child)

  def copy(newChildren: IndexedSeq[NewAST]): FilterVariants[RPK, RK, T] = {
    assert(newChildren.length == 1)
    FilterVariants(newChildren(0).asInstanceOf[MatrixAST[RPK, RK, T]], pred)
  }

  def typ: MatrixType = child.typ

  def execute(hc: HailContext): MatrixValue[RPK, RK, T] = {
    val prev = child.execute(hc)

    val localGlobalAnnotation = prev.localValue.globalAnnotation
    val ec = child.typ.variantEC

    val f: () => java.lang.Boolean = Parser.evalTypedExpr[java.lang.Boolean](pred, ec)

    val aggregatorOption = Aggregators.buildVariantAggregations[RPK, RK, T](
      prev.rdd.sparkContext, prev.localValue, ec)

    val p = (v: RK, va: Annotation, gs: Iterable[T]) => {
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
