package is.hail.utils

import is.hail.annotations.{Annotation, Inserter, Querier, UnsafeRow}
import is.hail.expr.{EvalContext, Parser}
import is.hail.expr.types._
import is.hail.sparkextras.OrderedRDD
import is.hail.variant.MatrixTable
import org.apache.spark.rdd.RDD

import scala.reflect.ClassTag

class RichMatrixTable(vsm: MatrixTable) {
  def expand(): RDD[(Annotation, Annotation, Annotation)] =
    mapWithKeys[(Annotation, Annotation, Annotation)]((v, s, g) => (v, s, g))

  def expandWithAll(): RDD[(Annotation, Annotation, Annotation, Annotation, Annotation)] =
    mapWithAll[(Annotation, Annotation, Annotation, Annotation, Annotation)]((v, va, s, sa, g) => (v, va, s, sa, g))

  def mapWithAll[U](f: (Annotation, Annotation, Annotation, Annotation, Annotation) => U)(implicit uct: ClassTag[U]): RDD[U] = {
    val localSampleIdsBc = vsm.sampleIdsBc
    val localSampleAnnotationsBc = vsm.sampleAnnotationsBc

    vsm.rdd
      .flatMap { case (v, (va, gs)) =>
        localSampleIdsBc.value.lazyMapWith2[Annotation, Annotation, U](localSampleAnnotationsBc.value, gs, { case (s, sa, g) => f(v, va, s, sa, g)
        })
      }
  }

  def mapWithKeys[U](f: (Annotation, Annotation, Annotation) => U)(implicit uct: ClassTag[U]): RDD[U] = {
    val localSampleIdsBc = vsm.sampleIdsBc

    vsm.rdd
      .flatMap { case (v, (va, gs)) =>
        localSampleIdsBc.value.lazyMapWith[Annotation, U](gs,
          (s, g) => f(v, s, g))
      }
  }

  def annotateSamplesF(signature: Type, path: List[String], annotation: (Annotation) => Annotation): MatrixTable = {
    val (t, i) = vsm.insertSA(signature, path)
    vsm.annotateSamples(annotation, t, i)
  }

  def insertVA(sig: Type, args: String*): (Type, Inserter) = vsm.insertVA(sig, args.toList)

  def querySA(code: String): (Type, Querier) = {
    val st = Map(Annotation.SAMPLE_HEAD -> (0, vsm.saSignature))
    val ec = EvalContext(st)
    val a = ec.a

    val (t, f) = Parser.parseExpr(code, ec)

    val f2: Annotation => Any = { annotation =>
      a(0) = annotation
      f()
    }

    (t, f2)
  }

  def queryGA(code: String): (Type, Querier) = {
    val st = Map(Annotation.GENOTYPE_HEAD -> (0, vsm.genotypeSignature))
    val ec = EvalContext(st)
    val a = ec.a

    val (t, f) = Parser.parseExpr(code, ec)

    val f2: Annotation => Any = { annotation =>
      a(0) = annotation
      f()
    }

    (t, f2)
  }

  def stringSampleIdsAndAnnotations: IndexedSeq[(Annotation, Annotation)] = vsm.stringSampleIds.zip(vsm.sampleAnnotations)

  def rdd: RDD[(Annotation, (Annotation, Iterable[Annotation]))] = ???

  def typedRDD[RK](implicit rkct: ClassTag[RK]): RDD[(RK, (Annotation, Iterable[Annotation]))] = {
    rdd.map { case (v, (va, gs)) =>
      (v.asInstanceOf[RK], (va, gs))
    }
  }

  def variants: RDD[Annotation] = vsm.rdd.keys

  def variantsAndAnnotations: OrderedRDD[Annotation, Annotation, Annotation] =
    rdd.mapValuesWithKey { case (v, (va, gs)) => va }
}
