package is.hail.expr.ir

import is.hail.annotations._
import is.hail.asm4s._
import is.hail.expr
import is.hail.expr.{Type, Aggregator, TAggregable, TArray, TBoolean, TContainer, TFloat32, TFloat64, TInt32, TInt64, TStruct}
import is.hail.utils._
import org.objectweb.asm.Opcodes._
import org.objectweb.asm.tree._

import scala.language.existentials

object ExtractAggregators {

  private case class IRAgg(in: In, t: Type, z: IR, seq: (IR, IR) => IR, comb: (IR, IR) => IR) { }

  trait Aggregable {
    def aggregate(
      zero: (MemoryBuffer) => Long,
      seq: (MemoryBuffer, Long, Boolean, Long, Boolean) => Long,
      comb: (MemoryBuffer, Long, Boolean, Long, Boolean) => Long): Long
  }

  def apply(ir: IR, t: TAggregable): (IR, TStruct, (MemoryBuffer, Aggregable) => Long) = {
    val (ir2, aggs) = extract(ir)
    val fields = aggs.map(_.t).zipWithIndex.map { case (t, i) => (i.toString -> t) }
    val tT: TStruct = t.carrierStruct
    val tU: TStruct = TStruct(fields:_*)
    def zipValues(irs: Iterable[IR]): IR =
      MakeStruct(fields.zip(irs).map { case ((n, t), v) => (n, t, v) })
    val zeroFb = FunctionBuilder.functionBuilder[MemoryBuffer, Long]
    Compile(zipValues(aggs.map(_.z)), zeroFb)
    val seqFb = FunctionBuilder.functionBuilder[MemoryBuffer, Long, Boolean, Long, Boolean, Long]
    println(zipValues(aggs.zipWithIndex.map { case (x, i) =>
      x.seq(GetField(In(0, tU), i.toString(), x.t), In(1, tT)) }))
    Compile(zipValues(aggs.zipWithIndex.map { case (x, i) =>
      x.seq(GetField(In(0, tU), i.toString(), x.t), In(1, tT)) }), seqFb)
    val combFb = FunctionBuilder.functionBuilder[MemoryBuffer, Long, Boolean, Long, Boolean, Long]
    Compile(zipValues(aggs.zipWithIndex.map { case (x, i) =>
      x.comb(GetField(In(0, tU), i.toString(), x.t), GetField(In(0, tU), i.toString(), x.t)) }), combFb)

    // update all the references to the intermediate
    aggs.map(_.in).foreach(_.typ = tU)

    val zero = zeroFb.result()
    val seq = seqFb.result()
    val comb = combFb.result()

    (ir2, tU, { (r, agg) =>
      // load classes into JVM
      val z = zero()
      val f = seq()
      val g = comb()
      agg.aggregate(
        r => z(r),
        (r, t, mt, u, mu) => f(r, t, mt, u, mu),
        (region, l, ml, r, mr) => g(region, l, ml, r, mr))
    })
  }

  private def extract(ir: IR): (IR, Array[IRAgg]) = {
    val ab = new ArrayBuilder[IRAgg]()
    val ir2 = extract(ir, ab)
    (ir2, ab.result())
  }

  private def extract(ir: IR, ab: ArrayBuilder[IRAgg]): IR = {
    def extract(ir: IR): IR = this.extract(ir, ab)
    ir match {
      case Ref(name, typ) =>
        assert(typ.isRealizable)
        ir
      case AggIn(_) =>
        throw new RuntimeException(s"AggMap must be used inside an AggSum, but found: $ir")
      case AggMap(a, name, body, typ) =>
        throw new RuntimeException(s"AggMap must be used inside an AggSum, but found: $ir")
      case AggSum(a, typ) =>
        val tAgg = a.typ.asInstanceOf[TAggregable]
        val in = In(0, null)
        ab += IRAgg(in,
          typ,
          zeroValue(typ),
          (u, t) => ApplyBinaryPrimOp(Add(), u, tAgg.getElement(lower(a, t)), typ),
          (l, r) => ApplyBinaryPrimOp(Add(), l, r, typ))

        GetField(in, (ab.length - 1).toString())
      case _ => Recur(extract)(ir)
    }
  }

  private def lower(ir: IR, aggIn: IR): IR = {
    def lower(ir: IR): IR = this.lower(ir, aggIn)
    ir match {
      case AggIn(typ) =>
        aggIn
      case AggMap(a, name, body, typ) =>
        val tAgg = a.typ.asInstanceOf[TAggregable]
        val tA = tAgg.carrierStruct
        val la = lower(a)
        assert(la.typ == tA, s"should have same type ${la.typ} $tA, $la")
        tAgg.inContext(la, e => Let(name, e, lower(body)))
      case AggSum(_, _) =>
        throw new RuntimeException(s"Found aggregator inside an aggregator: $ir")
      case In(i, typ) =>
        throw new RuntimeException(s"Referenced input inside an aggregator, that's a no-no: $ir")
      case InMissingness(i) =>
        throw new RuntimeException(s"Referenced input inside an aggregator, that's a no-no: $ir")
      case _ => Recur(lower)(ir)
    }
  }

  private def zeroValue(t: Type): IR = {
    t match {
      case _: TBoolean => False()
      case _: TInt32 => I32(0)
      case _: TInt64 => I64(0L)
      case _: TFloat32 => F32(0.0f)
      case _: TFloat64 => F64(0.0)
      case _ => throw new RuntimeException(s"no zero for $t")
    }
  }
}
