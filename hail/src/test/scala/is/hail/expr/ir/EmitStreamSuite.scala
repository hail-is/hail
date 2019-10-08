package is.hail.expr.ir

import is.hail.annotations.{Region, SafeRow, ScalaToRegionValue}
import is.hail.asm4s._
import is.hail.asm4s.joinpoint._
import is.hail.expr.types.physical._
import is.hail.expr.types.virtual._
import is.hail.utils._
import is.hail.variant.Call2

import org.apache.spark.sql.Row
import org.scalatest.testng.TestNGSuite
import org.testng.annotations.Test

class EmitStreamSuite extends TestNGSuite {

  private def compileStream(streamIR: IR, inputPType: PType): Any => IndexedSeq[Any] = {
    val fb = EmitFunctionBuilder[Region, Long, Boolean, Long]("eval_stream")
    val stream = EmitStream(fb, streamIR)
    val eltPType = stream.elementType
    fb.emit {
      val ait = stream.toArrayIterator(fb.apply_method)
      val arrayt = ait.toEmitTriplet(fb.apply_method, PArray(eltPType))
      Code(arrayt.setup, arrayt.m.mux(0L, arrayt.v))
    }
    val f = fb.resultWithIndex()
    ({ arg: Any => Region.scoped { r =>
      val off =
        if(arg == null)
          f(0, r)(r, 0L, true)
        else
          f(0, r)(r, ScalaToRegionValue(r, inputPType, arg), false)
      if(off == 0L)
        null
      else
        SafeRow.read(PArray(eltPType), r, off).asInstanceOf[IndexedSeq[Any]]
    } })
  }

  private def evalStream(streamIR: IR): IndexedSeq[Any] =
    compileStream(streamIR, PStruct.empty())(null)

  private def evalStreamLen(streamIR: IR): Option[Int] = {
    val fb = EmitFunctionBuilder[Region, Int]("eval_stream_len")
    val stream = EmitStream(fb, streamIR)
    fb.emit {
      JoinPoint.CallCC[Code[Int]] { (jb, ret) =>
        val str = stream.stream
        val mb = fb.apply_method
        str.init(mb, jb, ()) {
          case EmitStream.Missing | EmitStream.Empty => ret(0)
          case EmitStream.Start(s0) =>
            str.length(s0) match {
              case Some(len) => ret(len)
              case None => ret(-1)
            }
        }
      }
    }
    val f = fb.resultWithIndex()
    Region.scoped { r =>
      val len = f(0, r)(r)
      if(len < 0) None else Some(len)
    }
  }

  @Test def testEmitNA() {
    assert(evalStream(NA(TStream(TInt32()))) == null)
  }

  @Test def testEmitRange() {
    val tripleType = PStruct(false, "start" -> PInt32(), "stop" -> PInt32(), "step" -> PInt32())
    val triple = In(0, tripleType.virtualType)
    val range = compileStream(
      StreamRange(GetField(triple, "start"), GetField(triple, "stop"), GetField(triple, "step")),
      tripleType)
    for {
      start <- -2 to 2
      stop <- -2 to 8
      step <- 1 to 3
    } {
      assert(range(Row(start, stop, step)) == Array.range(start, stop, step).toFastIndexedSeq,
        s"($start, $stop, $step)")
    }
    assert(range(Row(null, 10, 1)) == null)
    assert(range(Row(0, null, 1)) == null)
    assert(range(Row(0, 10, null)) == null)
    assert(range(null) == null)
  }

  @Test def testEmitToStream() {
    val tests: Array[(IR, IndexedSeq[Any])] = Array(
      ToStream(MakeArray(Seq[IR](), TArray(TInt32()))) -> IndexedSeq(),
      ToStream(MakeArray(Seq[IR](1, 2, 3, 4), TArray(TInt32()))) -> IndexedSeq(1, 2, 3, 4),
      ToStream(NA(TArray(TInt32()))) -> null
    )
    for ((ir, v) <- tests) {
      val expectedLen = Some(if(v == null) 0 else v.length)
      assert(evalStream(ir) == v, Pretty(ir))
      assert(evalStreamLen(ir) == expectedLen, Pretty(ir))
    }
  }

  @Test def testEmitLet() {
    val Seq(start, end, i) = Seq("start", "end", "i").map(Ref(_, TInt32()))
    val ir =
      Let("end", 10,
        Let("start", 3,
          StreamRange(start, end, 1))
      )
    assert(evalStream(ir) == (3 until 10).toIndexedSeq, Pretty(ir))
    assert(evalStreamLen(ir) == Some(10 - 3), Pretty(ir))
  }
}
