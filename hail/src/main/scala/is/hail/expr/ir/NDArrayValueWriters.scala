package is.hail.expr.ir

import is.hail.asm4s._
import is.hail.expr.Nat
import is.hail.io.BufferSpec
import is.hail.types.encoded._
import is.hail.types.physical._
import is.hail.types.physical.stypes.{SCode, SType, SValue}
import is.hail.types.physical.stypes.interfaces.SNDArray
import is.hail.types.virtual._
import is.hail.utils._

import java.io.{BufferedOutputStream, OutputStream, OutputStreamWriter}

final case class NumericMatrixTextWriter(delimiter: String) extends ValueWriter {
  def writeValue(cb: EmitCodeBuilder, value: SValue, os: Value[OutputStream]): Unit = {
    require(value.st.virtualType == TNDArray(TFloat64, Nat(2)))
    val nd = value.asNDArray
    val IndexedSeq(nCols, _) = nd.shapes
    val numElts = SNDArray.numElements(nd.shapes)
    val sb = cb.newLocal[StringBuilder]("sb", Code.newInstance[StringBuilder, Long](numElts * 4L))
    val ow = cb.memoize(Code.newInstance[OutputStreamWriter, OutputStream](os))
    val k = cb.newLocal[Long]("k", 0L)

    SNDArray.forEachIndexRowMajor(cb, nd.shapes, "nd_matrix_text_writer") { case (cb, idx) =>
      val item = nd.loadElement(idx, cb).asDouble.value
      cb.assign(sb, sb.invoke[Double, StringBuilder]("append", item))
      cb.ifx((k % nCols).ceq(nCols - 1),
        cb.assign(sb, sb.invoke[Char, StringBuilder]("append", '\n')),
        cb.assign(sb, sb.invoke[String, StringBuilder]("append", delimiter)))
      cb.assign(k, k + 1)
    }

    cb += ow.invoke[String, Unit]("write", sb.invoke[String]("result"))
    cb += ow.invoke[Unit]("flush")
  }
}

final case object NumericNDArrayFlatBinaryWriter extends ValueWriter {
  def writeValue(cb: EmitCodeBuilder, value: SValue, os: Value[OutputStream]): Unit = {
    require(value.st.virtualType.asInstanceOf[TNDArray].elementType == TFloat64)
    val nd = value.asNDArray
    val ob = cb.memoize(BufferSpec.unblockedUncompressed.buildCodeOutputBuffer(Code.newInstance[BufferedOutputStream, OutputStream](os)))

    SNDArray.forEachIndexRowMajor(cb, nd.shapes, "nd_binary_flat_writer") { (cb, idx) =>
      cb += ob.writeDouble(nd.loadElement(idx, cb).asDouble.value)
    }

    cb += ob.invoke[Unit]("flush")
  }
}
