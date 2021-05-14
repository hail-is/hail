package is.hail.expr.ir.functions

import is.hail.annotations.{Memory, Region}
import is.hail.asm4s.{Code, Value}
import is.hail.expr.ir._
import is.hail.expr.{Nat, NatVariable}
import is.hail.linalg.{LAPACK, LinalgCodeUtils}
import is.hail.types.coerce
import is.hail.types.physical.stypes.EmitType
import is.hail.types.physical.stypes.concrete.{SBaseStructPointer, SNDArrayPointer}
import is.hail.types.physical.stypes.interfaces._
import is.hail.types.physical.{PBooleanRequired, PCanonicalNDArray, PCanonicalStruct, PFloat64Required, PNDArrayCode, PType}
import is.hail.types.virtual._

object NDArrayFunctions extends RegistryFunctions {
  override def registerAll() {
    for ((stringOp, argType, retType, irOp) <- ArrayFunctions.arrayOps) {
      val nDimVar = NatVariable()
      registerIR2(stringOp, TNDArray(argType, nDimVar), argType, TNDArray(retType, nDimVar)) { (_, a, c) =>
        val i = genUID()
        NDArrayMap(a, i, irOp(Ref(i, c.typ), c))
      }

      registerIR2(stringOp, argType, TNDArray(argType, nDimVar), TNDArray(retType, nDimVar)) { (_, c, a) =>
        val i = genUID()
        NDArrayMap(a, i, irOp(c, Ref(i, c.typ)))
      }

      registerIR2(stringOp, TNDArray(argType, nDimVar), TNDArray(argType, nDimVar), TNDArray(retType, nDimVar)) { (_, l, r) =>
        val lid = genUID()
        val rid = genUID()
        val lElemRef = Ref(lid, coerce[TNDArray](l.typ).elementType)
        val rElemRef = Ref(rid, coerce[TNDArray](r.typ).elementType)

        NDArrayMap2(l, r, lid, rid, irOp(lElemRef, rElemRef))
      }
    }

    def linear_solve(a: PNDArrayCode, b: PNDArrayCode, outputPt: PType, cb: EmitCodeBuilder, region: Value[Region]): (PNDArrayCode, Value[Int]) = {
      val aInput = a.asNDArray.memoize(cb, "A")
      val bInput = b.asNDArray.memoize(cb, "B")

      val aColMajor = LinalgCodeUtils.checkColMajorAndCopyIfNeeded(aInput, cb, region)
      val bColMajor = LinalgCodeUtils.checkColMajorAndCopyIfNeeded(bInput, cb, region)

      val IndexedSeq(n0, n1) = aColMajor.shapes(cb)

      cb.ifx(n0 cne n1, cb._fatal("hail.nd.solve: matrix a must be square."))

      val IndexedSeq(n, nrhs) = bColMajor.shapes(cb)

      cb.ifx(n0 cne n, cb._fatal("hail.nd.solve: Solve dimensions incompatible"))

      val infoDGESVResult = cb.newLocal[Int]("dgesv_result")
      val ipiv = cb.newLocal[Long]("dgesv_ipiv")
      cb.assign(ipiv, Code.invokeStatic1[Memory, Long, Long]("malloc", n * 4L))

      val aCopy = cb.newLocal[Long]("dgesv_a_copy")

      def aNumBytes = n * n * 8L

      cb.assign(aCopy, Code.invokeStatic1[Memory, Long, Long]("malloc", aNumBytes))
      val aColMajorFirstElement = aColMajor.firstDataAddress(cb)

      cb.append(Region.copyFrom(aColMajorFirstElement, aCopy, aNumBytes))

      val outputPType = coerce[PCanonicalNDArray](outputPt)
      val outputShape = IndexedSeq(n, nrhs)
      val (outputAddress, outputFinisher) = outputPType.constructDataFunction(outputShape, outputPType.makeColumnMajorStrides(outputShape, region, cb), cb, region)

      cb.append(Region.copyFrom(bColMajor.firstDataAddress(cb), outputAddress, n * nrhs * 8L))

      cb.assign(infoDGESVResult, Code.invokeScalaObject7[Int, Int, Long, Int, Long, Long, Int, Int](LAPACK.getClass, "dgesv",
        n.toI,
        nrhs.toI,
        aCopy,
        n.toI,
        ipiv,
        outputAddress,
        n.toI
      ))

      cb.append(Code.invokeStatic1[Memory, Long, Unit]("free", ipiv.load()))
      cb.append(Code.invokeStatic1[Memory, Long, Unit]("free", aCopy.load()))

      (outputFinisher(cb), infoDGESVResult)
    }

    registerIEmitCode2("linear_solve_no_crash", TNDArray(TFloat64, Nat(2)), TNDArray(TFloat64, Nat(2)), TStruct(("solution", TNDArray(TFloat64, Nat(2))), ("failed", TBoolean)),
      { (t, p1, p2) => EmitType(PCanonicalStruct(false, ("solution", PCanonicalNDArray(PFloat64Required, 2, false)), ("failed", PBooleanRequired)).sType, false) }) {
      case (cb, region, SBaseStructPointer(outputStructType: PCanonicalStruct), aec, bec) =>
        aec.toI(cb).flatMap(cb) { apc =>
          bec.toI(cb).map(cb) { bpc =>
            val outputNDArrayPType = outputStructType.fieldType("solution")
            val (resNDPCode, info) = linear_solve(apc.asNDArray, bpc.asNDArray, outputNDArrayPType, cb, region)
            val ndEmitCode = EmitCode(Code._empty, info cne 0, resNDPCode)
            outputStructType.constructFromFields(cb, region, IndexedSeq[EmitCode](ndEmitCode, EmitCode(Code._empty, false, primitive(info cne 0))), false)
          }
        }
    }

    registerPCode2("linear_solve", TNDArray(TFloat64, Nat(2)), TNDArray(TFloat64, Nat(2)), TNDArray(TFloat64, Nat(2)),
      { (t, p1, p2) => PCanonicalNDArray(PFloat64Required, 2, true).sType }) {
      case (er, cb, SNDArrayPointer(pt), apc, bpc) =>
        val (resPCode, info) = linear_solve(apc.asNDArray, bpc.asNDArray, pt, cb, er.region)
        cb.ifx(info cne 0, cb._fatal(s"hl.nd.solve: Could not solve, matrix was singular. dgesv error code ", info.toS))
        resPCode
    }
  }
}
