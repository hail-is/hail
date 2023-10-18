package is.hail.expr.ir.functions

import is.hail.annotations.{Memory, Region}
import is.hail.asm4s._
import is.hail.expr.ir._
import is.hail.expr.{Nat, NatVariable}
import is.hail.linalg.{LAPACK, LinalgCodeUtils}
import is.hail.types.tcoerce
import is.hail.types.physical.stypes.EmitType
import is.hail.types.physical.stypes.concrete.{SBaseStructPointer, SNDArrayPointer, SNDArrayPointerValue}
import is.hail.types.physical.stypes.interfaces._
import is.hail.types.physical.stypes.primitives._
import is.hail.types.physical._
import is.hail.types.virtual._
import is.hail.utils._

object  NDArrayFunctions extends RegistryFunctions {
  override def registerAll() {
    for ((stringOp, argType, retType, irOp) <- ArrayFunctions.arrayOps) {
      val nDimVar = NatVariable()
      registerIR2(stringOp, TNDArray(argType, nDimVar), argType, TNDArray(retType, nDimVar)) { (_, a, c, errorID) =>
        val i = genUID()
        NDArrayMap(a, i, irOp(Ref(i, c.typ), c, errorID))
      }

      registerIR2(stringOp, argType, TNDArray(argType, nDimVar), TNDArray(retType, nDimVar)) { (_, c, a, errorID) =>
        val i = genUID()
        NDArrayMap(a, i, irOp(c, Ref(i, c.typ), errorID))
      }

      registerIR2(stringOp, TNDArray(argType, nDimVar), TNDArray(argType, nDimVar), TNDArray(retType, nDimVar)) { (_, l, r, errorID) =>
        val lid = genUID()
        val rid = genUID()
        val lElemRef = Ref(lid, tcoerce[TNDArray](l.typ).elementType)
        val rElemRef = Ref(rid, tcoerce[TNDArray](r.typ).elementType)

        NDArrayMap2(l, r, lid, rid, irOp(lElemRef, rElemRef, errorID), errorID)
      }
    }

    def linear_triangular_solve(ndCoef: SNDArrayValue, ndDep: SNDArrayValue, lower: SBooleanValue, outputPt: PType, cb: EmitCodeBuilder, region: Value[Region], errorID: Value[Int]): (SNDArrayValue, Value[Int]) = {
      val ndCoefColMajor = LinalgCodeUtils.checkColMajorAndCopyIfNeeded(ndCoef, cb, region)
      val ndDepColMajor = LinalgCodeUtils.checkColMajorAndCopyIfNeeded(ndDep, cb, region)

      val IndexedSeq(ndCoefRow, ndCoefCol) = ndCoefColMajor.shapes
      cb.if_(ndCoefRow cne ndCoefCol, cb._fatalWithError(errorID, "hail.nd.solve_triangular: matrix a must be square."))

      val IndexedSeq(ndDepRow, ndDepCol) = ndDepColMajor.shapes
      cb.if_(ndCoefRow  cne ndDepRow, cb._fatalWithError(errorID,"hail.nd.solve_triangular: Solve dimensions incompatible"))

      val uplo = cb.newLocal[String]("dtrtrs_uplo")
      cb.if_(lower.value, cb.assign(uplo, const("L")), cb.assign(uplo, const("U")))

      val infoDTRTRSResult = cb.newLocal[Int]("dtrtrs_result")

      val outputPType = tcoerce[PCanonicalNDArray](outputPt)
      val output = outputPType.constructByActuallyCopyingData(ndDepColMajor, cb, region)

      cb.assign(infoDTRTRSResult, Code.invokeScalaObject9[String, String, String, Int, Int, Long, Int, Long, Int, Int](LAPACK.getClass, "dtrtrs",
        uplo,
        const("N"),
        const("N"),
        ndDepRow.toI,
        ndDepCol.toI,
        ndCoefColMajor.firstDataAddress,
        ndDepRow.toI,
        output.firstDataAddress,
        ndDepRow.toI
      ))

      (output, infoDTRTRSResult)
    }

    def linear_solve(a: SNDArrayValue, b: SNDArrayValue, outputPt: PType, cb: EmitCodeBuilder, region: Value[Region], errorID: Value[Int]): (SNDArrayValue, Value[Int]) = {
      val aColMajor = LinalgCodeUtils.checkColMajorAndCopyIfNeeded(a, cb, region)
      val bColMajor = LinalgCodeUtils.checkColMajorAndCopyIfNeeded(b, cb, region)

      val IndexedSeq(n0, n1) = aColMajor.shapes

      cb.if_(n0 cne n1, cb._fatalWithError(errorID, "hail.nd.solve: matrix a must be square."))

      val IndexedSeq(n, nrhs) = bColMajor.shapes

      cb.if_(n0 cne n, cb._fatalWithError(errorID, "hail.nd.solve: Solve dimensions incompatible"))

      val infoDGESVResult = cb.newLocal[Int]("dgesv_result")
      val ipiv = cb.newLocal[Long]("dgesv_ipiv")
      cb.assign(ipiv, Code.invokeStatic1[Memory, Long, Long]("malloc", n * 4L))

      val aCopy = cb.newLocal[Long]("dgesv_a_copy")

      def aNumBytes = n * n * 8L

      cb.assign(aCopy, Code.invokeStatic1[Memory, Long, Long]("malloc", aNumBytes))
      val aColMajorFirstElement = aColMajor.firstDataAddress

      cb.append(Region.copyFrom(aColMajorFirstElement, aCopy, aNumBytes))

      val outputPType = tcoerce[PCanonicalNDArray](outputPt)
      val outputShape = IndexedSeq(n, nrhs)
      val (outputAddress, outputFinisher) = outputPType.constructDataFunction(outputShape, outputPType.makeColumnMajorStrides(outputShape, cb), cb, region)

      cb.append(Region.copyFrom(bColMajor.firstDataAddress, outputAddress, n * nrhs * 8L))

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
      case (cb, region, SBaseStructPointer(outputStructType: PCanonicalStruct), errorID, aec, bec) =>
        aec.toI(cb).flatMap(cb) { apc =>
          bec.toI(cb).map(cb) { bpc =>
            val outputNDArrayPType = outputStructType.fieldType("solution")
            val (resNDPCode, info) = linear_solve(apc.asNDArray, bpc.asNDArray, outputNDArrayPType, cb, region, errorID)
            val ndEmitCode = EmitCode(Code._empty, info cne 0, resNDPCode)
            outputStructType.constructFromFields(cb, region, IndexedSeq[EmitCode](ndEmitCode, EmitCode(Code._empty, false, primitive(cb.memoize(info cne 0)))), false)
          }
        }
    }

    registerSCode2("linear_solve", TNDArray(TFloat64, Nat(2)), TNDArray(TFloat64, Nat(2)), TNDArray(TFloat64, Nat(2)),
      { (t, p1, p2) => PCanonicalNDArray(PFloat64Required, 2, true).sType }) {
      case (er, cb, SNDArrayPointer(pt), apc, bpc, errorID) =>
        val (resPCode, info) = linear_solve(apc.asNDArray, bpc.asNDArray, pt, cb, er.region, errorID)
        cb.if_(info cne 0, cb._fatalWithError(errorID,s"hl.nd.solve: Could not solve, matrix was singular. dgesv error code ", info.toS))
        resPCode
    }

    registerIEmitCode3("linear_triangular_solve_no_crash", TNDArray(TFloat64, Nat(2)), TNDArray(TFloat64, Nat(2)), TBoolean, TStruct(("solution", TNDArray(TFloat64, Nat(2))), ("failed", TBoolean)),
      { (t, p1, p2, p3) => EmitType(PCanonicalStruct(false, ("solution", PCanonicalNDArray(PFloat64Required, 2, false)), ("failed", PBooleanRequired)).sType, false) }) {
      case (cb, region, SBaseStructPointer(outputStructType: PCanonicalStruct), errorID, aec, bec, lowerec) =>
        aec.toI(cb).flatMap(cb) { apc =>
          bec.toI(cb).flatMap(cb) { bpc =>
            lowerec.toI(cb).map(cb) { lowerpc =>
              val outputNDArrayPType = outputStructType.fieldType("solution")
              val (resNDPCode, info) = linear_triangular_solve(apc.asNDArray, bpc.asNDArray, lowerpc.asBoolean, outputNDArrayPType, cb, region, errorID)
              val ndEmitCode = EmitCode(Code._empty, info cne 0, resNDPCode)
              outputStructType.constructFromFields(cb, region, IndexedSeq[EmitCode](ndEmitCode, EmitCode(Code._empty, false, primitive(cb.memoize(info cne 0)))), false)
            }
          }
        }
    }

    registerSCode3("linear_triangular_solve", TNDArray(TFloat64, Nat(2)), TNDArray(TFloat64, Nat(2)), TBoolean, TNDArray(TFloat64, Nat(2)),
      { (t, p1, p2, p3) => PCanonicalNDArray(PFloat64Required, 2, true).sType }) {
      case (er, cb, SNDArrayPointer(pt), apc, bpc, lower, errorID) =>
        val (resPCode, info) = linear_triangular_solve(apc.asNDArray, bpc.asNDArray, lower.asBoolean, pt, cb, er.region, errorID)
        cb.if_(info cne 0, cb._fatalWithError(errorID,s"hl.nd.solve: Could not solve, matrix was singular. dtrtrs error code ", info.toS))
        resPCode
    }

    registerSCode3("zero_band", TNDArray(TFloat64, Nat(2)), TInt64, TInt64, TNDArray(TFloat64, Nat(2)),
      { (_, _, _, _) => PCanonicalNDArray(PFloat64Required, 2, true).sType }) {
      case (er, cb, rst: SNDArrayPointer, block: SNDArrayValue, lower: SInt64Value, upper: SInt64Value, errorID) =>
        val newBlock = rst.coerceOrCopy(cb, er.region, block, deepCopy = false).asInstanceOf[SNDArrayPointerValue]
        val IndexedSeq(nRows, nCols) = newBlock.shapes
        val lowestDiagIndex = cb.memoize(- (nRows.get - 1L))
        val highestDiagIndex = cb.memoize(nCols.get - 1L)
        val iLeft = cb.newLocal[Long]("iLeft")
        val iRight = cb.newLocal[Long]("iRight")
        val i = cb.newLocal[Long]("i")
        val j = cb.newLocal[Long]("j")

        cb.if_(lower.value > lowestDiagIndex, {
          cb.assign(iLeft, (-lower.value).max(0L))
          cb.assign(iRight, (nCols.get - lower.value).min(nRows.get))

          cb.for_({
            cb.assign(i, iLeft)
            cb.assign(j, lower.value.max(0L))
          }, i < iRight, {
            cb.assign(i, i + 1L)
            cb.assign(j, j + 1L)
          }, {
            // block(i to i, 0 until j) := 0.0
            newBlock.slice(cb, i, (null, j)).coiterateMutate(cb, er.region) { _ =>
              primitive(0.0d)
            }
          })

          // block(iRight until nRows, ::) := 0.0
          newBlock.slice(cb, (iRight, null), ColonIndex).coiterateMutate(cb, er.region) { _ =>
            primitive(0.0d)
          }
        })

        cb.if_(upper.value < highestDiagIndex, {
          cb.assign(iLeft, (-upper.value).max(0L))
          cb.assign(iRight, (nCols.get - upper.value).min(nRows.get))

          // block(0 util iLeft, ::) := 0.0
          newBlock.slice(cb, (null, iLeft), ColonIndex).coiterateMutate(cb, er.region) { _ =>
            primitive(0.0d)
          }

          cb.for_({
            cb.assign(i, iLeft)
            cb.assign(j, upper.value.max(0L) + 1)
          }, i < iRight, {
            cb.assign(i, i + 1)
            cb.assign(j, j + 1)
          }, {
            // block(i to i, j to nCols) := 0.0
            newBlock.slice(cb, i, (j, null)).coiterateMutate(cb, er.region) { _ =>
              primitive(0.0d)
            }
          })
        })

        newBlock
    }

    registerSCode3("zero_row_intervals", TNDArray(TFloat64, Nat(2)), TArray(TInt64), TArray(TInt64), TNDArray(TFloat64, Nat(2)),
      { (_, _, _, _) => PCanonicalNDArray(PFloat64Required, 2, true).sType }) {
      case (er, cb, rst: SNDArrayPointer, block: SNDArrayValue, starts: SIndexableValue, stops: SIndexableValue, errorID) =>
        val newBlock = rst.coerceOrCopy(cb, er.region, block, deepCopy = false).asInstanceOf[SNDArrayPointerValue]
        val row = cb.newLocal[Long]("rowIdx")
        val IndexedSeq(nRows, nCols) = newBlock.shapes
        cb.for_(cb.assign(row, 0L), row < nRows.get, cb.assign(row, row + 1L), {
          val start = starts.loadElement(cb, row.toI).get(cb).asInt64.value
          val stop = stops.loadElement(cb, row.toI).get(cb).asInt64.value
          newBlock.slice(cb, row, (null, start)).coiterateMutate(cb, er.region) { _ =>
            primitive(0.0d)
          }
          newBlock.slice(cb, row, (stop, null)).coiterateMutate(cb, er.region) { _ =>
            primitive(0.0d)
          }
        })

        newBlock
      }
  }
}
