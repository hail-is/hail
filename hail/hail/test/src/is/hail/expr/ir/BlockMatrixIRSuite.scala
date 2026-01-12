package is.hail.expr.ir

import is.hail.{ExecStrategy, HailSuite}
import is.hail.ExecStrategy.ExecStrategy
import is.hail.collection.FastSeq
import is.hail.collection.implicits.toRichIterable
import is.hail.expr.Nat
import is.hail.expr.ir.defs.{
  Apply, ApplyBinaryPrimOp, ApplyUnaryPrimOp, BlockMatrixWrite, ErrorIDs, F64, I64, Literal,
  MakeArray, MakeNDArray, MakeTuple, ReadValue, Ref, Str, ToArray, True, UUID4, WriteValue,
}
import is.hail.io.TypedCodecSpec
import is.hail.linalg.BlockMatrix
import is.hail.types.encoded.{EBlockMatrixNDArray, EFloat64Required}
import is.hail.types.virtual._

import java.lang.Math.floorDiv

import breeze.linalg.{DenseMatrix => BDM}
import breeze.math.Ring.ringFromField
import org.scalatest.Inspectors.forAll
import org.testng.annotations.{DataProvider, Test}

class BlockMatrixIRSuite extends HailSuite {

  val N_ROWS = 3
  val N_COLS = 3
  val BLOCK_SIZE = 10
  val shape: Array[Long] = Array(N_ROWS.toLong, N_COLS.toLong)

  def toIR(bdm: BDM[Double], blockSize: Int = BLOCK_SIZE): BlockMatrixIR =
    ValueToBlockMatrix(
      Literal(TArray(TFloat64), bdm.t.toArray.toFastSeq),
      FastSeq(bdm.rows.toLong, bdm.cols.toLong),
      blockSize,
    )

  def fill(v: Double, nRows: Int = N_ROWS, nCols: Int = N_COLS, blockSize: Int = BLOCK_SIZE) =
    toIR(BDM.fill(nRows, nCols)(v), blockSize)

  val ones: BlockMatrixIR = fill(1)

  implicit val execStrats: Set[ExecStrategy] = ExecStrategy.allRelational

  def makeMap2(left: BlockMatrixIR, right: BlockMatrixIR, op: BinaryOp, strategy: SparsityStrategy)
    : BlockMatrixMap2 = {
    val l = Ref(freshName(), TFloat64)
    val r = Ref(freshName(), TFloat64)
    BlockMatrixMap2(left, right, l.name, r.name, ApplyBinaryPrimOp(op, l, r), strategy)
  }

  @DataProvider(name = "valToBMData")
  def valToBMData(): Array[Array[Any]] = Array(
    Array(
      ValueToBlockMatrix(F64(1), FastSeq(1, 1), BLOCK_SIZE),
      BDM.fill[Double](1, 1)(1),
    ),
    Array(ones, BDM.fill[Double](N_ROWS, N_COLS)(1)),
    Array(
      ValueToBlockMatrix(
        child = ToArray(mapIR(rangeIR(64))(it => it.toD)),
        shape = FastSeq(8, 8),
        blockSize = 3,
      ),
      BDM.tabulate[Double](8, 8)((i, j) => i * 8.0 + j),
    ),
    Array(
      ValueToBlockMatrix(
        child = ToArray(mapIR(rangeIR(27))(it => it.toD)),
        shape = FastSeq(3, 9),
        blockSize = 3,
      ),
      BDM.tabulate[Double](3, 9)((i, j) => i * 9.0 + j),
    ),
    Array(
      ValueToBlockMatrix(
        child = MakeNDArray.fill(F64(1), FastSeq(I64(8), I64(8)), True()),
        shape = FastSeq(8, 8),
        blockSize = 3,
      ),
      BDM.fill[Double](8, 8)(1),
    ),
    Array(
      ValueToBlockMatrix(
        child = MakeNDArray(
          ToArray(mapIR(rangeIR(27))(it => it.toD)),
          MakeTuple.ordered(FastSeq(I64(3), I64(9))),
          True(),
          ErrorIDs.NO_ERROR,
        ),
        shape = FastSeq(3, 9),
        blockSize = 3,
      ),
      BDM.tabulate[Double](3, 9)((i, j) => i * 9.0 + j),
    ),
  )

  @Test(dataProvider = "valToBMData")
  def testValueToBlockMatrix(bmir: ValueToBlockMatrix, bdm: BDM[Double]): Unit =
    assertBMEvalsTo(bmir, bdm)

  def rangeBlockMatrix(nRows: Int, nCols: Int, blockSize: Int = 2): BlockMatrixIR =
    ValueToBlockMatrix(
      ToArray(mapIR(rangeIR(nRows * nCols))(i => i.toD)),
      FastSeq(nRows.toLong, nCols.toLong),
      blockSize,
    )

  def sparseRangeBlockMatrix(
    nRows: Int,
    nCols: Int,
    blocks: IndexedSeq[(Int, Int)],
    blockSize: Int = 2,
  ): BlockMatrixIR =
    BlockMatrixSparsify(
      rangeBlockMatrix(nRows, nCols, blockSize),
      PerBlockSparsifier(blocks.map(p => p._1 + p._2 * 3)),
    )

  def rangeBreezeMatrix(nRows: Int, nCols: Int): BDM[Double] =
    BDM.tabulate[Double](nRows, nCols)((i, j) => i * 5.0 + j)

  def sparseRangeBreezeMatrix(nRows: Int, nCols: Int, blocks: IndexedSeq[(Int, Int)]): BDM[Double] =
    BDM.tabulate[Double](nRows, nCols) { (i, j) =>
      if (blocks.contains(floorDiv(i, 2) -> floorDiv(j, 2)))
        i * 5.0 + j
      else 0.0
    }

  @Test
  def testBlockMatrixSparsify(): Unit = {
    val blocks = FastSeq((1, 0), (0, 1), (2, 1), (0, 2))
    val bm = sparseRangeBlockMatrix(5, 5, blocks)
    val expected = sparseRangeBreezeMatrix(5, 5, blocks)
    assertBMEvalsTo(bm, expected)
  }

  @Test
  def testBlockMatrixSparseTranspose(): Unit = {
    val blocks = FastSeq((1, 0), (0, 1), (2, 1), (0, 2))
    val bm = sparseRangeBlockMatrix(5, 5, blocks)
    val expected = sparseRangeBreezeMatrix(5, 5, blocks)
    assertBMEvalsTo(
      BlockMatrixBroadcast(bm, FastSeq(1, 0), FastSeq(5, 5), 2),
      expected.t,
    )
  }

  @Test
  def testBlockMatrixSparseSumCols(): Unit = {
    val blocks = FastSeq((1, 0), (0, 1), (2, 1), (0, 2))
    val bm = sparseRangeBlockMatrix(5, 5, blocks)
    assertBMEvalsTo(
      BlockMatrixAgg(bm, FastSeq(0)),
      BDM.create(1, 5, Array(25.0, 27.0, 31.0, 34.0, 13.0)),
    )
  }

  @Test
  def testBlockMatrixSparseSumRows(): Unit = {
    val blocks = FastSeq((1, 0), (0, 1), (2, 1), (0, 2))
    val bm = sparseRangeBlockMatrix(5, 5, blocks)
    assertBMEvalsTo(
      BlockMatrixAgg(bm, FastSeq(1)),
      BDM.create(5, 1, Array(9.0, 24.0, 21.0, 31.0, 45.0)),
    )
  }

  @Test
  def testBlockMatrixSparseSumAll(): Unit = {
    val blocks = FastSeq((1, 0), (0, 1), (2, 1), (0, 2))
    val bm = sparseRangeBlockMatrix(5, 5, blocks)
    assertBMEvalsTo(
      BlockMatrixAgg(bm, FastSeq(0, 1)),
      BDM.create(1, 1, Array(130.0)),
    )
  }

  @Test def testBlockMatrixWriteRead(): Unit = {
    implicit val execStrats: Set[ExecStrategy] = ExecStrategy.interpretOnly
    val tempPath = ctx.createTmpPath("test-blockmatrix-write-read", "bm")
    Interpret[Unit](
      ctx,
      BlockMatrixWrite(ones, BlockMatrixNativeWriter(tempPath, false, false, false)),
    )

    assertBMEvalsTo(
      BlockMatrixRead(BlockMatrixNativeReader(fs, tempPath)),
      BDM.fill[Double](N_ROWS, N_COLS)(1),
    )
  }

  @Test def testBlockMatrixMap(): Unit = {
    val element = Ref(freshName(), TFloat64)
    val sqrtIR = BlockMatrixMap(
      ones,
      element.name,
      Apply("sqrt", FastSeq(), FastSeq(element), TFloat64, ErrorIDs.NO_ERROR),
      false,
    )
    val negIR =
      BlockMatrixMap(ones, element.name, ApplyUnaryPrimOp(Negate, element), false)
    val logIR = BlockMatrixMap(
      ones,
      element.name,
      Apply("log", FastSeq(), FastSeq(element), TFloat64, ErrorIDs.NO_ERROR),
      true,
    )
    val absIR = BlockMatrixMap(
      ones,
      element.name,
      Apply("abs", FastSeq(), FastSeq(element), TFloat64, ErrorIDs.NO_ERROR),
      false,
    )

    assertBMEvalsTo(sqrtIR, BDM.fill[Double](3, 3)(1))
    assertBMEvalsTo(negIR, BDM.fill[Double](3, 3)(-1))
    assertBMEvalsTo(logIR, BDM.fill[Double](3, 3)(0))
    assertBMEvalsTo(absIR, BDM.fill[Double](3, 3)(1))
  }

  @Test def testBlockMatrixMap2(): Unit = {
    val onesAddOnes = makeMap2(ones, ones, Add(), UnionBlocks)
    val onesSubOnes = makeMap2(ones, ones, Subtract(), UnionBlocks)
    val onesMulOnes = makeMap2(ones, ones, Multiply(), IntersectionBlocks)
    val onesDivOnes = makeMap2(ones, ones, FloatingPointDivide(), NeedsDense)

    assertBMEvalsTo(onesAddOnes, BDM.fill[Double](3, 3)(1.0 + 1.0))
    assertBMEvalsTo(onesSubOnes, BDM.fill[Double](3, 3)(1.0 - 1.0))
    assertBMEvalsTo(onesMulOnes, BDM.fill[Double](3, 3)(1.0 * 1.0))
    assertBMEvalsTo(onesDivOnes, BDM.fill[Double](3, 3)(1.0 / 1.0))
  }

  @Test def testBlockMatrixBroadcastValue_Scalars(): Unit = {
    val broadcastTwo = BlockMatrixBroadcast(
      ValueToBlockMatrix(
        MakeArray(IndexedSeq[F64](F64(2)), TArray(TFloat64)),
        Array[Long](1, 1),
        ones.typ.blockSize,
      ),
      FastSeq(),
      shape,
      ones.typ.blockSize,
    )

    val onesAddTwo = makeMap2(ones, broadcastTwo, Add(), UnionBlocks)
    val onesSubTwo = makeMap2(ones, broadcastTwo, Subtract(), UnionBlocks)
    val onesMulTwo = makeMap2(ones, broadcastTwo, Multiply(), IntersectionBlocks)
    val onesDivTwo = makeMap2(ones, broadcastTwo, FloatingPointDivide(), NeedsDense)

    assertBMEvalsTo(onesAddTwo, BDM.fill[Double](3, 3)(1.0 + 2.0))
    assertBMEvalsTo(onesSubTwo, BDM.fill[Double](3, 3)(1.0 - 2.0))
    assertBMEvalsTo(onesMulTwo, BDM.fill[Double](3, 3)(1.0 * 2.0))
    assertBMEvalsTo(onesDivTwo, BDM.fill[Double](3, 3)(1.0 / 2.0))
  }

  @Test def testBlockMatrixBroadcastValue_Vectors(): Unit = {
    val vectorLiteral = MakeArray(IndexedSeq[F64](F64(1), F64(2), F64(3)), TArray(TFloat64))

    val broadcastRowVector = BlockMatrixBroadcast(
      ValueToBlockMatrix(vectorLiteral, Array[Long](1, 3), ones.typ.blockSize),
      FastSeq(1),
      shape,
      ones.typ.blockSize,
    )
    val broadcastColVector = BlockMatrixBroadcast(
      ValueToBlockMatrix(vectorLiteral, Array[Long](3, 1), ones.typ.blockSize),
      FastSeq(0),
      shape,
      ones.typ.blockSize,
    )

    val ops = Array(
      (Add(), UnionBlocks, (i: Double, j: Double) => i + j),
      (Subtract(), UnionBlocks, (i: Double, j: Double) => i - j),
      (Multiply(), IntersectionBlocks, (i: Double, j: Double) => i * j),
      (FloatingPointDivide(), NeedsDense, (i: Double, j: Double) => i / j),
    )

    forAll(ops) { case (op, merge, f) =>
      val rightRowOp = makeMap2(ones, broadcastRowVector, op, merge)
      val rightColOp = makeMap2(ones, broadcastColVector, op, merge)
      val leftRowOp = makeMap2(broadcastRowVector, ones, op, merge)
      val leftColOp = makeMap2(broadcastColVector, ones, op, merge)

      val expectedRightRowOp = BDM.tabulate(3, 3)((_, j) => f(1.0, j.toDouble + 1))
      val expectedRightColOp = BDM.tabulate(3, 3)((i, _) => f(1.0, i.toDouble + 1))
      val expectedLeftRowOp = BDM.tabulate(3, 3)((_, j) => f(j.toDouble + 1, 1.0))
      val expectedLeftColOp = BDM.tabulate(3, 3)((i, _) => f(i.toDouble + 1, 1.0))

      assertBMEvalsTo(rightRowOp, expectedRightRowOp)
      assertBMEvalsTo(rightColOp, expectedRightColOp)
      assertBMEvalsTo(leftRowOp, expectedLeftRowOp)
      assertBMEvalsTo(leftColOp, expectedLeftColOp)
    }
  }

  @Test def testBlockMatrixFilter(): Unit = {
    val nRows = 5
    val nCols = 8
    val original = BDM.tabulate[Double](nRows, nCols)((i, j) => i.toDouble * nCols + j)
    val unfiltered = toIR(original, blockSize = 3)

    val keepRows = Array(0L, 1L, 4L)
    val keepCols = Array(0L, 2L, 7L)

    assertBMEvalsTo(
      BlockMatrixFilter(unfiltered, Array(keepRows, Array())),
      original(keepRows.map(_.toInt).toFastSeq, ::).toDenseMatrix,
    )
    assertBMEvalsTo(
      BlockMatrixFilter(unfiltered, Array(Array(), keepCols)),
      original(::, keepCols.map(_.toInt).toFastSeq).toDenseMatrix,
    )
    assertBMEvalsTo(
      BlockMatrixFilter(unfiltered, Array(keepRows, keepCols)),
      original(keepRows.map(_.toInt).toFastSeq, keepCols.map(_.toInt).toFastSeq).toDenseMatrix,
    )
  }

  @Test def testBlockMatrixSlice(): Unit = {
    val nRows = 12
    val nCols = 8
    val original = BDM.tabulate[Double](nRows, nCols)((i, j) => i.toDouble * nCols + j)
    val unsliced = toIR(original, blockSize = 3)

    val rowSlice = FastSeq(1L, 10L, 3L)
    val colSlice = FastSeq(4L, 8L, 2L)
    assertBMEvalsTo(
      BlockMatrixSlice(unsliced, FastSeq(rowSlice, colSlice)),
      original(
        Array.range(rowSlice(0).toInt, rowSlice(1).toInt, rowSlice(2).toInt).toFastSeq,
        Array.range(colSlice(0).toInt, colSlice(1).toInt, colSlice(2).toInt).toFastSeq,
      ).toDenseMatrix,
    )
  }

  @Test def testBlockMatrixDot(): Unit = {
    val m1 = BDM.tabulate[Double](5, 4)((i, j) => (i.toDouble + 1) * j)
    val m2 = BDM.tabulate[Double](4, 6)((i, j) => (i.toDouble + 5) * (j - 2))
    assertBMEvalsTo(BlockMatrixDot(toIR(m1), toIR(m2)), m1 * m2)
  }

  @Test def testBlockMatrixRandom(): Unit = {
    val gaussian = BlockMatrixRandom(0, gaussian = true, shape = Array(5L, 6L), blockSize = 3)
    val uniform = BlockMatrixRandom(0, gaussian = false, shape = Array(5L, 6L), blockSize = 3)

    val l = Ref(freshName(), TFloat64)
    val r = Ref(freshName(), TFloat64)
    assertBMEvalsTo(
      BlockMatrixMap2(gaussian, gaussian, l.name, r.name, l - r, NeedsDense),
      BDM.fill(5, 6)(0.0),
    )
    assertBMEvalsTo(
      BlockMatrixMap2(uniform, uniform, l.name, r.name, l - r, NeedsDense),
      BDM.fill(5, 6)(0.0),
    )
  }

  @Test def readBlockMatrixIR(): Unit = {
    implicit val execStrats: Set[ExecStrategy] = ExecStrategy.compileOnly
    val etype = EBlockMatrixNDArray(EFloat64Required, required = true)
    val path =
      getTestResource(
        "blockmatrix_example/0/parts/part-0-28-0-0-0feb7ac2-ab02-6cd4-5547-bfcb94dacb33"
      )
    val matrix = BlockMatrix.read(ctx, getTestResource("blockmatrix_example/0")).toBreezeMatrix()
    val expected = Array.tabulate(2)(i => Array.tabulate(2)(j => matrix(i, j)).toFastSeq).toFastSeq

    val typ = TNDArray(TFloat64, Nat(2))
    val spec = TypedCodecSpec(etype, typ, BlockMatrix.bufferSpec)
    val reader = ETypeValueReader(spec)
    val read = ReadValue(Str(path), reader, typ)
    assertNDEvals(read, expected)
    assertNDEvals(
      ReadValue(
        WriteValue(
          read,
          Str(ctx.createTmpPath("read-blockmatrix-ir", "hv")) + UUID4(),
          ETypeValueWriter(spec),
        ),
        reader,
        typ,
      ),
      expected,
    )
  }

  @Test def readWriteBlockMatrix(): Unit = {
    val original = getTestResource("blockmatrix_example/0")
    val expected = BlockMatrix.read(ctx, original).toBreezeMatrix()

    val path = ctx.createTmpPath("read-blockmatrix-ir", "bm")

    assertEvalsTo(
      BlockMatrixWrite(
        BlockMatrixRead(BlockMatrixNativeReader(ctx.fs, original)),
        BlockMatrixNativeWriter(path, overwrite = true, forceRowMajor = false, stageLocally = false),
      ),
      (),
    )

    assertBMEvalsTo(BlockMatrixRead(BlockMatrixNativeReader(ctx.fs, path)), expected)
  }

  @Test def testBlockMatrixDensify(): Unit = {
    val dense = fill(1, nRows = 10, nCols = 10, blockSize = 5)
    val sparse = BlockMatrixSparsify(dense, PerBlockSparsifier(FastSeq(0, 1, 2)))
    val densified = BlockMatrixDensify(sparse)
    val expected = BDM.fill(10, 10)(1.0)
    expected(5 until 10, 5 until 10) := BDM.fill(5, 5)(0.0): Unit
    assertBMEvalsTo(densified, expected)
  }
}
