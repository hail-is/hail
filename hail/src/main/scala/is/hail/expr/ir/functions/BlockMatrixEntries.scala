package is.hail.expr.ir.functions
import is.hail.HailContext
import is.hail.expr.ir.TableValue
import is.hail.expr.types.virtual.{TFloat64Optional, TInt64Optional, TStruct}
import is.hail.expr.types.{BlockMatrixType, TableType}
import is.hail.linalg.BlockMatrix

case class BlockMatrixEntries() extends BlockMatrixToTableFunction {
  override def typ(childType: BlockMatrixType): TableType = {
    //Copied from the BlockMatrix.entriesTable implementation
    val rvType = TStruct("i" -> TInt64Optional, "j" -> TInt64Optional, "entry" -> TFloat64Optional)
    TableType(rvType, Array[String](), TStruct.empty())
  }

  override def execute(bm: BlockMatrix): TableValue = bm.entriesTable(HailContext.get).value
}
