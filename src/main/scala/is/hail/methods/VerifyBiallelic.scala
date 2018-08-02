package is.hail.methods

import is.hail.expr.ir._
import is.hail.expr.types.TInt32
import is.hail.variant.MatrixTable

object VerifyBiallelic {
  def apply(vsm: MatrixTable, method: String): MatrixTable = {
    new MatrixTable(vsm.hc, MatrixMapRows(vsm.ast,
      If(ApplyComparisonOp(NEQ(TInt32()), ArrayLen(GetField(Ref("va", vsm.rvRowType), "alleles")), I32(2)),
        // TODO: message should include key when we can put IR in Die
        Die(s"'$method' expects biallelic variants ('alleles' field has length 2)", vsm.rvRowType),
        Ref("va", vsm.rvRowType)), None))
  }
}
