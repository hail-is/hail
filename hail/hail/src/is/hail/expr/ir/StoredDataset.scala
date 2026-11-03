package is.hail.expr.ir

import is.hail.io.fs.FS

object StoredDataset {

  /** Whether the table or matrix table at `path` was completely written.
    *
    * The native format commits by writing `_SUCCESS`. The columnar format writes no `_SUCCESS`; it
    * commits by writing its manifest root last.
    */
  def isCommitted(fs: FS, path: String): Boolean =
    fs.isFile(s"$path/_SUCCESS") || fs.isFile(s"$path/manifest.json")
}
