package is.hail.expr.ir

import is.hail.backend.ExecuteContext

import org.junit.jupiter.api.Test

class StoredDatasetSuite {

  private def datasetWith(files: String*)(implicit ctx: ExecuteContext): String = {
    val path = ctx.createTmpPath("stored-dataset", "ht")
    ctx.fs.mkDir(path)
    files.foreach(file => ctx.fs.touch(s"$path/$file"))
    path
  }

  @Test def testNativeFormatIsCommittedBySuccessFile(implicit ctx: ExecuteContext): Unit =
    assert(StoredDataset.isCommitted(ctx.fs, datasetWith("_SUCCESS")))

  @Test def testColumnarFormatIsCommittedByManifestRoot(implicit ctx: ExecuteContext): Unit =
    assert(StoredDataset.isCommitted(ctx.fs, datasetWith("manifest.json")))

  @Test def testDirectoryWithoutCommitMarkerIsNotCommitted(implicit ctx: ExecuteContext): Unit =
    assert(!StoredDataset.isCommitted(ctx.fs, datasetWith("metadata.json.gz")))

  @Test def testMissingPathIsNotCommitted(implicit ctx: ExecuteContext): Unit =
    assert(!StoredDataset.isCommitted(ctx.fs, ctx.createTmpPath("stored-dataset", "ht")))
}
