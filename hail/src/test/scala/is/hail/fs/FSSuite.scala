package is.hail.fs

import java.io.FileNotFoundException
import is.hail.fs.azure.AzureStorageFSSuite
import is.hail.HailSuite
import is.hail.backend.ExecuteContext
import is.hail.io.fs.FSUtil.dropTrailingSlash
import is.hail.io.fs.{FS, FileStatus, GoogleStorageFS, Seekable}
import is.hail.utils._
import org.apache.commons.codec.binary.Hex
import org.apache.commons.io.IOUtils
import org.scalatest.Inspectors.forAll
import org.scalatest.testng.TestNGSuite
import org.testng.annotations.Test

import java.nio.charset.StandardCharsets

trait FSSuite extends TestNGSuite {
  def root: String

  def fsResourcesRoot: String

  def fs: FS

  def tmpdir: String

  /* Structure of src/test/resources/fs:
     /a
     /adir
     /adir/x
     /az
     /dir
     /dir/x
     /zzz
   */
  def r(s: String): String = s"$fsResourcesRoot$s"

  def t(extension: String = null): String = ExecuteContext.createTmpPathNoCleanup(tmpdir, "fs-suite-tmp", extension)

  def pathsRelRoot(root: String, statuses: Array[FileStatus]): Set[String] = {
    statuses.map { status =>
      var p = status.getPath
      assert(p.startsWith(root), s"$p $root")
      p.drop(root.length)
    }.toSet
  }

  def pathsRelResourcesRoot(statuses: Array[FileStatus]): Set[String] = pathsRelRoot(fsResourcesRoot, statuses)

  @Test def testExists(): Unit = {
    assert(fs.exists(r("/a")))

    assert(fs.exists(r("/zzz")))
    assert(!fs.exists(r("/z"))) // prefix
  }

  @Test def testFileChecksum(): Unit = {
    val digest = fs.fileChecksum(s"$fsResourcesRoot/a")
    assert(new String(digest, StandardCharsets.UTF_8) == "yR+pThTPL6VHS+zv94jA3A==")
  }
}

class HadoopFSSuite extends HailSuite with FSSuite {
  val root: String = "file:/"

  lazy val fsResourcesRoot: String = "file:" + new java.io.File("./src/test/resources/fs").getCanonicalPath

  lazy val tmpdir: String = ctx.tmpdir
}
