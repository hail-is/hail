package is.hail.fs.gs

import java.io.FileInputStream
import is.hail.fs.BlobStorageFSSuite
import is.hail.io.fs.GoogleStorageFS
import is.hail.utils._
import org.apache.commons.io.IOUtils
import org.scalatest.testng.TestNGSuite
import org.testng.annotations.{BeforeClass, Test}
import org.testng.SkipException

class GoogleStorageFSSuite extends BlobStorageFSSuite {
  @BeforeClass
  def beforeclass(): Unit = {
    if (System.getenv("HAIL_CLOUD") != "gcp") {
      throw new SkipException("This test suite is only run in GCP.");
    } else {
      assert(root != null)
      assert(fsResourcesRoot != null)
    }
  }

  lazy val fs = {
    val gac = System.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    if (gac == null) {
      new GoogleStorageFS()
    } else {
      new GoogleStorageFS(
        Some(new String(IOUtils.toByteArray(new FileInputStream(gac)))))
    }
  }

  @Test def testMakeQualified(): Unit = {
    val qualifiedFileName = "gs://bucket/path"
    assert(fs.makeQualified(qualifiedFileName) == qualifiedFileName)

    val unqualifiedFileName = "not-gs://bucket/path"
    try {
      fs.makeQualified(unqualifiedFileName)
    }
    catch {
      case _: IllegalArgumentException =>
        return
    }
    assert(false)
  }

  @Test
  def deleteManyFiles(): Unit = {
    val prefix = s"$hail_test_storage_uri/google-storage-fs-suite/delete-many-files/${ java.util.UUID.randomUUID() }"
    for (i <- 0 until 2000) {
      fs.touch(s"$prefix/$i")
    }
    assert(fs.exists(prefix))
    fs.delete(prefix, recursive = true)
    assert(!fs.exists(prefix), s"files not deleted:\n${ fs.listDirectory(prefix).map(_.getPath).mkString("\n") }")
  }

  @Test def testSeekAfterEOF(): Unit = {
    val prefix = s"$hail_test_storage_uri/google-storage-fs-suite/delete-many-files/${ java.util.UUID.randomUUID() }"
    val p = s"$prefix/seek_file"
    using(fs.createCachedNoCompression(p)) { os =>
      os.write(1.toByte)
      os.write(2.toByte)
      os.write(3.toByte)
      os.write(4.toByte)
    }

    using(fs.openNoCompression(p)) { is =>
      assert(is.read() == 1.toByte)
      is.seek(3)
      assert(is.read() == 4.toByte)
      assert(is.read() == (-1).toByte)
      is.seek(0)
      assert(is.read() == 1.toByte)
    }
  }
}
