package is.hail.fs

import java.io.{FileInputStream, FileNotFoundException}

import is.hail.HailSuite
import is.hail.io.fs.{FS, FileStatus, GoogleStorageFS}
import is.hail.utils._
import org.apache.commons.io.IOUtils
import org.scalatest.testng.TestNGSuite
import org.testng.annotations.Test

trait FSSuite {
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

  def t(extension: String = ""): String = fs.getTemporaryFile(tmpdir, suffix = Some(extension))

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

    assert(fs.exists(r("/dir")))
    assert(fs.exists(r("/dir/")))

    assert(!fs.exists(r("/does_not_exist")))
    assert(!fs.exists(r("/does_not_exist_dir/")))
  }

  @Test def testFileStatusOnFile(): Unit = {
    // file
    val f = r("/a")
    val s = fs.fileStatus(f)
    assert(s.getPath == f)
    assert(s.isFile)
    assert(!s.isDirectory)
    assert(s.getLen == 12)
  }

  @Test def testFileStatusOnDir(): Unit = {
    // file
    val f = r("/dir")
    val s = fs.fileStatus(f)
    assert(s.getPath == f)
    assert(!s.isFile)
    assert(s.isDirectory)
  }

  @Test def testFileStatusOnDirWithSlash(): Unit = {
    // file
    val f = r("/dir/")
    val s = fs.fileStatus(f)
    assert(s.getPath == f.dropRight(1))
    assert(!s.isFile)
    assert(s.isDirectory)
  }

  @Test def testFileStatusOnMissingFile(): Unit = {
    try {
      fs.fileStatus(r("/does_not_exist"))
    } catch {
      case _: FileNotFoundException =>
        return
    }
    assert(false)
  }

  @Test def testFileStatusRoot(): Unit = {
    val s = fs.fileStatus(root)
    assert(s.getPath == root)
  }

  @Test def testFileStatusRootWithSlash(): Unit = {
    if (root.endsWith("/"))
      return

    val s = fs.fileStatus(s"$root/")
    assert(s.getPath == root)
  }

  @Test def testDeleteRecursive(): Unit = {
    val d = t()
    fs.mkDir(d)
    fs.touch(s"$d/x")
    fs.touch(s"$d/y")
    fs.mkDir(s"$d/subdir")
    fs.touch(s"$d/subdir/z")

    assert(fs.exists(s"$d/subdir/z"))

    fs.delete(d, recursive = true)

    assert(!fs.exists(s"$d"))
    assert(!fs.exists(s"$d/subdir/z"))
  }

  @Test def testListStatusDir(): Unit = {
    val statuses = fs.listStatus(r(""))
    assert(pathsRelResourcesRoot(statuses) == Set("/a", "/adir", "/az", "/dir", "/zzz"))
  }

  @Test def testListStatusDirWithSlash(): Unit = {
    val statuses = fs.listStatus(r("/"))
    assert(pathsRelResourcesRoot(statuses) == Set("/a", "/adir", "/az", "/dir", "/zzz"))
  }

  @Test def testGlobOnDir(): Unit = {
    val statuses = fs.glob(r(""))
    assert(pathsRelResourcesRoot(statuses) == Set(""))
  }

  @Test def testGlobMissingFile(): Unit = {
    val statuses = fs.glob(r("/does_not_exist_dir/does_not_exist"))
    assert(pathsRelResourcesRoot(statuses) == Set())
  }
  
  @Test def testGlobFilename(): Unit = {
    val statuses = fs.glob(r("/a*"))
    assert(pathsRelResourcesRoot(statuses) == Set("/a", "/adir", "/az"))
  }

  @Test def testGlobMatchDir(): Unit = {
    val statuses = fs.glob(r("/*dir/x"))
    assert(pathsRelResourcesRoot(statuses) == Set("/adir/x", "/dir/x"))
  }

  @Test def testGlobRoot(): Unit = {
    val statuses = fs.glob(root)
    // empty with respect to root (self)
    assert(pathsRelRoot(root, statuses) == Set(""))
  }

  @Test def testGlobRootWithSlash(): Unit = {
    if (root.endsWith("/"))
      return

    val statuses = fs.glob(s"$root/")
    assert(pathsRelRoot(root, statuses) == Set(""))
  }

  @Test def testWriteRead(): Unit = {
    val s = "this is a test string"
    val f = t()

    using(fs.createNoCompression(f)) { os =>
      val b = s.getBytes
      os.write(b)
    }

    assert(fs.exists(f))

    using(fs.openNoCompression(f)) { is =>
      val read = new String(IOUtils.toByteArray(is))
      assert(read == s)
    }

    fs.delete(f, false)

    assert(!fs.exists(f))
  }

  @Test def testWriteReadCompressed(): Unit = {
    val s = "this is a test string"
    val f = t(extension = ".bgz")

    using(fs.create(f)) { os =>
      val b = s.getBytes
      os.write(b)
    }

    assert(fs.exists(f))

    using(fs.open(f)) { is =>
      val read = new String(IOUtils.toByteArray(is))
      assert(read == s)
    }

    fs.delete(f, false)

    assert(!fs.exists(f))
  }

  def testGetCodecExtension(): Unit = {
    assert(fs.getCodecExtension("foo.vcf.bgz") == ".bgz")
  }

  def testStripCodecExtension(): Unit = {
    assert(fs.getCodecExtension("foo.vcf.bgz") == "foo.vcf")
  }
}

class TestHadoopFS extends HailSuite with FSSuite {
  val root: String = "file:/"
  
  lazy val fsResourcesRoot: String = "file:" + new java.io.File("./src/test/resources/fs").getCanonicalPath
  
  lazy val tmpdir: String = tmpDir.createTempFile()
}

class TestGoogleStorageFS extends TestNGSuite with FSSuite {
  val bucket: String = "hail-test-dmk9z"
  
  val root: String = s"gs://$bucket"
  
  val fsResourcesRoot: String = System.getenv("HAIL_GS_FS_TEST_RESOURCES")

  private val keyFile = "/test-gsa-key/key.json"
  
  lazy val fs = new GoogleStorageFS(
    new String(IOUtils.toByteArray(new FileInputStream(keyFile))))

  lazy val tmpdir: String = s"gs://$bucket/tmp"

  @Test def testDropTailingSlash(): Unit = {
    import GoogleStorageFS._

    assert(dropTrailingSlash("") == "")
    assert(dropTrailingSlash("/foo/bar") == "/foo/bar")
    assert(dropTrailingSlash("foo/bar/") == "foo/bar")
    assert(dropTrailingSlash("/foo///") == "/foo")
    assert(dropTrailingSlash("///") == "")
  }
}
