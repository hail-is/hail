package is.hail.fs

import java.io.FileNotFoundException
import is.hail.HailSuite
import is.hail.backend.ExecuteContext
import is.hail.io.fs.{FS, FileStatus}
import is.hail.utils._
import org.apache.commons.io.IOUtils
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
    println(s"actual path: ${s.getPath}")
    println(s"expected path: $f")
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
    println(s"expected root to be $s")
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
    assert(pathsRelResourcesRoot(statuses) == Set("/a", "/adir", "/az"),
      s"${statuses} ${pathsRelResourcesRoot(statuses)} ${Set("/a", "/adir", "/az")}")
  }

  @Test def testGlobMatchDir(): Unit = {
    val statuses = fs.glob(r("/*dir/x"))
    assert(pathsRelResourcesRoot(statuses) == Set("/adir/x", "/dir/x"),
      s"${statuses} ${pathsRelResourcesRoot(statuses)} ${Set("/adir/x", "/dir/x")}")
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

    println("creating a thing")
    using(fs.createNoCompression(f)) { os =>
      println("getting some bytes")
      val b = s.getBytes
      println(s"writing ${b.length} bytes")
      println("writing some bytes")
      os.write(b)
      println("finished writing bytes")
    }

    assert(fs.exists(f))

    using(fs.openNoCompression(f)) { is =>
      println("getting an inputstream")
      val read = new String(IOUtils.toByteArray(is))
      println("asserting contents are the same")
      println(s"Expected: $s")
      println(s"$read")
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

  @Test def testGetCodecExtension(): Unit = {
    assert(fs.getCodecExtension("foo.vcf.bgz") == ".bgz")
  }

  @Test def testStripCodecExtension(): Unit = {
    assert(fs.stripCodecExtension("foo.vcf.bgz") == "foo.vcf")
  }

  @Test def testReadWriteBytes(): Unit = {
    val f = t()

    println(s"255 as a byte is ${255.toByte}")

    println("creating some stuff")
    using(fs.create(f)) { os =>
      println("writing first")
      os.write(1)
      println("writing second")
      os.write(127)
      println("writing third")
      os.write(255)
    }

    println("checking existence")
    assert(fs.exists(f))

    println("opening a thing")
    using(fs.open(f)) { is =>
      println("reading first")
      assert(is.read() == 1)
      println("reading second")
      assert(is.read() == 127)
      println("reading third")
      val third = is.read()
      println(s"third is $third")
      assert(third == 255)
    }

    println("deleting a thing")
    fs.delete(f, false)

    println("checking non-existence")
    assert(!fs.exists(f))
  }
}

class HadoopFSSuite extends HailSuite with FSSuite {
  val root: String = "file:/"

  lazy val fsResourcesRoot: String = "file:" + new java.io.File("./src/test/resources/fs").getCanonicalPath

  lazy val tmpdir: String = ctx.tmpdir
}
