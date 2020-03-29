package is.hail.backend.service

import java.io.{FileInputStream, FileReader}

import com.google.auth.oauth2.ServiceAccountCredentials
import com.google.cloud.storage.StorageOptions
import is.hail.backend.{Backend, BroadcastValue}
import is.hail.expr.ir.ExecuteContext
import is.hail.io.bgen.IndexBgen
import is.hail.io.fs.FS
import is.hail.utils._

import scala.collection.JavaConverters._
import scala.collection.mutable
import scala.reflect.ClassTag

class GoogleFS extends FS {

}

object ServiceBackend {
  def main(args: Array[String]): Unit = {
    val storage = StorageOptions.newBuilder()
      .setCredentials(
        ServiceAccountCredentials.fromStream(new FileInputStream("/home/cotton/cseed-gsa-key.json")))
      .build()
      .getService()

    val bucket = storage.get("hail-cseed-k0ox4")

    val blob = bucket.get("hello.txt")

    val content = new String(blob.getContent())

    println(s"content $content")
  }
}

class User(
  username: String,
  fs: GoogleFS)

class ServiceBackend extends Backend {
  def broadcast[T: ClassTag](_value: T): BroadcastValue[T] = new BroadcastValue[T] {
    def value: T = _value
  }

  def parallelizeAndComputeWithIndex[T: ClassTag, U : ClassTag](collection: Array[T])(f: (T, Int) => U): Array[U] = {
    collection.zipWithIndex.map { case (t, i) => f(t, i) }
  }

  def stop(): Unit = ()

  private[this] users: mutable.Map[String, User]

  def fs: FS = ???

  def pyIndexBgen(
    files: java.util.List[String],
    indexFileMap: java.util.Map[String, String],
    rg: Option[String],
    contigRecoding: java.util.Map[String, String],
    skipInvalidLoci: Boolean) {
    ExecuteContext.scoped(this, fs) { ctx =>
      IndexBgen(ctx, files.asScala.toArray, indexFileMap.asScala.toMap, rg, contigRecoding.asScala.toMap, skipInvalidLoci)
    }
    info(s"Number of BGEN files indexed: ${ files.size() }")
  }
}
