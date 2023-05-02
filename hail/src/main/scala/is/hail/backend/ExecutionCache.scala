package is.hail.backend

import is.hail.expr.ir.analyses.SemanticHash.Hash
import is.hail.io.fs.FS

import java.io.OutputStream
import java.util.concurrent.ConcurrentHashMap
import scala.collection.mutable
import scala.io.Source
import scala.util.Using

case object ExecutionCache {
  def fsCache(fs: FS, cachedir: String): ExecutionCache = {
    assert(fs.validUrl(cachedir))
    FSExecutionCache(fs, cachedir)
  }

  def forTesting: ExecutionCache = new ExecutionCache {
    val storage = new ConcurrentHashMap[Hash.Type, IndexedSeq[(Int, Array[Byte])]]

    override def lookup(s: Hash.Type): IndexedSeq[(Int, Array[Byte])] =
      storage.getOrDefault(s, mutable.IndexedSeq.empty)

    override def put(s: Hash.Type, r: IndexedSeq[(Int, Array[Byte])]): Unit =
      storage.put(s, r)
  }
}

trait ExecutionCache {
  def lookup(s: Hash.Type): IndexedSeq[(Int, Array[Byte])]
  def put(s: Hash.Type, r: IndexedSeq[(Int, Array[Byte])]): Unit
}

private case class FSExecutionCache(fs: FS, cacheDir: String) extends ExecutionCache {

  override def lookup(s: Hash.Type): IndexedSeq[(Int, Array[Byte])] =
    Using(fs.open(at(s))) {
      Source.fromInputStream(_).getLines().map(Line.read).toIndexedSeq
    }.getOrElse(IndexedSeq.empty)

  override def put(s: Hash.Type, r: IndexedSeq[(Int, Array[Byte])]): Unit =
    fs.write(at(s)) { ostream => r.foreach(Line.write(_, ostream)) }

  def at(s: Hash.Type): String =
    s"$cacheDir/$s"

  case object Line {
    private type Type = (Hash.Type, Array[Byte])
    def write(line: Type, ostream: OutputStream): Unit = {
      assert(!line._2.contains('\n'))
      ostream.write(line._1)
      ostream.write(", ".getBytes)
      ostream.write(line._2)
      ostream.write("\n".getBytes)
    }

    def read(string: String): Type = {
      val Array(index, bytes) = string.split(", ")
      (index.toInt, bytes.getBytes)
    }
  }
}

